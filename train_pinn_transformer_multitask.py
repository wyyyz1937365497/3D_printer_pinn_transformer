# train_pinn_transformer_multitask.py
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import time
import gc
import argparse
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import signal
import atexit
import warnings
warnings.filterwarnings('ignore')
import os
import torch

# Windows多GPU优化配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 指定使用两张卡
os.environ["NCCL_P2P_DISABLE"] = "1"         # Windows下NCCL优化
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"     # 非阻塞模式

# 检查GPU可用性
# print(f"可用GPU数量: {torch.cuda.device_count()}")
# for i in range(torch.cuda.device_count()):
#     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
#     print(f"  显存: {torch.cuda.get_device_properties(i).total_memory/1024**3:.1f}GB")


# ==================== 配置参数 ====================
class Config:
    def __init__(self):
        self.data_path = 'printer_dataset/nozzle_simulation_gear_print.csv'
        self.seq_len = 200          # 历史窗口长度（200ms @ 1ms步长）
        self.pred_len = 50          # 预测长度（50ms）
        self.batch_size = 1024
        self.gradient_accumulation_steps = 2
        self.model_dim = 256
        self.num_heads = 8
        self.num_layers = 6
        self.dim_feedforward = 1024
        self.dropout = 0.1
        self.lr = 1e-4
        self.epochs = 50
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_workers = 4
        self.max_samples = 500000
        self.lambda_physics = 0.1   # 物理损失权重
        self.lambda_classification = 1.0  # 分类损失权重
        self.lambda_rul = 0.5       # RUL损失权重
        self.warmup_epochs = 5
        self.checkpoint_dir = './checkpoints_multitask'
        self.resume_from = None
        self.save_on_exit = True
        self.save_interval = 5
        self.start_epoch = 0
        self.load_optimizer_state = True
        self.pin_memory = True
        # 列定义
        self.ctrl_cols = ['ctrl_T_target', 'ctrl_speed_set', 'ctrl_pos_x', 'ctrl_pos_y', 'ctrl_pos_z']
        self.state_cols = ['temperature_C', 'vibration_disp_x_m', 'vibration_disp_y_m', 
                          'vibration_vel_x_m_s', 'vibration_vel_y_m_s', 
                          'motor_current_x_A', 'motor_current_y_A', 'motor_current_z_A',
                          'pressure_bar', 'nozzle_pos_x_mm', 'nozzle_pos_y_mm', 'nozzle_pos_z_mm',
                          'print_quality']
        self.target_cols = ['fault_label', 'fault_type', 'print_quality']
        
        # 维度定义
        self.input_dim = len(self.ctrl_cols) + len(self.state_cols) + 1  # +1 for hour feature
        self.output_dim = len(self.state_cols)
        self.ctrl_dim = len(self.ctrl_cols)
        self.class_dim = 4  # 3种故障类型 + 正常
        self.rul_dim = 1    # RUL预测

# ==================== 位置编码 ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ==================== 多任务PINN-Transformer模型 ====================
class PrinterPINN_MultiTask(nn.Module):
    def __init__(self, config):
        super(PrinterPINN_MultiTask, self).__init__()
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.ctrl_dim = config.ctrl_dim
        self.d_model = config.model_dim
        self.pred_len = config.pred_len
        self.class_dim = config.class_dim
        self.rul_dim = config.rul_dim
        
        # 共享编码器
        self.encoder_embedding = nn.Linear(self.input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # 解码器（用于物理场重构）
        self.decoder_embedding = nn.Linear(self.ctrl_dim, self.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.fc_out = nn.Linear(self.d_model, self.output_dim)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.class_dim)
        )
        
        # RUL回归头
        self.rul_predictor = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.rul_dim),
            nn.ReLU()  # RUL应为正数
        )
        
    def forward(self, src, tgt_ctrl):
        # 共享编码器
        src_emb = self.encoder_embedding(src)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb)  # [batch, seq_len, d_model]
        
        # 物理场重构解码器
        tgt_emb = self.decoder_embedding(tgt_ctrl)
        tgt_emb = self.pos_encoder(tgt_emb)
        decoder_output = self.decoder(tgt_emb, memory)
        physics_pred = self.fc_out(decoder_output)  # [batch, pred_len, output_dim]
        
        # 使用编码器的最终状态进行分类和RUL预测
        # 取序列的最后一个时间步
        last_hidden = memory[:, -1, :]  # [batch, d_model]
        
        # 故障分类
        class_pred = self.classifier(last_hidden)  # [batch, class_dim]
        
        # RUL预测
        rul_pred = self.rul_predictor(last_hidden)  # [batch, 1]
        
        return {
            'physics_pred': physics_pred,
            'class_pred': class_pred,
            'rul_pred': rul_pred,
            'memory': memory
        }
    
    def physics_loss(self, outputs, targets, device='cuda'):
        """物理约束损失（针对3D打印喷头动力学）"""
        physics_pred = outputs['physics_pred']
        y_true = targets
        
        loss = 0.0
        batch_size, seq_len, _ = physics_pred.shape
        
        # 1. 热传导方程约束（温度变化应平滑）
        temp_pred = physics_pred[:, :, 0]  # temperature_C
        if seq_len > 1:
            dT_dt = torch.diff(temp_pred, dim=1) / 0.001  # 1ms步长
            if dT_dt.shape[1] > 1:  # 确保有足够的元素进行二次微分
                d2T_dt2 = torch.diff(dT_dt, dim=1) / 0.001
                # 温度加速度应有限（避免不合理的剧烈变化）
                thermal_loss = torch.mean(torch.abs(d2T_dt2))
                # 添加阈值防止无穷大
                thermal_loss = torch.clamp(thermal_loss, max=1e3)
                loss += thermal_loss
        
        # 2. 振动动力学约束（质量-弹簧-阻尼系统）
        if seq_len > 1:
            disp_x_pred = physics_pred[:, :, 1]  # vibration_disp_x_m
            disp_y_pred = physics_pred[:, :, 2]  # vibration_disp_y_m
            vel_x_pred = physics_pred[:, :, 3]   # vibration_vel_x_m_s
            vel_y_pred = physics_pred[:, :, 4]   # vibration_vel_y_m_s
            
            # 从位移计算速度（应与预测的速度一致）
            dt = 0.001  # 1ms
            if disp_x_pred.shape[1] > 1 and disp_y_pred.shape[1] > 1:
                vel_x_from_disp = torch.diff(disp_x_pred, dim=1) / dt
                vel_y_from_disp = torch.diff(disp_y_pred, dim=1) / dt
                
                # 速度一致性损失
                vibration_loss = torch.mean((vel_x_from_disp - vel_x_pred[:, :-1])**2) + \
                                torch.mean((vel_y_from_disp - vel_y_pred[:, :-1])**2)
                # 添加阈值防止无穷大
                vibration_loss = torch.clamp(vibration_loss, max=1e3)
                loss += vibration_loss
        
        # 3. 能量守恒约束（简化的）
        if seq_len > 1:
            vel_x_pred = physics_pred[:, :, 3]   # vibration_vel_x_m_s
            vel_y_pred = physics_pred[:, :, 4]   # vibration_vel_y_m_s
            kinetic_energy = vel_x_pred**2 + vel_y_pred**2
            if kinetic_energy.shape[1] > 1:
                d_energy_dt = torch.diff(kinetic_energy, dim=1) / dt
                energy_loss = torch.mean(torch.abs(d_energy_dt))
                # 添加阈值防止无穷大
                energy_loss = torch.clamp(energy_loss, max=1e2)
                loss += 0.1 * energy_loss
        
        # 4. 电机电流-振动耦合约束
        if seq_len > 1:
            current_x_pred = physics_pred[:, :, 5]  # motor_current_x_A
            current_y_pred = physics_pred[:, :, 6]  # motor_current_y_A
            
            # 电流应与加速度相关（F=ma，而F与电流成正比）
            dt = 0.001  # 1ms
            if vel_x_pred.shape[1] > 1 and vel_y_pred.shape[1] > 1:
                accel_x_pred = torch.diff(vel_x_pred, dim=1) / dt
                accel_y_pred = torch.diff(vel_y_pred, dim=1) / dt
                
                if accel_x_pred.shape[1] > 0 and accel_y_pred.shape[1] > 0:
                    current_accel_corr_x = torch.mean(current_x_pred[:, :-1] * accel_x_pred)
                    current_accel_corr_y = torch.mean(current_y_pred[:, :-1] * accel_y_pred)
                    
                    # 确保相关性合理（避免完全不相关的预测）
                    coupling_loss = torch.abs(1.0 - torch.abs(current_accel_corr_x)) + \
                                   torch.abs(1.0 - torch.abs(current_accel_corr_y))
                    # 添加阈值防止无穷大
                    coupling_loss = torch.clamp(coupling_loss, max=1e2)
                    loss += 0.2 * coupling_loss
        
        return loss

# ==================== 数据处理器 ====================
class MultiTaskDataProcessor:
    def __init__(self, data_path, seq_len, pred_len, max_samples, config):
        self.data_path = data_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.max_samples = max_samples
        self.config = config
        
        print(f"🔄 开始处理多任务数据...")
        print(f"📊 历史长度: {seq_len}, 预测长度: {pred_len}")
        self.process_data()
    
    def process_data(self):
        """处理数据用于多任务训练"""
        df = pd.read_csv(self.data_path)
        print(f"✅ 原始数据加载: {df.shape}")
        
        # 转换为数值类型
        numeric_cols = self.config.ctrl_cols + self.config.state_cols + ['fault_label', 'fault_type', 'timestamp']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
        
        # 特征工程：添加时间特征
        df['hour'] = (df['timestamp'] % 3600) / 3600  # 小时周期
        
        # 选择相关列
        all_cols = self.config.ctrl_cols + self.config.state_cols + ['hour']
        target_cols = ['fault_label', 'fault_type', 'print_quality']
        
        grouped = df.groupby('machine_id')
        samples = []
        count = 0
        
        print("📊 收集样本索引...")
        for machine_id, group in grouped:
            group = group.sort_values('timestamp').reset_index(drop=True)
            
            # 数据数组
            data_array = group[all_cols].values
            ctrl_array = group[self.config.ctrl_cols].values
            target_array = group[target_cols].values
            
            total_len = len(group)
            required_len = self.seq_len + self.pred_len
            
            if total_len < required_len:
                continue
            
            n_windows = total_len - required_len + 1
            
            for i in range(n_windows):
                if count >= self.max_samples:
                    break
                
                # 检查窗口内是否有故障
                window_fault = target_array[i:i+required_len, 0]  # fault_label
                
                # 如果窗口内有故障，只在故障发生后的窗口使用
                if np.any(window_fault == 1):
                    fault_indices = np.where(window_fault == 1)[0]
                    first_fault_idx = fault_indices[0]
                    if first_fault_idx < self.seq_len:  # 故障在历史窗口内
                        continue
                
                # 提取样本
                x_hist = data_array[i:i+self.seq_len]
                x_future_ctrl = ctrl_array[i+self.seq_len:i+required_len]
                y_future_state = data_array[i+self.seq_len:i+required_len, len(self.config.ctrl_cols):len(self.config.ctrl_cols)+len(self.config.state_cols)]
                y_targets = target_array[i+self.seq_len:i+required_len]
                
                # 计算RUL（剩余使用寿命）
                # 简化：如果当前无故障，RUL为到故障发生的时间；如果有故障，RUL为0
                current_fault = target_array[i+self.seq_len, 0]  # 预测起点的故障状态
                if current_fault == 0:
                    future_faults = target_array[i+self.seq_len:, 0]
                    fault_indices = np.where(future_faults == 1)[0]
                    if len(fault_indices) > 0:
                        first_fault_idx = fault_indices[0]
                        rul = first_fault_idx * 0.001  # 转换为秒
                    else:
                        rul = 3600  # 默认1小时
                else:
                    rul = 0
                
                # RUL归一化（简化）
                rul_normalized = min(rul, 3600) / 3600
                
                samples.append({
                    'x_hist': x_hist,
                    'x_future_ctrl': x_future_ctrl,
                    'y_future_state': y_future_state,
                    'y_fault_label': y_targets[0, 0],  # 当前步的故障标签
                    'y_fault_type': y_targets[0, 1],   # 当前步的故障类型
                    'y_rul': rul_normalized
                })
                
                count += 1
                if count % 10000 == 0:
                    print(f"  已收集 {count} 个样本...")
                
                if count >= self.max_samples:
                    break
            
            if count >= self.max_samples:
                break
        
        self.total_samples = len(samples)
        self.split_idx = int(self.total_samples * 0.8)
        train_samples = samples[:self.split_idx]
        val_samples = samples[self.split_idx:]
        
        print(f"📊 总样本数: {self.total_samples}")
        print(f"   训练集: {len(train_samples)}, 验证集: {len(val_samples)}")
        
        # 计算统计量（仅使用训练集）
        print("📊 计算统计量...")
        all_x_hist = np.array([s['x_hist'] for s in train_samples])
        
        self.mean_X = all_x_hist.mean(axis=(0, 1))
        self.std_X = all_x_hist.std(axis=(0, 1))
        self.std_X[self.std_X < 1e-8] = 1.0
        
        print(f"   Input Mean: {self.mean_X}")
        print(f"   Input Std: {self.std_X}")
        
        # 更新配置中的input_dim以反映实际维度
        self.config.input_dim = all_x_hist.shape[2]  # 应该是19（18个特征+1小时特征）
        print(f"   实际输入维度: {self.config.input_dim}")
        
        # 准备训练和验证数据
        self.prepare_datasets(train_samples, val_samples)
        
        # 保存归一化参数
        self.save_normalization_params()
        
        print(f"✅ 数据处理完成！")
    
    def prepare_datasets(self, train_samples, val_samples):
        """准备训练和验证数据集"""
        def prepare_batch(samples):
            X_hist = np.zeros((len(samples), self.seq_len, self.config.input_dim), dtype=np.float32)
            X_ctrl = np.zeros((len(samples), self.pred_len, self.config.ctrl_dim), dtype=np.float32)
            Y_state = np.zeros((len(samples), self.pred_len, len(self.config.state_cols)), dtype=np.float32)
            Y_fault = np.zeros(len(samples), dtype=np.int64)
            Y_fault_type = np.zeros(len(samples), dtype=np.int64)
            Y_rul = np.zeros(len(samples), dtype=np.float32)
            
            for idx, sample in enumerate(samples):
                # 确保sample['x_hist']的维度与input_dim一致
                hist_data = sample['x_hist']
                if hist_data.shape[1] != self.config.input_dim:
                    print(f"警告: x_hist维度不匹配。期望: {self.config.input_dim}，实际: {hist_data.shape[1]}")
                    # 根据实际维度调整mean_X和std_X
                    if hist_data.shape[1] > self.config.input_dim:
                        # 如果实际维度更大，截取到配置的维度
                        hist_data = hist_data[:, :self.config.input_dim]
                    elif hist_data.shape[1] < self.config.input_dim:
                        # 如果实际维度更小，需要重新计算统计值
                        print(f"错误: 实际维度小于配置维度，无法处理")
                        raise ValueError(f"数据维度不匹配: 期望至少{self.config.input_dim}，实际{hist_data.shape[1]}")
                
                X_hist[idx] = (hist_data - self.mean_X) / self.std_X
                
                ctrl_data = sample['x_future_ctrl']
                if ctrl_data.shape[1] != self.config.ctrl_dim:
                    print(f"警告: x_future_ctrl维度不匹配。期望: {self.config.ctrl_dim}，实际: {ctrl_data.shape[1]}")
                    if ctrl_data.shape[1] > self.config.ctrl_dim:
                        ctrl_data = ctrl_data[:, :self.config.ctrl_dim]
                    else:
                        print(f"错误: 实际控制维度小于配置维度，无法处理")
                        raise ValueError(f"控制数据维度不匹配: 期望至少{self.config.ctrl_dim}，实际{ctrl_data.shape[1]}")
                
                X_ctrl[idx] = (ctrl_data - self.mean_X[:self.config.ctrl_dim]) / self.std_X[:self.config.ctrl_dim]
                
                # Y_state 也需要归一化，使用相同的方法
                state_data = sample['y_future_state']
                if state_data.shape[1] != len(self.config.state_cols):
                    print(f"警告: y_future_state维度不匹配。期望: {len(self.config.state_cols)}，实际: {state_data.shape[1]}")
                    if state_data.shape[1] > len(self.config.state_cols):
                        state_data = state_data[:, :len(self.config.state_cols)]
                    else:
                        print(f"错误: 实际状态维度小于配置维度，无法处理")
                        raise ValueError(f"状态数据维度不匹配: 期望至少{len(self.config.state_cols)}，实际{state_data.shape[1]}")
                
                Y_state[idx] = (state_data - self.mean_X[len(self.config.ctrl_cols):len(self.config.ctrl_cols)+len(self.config.state_cols)]) / \
                              self.std_X[len(self.config.ctrl_cols):len(self.config.ctrl_cols)+len(self.config.state_cols)]
                Y_fault[idx] = int(sample['y_fault_label'])
                Y_fault_type[idx] = int(sample['y_fault_type'])
                Y_rul[idx] = sample['y_rul']
            
            return X_hist, X_ctrl, Y_state, Y_fault, Y_fault_type, Y_rul
        
        self.train_X, self.train_ctrl, self.train_Y_state, self.train_Y_fault, self.train_Y_fault_type, self.train_Y_rul = prepare_batch(train_samples)
        self.val_X, self.val_ctrl, self.val_Y_state, self.val_Y_fault, self.val_Y_fault_type, self.val_Y_rul = prepare_batch(val_samples)
    
    def save_normalization_params(self):
        """保存归一化参数"""
        params = {
            'mean_X': self.mean_X,
            'std_X': self.std_X,
            'ctrl_cols': self.config.ctrl_cols,
            'state_cols': self.config.state_cols
        }
        
        params_path = os.path.join(self.config.checkpoint_dir, 'normalization_params.pkl')
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)
        
        print(f"💾 归一化参数已保存: {params_path}")

# ==================== 数据集类 ====================
class MultiTaskDataset(Dataset):
    def __init__(self, X_hist, X_ctrl, Y_state, Y_fault, Y_fault_type, Y_rul):
        self.X_hist = torch.from_numpy(X_hist)
        self.X_ctrl = torch.from_numpy(X_ctrl)
        self.Y_state = torch.from_numpy(Y_state)
        self.Y_fault = torch.from_numpy(Y_fault)
        self.Y_fault_type = torch.from_numpy(Y_fault_type)
        self.Y_rul = torch.from_numpy(Y_rul)
    
    def __len__(self):
        return self.X_hist.shape[0]
    
    def __getitem__(self, idx):
        return {
            'x_hist': self.X_hist[idx],
            'x_ctrl': self.X_ctrl[idx],
            'y_state': self.Y_state[idx],
            'y_fault': self.Y_fault[idx],
            'y_fault_type': self.Y_fault_type[idx],
            'y_rul': self.Y_rul[idx]
        }

def multitask_collate_fn(batch):
    """自定义的collate函数"""
    x_hist = torch.stack([item['x_hist'] for item in batch])
    x_ctrl = torch.stack([item['x_ctrl'] for item in batch])
    y_state = torch.stack([item['y_state'] for item in batch])
    y_fault = torch.stack([item['y_fault'] for item in batch])
    y_fault_type = torch.stack([item['y_fault_type'] for item in batch])
    y_rul = torch.stack([item['y_rul'] for item in batch])
    
    return {
        'x_hist': x_hist,
        'x_ctrl': x_ctrl,
        'y_state': y_state,
        'y_fault': y_fault,
        'y_fault_type': y_fault_type,
        'y_rul': y_rul
    }

# ==================== 训练函数 ====================
def train_multitask_pinn(config):
    print("=" * 80)
    print("🚀 PrinterPINN 多任务训练 (物理场重构 + 故障分类 + RUL预测)")
    print("=" * 80)
    
    # 创建检查点目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # 数据处理
    processor = MultiTaskDataProcessor(
        config.data_path,
        config.seq_len,
        config.pred_len,
        config.max_samples,
        config
    )
    
    # 创建数据集和数据加载器
    train_dataset = MultiTaskDataset(
        processor.train_X, processor.train_ctrl, processor.train_Y_state,
        processor.train_Y_fault, processor.train_Y_fault_type, processor.train_Y_rul
    )
    
    val_dataset = MultiTaskDataset(
        processor.val_X, processor.val_ctrl, processor.val_Y_state,
        processor.val_Y_fault, processor.val_Y_fault_type, processor.val_Y_rul
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,  # Windows设为0或2
        pin_memory=config.pin_memory,    # 必须为True
        persistent_workers=True,         # 减少worker重建开销
        prefetch_factor=2 if config.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=multitask_collate_fn
    )
    
    # 模型
    model = PrinterPINN_MultiTask(config)
    if torch.cuda.device_count() > 1:
        print(f"🎮 使用 {torch.cuda.device_count()} 个 GPU!")
        model = nn.DataParallel(model)
    model = model.to(config.device)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-5
    )
    
    # 学习率调度器
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.warmup_epochs * len(train_loader)
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(config.epochs - config.warmup_epochs) * len(train_loader),
        eta_min=1e-6
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_epochs * len(train_loader)]
    )
    
    # 损失函数
    physics_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    rul_criterion = nn.MSELoss()
    
    # 混合精度训练
    scaler = GradScaler('cuda', enabled=True)
    
    # TensorBoard
    log_dir = os.path.join("runs", f"multitask_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 从检查点恢复
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config.resume_from is not None and os.path.exists(config.resume_from):
        checkpoint = torch.load(config.resume_from, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_loss']
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"✅ 检查点已加载: {config.resume_from}")
        print(f"   从Epoch {start_epoch}开始继续训练")
        print(f"   最佳验证损失: {best_val_loss:.6f}")

    # 训练循环
    print_every = 50
    print("\n🚀 开始多任务训练...")
    print(f"{'='*80}")
    
    # 记录每个epoch的时间，用于预测剩余时间
    epoch_times = []
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        model.train()
        
        total_physics_loss = 0
        total_class_loss = 0
        total_rul_loss = 0
        total_physics_loss_term = 0
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # 将数据移动到设备
            x_hist = batch['x_hist'].to(config.device)
            x_ctrl = batch['x_ctrl'].to(config.device)
            y_state = batch['y_state'].to(config.device)
            y_fault = batch['y_fault'].to(config.device)
            y_fault_type = batch['y_fault_type'].to(config.device)
            y_rul = batch['y_rul'].to(config.device)
            
            # 启用自动混合精度
            with autocast(device_type='cuda', enabled=True):
                # 前向传播
                outputs = model(x_hist, x_ctrl)
                
                # 1. 物理场重构损失
                physics_loss = physics_criterion(outputs['physics_pred'], y_state)
                
                # 2. 故障分类损失
                # 将故障类型转换为分类标签（0=正常，1-3=故障类型）
                class_labels = torch.zeros_like(y_fault, dtype=torch.long)
                mask_fault = (y_fault == 1)
                class_labels[mask_fault] = y_fault_type[mask_fault].long()
                # 确保标签在[0, class_dim-1]范围内
                class_labels = torch.clamp(class_labels, 0, config.class_dim-1)
                
                class_loss = class_criterion(outputs['class_pred'], class_labels)
                
                # 3. RUL回归损失
                rul_loss = rul_criterion(outputs['rul_pred'].squeeze(), y_rul)
                
                # 4. 物理约束损失
                # 解决DataParallel无法访问自定义方法的问题
                if isinstance(model, nn.DataParallel):
                    physics_constraint_loss = model.module.physics_loss(outputs, y_state, config.device)
                else:
                    physics_constraint_loss = model.physics_loss(outputs, y_state, config.device)
                
                # 防止物理约束损失为无穷大或NaN
                if torch.isnan(physics_constraint_loss) or torch.isinf(physics_constraint_loss):
                    print(f"⚠️  检测到物理约束损失异常: {physics_constraint_loss}")
                    physics_constraint_loss = torch.tensor(0.0, device=physics_constraint_loss.device, dtype=physics_constraint_loss.dtype)
                
                # 总损失
                total_loss = (physics_loss + 
                             config.lambda_classification * class_loss + 
                             config.lambda_rul * rul_loss + 
                             config.lambda_physics * physics_constraint_loss)
                
                # 检查总损失是否正常
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"⚠️  检测到总损失异常: {total_loss}")
                    continue  # 跳过这个批次
            
            # 反向传播 - 使用scaler进行缩放
            scaler.scale(total_loss).backward()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Unscales gradients for the optimizer step
                scaler.unscale_(optimizer)
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # 更新优化器参数
                scaler.step(optimizer)
                # 更新scaler状态
                scaler.update()
                # 清零梯度
                optimizer.zero_grad()
                # 更新学习率
                scheduler.step()
            
            # 累积损失
            total_physics_loss += physics_loss.item()
            total_class_loss += class_loss.item()
            total_rul_loss += rul_loss.item()
            total_physics_loss_term += physics_constraint_loss.item()
            
            # 打印进度
            if (batch_idx + 1) % print_every == 0:
                avg_physics = total_physics_loss / (batch_idx + 1)
                avg_class = total_class_loss / (batch_idx + 1)
                avg_rul = total_rul_loss / (batch_idx + 1)
                avg_physics_term = total_physics_loss_term / (batch_idx + 1)
                
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  🔵 Epoch {epoch+1:2d}/{config.epochs} | Batch {batch_idx+1:4d}/{len(train_loader):4d} | "
                      f"Physics: {avg_physics:.4f} | Class: {avg_class:.4f} | RUL: {avg_rul:.4f} | "
                      f"PhysicsTerm: {avg_physics_term:.4f} | LR: {current_lr:.2e}")
        
        # 验证
        model.eval()
        val_physics_loss = 0
        val_class_loss = 0
        val_rul_loss = 0
        val_physics_term = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                x_hist = batch['x_hist'].to(config.device)
                x_ctrl = batch['x_ctrl'].to(config.device)
                y_state = batch['y_state'].to(config.device)
                y_fault = batch['y_fault'].to(config.device)
                y_fault_type = batch['y_fault_type'].to(config.device)
                y_rul = batch['y_rul'].to(config.device)
                
                # 在验证期间也使用混合精度
                with autocast(device_type='cuda', enabled=True):
                    outputs = model(x_hist, x_ctrl)
                    
                    # 物理场重构损失
                    physics_loss = physics_criterion(outputs['physics_pred'], y_state)
                    
                    # 故障分类
                    class_labels = torch.zeros_like(y_fault, dtype=torch.long)
                    mask_fault = (y_fault == 1)
                    class_labels[mask_fault] = y_fault_type[mask_fault].long()
                    class_labels = torch.clamp(class_labels, 0, config.class_dim-1)
                    
                    class_loss = class_criterion(outputs['class_pred'], class_labels)
                    
                    # RUL损失
                    rul_loss = rul_criterion(outputs['rul_pred'].squeeze(), y_rul)
                    
                    # 物理约束
                    # 解决DataParallel无法访问自定义方法的问题
                    if isinstance(model, nn.DataParallel):
                        physics_term = model.module.physics_loss(outputs, y_state, config.device)
                    else:
                        physics_term = model.physics_loss(outputs, y_state, config.device)
                    
                    # 检查物理项是否为异常值
                    if torch.isnan(physics_term) or torch.isinf(physics_term):
                        print(f"⚠️  验证期间检测到物理约束损失异常: {physics_term}")
                        physics_term = torch.tensor(0.0, device=physics_term.device, dtype=physics_term.dtype)
                
                val_physics_loss += physics_loss.item()
                val_class_loss += class_loss.item()
                val_rul_loss += rul_loss.item()
                val_physics_term += physics_term.item()
                
                # 收集预测结果用于评估
                _, predicted = torch.max(outputs['class_pred'], 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(class_labels.cpu().numpy())
        
        # 计算当前epoch耗时
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # 计算平均epoch时间并预测剩余时间
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = config.epochs - (epoch + 1)
        remaining_time = avg_epoch_time * remaining_epochs
        
        # 将剩余时间转换为小时、分钟、秒
        hours = int(remaining_time // 3600)
        minutes = int((remaining_time % 3600) // 60)
        seconds = int(remaining_time % 60)
        
        # 计算平均验证损失
        avg_val_physics = val_physics_loss / len(val_loader)
        avg_val_class = val_class_loss / len(val_loader)
        avg_val_rul = val_rul_loss / len(val_loader)
        avg_val_physics_term = val_physics_term / len(val_loader)
        
        total_val_loss = avg_val_physics + config.lambda_classification * avg_val_class + \
                        config.lambda_rul * avg_val_rul + config.lambda_physics * avg_val_physics_term

        # 打印epoch摘要
        print(f"🟢 Epoch {epoch+1:2d}/{config.epochs} | Time: {epoch_time:.2f}s | ETA: {hours:02d}h {minutes:02d}m {seconds:02d}s")
        print(f"   Train - Physics: {total_physics_loss/len(train_loader):.4f} | "
              f"Class: {total_class_loss/len(train_loader):.4f} | "
              f"RUL: {total_rul_loss/len(train_loader):.4f} | "
              f"PhysicsTerm: {total_physics_loss_term/len(train_loader):.4f}")
        print(f"   Val   - Physics: {avg_val_physics:.4f} | "
              f"Class: {avg_val_class:.4f} | "
              f"RUL: {avg_val_rul:.4f} | "
              f"PhysicsTerm: {avg_val_physics_term:.4f} | "
              f"Total: {total_val_loss:.4f}")
        
        # 分类性能评估
        if len(all_preds) > 0 and len(set(all_labels)) > 1:  # 确保至少有两个不同的标签
            print("\n📊 分类报告:")
            # 检查实际的标签数量，只显示实际存在的类别
            unique_labels = sorted(set(all_labels))
            if len(unique_labels) > 1:  # 确保有多个类别
                target_names_map = {
                    0: 'Normal', 
                    1: 'Nozzle Clog', 
                    2: 'Mechanical Loose', 
                    3: 'Motor Fault'
                }
                actual_target_names = [target_names_map[i] for i in unique_labels if i in target_names_map]
                
                print(classification_report(
                    all_labels, 
                    all_preds, 
                    labels=unique_labels,
                    target_names=actual_target_names
                ))
            else:
                print(f"⚠️  只有一个类别被预测，无法生成分类报告。唯一标签: {unique_labels[0]}")
            
            # 绘制混淆矩阵
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(all_labels, all_preds)
            # 确保标签顺序正确
            unique_all = sorted(set(all_labels + all_preds))
            target_names_map = {
                0: 'Normal', 
                1: 'Nozzle Clog', 
                2: 'Mechanical Loose', 
                3: 'Motor Fault'
            }
            tick_labels = [target_names_map.get(i, f'Class {i}') for i in unique_all]
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=tick_labels,
                yticklabels=tick_labels
            )
            plt.title(f'Confusion Matrix - Epoch {epoch+1}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            cm_path = os.path.join(config.checkpoint_dir, f'confusion_matrix_epoch{epoch+1}.png')
            plt.savefig(cm_path)
            plt.close()
            
            print(f"   混淆矩阵已保存: {cm_path}")
        else:
            print(f"⚠️  Epoch {epoch+1}: 无法生成分类报告，预测数据不足或类别不全")
        
        # TensorBoard记录
        writer.add_scalar("Loss/train_physics", total_physics_loss/len(train_loader), epoch)
        writer.add_scalar("Loss/train_class", total_class_loss/len(train_loader), epoch)
        writer.add_scalar("Loss/train_rul", total_rul_loss/len(train_loader), epoch)
        writer.add_scalar("Loss/train_physics_term", total_physics_loss_term/len(train_loader), epoch)
        
        writer.add_scalar("Loss/val_physics", avg_val_physics, epoch)
        writer.add_scalar("Loss/val_class", avg_val_class, epoch)
        writer.add_scalar("Loss/val_rul", avg_val_rul, epoch)
        writer.add_scalar("Loss/val_physics_term", avg_val_physics_term, epoch)
        writer.add_scalar("Loss/val_total", total_val_loss, epoch)
        
        writer.add_scalar("Time/epoch", epoch_time, epoch)
        
        # 保存最佳模型
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            checkpoint_path = os.path.join(config.checkpoint_dir, "best_multitask_model.pth")
            save_checkpoint(epoch+1, model, optimizer, scheduler, total_val_loss, best_val_loss, config, checkpoint_path, scaler)
            print(f"  💾 最佳模型已保存 (验证损失: {best_val_loss:.4f})")
        
        # 定期保存
        if (epoch + 1) % config.save_interval == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch{epoch+1}.pth")
            save_checkpoint(epoch+1, model, optimizer, scheduler, total_val_loss, best_val_loss, config, checkpoint_path, scaler)
    
    print(f"\n{'='*80}")
    print("🎉 训练完成！")
    print(f"{'='*80}")

def save_checkpoint(epoch, model, optimizer, scheduler, current_loss, best_loss, config, filename, scaler=None):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'current_loss': current_loss,
        'best_loss': best_loss,
        'config': config.__dict__,
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    torch.save(checkpoint, filename)
    print(f"💾 检查点已保存: {filename}")

def load_checkpoint(model, optimizer, scheduler, filename):
    """加载检查点"""
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    
    print(f"✅ 检查点已加载: {filename}")
    print(f"   从Epoch {start_epoch}开始继续训练")
    print(f"   最佳验证损失: {best_loss:.6f}")
    
    return start_epoch, best_loss

# ==================== 主函数 ====================
def get_args():
    parser = argparse.ArgumentParser(description='训练3D打印机多任务PINN模型')
    parser.add_argument('--data_path', type=str, default='printer_dataset/nozzle_simulation_gear_print.csv',
                        help='数据文件路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2048, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--resume_from', type=str, help='从指定检查点恢复训练')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_multitask', help='检查点保存目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    config = Config()
    
    # 更新配置
    config.data_path = args.data_path
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.resume_from = args.resume_from
    config.checkpoint_dir = args.checkpoint_dir
    config.device = args.device
    
    train_multitask_pinn(config)