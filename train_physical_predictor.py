# train_physical_predictor_enhanced.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
import argparse
from datetime import datetime, timedelta
import seaborn as sns
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
class Config:
    def __init__(self, resume_from=None, gpu_ids=[0]):
        self.data_path = 'printer_dataset_correction/printer_gear_correction_dataset.csv'
        self.seq_len = 250           # 历史窗口长度 (250ms)
        self.pred_len = 50           # 预测长度 (50ms)
        self.batch_size = 1024
        self.gradient_accumulation_steps = 2
        self.model_dim = 192
        self.num_heads = 8
        self.num_layers = 5
        self.dim_feedforward = 768
        self.dropout = 0.1
        self.lr = 5e-5
        self.epochs = 60
        self.gpu_ids = gpu_ids
        self.resume_from = resume_from
        
        if len(gpu_ids) > 1:
            self.device = f'cuda:{gpu_ids[0]}'  # 主GPU
        else:
            self.device = f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu'
            
        self.lambda_physics = 0.4    # 物理约束权重
        self.lambda_freq = 0.3       # 频域约束权重
        self.checkpoint_dir = './checkpoints_physical_predictor_enhanced'
        self.max_samples = 300000
        self.warmup_epochs = 5
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 特征列
        self.feature_cols = [
            'ctrl_T_target', 'ctrl_speed_set', 'ctrl_pos_x', 'ctrl_pos_y', 'ctrl_pos_z',
            'temperature_C', 'vibration_disp_x_m', 'vibration_disp_y_m',
            'vibration_vel_x_m_s', 'vibration_vel_y_m_s',
            'motor_current_x_A', 'motor_current_y_A',
            'pressure_bar'
        ]
        
        # 目标列 (需要预测的物理量)
        self.target_cols = [
            'vibration_disp_x_m', 'vibration_disp_y_m',
            'temperature_C', 'motor_current_x_A', 'motor_current_y_A'
        ]
        
        # 频域特征参数
        self.freq_bands = 8  # 频率带数量
        self.sampling_rate = 1000  # 1kHz 采样率 (1ms步长)
        self.max_freq = 500  # 最大频率 (Hz)
        
        # 计算总输入维度 (时域 + 频域)
        self.time_domain_dim = len(self.feature_cols)
        self.freq_domain_dim = self.freq_bands * len(self.feature_cols)
        self.input_dim = self.time_domain_dim + self.freq_domain_dim
        self.output_dim = len(self.target_cols)

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

# 自定义滤波器模块
class FilterModule(nn.Module):
    def __init__(self, b, a):
        super().__init__()
        self.register_buffer('b', torch.tensor(b, dtype=torch.float32))
        self.register_buffer('a', torch.tensor(a, dtype=torch.float32))
    
    def forward(self, x):
        # 这里使用简化的频域分析，因为实际的滤波实现比较复杂
        # 我们直接返回输入，因为实际的频域特征已经在预处理中计算了
        return x

# ==================== 频域特征提取器 ====================
class FrequencyFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FrequencyFeatureExtractor, self).__init__()
        self.config = config
        
        # 定义频率带 (对数尺度)
        self.freq_bands = np.logspace(np.log10(1), np.log10(config.max_freq), config.freq_bands + 1)

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]
        由于频域特征已在预处理阶段计算并添加到输入中，这里我们提取这些特征
        """
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        
        # 计算原始时域特征数量
        original_features = len(self.config.feature_cols)
        freq_features_count = input_dim - original_features
        
        if freq_features_count <= 0:
            # 如果没有频域特征，返回空张量
            return torch.zeros(batch_size, seq_len, 0, device=device)
        
        # 提取频域特征部分（即除了原始特征之外的部分）
        freq_part = x[:, :, len(self.config.feature_cols):]  # 提取频域部分
        
        return freq_part

# ==================== 增强版物理预测模型 ====================
class EnhancedPhysicalPredictor(nn.Module):
    def __init__(self, config):
        super(EnhancedPhysicalPredictor, self).__init__()
        self.config = config
        
        # 频域特征提取 - 现在只是提取预计算的频域特征
        self.freq_extractor = FrequencyFeatureExtractor(config)
        
        # 使用配置中的总输入维度（已包含频域特征）
        total_input_dim = config.input_dim
        
        # 编码器 - 使用更新后的输入维度
        self.encoder_embedding = nn.Linear(total_input_dim, config.model_dim)
        self.pos_encoder = PositionalEncoding(config.model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # 多头输出 (针对不同物理量)
        self.vibration_head = nn.Sequential(
            nn.Linear(config.model_dim, 96),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(96, 2)  # x, y振动
        )
        
        self.thermal_head = nn.Sequential(
            nn.Linear(config.model_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1)  # 温度
        )
        
        self.motor_head = nn.Sequential(
            nn.Linear(config.model_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 2)  # x, y电机电流
        )
        
        # 物理参数 (可学习)
        self.register_buffer('mass', torch.tensor(0.045))  # 喷头质量 (kg)
        self.register_buffer('stiffness', torch.tensor(1500.0))  # 刚度 (N/m)
        self.register_buffer('damping', torch.tensor(0.48))  # 阻尼系数

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim] - 已包含预计算的频域特征
        """
        # x已经包含了时域和频域特征，直接使用即可
        
        # 编码器
        x_emb = self.encoder_embedding(x)
        x_emb = self.pos_encoder(x_emb)
        memory = self.encoder(x_emb)  # [batch, seq_len, model_dim]
        
        # 使用序列的最后一个时间步进行预测
        last_state = memory[:, -1, :]  # [batch, model_dim]
        
        # 多头预测
        vib_pred = self.vibration_head(last_state)  # [2]
        temp_pred = self.thermal_head(last_state)   # [1]
        motor_pred = self.motor_head(last_state)    # [2]
        
        # 合并预测结果
        prediction = torch.cat([vib_pred, temp_pred, motor_pred], dim=1)  # [batch, 5]
        
        return prediction

    def physics_loss(self, predictions, targets, dt=0.001):
        """增强的物理约束损失"""
        loss = 0.0
        
        # 1. 振动动力学约束 (质量-弹簧-阻尼系统)
        vib_x_pred = predictions[:, 0]
        vib_y_pred = predictions[:, 1]
        
        # 原始目标值
        vib_x_target = targets[:, 0]
        vib_y_target = targets[:, 1]
        
        # 振动应该平滑变化
        if len(vib_x_pred) > 1:
            vib_x_smoothness = torch.mean(torch.abs(torch.diff(vib_x_pred)))
            vib_y_smoothness = torch.mean(torch.abs(torch.diff(vib_y_pred)))
            loss += 0.3 * (vib_x_smoothness + vib_y_smoothness)
        
        # 2. 热传导方程约束
        temp_pred = predictions[:, 2]
        temp_target = targets[:, 2]
        
        if len(temp_pred) > 1:
            dT_dt = torch.diff(temp_pred) / dt
            # 温度变化率应该平滑
            d2T_dt2 = torch.diff(dT_dt) / dt
            thermal_smoothness = torch.mean(torch.abs(d2T_dt2))
            loss += 0.2 * thermal_smoothness
            
            # 温度不应该突变
            temp_change = torch.mean(torch.abs(torch.diff(temp_pred)))
            loss += 0.3 * torch.clamp(temp_change - 1.0, min=0)  # 每毫秒变化不超过1°C
        
        # 3. 电机电流-振动耦合约束
        current_x_pred = predictions[:, 3]
        current_y_pred = predictions[:, 4]
        
        # 电机电流应该与振动幅度相关
        vib_magnitude = torch.sqrt(vib_x_pred**2 + vib_y_pred**2)
        current_magnitude = torch.sqrt(current_x_pred**2 + current_y_pred**2)
        
        # 计算相关性
        if len(vib_magnitude) > 1:
            vib_mean = torch.mean(vib_magnitude)
            current_mean = torch.mean(current_magnitude)
            
            vib_centered = vib_magnitude - vib_mean
            current_centered = current_magnitude - current_mean
            
            correlation = torch.sum(vib_centered * current_centered) / (
                torch.sqrt(torch.sum(vib_centered**2)) * 
                torch.sqrt(torch.sum(current_centered**2)) + 1e-8
            )
            # 确保相关性值在合理范围内
            correlation = torch.clamp(correlation, -1.0, 1.0)
            
            # 希望有正相关
            loss += 0.2 * torch.relu(0.3 - correlation)
        
        return loss

    def frequency_loss(self, predictions, targets):
        """频域一致性损失"""
        loss = 0.0
        
        # 确保张量是float32精度以避免FFT的半精度问题
        predictions = predictions.float()
        targets = targets.float()
        
        # 计算预测值和目标值的FFT
        pred_x_fft = torch.fft.rfft(predictions[:, 0])  # 预测的x振动
        pred_y_fft = torch.fft.rfft(predictions[:, 1])  # 预测的y振动
        target_x_fft = torch.fft.rfft(targets[:, 0])    # 目标的x振动
        target_y_fft = torch.fft.rfft(targets[:, 1])    # 目标的y振动
        
        # 取前N个频率分量进行比较，确保不会超出索引范围
        max_freq_bins = min(10, pred_x_fft.shape[0])
        pred_x_mag = torch.abs(pred_x_fft[:max_freq_bins])
        pred_y_mag = torch.abs(pred_y_fft[:max_freq_bins])
        target_x_mag = torch.abs(target_x_fft[:max_freq_bins])
        target_y_mag = torch.abs(target_y_fft[:max_freq_bins])
        
        # 频域幅度损失
        freq_mag_loss = nn.MSELoss()(pred_x_mag, target_x_mag) + \
                        nn.MSELoss()(pred_y_mag, target_y_mag)
        
        loss += freq_mag_loss
        
        return loss

# ==================== 数据集类 ====================
class PhysicalPredictionDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# ==================== 频域特征处理器 ====================
class FrequencyFeatureProcessor:
    """内存友好的频域特征处理器"""
    def __init__(self, config, feature_mean, feature_std):
        self.config = config
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        
        # 定义频率带 (对数尺度)
        self.freq_bands = np.logspace(np.log10(1), np.log10(config.max_freq), 
                                     config.freq_bands + 1)
        
        # 预计算FFT频率
        self.fft_freqs = np.fft.rfftfreq(config.seq_len, 1.0/config.sampling_rate)
    
    def compute_frequency_features(self, sequences):
        """
        按批次计算频域特征，避免内存溢出
        sequences: [n_samples, seq_len, n_features]
        """
        n_samples, seq_len, n_features = sequences.shape
        n_freq_features = self.config.freq_bands * n_features
        
        # 创建空数组（只分配输出空间）
        freq_features = np.zeros((n_samples, seq_len, n_freq_features), dtype=np.float32)
        
        # 分批次处理（每次处理5000个样本）
        batch_size = 5000
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_sequences = sequences[start_idx:end_idx]
            
            # 处理当前批次
            self._process_batch(batch_sequences, freq_features, start_idx, end_idx)
            
            print(f"  处理频域特征: {end_idx}/{n_samples} samples")
        
        return freq_features
    
    def _process_batch(self, batch_sequences, freq_features, start_idx, end_idx):
        """处理单个批次的频域特征"""
        batch_size, seq_len, n_features = batch_sequences.shape
        
        for feature_idx in range(n_features):
            # 对每个特征计算FFT
            feature_data = batch_sequences[:, :, feature_idx]  # [batch_size, seq_len]
            fft_result = np.fft.rfft(feature_data, axis=1)  # [batch_size, fft_coefficients]
            fft_magnitude = np.abs(fft_result)  # [batch_size, fft_coefficients]
            
            # 为每个频率带计算能量
            for band_idx in range(self.config.freq_bands):
                low_freq = self.freq_bands[band_idx]
                high_freq = self.freq_bands[band_idx+1]
                
                # 找到对应频率范围的索引
                band_indices = np.where((self.fft_freqs >= low_freq) & 
                                       (self.fft_freqs < high_freq))[0]
                
                if len(band_indices) > 0:
                    # 计算该频带的平均能量 - [batch_size, len(band_indices)] -> [batch_size]
                    band_energy = np.mean(fft_magnitude[:, band_indices], axis=1)
                    
                    # 将能量分配到所有时间步
                    start_col = band_idx * n_features + feature_idx
                    # band_energy: [batch_size] -> [batch_size, 1]
                    # 然后广播到 [batch_size, seq_len]
                    freq_features[start_idx:end_idx, :, start_col] = \
                        np.broadcast_to(band_energy[:, np.newaxis], 
                                       (batch_size, seq_len))

# ==================== 数据处理器 ====================
def prepare_data(config):
    print("🔄 加载和处理数据...")
    df = pd.read_csv(config.data_path)
    
    # 选择正常机器的数据（无故障）
    normal_df = df[df['fault_label'] == 0].copy()
    print(f"   正常机器数据: {len(normal_df)} / {len(df)}")
    
    # 限制总样本数以节省内存
    max_total_samples = 100000
    if len(normal_df) > max_total_samples:
        normal_df = normal_df.sample(n=max_total_samples, random_state=42)
        print(f"   限制数据量: {len(normal_df)}")
    
    # 提取特征和目标
    features = normal_df[config.feature_cols].values
    targets = normal_df[config.target_cols].values
    
    # 标准化
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0
    
    target_mean = targets.mean(axis=0)
    target_std = targets.std(axis=0)
    target_std[target_std < 1e-8] = 1.0
    
    features_norm = (features - feature_mean) / feature_std
    targets_norm = (targets - target_mean) / target_std
    
    # 创建序列样本
    sequences = []
    target_values = []
    
    machine_ids = normal_df['machine_id'].unique()
    
    for mid in machine_ids:
        machine_data = normal_df[normal_df['machine_id'] == mid]
        if len(machine_data) < config.seq_len + config.pred_len:
            continue
        
        machine_features = features_norm[normal_df['machine_id'] == mid]
        machine_targets = targets_norm[normal_df['machine_id'] == mid]
        
        # 限制每台机器生成的样本数量
        max_samples_per_machine = 1000
        n_windows = min(len(machine_data) - config.seq_len - config.pred_len + 1, 
                        max_samples_per_machine)
        
        for i in range(n_windows):
            seq = machine_features[i:i+config.seq_len]
            target_idx = i + config.seq_len + config.pred_len - 1
            target_val = machine_targets[target_idx]
            
            sequences.append(seq)
            target_values.append(target_val)
    
    sequences = np.array(sequences)
    target_values = np.array(target_values)
    
    # 最终限制总样本数
    max_final_samples = 50000
    if len(sequences) > max_final_samples:
        idx = np.random.choice(len(sequences), max_final_samples, replace=False)
        sequences = sequences[idx]
        target_values = target_values[idx]
    
    # 预计算频域特征
    print("📊 计算频域特征...")
    freq_processor = FrequencyFeatureProcessor(config, feature_mean, feature_std)
    freq_features = freq_processor.compute_frequency_features(sequences)
    
    # 合并时域和频域特征
    combined_sequences = np.concatenate([sequences, freq_features], axis=2)
    print(f"   合并后特征维度: {combined_sequences.shape[2]} (时域: {sequences.shape[2]}, 频域: {freq_features.shape[2]})")
    
    # 分割训练集和验证集
    train_seq, val_seq, train_targets, val_targets = train_test_split(
        combined_sequences, target_values, test_size=0.2, random_state=42
    )
    
    print(f"📊 总样本数: {len(sequences)}")
    print(f"   训练集: {len(train_seq)}, 验证集: {len(val_seq)}")
    
    # 保存标准化参数
    normalization_params = {
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'target_mean': target_mean,
        'target_std': target_std,
        'feature_cols': config.feature_cols,
        'target_cols': config.target_cols,
        'freq_bands': config.freq_bands,
        'sampling_rate': config.sampling_rate
    }
    
    with open(os.path.join(config.checkpoint_dir, 'normalization_params.pkl'), 'wb') as f:
        pickle.dump(normalization_params, f)
    
    return (train_seq, train_targets), (val_seq, val_targets), normalization_params

# ==================== 训练函数 ====================
def train_model(config):
    print("=" * 80)
    print("🚀 训练增强版物理预测模型")
    print("=" * 80)
    
    # 准备数据
    (train_seq, train_targets), (val_seq, val_targets), norm_params = prepare_data(config)
    
    # 创建数据集和数据加载器
    train_dataset = PhysicalPredictionDataset(train_seq, train_targets)
    val_dataset = PhysicalPredictionDataset(val_seq, val_targets)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 创建模型
    model = EnhancedPhysicalPredictor(config)
    print(f"✅ 模型创建完成 | 参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 检查是否使用多GPU
    if len(config.gpu_ids) > 1:
        print(f"✅ 使用多GPU训练: {config.gpu_ids}")
        model = nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(config.device)
    else:
        model = model.to(config.device)
    
    # 优化器和学习率调度
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=1e-5
    )
    
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
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda')
    
    # 从检查点恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    if config.resume_from and os.path.exists(config.resume_from):
        print(f"🔄 从检查点恢复训练: {config.resume_from}")
        checkpoint = torch.load(config.resume_from)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print(f"✅ 恢复训练成功 | 从第 {start_epoch} 个epoch开始")
    
    print("\n🔥 开始训练...")
    print("-" * 80)
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        total_physics_loss = 0
        total_freq_loss = 0
        
        for batch_idx, (seq, target) in enumerate(train_loader):
            seq, target = seq.to(config.device), target.to(config.device)
            
            with autocast('cuda'):
                pred = model(seq)
                data_loss = criterion(pred, target)
                
                # 物理约束损失 - 检查是否使用DataParallel
                if isinstance(model, nn.DataParallel):
                    physics_loss = model.module.physics_loss(pred, target)
                    freq_loss = model.module.frequency_loss(pred, target)
                else:
                    physics_loss = model.physics_loss(pred, target)
                    freq_loss = model.frequency_loss(pred, target)
                
                # 总损失
                total_batch_loss = data_loss + config.lambda_physics * physics_loss + config.lambda_freq * freq_loss
            
            # 反向传播
            scaler.scale(total_batch_loss).backward()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += total_batch_loss.item()
            total_physics_loss += physics_loss.item()
            total_freq_loss += freq_loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_physics_loss = total_physics_loss / len(train_loader)
        avg_freq_loss = total_freq_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        val_physics_loss = 0
        
        with torch.no_grad():
            for seq, target in val_loader:
                seq, target = seq.to(config.device), target.to(config.device)
                pred = model(seq)
                loss = criterion(pred, target)
                
                # 物理约束损失 - 检查是否使用DataParallel
                if isinstance(model, nn.DataParallel):
                    physics_loss = model.module.physics_loss(pred, target)
                else:
                    physics_loss = model.physics_loss(pred, target)
                
                val_loss += loss.item()
                val_physics_loss += physics_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_physics_loss = val_physics_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # 计算剩余时间
        elapsed_time = time.time() - epoch_start
        remaining_epochs = config.epochs - epoch - 1
        remaining_time = elapsed_time * remaining_epochs
        remaining_time_str = str(timedelta(seconds=int(remaining_time)))
        
        print(f"✅ Epoch {epoch+1:2d}/{config.epochs} | "
              f"Train Loss: {avg_train_loss:.6f} (Physics: {avg_physics_loss:.6f}, Freq: {avg_freq_loss:.6f}) | "
              f"Val Loss: {avg_val_loss:.6f} (Physics: {avg_val_physics_loss:.6f}) | "
              f"Time: {epoch_time:.2f}s | "
              f"剩余时间: {remaining_time_str}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': config.__dict__
            }
            torch.save(checkpoint_data, os.path.join(config.checkpoint_dir, 'best_physical_predictor.pth'))
            print(f"   💾 保存最佳模型 (验证损失: {best_val_loss:.6f})")
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': config.__dict__
            }
            torch.save(checkpoint_data, os.path.join(config.checkpoint_dir, f'checkpoint_epoch{epoch+1}.pth'))
            print(f"   💾 保存检查点: epoch {epoch+1}")
        
        # 每10个epoch生成一次可视化
        if (epoch + 1) % 10 == 0:
            visualize_training_progress(model, val_loader, config, epoch + 1)
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('增强版物理预测模型训练过程')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.checkpoint_dir, 'training_curve_enhanced.png'))
    
    print("\n" + "=" * 80)
    print(f"🎉 训练完成! 最佳验证损失: {best_val_loss:.6f}")
    print("=" * 80)

def visualize_training_progress(model, val_loader, config, epoch):
    """可视化验证集上的预测结果"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for seq, target in val_loader:
            seq = seq.to(config.device)
            pred = model(seq)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            
            if len(all_preds) * config.batch_size > 1000:  # 限制样本数量
                break
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'增强版物理预测模型 - Epoch {epoch}', fontsize=16)
    
    target_names = ['Vib X', 'Vib Y', 'Temp', 'Motor X', 'Motor Y']
    
    for i in range(min(5, len(target_names))):
        ax = axes[i//3, i%3]
        ax.scatter(targets[:200, i], preds[:200, i], alpha=0.6, s=10)
        ax.plot([targets[:200, i].min(), targets[:200, i].max()], 
                [targets[:200, i].min(), targets[:200, i].max()], 'r--')
        ax.set_xlabel('真实值')
        ax.set_ylabel('预测值')
        ax.set_title(target_names[i])
        ax.grid(True)
    
    # 物理一致性检查
    ax = axes[1, 2]
    
    # 计算振动和电机电流的相关性
    vib_magnitude = np.sqrt(preds[:, 0]**2 + preds[:, 1]**2)
    motor_magnitude = np.sqrt(preds[:, 3]**2 + preds[:, 4]**2)
    
    ax.scatter(vib_magnitude[:200], motor_magnitude[:200], alpha=0.6, s=10)
    ax.set_xlabel('振动幅度')
    ax.set_ylabel('电机电流幅度')
    ax.set_title('物理一致性: 振动-电流关系')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.checkpoint_dir, f'validation_results_epoch{epoch}.png'))
    plt.close()

# ==================== 主函数 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练增强版物理预测模型')
    parser.add_argument('--resume', type=str, default=None, help='从指定路径恢复训练')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='GPU IDs (例如: "0,1,2,3")')
    args = parser.parse_args()
    
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    config = Config(resume_from=args.resume, gpu_ids=gpu_ids)
    
    train_model(config)