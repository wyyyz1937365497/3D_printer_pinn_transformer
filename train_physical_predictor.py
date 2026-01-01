# train_physical_predictor_streaming.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import os
import time
import pickle
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
import argparse
from datetime import datetime, timedelta
import warnings
from scipy import signal
from tqdm import tqdm  # 添加进度条库
warnings.filterwarnings('ignore')

class Config:
    def __init__(self, resume_from=None, gpu_ids=[0]):
        self.data_dir = 'printer_dataset_correction/'  # 数据目录
        self.seq_len = 250
        self.pred_len = 50
        self.batch_size = 1024  # 增加batch size以提高训练效率
        self.gradient_accumulation_steps = 1
        self.model_dim = 192
        self.num_heads = 8
        self.num_layers = 6
        self.dim_feedforward = 768
        self.dropout = 0.1
        self.lr = 5e-5
        self.epochs = 30  # 可能需要减少epochs，因为数据量大
        self.gpu_ids = gpu_ids
        self.resume_from = resume_from
        self.device = f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu'
        self.lambda_physics = 0.4
        self.lambda_freq = 0.3
        self.checkpoint_dir = './checkpoints_physical_predictor_enhanced'  # 修改检查点目录
        self.warmup_epochs = 5
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # 修改特征列，使用实际数据集中的列名
        self.feature_cols = [
            'nozzle_x', 'nozzle_y', 'nozzle_z',  # 替换原来的控制列
            'temperature_C', 'vibration_disp_x_m', 'vibration_disp_y_m',
            'vibration_vel_x_m_s', 'vibration_vel_y_m_s',
            'motor_current_x_A', 'motor_current_y_A',
            'pressure_bar'
        ]
        self.target_cols = [
            'vibration_disp_x_m', 'vibration_disp_y_m',
            'temperature_C', 'motor_current_x_A', 'motor_current_y_A'
        ]
        self.freq_bands = 8
        self.sampling_rate = 1000
        self.max_freq = 500
        self.time_domain_dim = len(self.feature_cols)
        self.freq_domain_dim = self.freq_bands * len(self.feature_cols)
        self.input_dim = self.time_domain_dim + self.freq_domain_dim
        self.output_dim = len(self.target_cols)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class FrequencyFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        original_features = len(self.config.feature_cols)
        if input_dim <= original_features:
            return torch.zeros(batch_size, seq_len, 0, device=x.device)
        return x[:, :, original_features:]

class EnhancedPhysicalPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.freq_extractor = FrequencyFeatureExtractor(config)
        self.encoder_embedding = nn.Linear(config.input_dim, config.model_dim)
        self.pos_encoder = PositionalEncoding(config.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.vibration_head = nn.Sequential(
            nn.Linear(config.model_dim, 96),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(96, 2)
        )
        self.thermal_head = nn.Sequential(
            nn.Linear(config.model_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1)
        )
        self.motor_head = nn.Sequential(
            nn.Linear(config.model_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 2)
        )
        self.register_buffer('mass', torch.tensor(0.045))
        self.register_buffer('stiffness', torch.tensor(1500.0))
        self.register_buffer('damping', torch.tensor(0.48))
    
    def forward(self, x):
        x_emb = self.encoder_embedding(x)
        x_emb = self.pos_encoder(x_emb)
        memory = self.encoder(x_emb)
        last_state = memory[:, -1, :]
        vib_pred = self.vibration_head(last_state)
        temp_pred = self.thermal_head(last_state)
        motor_pred = self.motor_head(last_state)
        return torch.cat([vib_pred, temp_pred, motor_pred], dim=1)
    
    def physics_loss(self, pred, target, dt=0.001):
        loss = 0.0
        if len(pred) > 1:
            # 振动平滑损失
            if len(pred) > 1:
                vib_x_smooth = torch.mean(torch.abs(torch.diff(pred[:, 0])))
                vib_y_smooth = torch.mean(torch.abs(torch.diff(pred[:, 1])))
                # 使用clamp防止数值不稳定
                vib_x_smooth = torch.clamp(vib_x_smooth, max=1e3)
                vib_y_smooth = torch.clamp(vib_y_smooth, max=1e3)
                loss += 0.3 * (vib_x_smooth + vib_y_smooth)
            
            # 温度平滑损失
            if len(pred) > 2:
                dT_dt = torch.diff(pred[:, 2]) / dt
                if len(dT_dt) > 1:
                    d2T_dt2 = torch.diff(dT_dt) / dt
                    thermal_smooth = torch.mean(torch.abs(d2T_dt2))
                    # 使用clamp防止数值不稳定
                    thermal_smooth = torch.clamp(thermal_smooth, max=1e3)
                    loss += 0.2 * thermal_smooth
            
            # 温度变化限制
            if len(pred) > 1:
                temp_change = torch.mean(torch.abs(torch.diff(pred[:, 2])))
                temp_change = torch.clamp(temp_change, max=1e2)
                loss += 0.3 * torch.clamp(temp_change - 1.0, min=0, max=1e2)
            
            # 振动-电流相关性损失
            if len(pred) > 3:
                vib_mag = torch.sqrt(pred[:,0]**2 + pred[:,1]**2 + 1e-8)  # 防止sqrt(0)
                cur_mag = torch.sqrt(pred[:,3]**2 + pred[:,4]**2 + 1e-8)  # 防止sqrt(0)
                if len(vib_mag) > 1:
                    vib_mean = torch.mean(vib_mag)
                    cur_mean = torch.mean(cur_mag)
                    vib_c = vib_mag - vib_mean
                    cur_c = cur_mag - cur_mean
                    # 计算相关系数时防止除零
                    vib_var = torch.sum(vib_c**2)
                    cur_var = torch.sum(cur_c**2)
                    if vib_var > 1e-8 and cur_var > 1e-8:
                        corr = torch.sum(vib_c * cur_c) / (torch.sqrt(vib_var * cur_var) + 1e-8)
                        loss += 0.2 * torch.relu(0.3 - corr)
        
        # 使用clamp防止loss过大
        loss = torch.clamp(loss, max=1e4)
        return loss
    
    def frequency_loss(self, pred, target):
        # 确保输入是float32以避免精度问题
        pred = pred.float()
        target = target.float()
        
        # 检查输入长度，避免微分操作失败
        if len(pred) < 2:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        try:
            pred_x_fft = torch.fft.rfft(pred[:, 0])
            pred_y_fft = torch.fft.rfft(pred[:, 1])
            target_x_fft = torch.fft.rfft(target[:, 0])
            target_y_fft = torch.fft.rfft(target[:, 1])
            
            # 限制FFT的最大频率分量数量
            max_bins = min(10, pred_x_fft.shape[0])
            pred_x_mag = torch.abs(pred_x_fft[:max_bins])
            pred_y_mag = torch.abs(pred_y_fft[:max_bins])
            target_x_mag = torch.abs(target_x_fft[:max_bins])
            target_y_mag = torch.abs(target_y_fft[:max_bins])
            
            # 使用clamp防止数值不稳定
            pred_x_mag = torch.clamp(pred_x_mag, max=1e3)
            pred_y_mag = torch.clamp(pred_y_mag, max=1e3)
            target_x_mag = torch.clamp(target_x_mag, max=1e3)
            target_y_mag = torch.clamp(target_y_mag, max=1e3)
            
            mse_loss = nn.MSELoss()
            loss = mse_loss(pred_x_mag, target_x_mag) + mse_loss(pred_y_mag, target_y_mag)
            
            # 再次clamp防止loss过大
            loss = torch.clamp(loss, max=1e4)
            return loss
        except RuntimeError:
            # 如果FFT失败，返回0损失
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

class FrequencyFeatureProcessor:
    """频域特征处理器，支持缓存到.npy文件"""
    def __init__(self, config, feature_mean, feature_std):
        self.config = config
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.freq_bands = np.logspace(np.log10(1), np.log10(config.max_freq), config.freq_bands + 1)
        self.fft_freqs = np.fft.rfftfreq(config.seq_len, 1.0/config.sampling_rate)
    
    def compute_and_cache_features(self, data_dir, file_pattern='machine_*.csv'):
        """为所有CSV文件计算并缓存频域特征"""
        csv_files = [f for f in os.listdir(data_dir) if f.startswith('machine_') and f.endswith('.csv')]
        cache_dir = os.path.join(data_dir, 'freq_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        for csv_file in csv_files:
            csv_path = os.path.join(data_dir, csv_file)
            cache_path = os.path.join(cache_dir, csv_file.replace('.csv', '_freq.npy'))
            
            if os.path.exists(cache_path):
                print(f"✅ 频域特征缓存已存在: {cache_path}")
                continue
            
            print(f"🔄 计算频域特征: {csv_file}")
            df = pd.read_csv(csv_path)
            normal_df = df[df['fault_label'] == 0]
            
            if len(normal_df) < self.config.seq_len:
                print(f"   警告: {csv_file} 数据不足，跳过")
                continue
            
            # 提取并标准化特征
            features = normal_df[self.config.feature_cols].values
            features_norm = (features - self.feature_mean) / self.feature_std
            
            # 计算频域特征
            n_samples = features_norm.shape[0]
            n_freq_features = self.config.freq_bands * len(self.config.feature_cols)
            freq_features = np.zeros((n_samples, n_freq_features), dtype=np.float32)
            
            # 按批次处理
            batch_size = 5000
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                self._process_batch(features_norm[start:end], freq_features, start, end)
                print(f"   处理: {end}/{n_samples} 样本")
            
            # 保存到缓存
            np.save(cache_path, freq_features)
            print(f"   💾 保存频域特征缓存: {cache_path}")
    
    def _process_batch(self, batch_seq, freq_features, start, end):
        """处理单个批次的频域特征"""
        B, F = batch_seq.shape
        for f in range(F):
            # 为每个特征单独计算FFT
            fft_result = np.fft.rfft(batch_seq[:, f], axis=0)
            fft_mag = np.abs(fft_result)
            
            for b in range(self.config.freq_bands):
                low = self.freq_bands[b]
                high = self.freq_bands[b+1]
                idx = np.where((self.fft_freqs >= low) & (self.fft_freqs < high))[0]
                if len(idx) > 0:
                    energy = np.mean(fft_mag[idx])
                    col = b * F + f
                    freq_features[start:end, col] = energy

class StreamingPhysicalDataset(IterableDataset):
    def __init__(self, data_dir, config, split='train', val_ratio=0.2, norm_params=None):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.split = split
        self.val_ratio = val_ratio
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                           if f.startswith('machine_') and f.endswith('.csv')])
        self.cache_dir = os.path.join(data_dir, 'freq_cache')
        
        # 加载或初始化标准化参数
        if norm_params is None:
            self._init_normalization()
        else:
            self.feature_mean = norm_params['feature_mean']
            self.feature_std = norm_params['feature_std']
            self.target_mean = norm_params['target_mean']
            self.target_std = norm_params['target_std']
        
        # 预计算频域特征（如果尚未缓存）
        if not hasattr(self, 'feature_mean'):
            self.feature_mean = np.zeros(len(self.config.feature_cols))
            self.feature_std = np.ones(len(self.config.feature_cols))
        
        freq_processor = FrequencyFeatureProcessor(config, self.feature_mean, self.feature_std)
        freq_processor.compute_and_cache_features(data_dir)
    
    def _init_normalization(self):
        """从第一个有效文件初始化标准化参数"""
        for filepath in self.files:
            df = pd.read_csv(filepath)
            normal_df = df[df['fault_label'] == 0]
            if len(normal_df) > 1000:  # 确保有足够的正常数据
                features = normal_df[self.config.feature_cols].values
                targets = normal_df[self.config.target_cols].values
                
                self.feature_mean = features.mean(axis=0)
                self.feature_std = features.std(axis=0)
                self.feature_std[self.feature_std < 1e-8] = 1.0
                
                self.target_mean = targets.mean(axis=0)
                self.target_std = targets.std(axis=0)
                self.target_std[self.target_std < 1e-8] = 1.0
                
                # 保存标准化参数
                norm_params = {
                    'feature_mean': self.feature_mean,
                    'feature_std': self.feature_std,
                    'target_mean': self.target_mean,
                    'target_std': self.target_std,
                    'feature_cols': self.config.feature_cols,
                    'target_cols': self.config.target_cols
                }
                with open(os.path.join(self.config.checkpoint_dir, 'normalization_params.pkl'), 'wb') as f:
                    pickle.dump(norm_params, f)
                print("✅ 保存标准化参数")
                return
        
        raise ValueError("未找到有效数据文件")
    
    def _load_freq_features(self, machine_id):
        """加载特定机器的频域特征"""
        cache_path = os.path.join(self.cache_dir, f'machine_{machine_id:03d}_freq.npy')
        if os.path.exists(cache_path):
            return np.load(cache_path)
        return None
    
    def _process_file(self, filepath):
        """处理单个CSV文件，生成训练样本"""
        machine_id = int(os.path.basename(filepath).split('_')[1].split('.')[0])
        df = pd.read_csv(filepath)
        normal_df = df[df['fault_label'] == 0]
        
        if len(normal_df) < self.config.seq_len + self.config.pred_len:
            return
        
        # 加载缓存的频域特征
        freq_features = self._load_freq_features(machine_id)
        if freq_features is None or len(freq_features) != len(normal_df):
            print(f"⚠️  机器 {machine_id} 的频域特征缓存无效，跳过")
            return
        
        # 标准化时域特征
        features = normal_df[self.config.feature_cols].values
        targets = normal_df[self.config.target_cols].values
        features_norm = (features - self.feature_mean) / self.feature_std
        targets_norm = (targets - self.target_mean) / self.target_std
        
        # 合并时域+频域特征
        combined_features = np.concatenate([features_norm, freq_features], axis=1)
        
        # 生成序列
        n_samples = len(normal_df) - self.config.seq_len - self.config.pred_len + 1
        indices = np.arange(n_samples)
        
        # 划分训练/验证集
        val_size = int(n_samples * self.val_ratio)
        if self.split == 'train':
            indices = indices[:-val_size] if val_size > 0 else indices
        else:
            indices = indices[-val_size:] if val_size > 0 else []
        
        for idx in indices:
            seq = combined_features[idx:idx+self.config.seq_len]
            target_idx = idx + self.config.seq_len + self.config.pred_len - 1
            target_val = targets_norm[target_idx]
            yield torch.from_numpy(seq.astype(np.float32)), torch.from_numpy(target_val.astype(np.float32))
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # 单进程
            for filepath in self.files:
                yield from self._process_file(filepath)
        else:
            # 多进程：每个worker处理不同的文件
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files_per_worker = len(self.files) // num_workers
            start_idx = worker_id * files_per_worker
            end_idx = start_idx + files_per_worker if worker_id < num_workers - 1 else len(self.files)
            
            for filepath in self.files[start_idx:end_idx]:
                yield from self._process_file(filepath)

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def validate_model(model, config, scaler):
    """验证模型"""
    val_dataset = StreamingPhysicalDataset(config.data_dir, config, split='val')
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=2, pin_memory=True)
    
    model.eval()
    total_loss = 0
    total_batches = 0
    max_val_batches = 500  # 限制验证批次数量以节省时间
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_loader):
            if batch_idx >= max_val_batches:
                break
                
            data, targets = data.to(config.device, non_blocking=True), targets.to(config.device, non_blocking=True)
            
            with autocast(device_type='cuda'):
                outputs = model(data)
                mse_loss = nn.MSELoss()(outputs, targets)
                
                # 物理损失
                if isinstance(model, nn.DataParallel):
                    physics_loss = model.module.physics_loss(outputs, targets)
                    freq_loss = model.module.frequency_loss(outputs, targets)
                else:
                    physics_loss = model.physics_loss(outputs, targets)
                    freq_loss = model.frequency_loss(outputs, targets)
                
                loss = mse_loss + config.lambda_physics * physics_loss + config.lambda_freq * freq_loss
            
            # 检查损失是否为NaN或无穷大
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  跳过验证批次 {batch_idx}，检测到无效损失值")
                continue
            
            total_loss += loss.item()
            total_batches += 1
    
    return total_loss / total_batches if total_batches > 0 else float('inf')

def train_model(config):
    print("="*80)
    print("🚀 启动流式训练（低内存占用 + 全量数据）")
    print("="*80)
    
    # 创建模型
    model = EnhancedPhysicalPredictor(config)
    if len(config.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=config.gpu_ids)
    model = model.to(config.device)
    print(f"✅ 模型创建完成 | 参数量: {count_parameters(model):,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 计算训练步数（估算）
    total_samples = 0
    for filename in os.listdir(config.data_dir):
        if filename.startswith('machine_') and filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(config.data_dir, filename))
            total_samples += len(df[df['fault_label'] == 0])  # 仅使用正常样本
    
    steps_per_epoch = total_samples // (config.batch_size * config.gradient_accumulation_steps)
    if steps_per_epoch == 0:
        steps_per_epoch = 1  # 确保至少有一个步骤
    
    print(f"📊 估算训练样本总数: {total_samples:,}")
    print(f"📊 每个epoch步数: {steps_per_epoch}")
    
    # 训练循环
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # 添加提前停止相关参数
    patience = 5  # 允许连续5个epoch验证损失不下降后停止训练
    patience_counter = 0  # 计数器
    min_delta = 0.001  # 验证损失需要下降的最小值，否则视为停滞
    
    for epoch in range(config.epochs):
        # 创建训练数据集和加载器
        train_dataset = StreamingPhysicalDataset(config.data_dir, config, split='train')
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)
        
        model.train()
        total_loss = 0
        total_mse_loss = 0
        total_physics_loss = 0
        total_freq_loss = 0
        num_batches = 0
        
        # 设置进度条
        train_pbar = tqdm(enumerate(train_loader), total=steps_per_epoch, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for batch_idx, (data, targets) in train_pbar:
            data, targets = data.to(config.device, non_blocking=True), targets.to(config.device, non_blocking=True)
            
            with autocast(device_type='cuda'):
                outputs = model(data)
                mse_loss = nn.MSELoss()(outputs, targets)
                
                # 物理损失
                if isinstance(model, nn.DataParallel):
                    physics_loss = model.module.physics_loss(outputs, targets)
                    freq_loss = model.module.frequency_loss(outputs, targets)
                else:
                    physics_loss = model.physics_loss(outputs, targets)
                    freq_loss = model.frequency_loss(outputs, targets)
                
                loss = mse_loss + config.lambda_physics * physics_loss + config.lambda_freq * freq_loss
            
            # 检查损失是否为NaN或无穷大，如果是则跳过此批次
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  跳过批次 {batch_idx}，检测到无效损失值")
                continue
            
            # 反向传播
            scaler.scale(loss / config.gradient_accumulation_steps).backward()
            
            # 梯度累积更新
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # 梯度裁剪以防止梯度爆炸
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_physics_loss += physics_loss.item()
            total_freq_loss += freq_loss.item()
            num_batches += 1
            
            # 更新进度条
            train_pbar.set_postfix({
                'Loss': f"{total_loss/num_batches:.6f}",
                'MSE': f"{total_mse_loss/num_batches:.6f}",
                'Physics': f"{total_physics_loss/num_batches:.6f}",
                'Freq': f"{total_freq_loss/num_batches:.6f}"
            })
            
            # 限制每轮训练的步数，避免过长的训练时间
            if num_batches >= steps_per_epoch:
                break
        
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        
        # 验证
        val_loss = validate_model(model, config, scaler)
        val_losses.append(val_loss)
        
        # 手动更新学习率
        if epoch < config.warmup_epochs:
            # Warmup阶段：线性增长学习率
            lr = config.lr * (epoch + 1) / config.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # 主训练阶段：余弦退火
            import math
            lr = config.lr * 0.5 * (1 + math.cos(math.pi * (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存检查点
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(config.checkpoint_dir, 'best_physical_predictor.pth'))
            print(f"✅ 最佳模型已保存 (Loss: {best_loss:.6f})")
            patience_counter = 0  # 重置计数器
        else:
            patience_counter += 1
            
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(config.checkpoint_dir, f'checkpoint_epoch{epoch+1}.pth'))
            print(f"💾 epoch {epoch+1} 的检查点已保存")
        
        # 检查是否需要提前停止
        if patience_counter >= patience:
            print(f"⚠️  验证损失连续 {patience} 个epoch未改善，停止训练...")
            break
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('模型训练曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses[1:], label='Train Loss (zoomed)')
    plt.plot(val_losses[1:], label='Validation Loss (zoomed)')
    plt.title('模型训练曲线 (缩放)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.checkpoint_dir, 'training_curve_enhanced.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎉 训练完成！最佳验证损失: {best_loss:.6f}")

# ==================== 主函数 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练增强版物理预测模型')
    parser.add_argument('--resume', type=str, default=None, help='从指定路径恢复训练')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='GPU IDs (例如: "0,1,2,3")')
    args = parser.parse_args()
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    
    config = Config(resume_from=args.resume, gpu_ids=gpu_ids)
    train_model(config)