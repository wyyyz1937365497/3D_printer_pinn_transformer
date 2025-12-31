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
warnings.filterwarnings('ignore')

class Config:
    def __init__(self, resume_from=None, gpu_ids=[0]):
        self.data_dir = 'printer_dataset_correction/'  # 数据目录
        self.seq_len = 250
        self.pred_len = 50
        self.batch_size = 2048
        self.gradient_accumulation_steps = 2
        self.model_dim = 192
        self.num_heads = 8
        self.num_layers = 6
        self.dim_feedforward = 768
        self.dropout = 0.1
        self.lr = 5e-5
        self.epochs = 60
        self.gpu_ids = gpu_ids
        self.resume_from = resume_from
        self.device = f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu'
        self.lambda_physics = 0.4
        self.lambda_freq = 0.3
        self.checkpoint_dir = './checkpoints_physical_predictor_streaming'
        self.warmup_epochs = 5
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.feature_cols = [
            'ctrl_T_target', 'ctrl_speed_set', 'ctrl_pos_x', 'ctrl_pos_y', 'ctrl_pos_z',
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
            vib_x_smooth = torch.mean(torch.abs(torch.diff(pred[:, 0])))
            vib_y_smooth = torch.mean(torch.abs(torch.diff(pred[:, 1])))
            loss += 0.3 * (vib_x_smooth + vib_y_smooth)
            
            dT_dt = torch.diff(pred[:, 2]) / dt
            d2T_dt2 = torch.diff(dT_dt) / dt
            thermal_smooth = torch.mean(torch.abs(d2T_dt2))
            loss += 0.2 * thermal_smooth
            
            temp_change = torch.mean(torch.abs(torch.diff(pred[:, 2])))
            loss += 0.3 * torch.clamp(temp_change - 1.0, min=0)
            
            vib_mag = torch.sqrt(pred[:,0]**2 + pred[:,1]**2)
            cur_mag = torch.sqrt(pred[:,3]**2 + pred[:,4]**2)
            if len(vib_mag) > 1:
                vib_mean = torch.mean(vib_mag)
                cur_mean = torch.mean(cur_mag)
                vib_c = vib_mag - vib_mean
                cur_c = cur_mag - cur_mean
                corr = torch.sum(vib_c * cur_c) / (torch.sqrt(torch.sum(vib_c**2)) * torch.sqrt(torch.sum(cur_c**2)) + 1e-8)
                loss += 0.2 * torch.relu(0.3 - corr)
        return loss
    
    def frequency_loss(self, pred, target):
        pred = pred.float()
        target = target.float()
        pred_x_fft = torch.fft.rfft(pred[:, 0])
        pred_y_fft = torch.fft.rfft(pred[:, 1])
        target_x_fft = torch.fft.rfft(target[:, 0])
        target_y_fft = torch.fft.rfft(target[:, 1])
        
        max_bins = min(10, pred_x_fft.shape[0])
        pred_x_mag = torch.abs(pred_x_fft[:max_bins])
        pred_y_mag = torch.abs(pred_y_fft[:max_bins])
        target_x_mag = torch.abs(target_x_fft[:max_bins])
        target_y_mag = torch.abs(target_y_fft[:max_bins])
        
        return nn.MSELoss()(pred_x_mag, target_x_mag) + nn.MSELoss()(pred_y_mag, target_y_mag)

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

def train_model(config):
    print("=" * 80)
    print("🚀 启动流式训练（低内存占用 + 全量数据）")
    print("=" * 80)
    
    # 创建数据集
    train_dataset = StreamingPhysicalDataset(config.data_dir, config, split='train')
    val_dataset = StreamingPhysicalDataset(config.data_dir, config, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    # 创建模型
    model = EnhancedPhysicalPredictor(config).to(config.device)
    if len(config.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=config.gpu_ids)
    
    print(f"✅ 模型创建完成 | 参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
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
    
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda')
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    if config.resume_from and os.path.exists(config.resume_from):
        print(f"🔄 从检查点恢复训练: {config.resume_from}")
        checkpoint = torch.load(config.resume_from, map_location=config.device)
        model_state = checkpoint['model_state_dict']
        
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        
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
                
                # 物理约束损失
                if isinstance(model, nn.DataParallel):
                    physics_loss = model.module.physics_loss(pred, target)
                    freq_loss = model.module.frequency_loss(pred, target)
                else:
                    physics_loss = model.physics_loss(pred, target)
                    freq_loss = model.frequency_loss(pred, target)
                
                total_batch_loss = data_loss + config.lambda_physics * physics_loss + config.lambda_freq * freq_loss
            
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
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('物理预测模型训练过程')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.checkpoint_dir, 'training_curve.png'))
    
    print("\n" + "=" * 80)
    print(f"🎉 训练完成! 最佳验证损失: {best_val_loss:.6f}")
    print("=" * 80)

# ==================== 主函数 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练增强版物理预测模型')
    parser.add_argument('--resume', type=str, default=None, help='从指定路径恢复训练')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='GPU IDs (例如: "0,1,2,3")')
    args = parser.parse_args()
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    
    config = Config(resume_from=args.resume, gpu_ids=gpu_ids)
    train_model(config)