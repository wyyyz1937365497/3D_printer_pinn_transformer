# train_physical_predictor.py
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

# ==================== 配置参数 ====================
class Config:
    def __init__(self):
        self.data_path = 'printer_dataset_correction/printer_gear_correction_dataset.csv'
        self.seq_len = 100           # 历史窗口长度 (100ms)
        self.pred_len = 20           # 预测长度 (20ms)
        self.batch_size = 512
        self.gradient_accumulation_steps = 2
        self.model_dim = 128
        self.num_heads = 8
        self.num_layers = 4
        self.dim_feedforward = 512
        self.dropout = 0.1
        self.lr = 1e-4
        self.epochs = 40
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lambda_physics = 0.2    # 物理约束权重
        self.checkpoint_dir = './checkpoints_physical_predictor'
        self.max_samples = 100000
        self.warmup_epochs = 3
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
        
        self.input_dim = len(self.feature_cols)
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

# ==================== 物理预测模型 ====================
class PhysicalPredictor(nn.Module):
    def __init__(self, config):
        super(PhysicalPredictor, self).__init__()
        self.config = config
        
        # 编码器
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim // 2, config.output_dim)
        )
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x_emb = self.encoder_embedding(x)
        x_emb = self.pos_encoder(x_emb)
        memory = self.encoder(x_emb)  # [batch, seq_len, model_dim]
        
        # 使用序列的最后一个时间步进行预测
        last_state = memory[:, -1, :]  # [batch, model_dim]
        prediction = self.decoder(last_state)  # [batch, output_dim]
        
        return prediction
    
    def physics_loss(self, predictions, targets):
        """物理约束损失"""
        loss = 0.0
        
        # 1. 振动平滑约束
        vib_x_pred = predictions[:, 0]
        vib_y_pred = predictions[:, 1]
        
        # 振动应该平滑变化
        if len(vib_x_pred) > 1:
            vib_x_smoothness = torch.mean(torch.abs(torch.diff(vib_x_pred)))
            vib_y_smoothness = torch.mean(torch.abs(torch.diff(vib_y_pred)))
            loss += 0.5 * (vib_x_smoothness + vib_y_smoothness)
        
        # 2. 温度变化约束
        temp_pred = predictions[:, 2]
        if len(temp_pred) > 1:
            temp_change = torch.mean(torch.abs(torch.diff(temp_pred)))
            # 温度不应该变化太快
            loss += 0.3 * torch.clamp(temp_change - 0.5, min=0)
        
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
            
            # 希望有正相关
            loss += 0.2 * torch.relu(0.3 - correlation)
        
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

# ==================== 数据处理器 ====================
def prepare_data(config):
    print("🔄 加载和处理数据...")
    df = pd.read_csv(config.data_path)
    
    # 选择正常机器的数据（无故障）
    normal_df = df[df['fault_label'] == 0].copy()
    print(f"   正常机器数据: {len(normal_df)} / {len(df)}")
    
    # 采样以减少数据量
    if len(normal_df) > config.max_samples * 2:
        normal_df = normal_df.sample(n=config.max_samples * 2, random_state=42)
        print(f"   采样后数据量: {len(normal_df)}")
    
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
        
        for i in range(len(machine_data) - config.seq_len - config.pred_len + 1):
            seq = machine_features[i:i+config.seq_len]
            target_idx = i + config.seq_len + config.pred_len - 1
            target_val = machine_targets[target_idx]
            
            sequences.append(seq)
            target_values.append(target_val)
    
    sequences = np.array(sequences)
    target_values = np.array(target_values)
    
    # 限制样本数量
    if len(sequences) > config.max_samples:
        idx = np.random.choice(len(sequences), config.max_samples, replace=False)
        sequences = sequences[idx]
        target_values = target_values[idx]
    
    # 分割训练集和验证集
    train_seq, val_seq, train_targets, val_targets = train_test_split(
        sequences, target_values, test_size=0.2, random_state=42
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
        'target_cols': config.target_cols
    }
    
    with open(os.path.join(config.checkpoint_dir, 'normalization_params.pkl'), 'wb') as f:
        pickle.dump(normalization_params, f)
    
    return (train_seq, train_targets), (val_seq, val_targets), normalization_params

# ==================== 训练函数 ====================
def train_model(config):
    print("=" * 80)
    print("🚀 训练物理预测模型")
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
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    model = PhysicalPredictor(config).to(config.device)
    print(f"✅ 模型创建完成 | 参数量: {sum(p.numel() for p in model.parameters())}")
    
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
    
    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("\n🔥 开始训练...")
    print("-" * 80)
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        
        for batch_idx, (seq, target) in enumerate(train_loader):
            seq, target = seq.to(config.device), target.to(config.device)
            
            with autocast('cuda'):
                pred = model(seq)
                data_loss = criterion(pred, target)
                
                # 物理约束损失
                physics_loss = model.physics_loss(pred, target)
                total_batch_loss = data_loss + config.lambda_physics * physics_loss
            
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
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for seq, target in val_loader:
                seq, target = seq.to(config.device), target.to(config.device)
                pred = model(seq)
                loss = criterion(pred, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - epoch_start
        print(f"✅ Epoch {epoch+1:2d}/{config.epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'config': config.__dict__
            }, os.path.join(config.checkpoint_dir, 'best_physical_predictor.pth'))
            print(f"   💾 保存最佳模型 (验证损失: {best_val_loss:.6f})")
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练过程')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.checkpoint_dir, 'training_curve.png'))
    
    print("\n" + "=" * 80)
    print(f"🎉 训练完成! 最佳验证损失: {best_val_loss:.6f}")
    print("=" * 80)

# ==================== 主函数 ====================
if __name__ == "__main__":
    config = Config()
    train_model(config)