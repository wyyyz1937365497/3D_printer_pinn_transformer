# train_fault_classifier.py
# 独立的故障分类器，使用Transformer编码器提取特征
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import time
import pickle
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from datetime import datetime, timedelta

# ==================== 配置参数 ====================
class Config:
    def __init__(self, resume_from=None, gpu_ids=[0]):
        self.data_path = 'printer_dataset_correction/printer_gear_correction_dataset.csv'
        self.batch_size = 256
        self.lr = 1e-4
        self.epochs = 25
        self.gpu_ids = gpu_ids
        self.resume_from = resume_from  # 添加继续训练的路径
        if len(gpu_ids) > 1:
            self.device = f'cuda:{gpu_ids[0]}'  # 主GPU
        else:
            self.device = f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu'
        self.checkpoint_dir = './checkpoints_fault_classifier'
        self.max_samples = 30000
        self.seq_len = 150  # 长序列，捕获故障模式
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 特征列
        self.feature_cols = [
            'temperature_C', 'vibration_disp_x_m', 'vibration_disp_y_m',
            'vibration_vel_x_m_s', 'vibration_vel_y_m_s',
            'motor_current_x_A', 'motor_current_y_A',
            'pressure_bar'
        ]
        
        # 故障类型: 0=正常, 1=喷嘴堵塞, 2=机械松动, 3=电机故障
        self.n_classes = 4
        self.input_dim = len(self.feature_cols)
        self.model_dim = 128

# ==================== 故障分类模型 ====================
class FaultClassifier(nn.Module):
    def __init__(self, config):
        super(FaultClassifier, self).__init__()
        self.embedding = nn.Linear(config.input_dim, config.model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.classifier = nn.Sequential(
            nn.Linear(config.model_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, config.n_classes)
        )
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.embedding(x)  # [batch, seq_len, model_dim]
        memory = self.encoder(x)  # [batch, seq_len, model_dim]
        
        # 使用序列所有时间步的平均值
        seq_avg = torch.mean(memory, dim=1)  # [batch, model_dim]
        
        return self.classifier(seq_avg)

# ==================== 数据集类 ====================
class FaultDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# ==================== 数据处理器 ====================
def prepare_fault_data(config):
    print("🔄 加载故障数据...")
    df = pd.read_csv(config.data_path)
    
    # 特征和标签
    features = df[config.feature_cols].values
    labels = df['fault_label'].values
    
    # 标准化
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0
    features_norm = (features - feature_mean) / feature_std
    
    # 创建序列样本
    sequences = []
    sequence_labels = []
    
    machine_ids = df['machine_id'].unique()
    
    for mid in machine_ids:
        machine_data = df[df['machine_id'] == mid]
        machine_features = features_norm[df['machine_id'] == mid]
        machine_labels = labels[df['machine_id'] == mid]
        
        # 按seq_len长度切分序列
        for i in range(0, len(machine_data) - config.seq_len, config.seq_len):
            seq = machine_features[i:i+config.seq_len]
            # 使用序列最后一个位置的标签
            label = machine_labels[i+config.seq_len-1]
            
            if len(seq) == config.seq_len:  # 确保序列长度正确
                sequences.append(seq)
                sequence_labels.append(label)
    
    sequences = np.array(sequences)
    sequence_labels = np.array(sequence_labels)
    
    # 限制样本数量
    if len(sequences) > config.max_samples:
        idx = np.random.choice(len(sequences), config.max_samples, replace=False)
        sequences = sequences[idx]
        sequence_labels = sequence_labels[idx]
    
    # 分割训练集和验证集
    train_seq, val_seq, train_labels, val_labels = train_test_split(
        sequences, sequence_labels, test_size=0.2, random_state=42
    )
    
    print(f"📊 总样本数: {len(sequences)}")
    print(f"   训练集: {len(train_seq)}, 验证集: {len(val_seq)}")
    print(f"   故障分布: {np.bincount(sequence_labels)}")
    
    return (train_seq, train_labels), (val_seq, val_labels)

# ==================== 训练函数 ====================
def train_fault_classifier(config):
    print("=" * 80)
    print("🚀 训练故障分类器")
    print("=" * 80)
    
    # 准备数据
    (train_seq, train_labels), (val_seq, val_labels) = prepare_fault_data(config)
    
    # 创建数据集和数据加载器
    train_dataset = FaultDataset(train_seq, train_labels)
    val_dataset = FaultDataset(val_seq, val_labels)
    
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
    model = FaultClassifier(config)
    print(f"✅ 模型创建完成 | 参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 检查是否使用多GPU
    if len(config.gpu_ids) > 1:
        print(f"✅ 使用多GPU训练: {config.gpu_ids}")
        model = nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(config.device)
    else:
        model = model.to(config.device)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
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
    
    print("\n🔥 开始训练故障分类器...")
    print("-" * 80)
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        
        for batch_idx, (seq, label) in enumerate(train_loader):
            seq, label = seq.to(config.device), label.to(config.device)
            
            pred = model(seq)
            loss = criterion(pred, label)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for seq, label in val_loader:
                seq, label = seq.to(config.device), label.to(config.device)
                pred = model(seq)
                loss = criterion(pred, label)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        epoch_time = time.time() - epoch_start
        
        # 计算剩余时间
        elapsed_time = time.time() - epoch_start
        remaining_epochs = config.epochs - epoch - 1
        remaining_time = elapsed_time * remaining_epochs
        remaining_time_str = str(timedelta(seconds=int(remaining_time)))
        
        print(f"✅ Epoch {epoch+1:2d}/{config.epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
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
            torch.save(checkpoint_data, os.path.join(config.checkpoint_dir, 'best_fault_classifier.pth'))
            print(f"   💾 保存最佳故障分类器 (验证损失: {best_val_loss:.6f})")
    
    print("\n" + "=" * 80)
    print(f"🎉 故障分类器训练完成! 最佳验证损失: {best_val_loss:.6f}")
    print("=" * 80)

# ==================== 主函数 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练故障分类器')
    parser.add_argument('--resume', type=str, default=None, help='从指定路径恢复训练')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='GPU IDs (例如: "0,1,2,3")')
    args = parser.parse_args()
    
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    config = Config(resume_from=args.resume, gpu_ids=gpu_ids)
    train_fault_classifier(config)