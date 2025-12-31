# train_fault_classifier_streaming.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import os
import time
import pickle
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from datetime import datetime, timedelta

class Config:
    def __init__(self, resume_from=None, gpu_ids=[0]):
        self.data_dir = 'printer_dataset_correction/'
        self.batch_size = 256
        self.lr = 1e-4
        self.epochs = 25
        self.gpu_ids = gpu_ids
        self.resume_from = resume_from
        self.device = f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu'
        self.checkpoint_dir = './checkpoints_fault_classifier_streaming'
        self.seq_len = 150
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.feature_cols = [
            'temperature_C', 'vibration_disp_x_m', 'vibration_disp_y_m',
            'vibration_vel_x_m_s', 'vibration_vel_y_m_s',
            'motor_current_x_A', 'motor_current_y_A',
            'pressure_bar'
        ]
        self.n_classes = 4  # 0=正常, 1=喷嘴堵塞, 2=机械松动, 3=电机故障
        self.input_dim = len(self.feature_cols)
        self.model_dim = 128

class FaultClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
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
        x = self.embedding(x)
        memory = self.encoder(x)
        seq_avg = torch.mean(memory, dim=1)
        return self.classifier(seq_avg)

class StreamingFaultDataset(IterableDataset):
    def __init__(self, data_dir, config, split='train', val_ratio=0.2, norm_params=None):
        self.data_dir = data_dir
        self.config = config
        self.split = split
        self.val_ratio = val_ratio
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                           if f.startswith('machine_') and f.endswith('.csv')])
        
        # 加载标准化参数
        if norm_params is None:
            self._load_norm_params()
        else:
            self.feature_mean = norm_params['feature_mean']
            self.feature_std = norm_params['feature_std']
    
    def _load_norm_params(self):
        path = './checkpoints_physical_predictor_streaming/normalization_params.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                params = pickle.load(f)
                # 只取故障分类需要的特征
                relevant_idx = [i for i, col in enumerate(params['feature_cols']) 
                               if col in self.config.feature_cols]
                self.feature_mean = params['feature_mean'][relevant_idx]
                self.feature_std = params['feature_std'][relevant_idx]
        else:
            # 估算
            self.feature_mean = np.zeros(len(self.config.feature_cols))
            self.feature_std = np.ones(len(self.config.feature_cols))
    
    def _process_file(self, filepath):
        df = pd.read_csv(filepath)
        features = df[self.config.feature_cols].values
        labels = df['fault_label'].values
        
        features = (features - self.feature_mean) / self.feature_std
        
        n = len(features)
        step = self.config.seq_len // 2  # 50% 重叠采样
        
        for i in range(0, n - self.config.seq_len, step):
            seq = features[i:i+self.config.seq_len]
            label = labels[i+self.config.seq_len-1]  # 使用序列末尾的标签
            
            # 划分训练/验证
            total_windows = (n - self.config.seq_len) // step
            val_start = int(total_windows * (1 - self.val_ratio))
            current_window = i // step
            
            if (self.split == 'train' and current_window < val_start) or \
               (self.split == 'val' and current_window >= val_start):
                yield torch.from_numpy(seq.astype(np.float32)), torch.tensor(label, dtype=torch.long)
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            for f in self.files:
                yield from self._process_file(f)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files_per_worker = len(self.files) // num_workers
            start_idx = worker_id * files_per_worker
            end_idx = start_idx + files_per_worker if worker_id < num_workers - 1 else len(self.files)
            
            for f in self.files[start_idx:end_idx]:
                yield from self._process_file(f)

def train_fault_classifier(config):
    print("=" * 80)
    print("🚀 训练流式故障分类器")
    print("=" * 80)
    
    # 创建数据集
    train_dataset = StreamingFaultDataset(config.data_dir, config, split='train')
    val_dataset = StreamingFaultDataset(config.data_dir, config, split='val')
    
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
    model = FaultClassifier(config).to(config.device)
    if len(config.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=config.gpu_ids)
    
    print(f"✅ 模型创建完成 | 参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    criterion = nn.CrossEntropyLoss()
    
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
        correct = 0
        total = 0
        
        with torch.no_grad():
            for seq, label in val_loader:
                seq, label = seq.to(config.device), label.to(config.device)
                pred = model(seq)
                loss = criterion(pred, label)
                val_loss += loss.item()
                
                _, predicted = torch.max(pred, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        accuracy = 100 * correct / total
        scheduler.step(avg_val_loss)
        
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - epoch_start
        remaining_epochs = config.epochs - epoch - 1
        remaining_time = elapsed_time * remaining_epochs
        remaining_time_str = str(timedelta(seconds=int(remaining_time)))
        
        print(f"✅ Epoch {epoch+1:2d}/{config.epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Accuracy: {accuracy:.2f}% | "
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
            print(f"   💾 保存最佳故障分类器 (验证损失: {best_val_loss:.6f}, 准确率: {accuracy:.2f}%)")
    
    print("\n" + "=" * 80)
    print(f"🎉 故障分类器训练完成! 最佳验证损失: {best_val_loss:.6f}")
    print("=" * 80)

# ==================== 主函数 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练流式故障分类器')
    parser.add_argument('--resume', type=str, default=None, help='从指定路径恢复训练')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='GPU IDs (例如: "0,1,2,3")')
    args = parser.parse_args()
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    
    config = Config(resume_from=args.resume, gpu_ids=gpu_ids)
    train_fault_classifier(config)