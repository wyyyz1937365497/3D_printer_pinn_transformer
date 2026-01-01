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
        self.batch_size = 256  # 增加batch size
        self.lr = 1e-4
        self.epochs = 25  # 调整epoch数
        self.gpu_ids = gpu_ids
        self.resume_from = resume_from
        self.device = f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu'
        self.checkpoint_dir = './checkpoints_fault_classifier'  # 修改检查点目录
        self.seq_len = 150
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # 修改特征列，使用实际数据集中的列名
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
        path = './checkpoints_physical_predictor_enhanced/normalization_params.pkl'  # 修改为正确的路径
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
    model = FaultClassifier(config)
    if len(config.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=config.gpu_ids)
    model = model.to(config.device)
    
    # 损失函数（使用标签平滑）
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 训练循环
    best_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    # 添加提前停止相关参数
    patience = 5  # 允许连续5个epoch验证准确率不提升后停止训练
    patience_counter = 0  # 计数器
    min_delta = 0.001  # 验证准确率需要提升的最小值
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_idx, (features, labels) in enumerate(train_pbar):
            features, labels = features.to(config.device, non_blocking=True), labels.to(config.device, non_blocking=True).long()
            
            with autocast(device_type='cuda'):
                outputs = model(features)
                loss = criterion(outputs, labels)
            
            # 检查损失是否为NaN或无穷大
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  跳过批次 {batch_idx}，检测到无效损失值")
                continue
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            
            train_pbar.set_postfix({'Loss': f"{loss.item():.6f}"})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        train_losses.append(avg_loss)
        
        # 验证
        val_acc, val_loss = validate_fault_classifier(model, val_loader, config, scaler, criterion)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_loss:.6f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.6f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, os.path.join(config.checkpoint_dir, 'best_fault_classifier.pth'))
            print(f"✅ 最佳模型已保存 (Acc: {best_acc:.4f})")
            patience_counter = 0  # 重置计数器
        else:
            patience_counter += 1
            
        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
            }, os.path.join(config.checkpoint_dir, f'fault_classifier_epoch{epoch+1}.pth'))
            print(f"💾 epoch {epoch+1} 的检查点已保存")
        
        # 检查是否需要提前停止
        if patience_counter >= patience:
            print(f"⚠️  验证准确率连续 {patience} 个epoch未提升，停止训练...")
            break
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('故障分类器训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.title('故障分类器验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.checkpoint_dir, 'fault_classifier_training_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎉 故障分类器训练完成！最佳验证准确率: {best_acc:.4f}")

# ==================== 主函数 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练流式故障分类器')
    parser.add_argument('--resume', type=str, default=None, help='从指定路径恢复训练')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='GPU IDs (例如: "0,1,2,3")')
    args = parser.parse_args()
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    
    config = Config(resume_from=args.resume, gpu_ids=gpu_ids)
    train_fault_classifier(config)