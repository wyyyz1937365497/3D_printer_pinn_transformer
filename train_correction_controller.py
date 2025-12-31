# train_correction_controller_streaming.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import os
import time
import pickle
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import argparse
from datetime import datetime, timedelta

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class Config:
    def __init__(self, resume_from=None, gpu_ids=[0]):
        self.data_dir = 'printer_dataset_correction/'
        self.batch_size = 1024  # 增加batch size
        self.lr = 3e-4
        self.epochs = 30  # 调整epoch数
        self.gpu_ids = gpu_ids
        self.resume_from = resume_from
        self.device = f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu'
        self.checkpoint_dir = './checkpoints_correction_controller'  # 修改检查点目录
        self.seq_len = 50
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # 修改特征列，使用实际数据集中的列名
        self.feature_cols = [
            'nozzle_x', 'nozzle_y', 'nozzle_z',
            'temperature_C', 'vibration_disp_x_m', 'vibration_disp_y_m',
            'vibration_vel_x_m_s', 'vibration_vel_y_m_s',
            'motor_current_x_A', 'motor_current_y_A',
            'pressure_bar'
        ]
        self.correction_cols = ['correction_x_mm', 'correction_y_mm', 'correction_temp_C']
        self.input_dim = len(self.feature_cols)
        self.output_dim = len(self.correction_cols)

class CorrectionController(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config.output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class StreamingCorrectionDataset(IterableDataset):
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
            self.correction_mean = norm_params.get('correction_mean', np.zeros(len(config.correction_cols)))
            self.correction_std = norm_params.get('correction_std', np.ones(len(config.correction_cols)))
    
    def _load_norm_params(self):
        path = './checkpoints_physical_predictor_enhanced/normalization_params.pkl'  # 修改为正确的路径
        if os.path.exists(path):
            with open(path, 'rb') as f:
                params = pickle.load(f)
                self.feature_mean = params['feature_mean']
                self.feature_std = params['feature_std']
        else:
            raise FileNotFoundError("请先训练物理预测模型以获取标准化参数")
        
        # 尝试加载矫正参数
        correction_path = './checkpoints_correction_controller/correction_params.pkl'
        if os.path.exists(correction_path):
            with open(correction_path, 'rb') as f:
                corr_params = pickle.load(f)
                self.correction_mean = corr_params['correction_mean']
                self.correction_std = corr_params['correction_std']
        else:
            # 估算
            self.correction_mean = np.array([0.0, 0.0, 0.0])
            self.correction_std = np.array([0.01, 0.01, 10.0])
    
    def _process_file(self, filepath):
        df = pd.read_csv(filepath)
        normal_df = df[df['fault_label'] == 0]
        # 只保留振动幅度大于阈值的样本（需要矫正的区域）
        mask = (normal_df['vibration_disp_x_m'].abs() + normal_df['vibration_disp_y_m'].abs()) > 0.0005
        normal_df = normal_df[mask]
        
        if len(normal_df) == 0:
            return
        
        features = normal_df[self.config.feature_cols].values
        corrections = normal_df[self.config.correction_cols].values
        
        features = (features - self.feature_mean) / self.feature_std
        corrections = (corrections - self.correction_mean) / self.correction_std
        
        indices = np.arange(len(features))
        val_size = int(len(indices) * self.val_ratio)
        
        if self.split == 'train':
            indices = indices[:-val_size] if val_size > 0 else indices
        else:
            indices = indices[-val_size:] if val_size > 0 else []
        
        for i in indices:
            yield torch.from_numpy(features[i].astype(np.float32)), torch.from_numpy(corrections[i].astype(np.float32))
    
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

def train_correction_controller(config):
    print("=" * 80)
    print("🚀 训练流式矫正控制器")
    print("=" * 80)
    
    # 创建数据集
    train_dataset = StreamingCorrectionDataset(config.data_dir, config, split='train')
    val_dataset = StreamingCorrectionDataset(config.data_dir, config, split='val')
    
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
    model = CorrectionController(config).to(config.device)
    if len(config.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=config.gpu_ids)
    
    print(f"✅ 模型创建完成 | 参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
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
    
    print("\n🔥 开始训练矫正控制器...")
    print("-" * 80)
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        
        for batch_idx, (feat, corr) in enumerate(train_loader):
            feat, corr = feat.to(config.device), corr.to(config.device)
            
            with autocast('cuda'):
                pred = model(feat)
                loss = criterion(pred, corr)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for feat, corr in val_loader:
                feat, corr = feat.to(config.device), corr.to(config.device)
                pred = model(feat)
                loss = criterion(pred, corr)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        epoch_time = time.time() - epoch_start
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
            torch.save(checkpoint_data, os.path.join(config.checkpoint_dir, 'best_correction_controller.pth'))
            print(f"   💾 保存最佳矫正控制器 (验证损失: {best_val_loss:.6f})")
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('矫正控制器训练过程')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.checkpoint_dir, 'correction_training_curve.png'))
    
    print("\n" + "=" * 80)
    print(f"🎉 矫正控制器训练完成! 最佳验证损失: {best_val_loss:.6f}")
    print("=" * 80)

# ==================== 主函数 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练流式矫正控制器')
    parser.add_argument('--resume', type=str, default=None, help='从指定路径恢复训练')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='GPU IDs (例如: "0,1,2,3")')
    args = parser.parse_args()
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    
    config = Config(resume_from=args.resume, gpu_ids=gpu_ids)
    train_correction_controller(config)