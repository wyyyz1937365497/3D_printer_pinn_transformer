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
    model = CorrectionController(config)
    if len(config.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=config.gpu_ids)
    model = model.to(config.device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 训练循环
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # 添加提前停止相关参数
    patience = 5  # 允许连续5个epoch验证损失不下降后停止训练
    patience_counter = 0  # 计数器
    min_delta = 0.001  # 验证损失需要下降的最小值
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_idx, (features, corrections) in enumerate(train_pbar):
            features, corrections = features.to(config.device, non_blocking=True), corrections.to(config.device, non_blocking=True)
            
            with autocast(device_type='cuda'):
                outputs = model(features)
                loss = nn.MSELoss()(outputs, corrections)
            
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
        val_loss = validate_correction_controller(model, val_loader, config, scaler)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(config.checkpoint_dir, 'best_correction_controller.pth'))
            print(f"✅ 最佳模型已保存 (Loss: {best_loss:.6f})")
            patience_counter = 0  # 重置计数器
        else:
            patience_counter += 1
            
        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(config.checkpoint_dir, f'correction_controller_epoch{epoch+1}.pth'))
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
    plt.title('校正控制器训练曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.checkpoint_dir, 'correction_controller_training_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎉 校正控制器训练完成！最佳验证损失: {best_loss:.6f}")

# ==================== 主函数 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练流式矫正控制器')
    parser.add_argument('--resume', type=str, default=None, help='从指定路径恢复训练')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='GPU IDs (例如: "0,1,2,3")')
    args = parser.parse_args()
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    
    config = Config(resume_from=args.resume, gpu_ids=gpu_ids)
    train_correction_controller(config)