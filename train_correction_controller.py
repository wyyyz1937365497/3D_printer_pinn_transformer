# train_correction_controller.py
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
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

# ==================== 配置参数 ====================
class Config:
    def __init__(self):
        self.data_path = 'printer_dataset_correction/printer_gear_correction_dataset.csv'
        self.pred_model_path = './checkpoints_physical_predictor/best_physical_predictor.pth'
        self.batch_size = 1024
        self.lr = 3e-4
        self.epochs = 30
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint_dir = './checkpoints_correction_controller'
        self.max_samples = 50000
        self.seq_len = 50  # 短序列，用于实时控制
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 特征列
        self.feature_cols = [
            'ctrl_T_target', 'ctrl_speed_set', 'ctrl_pos_x', 'ctrl_pos_y', 'ctrl_pos_z',
            'temperature_C', 'vibration_disp_x_m', 'vibration_disp_y_m',
            'vibration_vel_x_m_s', 'vibration_vel_y_m_s',
            'motor_current_x_A', 'motor_current_y_A',
            'pressure_bar'
        ]
        
        # 矫正目标列
        self.correction_cols = [
            'correction_x_mm', 'correction_y_mm', 'correction_temp_C'
        ]
        
        self.input_dim = len(self.feature_cols)
        self.output_dim = len(self.correction_cols)

# ==================== 矫正控制器模型 ====================
class CorrectionController(nn.Module):
    def __init__(self, config):
        super(CorrectionController, self).__init__()
        self.config = config
        
        # 两层MLP
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config.output_dim)
        )
    
    def forward(self, x):
        # x: [batch, input_dim]
        return self.net(x)

# ==================== 数据集类 ====================
class CorrectionDataset(Dataset):
    def __init__(self, features, corrections):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.corrections = torch.tensor(corrections, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.corrections[idx]

# ==================== 数据处理器 ====================
def prepare_correction_data(config):
    print("🔄 加载矫正数据...")
    df = pd.read_csv(config.data_path)
    
    # 选择正常机器但有矫正信号的数据
    normal_df = df[df['fault_label'] == 0].copy()
    
    # 我们只关心振动较大的区域（需要矫正的地方）
    normal_df = normal_df[normal_df['vibration_disp_x_m'].abs() + normal_df['vibration_disp_y_m'].abs() > 0.0005]
    
    print(f"   有效矫正样本: {len(normal_df)}")
    
    # 采样
    if len(normal_df) > config.max_samples:
        normal_df = normal_df.sample(n=config.max_samples, random_state=42)
        print(f"   采样后样本数: {len(normal_df)}")
    
    # 提取特征和矫正目标
    features = normal_df[config.feature_cols].values
    corrections = normal_df[config.correction_cols].values
    
    # 从物理预测模型加载标准化参数
    norm_params_path = './checkpoints_physical_predictor/normalization_params.pkl'
    if os.path.exists(norm_params_path):
        with open(norm_params_path, 'rb') as f:
            norm_params = pickle.load(f)
        
        feature_mean = norm_params['feature_mean']
        feature_std = norm_params['feature_std']
        features_norm = (features - feature_mean) / feature_std
        print("✅ 使用物理预测模型的标准化参数")
    else:
        # 如果没有，自己计算
        feature_mean = features.mean(axis=0)
        feature_std = features.std(axis=0)
        feature_std[feature_std < 1e-8] = 1.0
        features_norm = (features - feature_mean) / feature_std
    
    # 目标标准化
    correction_mean = corrections.mean(axis=0)
    correction_std = corrections.std(axis=0)
    correction_std[correction_std < 1e-8] = 1.0
    corrections_norm = (corrections - correction_mean) / correction_std
    
    # 保存矫正标准化参数
    correction_params = {
        'correction_mean': correction_mean,
        'correction_std': correction_std,
        'correction_cols': config.correction_cols
    }
    
    with open(os.path.join(config.checkpoint_dir, 'correction_params.pkl'), 'wb') as f:
        pickle.dump(correction_params, f)
    
    # 分割数据集
    train_feat, val_feat, train_corr, val_corr = train_test_split(
        features_norm, corrections_norm, test_size=0.2, random_state=42
    )
    
    print(f"📊 总样本数: {len(features_norm)}")
    print(f"   训练集: {len(train_feat)}, 验证集: {len(val_feat)}")
    
    return (train_feat, train_corr), (val_feat, val_corr), correction_params

# ==================== 训练函数 ====================
def train_correction_controller(config):
    print("=" * 80)
    print("🚀 训练矫正控制器")
    print("=" * 80)
    
    # 准备数据
    (train_feat, train_corr), (val_feat, val_corr), corr_params = prepare_correction_data(config)
    
    # 创建数据集和数据加载器
    train_dataset = CorrectionDataset(train_feat, train_corr)
    val_dataset = CorrectionDataset(val_feat, val_corr)
    
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
    model = CorrectionController(config).to(config.device)
    print(f"✅ 模型创建完成 | 参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 损失函数
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda')
    
    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("\n🔥 开始训练矫正控制器...")
    print("-" * 80)
    
    for epoch in range(config.epochs):
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
            }, os.path.join(config.checkpoint_dir, 'best_correction_controller.pth'))
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
    config = Config()
    train_correction_controller(config)