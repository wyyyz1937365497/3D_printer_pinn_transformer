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

# ==================== 配置参数 ====================
class Config:
    def __init__(self):
        self.data_path = 'printer_dataset_correction/printer_gear_correction_dataset.csv'
        self.batch_size = 256
        self.lr = 1e-4
        self.epochs = 25
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

# 其余代码实现类似，专注于故障分类
# 这里省略以节省空间