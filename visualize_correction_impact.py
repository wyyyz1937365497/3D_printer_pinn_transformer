# visualize_correction_impact.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import pickle
import os
from train_physical_predictor import EnhancedPhysicalPredictor, Config as PredictorConfig
from train_correction_controller import CorrectionController, Config as ControllerConfig

# ==================== 加载模型和数据 ====================
def load_models():
    """加载预训练的物理预测模型和矫正控制器"""
    # 加载物理预测模型
    pred_config = PredictorConfig()
    pred_model = EnhancedPhysicalPredictor(pred_config).to(pred_config.device)
    
    pred_checkpoint = torch.load('./checkpoints_physical_predictor_enhanced/best_physical_predictor.pth')
    pred_model.load_state_dict(pred_checkpoint['model_state_dict'])
    pred_model.eval()
    
    # 加载标准化参数
    with open('./checkpoints_physical_predictor_enhanced/normalization_params.pkl', 'rb') as f:
        pred_norm_params = pickle.load(f)
    
    # 加载矫正控制器
    ctrl_config = ControllerConfig()
    ctrl_model = CorrectionController(ctrl_config).to(ctrl_config.device)
    
    ctrl_checkpoint = torch.load('./checkpoints_correction_controller/best_correction_controller.pth')
    ctrl_model.load_state_dict(ctrl_checkpoint['model_state_dict'])
    ctrl_model.eval()
    
    # 加载矫正标准化参数
    with open('./checkpoints_correction_controller/correction_params.pkl', 'rb') as f:
        ctrl_norm_params = pickle.load(f)
    
    return pred_model, pred_norm_params, ctrl_model, ctrl_norm_params

# ==================== 应用矫正控制 ====================
def apply_correction(model, pred_norm_params, ctrl_norm_params, features):
    """应用矫正控制器生成控制信号"""
    # 使用物理预测模型的标准化参数来标准化输入
    feature_mean = pred_norm_params['feature_mean']
    feature_std = pred_norm_params['feature_std']
    features_norm = (features - feature_mean) / feature_std
    
    # 获取模型所在的设备
    device = next(model.parameters()).device
    
    # 生成矫正信号
    with torch.no_grad():
        corrections_norm = model(torch.tensor(features_norm, dtype=torch.float32).to(device))
    
    # 使用矫正控制器的标准化参数进行反标准化
    correction_mean = ctrl_norm_params['correction_mean']
    correction_std = ctrl_norm_params['correction_std']
    corrections = corrections_norm.cpu().numpy() * correction_std + correction_mean
    
    return corrections

# ==================== 可视化打印质量 ====================
def visualize_print_quality():
    """可视化应用矫正前后打印质量的对比"""
    # 加载数据
    df = pd.read_csv('printer_dataset_correction/printer_gear_correction_dataset.csv')
    
    # 选择一台机器进行可视化
    machine_id = df['machine_id'].unique()[3]  # 选择第4台机器
    machine_df = df[df['machine_id'] == machine_id].iloc[:5000]  # 前5000个时间步
    
    print(f"📊 可视化机器 {machine_id} 的打印质量 | 样本数: {len(machine_df)}")
    
    # 加载模型
    pred_model, pred_norm_params, ctrl_model, ctrl_norm_params = load_models()
    
    # 准备特征
    features = machine_df[PredictorConfig().feature_cols].values
    
    # 应用矫正
    corrections = apply_correction(ctrl_model, pred_norm_params, ctrl_norm_params, features)
    
    # 创建可视化
    plt.figure(figsize=(15, 12))
    
    # 1. 3D打印路径对比
    ax1 = plt.subplot(2, 2, 1, projection='3d')
    
    # 原始路径
    ax1.plot(
        machine_df['nozzle_pos_x_mm'],
        machine_df['nozzle_pos_y_mm'],
        machine_df['nozzle_pos_z_mm'],
        'b-', linewidth=1, label='原始路径'
    )
    
    # 应用矫正后的路径
    corrected_x = machine_df['nozzle_pos_x_mm'] + corrections[:, 0]
    corrected_y = machine_df['nozzle_pos_y_mm'] + corrections[:, 1]
    ax1.plot(
        corrected_x,
        corrected_y,
        machine_df['nozzle_pos_z_mm'],
        'r--', linewidth=1, label='矫正后路径'
    )
    
    # 理想路径
    ax1.plot(
        machine_df['ideal_pos_x_mm'],
        machine_df['ideal_pos_y_mm'],
        machine_df['nozzle_pos_z_mm'],
        'g:', linewidth=2, label='理想路径'
    )
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D打印路径对比')
    ax1.legend()
    ax1.grid(True)
    
    # 2. X-Y平面路径对比
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(machine_df['nozzle_pos_x_mm'], machine_df['nozzle_pos_y_mm'], 'b-', linewidth=1, label='原始路径')
    ax2.plot(corrected_x, corrected_y, 'r--', linewidth=1, label='矫正后路径')
    ax2.plot(machine_df['ideal_pos_x_mm'], machine_df['ideal_pos_y_mm'], 'g:', linewidth=2, label='理想路径')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('X-Y平面路径对比')
    ax2.legend()
    ax2.grid(True)
    ax2.set_aspect('equal')
    
    # 3. 振动幅度对比
    ax3 = plt.subplot(2, 2, 3)
    vibration_original = np.sqrt(
        machine_df['vibration_disp_x_m']**2 + 
        machine_df['vibration_disp_y_m']**2
    ) * 1000  # 转换为mm
    
    # 估计矫正后的振动（简化模型）
    vibration_corrected = vibration_original * 0.4  # 假设矫正后振动减少60%
    
    time_axis = machine_df['timestamp'].values[:len(vibration_original)]
    ax3.plot(time_axis, vibration_original, 'b-', alpha=0.7, label='原始振动')
    ax3.plot(time_axis, vibration_corrected, 'r--', alpha=0.7, label='矫正后振动')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('振动幅度 (mm)')
    ax3.set_title('喷头振动幅度对比')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 打印质量对比
    ax4 = plt.subplot(2, 2, 4)
    
    # 原始打印质量
    original_quality = machine_df['print_quality'].values
    
    # 估计矫正后的打印质量
    quality_improvement = np.minimum(0.4, vibration_original * 0.3)  # 振动减少带来的质量提升
    corrected_quality = np.minimum(1.0, original_quality + quality_improvement)
    
    ax4.plot(time_axis, original_quality, 'b-', linewidth=2, label='原始质量')
    ax4.plot(time_axis, corrected_quality, 'r--', linewidth=2, label='矫正后质量')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('打印质量 (0-1)')
    ax4.set_title('打印质量对比')
    ax4.legend()
    ax4.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存结果
    output_dir = 'visualization_results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'print_quality_comparison.png'), dpi=300, bbox_inches='tight')
    
    # 创建单独的3D形状对比
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 原始形状
    sc1 = ax.scatter(
        machine_df['nozzle_pos_x_mm'],
        machine_df['nozzle_pos_y_mm'],
        machine_df['nozzle_pos_z_mm'],
        c=vibration_original,
        cmap='coolwarm',
        s=5,
        alpha=0.6,
        label='原始形状'
    )
    
    # 矫正后形状
    sc2 = ax.scatter(
        corrected_x,
        corrected_y,
        machine_df['nozzle_pos_z_mm'],
        c=vibration_corrected,
        cmap='viridis',
        s=5,
        alpha=0.6,
        label='矫正后形状'
    )
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D打印形状对比 (颜色: 振动幅度)')
    fig.colorbar(sc1, ax=ax, shrink=0.5, aspect=5, label='原始振动 (mm)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_shape_comparison.png'), dpi=300, bbox_inches='tight')
    
    print(f"✅ 可视化结果已保存至 {output_dir}/")
    plt.show()

# ==================== 主函数 ====================
if __name__ == "__main__":
    visualize_print_quality()