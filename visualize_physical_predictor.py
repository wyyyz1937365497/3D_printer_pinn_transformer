# visualize_physical_predictor.py
# 物理预测模型推理效果可视化脚本

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os
import pickle
from train_physical_predictor import EnhancedPhysicalPredictor, Config

def load_model_and_data(checkpoint_path, data_path, config):
    """加载模型和数据"""
    print("  正在加载模型...")
    # 加载模型
    model = EnhancedPhysicalPredictor(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    # 检查是否是DataParallel模型
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('module.') for key in state_dict.keys()):
        # 如果是DataParallel保存的模型，去掉module.前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith('module.') else key  # 去掉'module.'前缀
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    
    if len(config.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=config.gpu_ids)
    model = model.to(config.device)
    model.eval()
    
    print("  正在加载标准化参数...")
    # 加载标准化参数
    norm_path = os.path.join(os.path.dirname(checkpoint_path), 'normalization_params.pkl')
    with open(norm_path, 'rb') as f:
        norm_params = pickle.load(f)
    
    print("  正在加载数据...")
    # 加载数据
    df = pd.read_csv(data_path)
    
    return model, df, norm_params

def preprocess_data(df, config, norm_params):
    """预处理数据"""
    print("  正在预处理数据...")
    # 选择特征列
    feature_cols = config.feature_cols
    target_cols = config.target_cols
    
    # 提取特征和目标
    features = df[feature_cols].values.astype(np.float32)
    targets = df[target_cols].values.astype(np.float32)
    
    # 计算频域特征
    freq_features = compute_frequency_features(features, config)
    
    # 合并时域和频域特征
    combined_features = np.concatenate([features, freq_features], axis=1)
    
    # 检查标准化参数维度是否匹配
    expected_feature_dim = norm_params['feature_mean'].shape[0]
    actual_feature_dim = combined_features.shape[1]
    
    if expected_feature_dim != actual_feature_dim:
        print(f"  警告: 标准化参数维度不匹配 - 期望: {expected_feature_dim}, 实际: {actual_feature_dim}")
        print("  正在重新计算标准化参数...")
        
        # 使用当前数据的统计量进行标准化
        feature_mean = np.mean(combined_features, axis=0)
        feature_std = np.std(combined_features, axis=0) + 1e-8
    else:
        # 使用保存的标准化参数
        feature_mean = norm_params['feature_mean']
        feature_std = norm_params['feature_std']
    
    combined_features = (combined_features - feature_mean) / feature_std
    
    return combined_features, targets

def compute_frequency_features(features, config):
    """计算频域特征"""
    # 使用滑动窗口计算频域特征
    window_size = min(config.seq_len, features.shape[0])
    stride = config.pred_len
    
    freq_features_list = []
    
    for i in range(0, max(1, features.shape[0] - window_size + 1), stride):
        window = features[i:i+window_size]
        
        # 对每个特征维度计算频域表示
        freq_data = []
        for j in range(window.shape[1]):
            # 使用FFT计算频域特征
            fft_vals = np.fft.fft(window[:, j])
            fft_magnitude = np.abs(fft_vals[:config.freq_bands])  # 取前freq_bands个频率成分
            freq_data.extend(fft_magnitude)
        
        freq_features_list.append(freq_data)
    
    # 如果序列长度不够一个窗口，复制最后一个窗口的数据
    if len(freq_features_list) == 0:
        # 计算整个序列的频域特征
        freq_data = []
        for j in range(features.shape[1]):
            fft_vals = np.fft.fft(features[:, j])
            fft_magnitude = np.abs(fft_vals[:config.freq_bands])
            freq_data.extend(fft_magnitude)
        freq_features_list = [freq_data] * features.shape[0]
    elif len(freq_features_list) < features.shape[0]:
        # 扩展频域特征以匹配原始序列长度
        last_freq_features = freq_features_list[-1]
        while len(freq_features_list) < features.shape[0]:
            freq_features_list.append(last_freq_features)
    
    return np.array(freq_features_list, dtype=np.float32)

def predict_with_model(model, features, config):
    """使用模型进行预测"""
    print("  正在进行模型预测...")
    model.eval()
    
    with torch.no_grad():
        # 重塑数据为序列格式
        seq_len = config.seq_len
        n_samples = (len(features) // seq_len) * seq_len  # 调整为能被seq_len整除的长度
        features = features[:n_samples]
        
        # 重塑为 (batch, seq_len, features)
        features = features.reshape(-1, seq_len, features.shape[-1])
        
        # 只取前10个序列进行预测，避免长时间运行
        features = features[:10]
        
        # 转换为tensor
        features_tensor = torch.FloatTensor(features).to(config.device)
        
        # 分批预测以避免内存问题
        batch_size = min(config.batch_size, len(features_tensor))
        predictions = []
        
        for i in range(0, len(features_tensor), batch_size):
            batch = features_tensor[i:i+batch_size]
            
            with torch.cuda.amp.autocast():
                pred = model(batch)
            
            predictions.append(pred.cpu().numpy())
    
    # 合并预测结果
    predictions = np.vstack(predictions)
    
    # 重塑回原始形状
    predictions = predictions.reshape(-1, predictions.shape[-1])
    
    return predictions

def denormalize_data(data, norm_params, target_cols):
    """反标准化数据"""
    print("  正在反标准化数据...")
    target_mean = norm_params['target_mean']
    target_std = norm_params['target_std']
    
    # 获取目标列的索引
    target_idx = [i for i, col in enumerate(norm_params['target_cols']) if col in target_cols]
    
    if len(target_idx) != len(target_cols):
        # 如果目标列不完全匹配，使用全部目标列
        target_idx = list(range(len(target_cols)))
    
    if len(target_idx) > 0:
        target_mean = target_mean[target_idx]
        target_std = target_std[target_idx]
    
    denorm_data = data * (target_std + 1e-8) + target_mean
    
    return denorm_data

def plot_predictions_vs_actual(y_true, y_pred, target_cols, title="物理预测模型效果"):
    """绘制预测值与真实值对比图"""
    print("  正在生成预测值与真实值对比图...")
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    n_targets = min(len(target_cols), y_pred.shape[1])
    
    fig, axes = plt.subplots(n_targets, 1, figsize=(12, 4*n_targets))
    if n_targets == 1:
        axes = [axes]
    
    for i in range(n_targets):
        axes[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=1)
        axes[i].plot([y_true[:, i].min(), y_true[:, i].max()], 
                    [y_true[:, i].min(), y_true[:, i].max()], 'r--', lw=2)
        axes[i].set_xlabel(f'真实值 - {target_cols[i]}')
        axes[i].set_ylabel(f'预测值 - {target_cols[i]}')
        axes[i].set_title(f'{target_cols[i]}: 预测值 vs 真实值')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join('./checkpoints_physical_predictor_enhanced', f'{title.replace(" ", "_")}.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # 关闭图形以释放内存
    print(f"  预测值与真实值对比图已保存至: {os.path.join('./checkpoints_physical_predictor_enhanced', f'{title.replace(' ', '_')}.png')}")

def plot_time_series(y_true, y_pred, target_cols, start_idx=0, end_idx=1000, title="时间序列预测"):
    """绘制时间序列预测结果"""
    print("  正在生成时间序列预测图...")
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    n_targets = min(len(target_cols), y_pred.shape[1])
    
    fig, axes = plt.subplots(n_targets, 1, figsize=(15, 3*n_targets))
    if n_targets == 1:
        axes = [axes]
    
    for i in range(n_targets):
        axes[i].plot(y_true[start_idx:end_idx, i], label='真实值', alpha=0.7)
        axes[i].plot(y_pred[start_idx:end_idx, i], label='预测值', alpha=0.7)
        axes[i].set_xlabel('时间步')
        axes[i].set_ylabel(target_cols[i])
        axes[i].set_title(f'{target_cols[i]} 时间序列预测效果')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join('./checkpoints_physical_predictor_enhanced', f'{title.replace(" ", "_")}_timeseries.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # 关闭图形以释放内存
    print(f"  时间序列预测图已保存至: {os.path.join('./checkpoints_physical_predictor_enhanced', f'{title.replace(' ', '_')}_timeseries.png')}")

def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    print("  正在计算评估指标...")
    # MSE
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    # MAE
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    # RMSE
    rmse = np.sqrt(mse)
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1 - (ss_res / ss_tot)
    
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R²': r2}

def main():
    print("="*80)
    print("🔬 物理预测模型推理效果可视化")
    print("="*80)
    
    # 配置参数
    config = Config()
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型检查点路径
    checkpoint_path = os.path.join(config.checkpoint_dir, 'best_physical_predictor.pth')
    if not os.path.exists(checkpoint_path):
        print(f"❌ 未找到最佳模型检查点文件: {checkpoint_path}")
        # 尝试使用最新的检查点
        checkpoint_files = [f for f in os.listdir(config.checkpoint_dir) if f.endswith('.pth') and 'best' not in f]
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(config.checkpoint_dir, x)), reverse=True)
            checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_files[0])
            print(f"✅ 使用最新检查点: {checkpoint_path}")
        else:
            print(f"❌ 未找到任何模型检查点文件")
            return
    
    # 选择测试数据文件
    data_dir = config.data_dir
    data_files = [f for f in os.listdir(data_dir) if f.startswith('machine_') and f.endswith('.csv')]
    if not data_files:
        print(f"❌ 未找到数据文件")
        return
    
    # 使用第一个数据文件进行可视化
    data_path = os.path.join(data_dir, data_files[0])
    print(f"📊 使用数据文件: {data_files[0]}")
    
    # 加载模型和数据
    print("🔄 加载模型和数据...")
    model, df, norm_params = load_model_and_data(checkpoint_path, data_path, config)
    
    print(f"   数据形状: {df.shape}")
    print(f"   特征列: {config.feature_cols}")
    print(f"   目标列: {config.target_cols}")
    
    # 预处理数据
    features, targets = preprocess_data(df, config, norm_params)
    print(f"   预处理后特征形状: {features.shape}")
    print(f"   预处理后目标形状: {targets.shape}")
    
    # 准备用于预测的数据 - 与预测函数中相同的处理
    seq_len = config.seq_len
    n_samples = (len(features) // seq_len) * seq_len
    targets_for_pred = targets[:n_samples]
    
    # 重塑为序列格式，然后提取用于预测的序列
    targets_for_pred = targets_for_pred.reshape(-1, seq_len, targets_for_pred.shape[-1])
    
    # 只取前10个序列用于预测（与预测函数一致）
    targets_for_pred = targets_for_pred[:10]
    
    # 重塑回原始形状 - 现在是 (10 * seq_len, features)，即 (2500, 5)
    targets_for_pred = targets_for_pred.reshape(-1, targets_for_pred.shape[-1])
    
    # 模型预测
    predictions = predict_with_model(model, features, config)
    print(f"   预测结果形状: {predictions.shape}")
    
    # 反标准化
    targets_denorm = denormalize_data(targets_for_pred, norm_params, config.target_cols)
    predictions_denorm = denormalize_data(predictions, norm_params, config.target_cols)
    print("   数据反标准化完成")
    
    # 确保两个数组形状一致
    print(f"   真实值形状: {targets_denorm.shape}")
    print(f"   预测值形状: {predictions_denorm.shape}")
    
    if targets_denorm.shape != predictions_denorm.shape:
        print(f"⚠️  形状不一致，调整到相同形状")
        min_len = min(len(targets_denorm), len(predictions_denorm))
        targets_denorm = targets_denorm[:min_len]
        predictions_denorm = predictions_denorm[:min_len]
    
    # 计算评估指标
    metrics = calculate_metrics(targets_denorm, predictions_denorm)
    
    print("\n📊 评估指标:")
    for metric_name, values in metrics.items():
        print(f"  {metric_name}:")
        for i, col in enumerate(config.target_cols[:len(values)]):
            print(f"    {col}: {values[i]:.6f}")
    
    # 绘制可视化图表
    print("\n🎨 生成可视化图表...")
    
    # 预测值与真实值对比
    plot_predictions_vs_actual(targets_denorm, predictions_denorm, config.target_cols)
    
    # 时间序列预测效果（前1000个点）
    plot_time_series(targets_denorm, predictions_denorm, config.target_cols, 
                    start_idx=0, end_idx=min(1000, len(targets_denorm)))
    
    # 如果数据量大，也可以绘制后面的片段
    if len(targets_denorm) > 2000:
        plot_time_series(targets_denorm, predictions_denorm, config.target_cols,
                        start_idx=len(targets_denorm)//2, 
                        end_idx=min(len(targets_denorm)//2 + 1000, len(targets_denorm)),
                        title="时间序列预测(后半段)")
    
    print(f"\n🎉 可视化完成！图表已保存至: {config.checkpoint_dir}")

if __name__ == "__main__":
    # 定义配置类，与训练时保持一致
    class Config:
        def __init__(self):
            self.checkpoint_dir = './checkpoints_physical_predictor_enhanced'
            self.data_dir = './printer_dataset_correction'
            self.seq_len = 250
            self.pred_len = 50
            self.batch_size = 1024
            self.gradient_accumulation_steps = 1
            self.model_dim = 192  # 与训练时保持一致
            self.num_heads = 8
            self.num_layers = 6  # 与训练时保持一致
            self.dim_feedforward = 768  # 前馈网络维度
            self.dropout = 0.1
            self.lr = 5e-5
            self.epochs = 30
            self.gpu_ids = [0]
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.lambda_physics = 0.4
            self.lambda_freq = 0.3
            self.warmup_epochs = 5
            
            # 特征和目标列
            self.feature_cols = [
                'nozzle_x', 'nozzle_y', 'nozzle_z',  # 替换原来的控制列
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
            self.input_dim = self.time_domain_dim + self.freq_domain_dim  # 实际输入维度
            self.output_dim = len(self.target_cols)
            
            # 添加其他必要参数
            self.n_heads = self.num_heads  # 与num_heads相同
            self.temperature = 1.0  # 注意力温度
            self.layer_norm_eps = 1e-5  # 层归一化epsilon
            self.max_len = 5000  # 最大序列长度
            self.num_freq_components = 10  # 频域组件数量
    
    # 调用主函数
    main()