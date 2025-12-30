# inference_evaluation.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import os
import pickle
import time
from train_pinn_transformer_multitask import PrinterPINN_MultiTask, Config

class NozzleInferenceEvaluator:
    def __init__(self, model_path, config_path=None):
        """
        初始化推理评估器
        :param model_path: 训练好的模型路径
        :param config_path: 配置文件路径（如果有）
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 初始化推理评估器 | 设备: {self.device}")
        
        # 加载模型
        self.model, self.config = self.load_model(model_path, config_path)
        
        # 加载归一化参数
        self.load_normalization_params()
        
        print("✅ 推理评估器初始化完成")
    
    def load_model(self, model_path, config_path=None):
        """加载训练好的模型和配置"""
        print(f"📥 加载模型: {model_path}")
        
        # 如果有单独的配置文件，加载它
        if config_path and os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
        else:
            # 使用默认配置
            config = Config()
        
        # 创建模型
        model = PrinterPINN_MultiTask(config)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(self.device)
        model.eval()
        
        print(f"✅ 模型加载成功 | 输入维度: {config.input_dim}, 输出维度: {config.output_dim}")
        return model, config
    
    def load_normalization_params(self):
        """加载归一化参数"""
        params_path = os.path.join(self.config.checkpoint_dir, 'normalization_params.pkl')
        
        if os.path.exists(params_path):
            with open(params_path, 'rb') as f:
                params = pickle.load(f)
            
            self.mean_X = params['mean_X']
            self.std_X = params['std_X']
            self.ctrl_cols = params['ctrl_cols']
            self.state_cols = params['state_cols']
            
            print("📊 归一化参数加载成功")
            print(f"   平均值: {self.mean_X}")
            print(f"   标准差: {self.std_X}")
        else:
            raise FileNotFoundError(f"归一化参数文件不存在: {params_path}")
    
    def preprocess_data(self, data_df, seq_len, pred_len):
        """
        预处理数据用于推理
        :param data_df: pandas DataFrame，包含传感器数据
        :param seq_len: 历史窗口长度
        :param pred_len: 预测长度
        :return: 处理后的张量
        """
        print("⚙️  预处理数据...")
        
        # 确保数据按时间戳排序
        if 'timestamp' in data_df.columns:
            data_df = data_df.sort_values('timestamp')
        
        # 选择需要的列
        all_cols = self.ctrl_cols + self.state_cols
        if 'hour' in data_df.columns:
            all_cols.append('hour')
        
        # 提取数据
        data_array = data_df[all_cols].values.astype(np.float32)
        
        # 归一化
        normalized_data = (data_array - self.mean_X) / self.std_X
        
        # 提取控制信号
        ctrl_data = data_df[self.ctrl_cols].values.astype(np.float32)
        normalized_ctrl = (ctrl_data - self.mean_X[:len(self.ctrl_cols)]) / self.std_X[:len(self.ctrl_cols)]
        
        # 创建滑动窗口
        n_samples = len(data_array) - seq_len - pred_len + 1
        if n_samples <= 0:
            raise ValueError(f"数据太短，无法创建窗口。需要至少 {seq_len + pred_len} 个样本，但只有 {len(data_array)} 个样本。")
        
        X_hist = np.zeros((n_samples, seq_len, self.config.input_dim), dtype=np.float32)
        X_ctrl = np.zeros((n_samples, pred_len, len(self.ctrl_cols)), dtype=np.float32)
        
        for i in range(n_samples):
            X_hist[i] = normalized_data[i:i+seq_len]
            X_ctrl[i] = normalized_ctrl[i+seq_len:i+seq_len+pred_len]
        
        print(f"✅ 数据预处理完成 | 样本数: {n_samples}")
        print(f"   X_hist shape: {X_hist.shape}")
        print(f"   X_ctrl shape: {X_ctrl.shape}")
        
        return torch.tensor(X_hist).to(self.device), torch.tensor(X_ctrl).to(self.device), data_df
    
    def predict(self, data_df, batch_size=32):
        """
        进行预测
        :param data_df: 输入数据 DataFrame
        :param batch_size: 批次大小
        :return: 预测结果字典
        """
        print("🧠 进行推理预测...")
        
        # 预处理数据
        X_hist, X_ctrl, original_df = self.preprocess_data(data_df, self.config.seq_len, self.config.pred_len)
        
        n_samples = X_hist.shape[0]
        physics_preds = []
        class_preds = []
        rul_preds = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_hist = X_hist[i:batch_end]
                batch_ctrl = X_ctrl[i:batch_end]
                
                outputs = self.model(batch_hist, batch_ctrl)
                
                # 物理场重构
                physics_pred = outputs['physics_pred'].cpu().numpy()
                physics_preds.append(physics_pred)
                
                # 故障分类
                class_pred = torch.softmax(outputs['class_pred'], dim=1).cpu().numpy()
                class_preds.append(class_pred)
                
                # RUL预测
                rul_pred = outputs['rul_pred'].cpu().numpy()
                rul_preds.append(rul_pred)
        
        # 合并结果
        physics_preds = np.concatenate(physics_preds, axis=0)
        class_preds = np.concatenate(class_preds, axis=0)
        rul_preds = np.concatenate(rul_preds, axis=0)
        
        inference_time = time.time() - start_time
        print(f"✅ 推理完成 | 样本数: {n_samples} | 耗时: {inference_time:.2f}s | 吞吐量: {n_samples/inference_time:.1f} 样本/秒")
        
        return {
            'physics_preds': physics_preds,
            'class_preds': class_preds,
            'rul_preds': rul_preds,
            'timestamps': original_df['timestamp'].values[self.config.seq_len:self.config.seq_len + n_samples],
            'machine_ids': original_df['machine_id'].values[self.config.seq_len:self.config.seq_len + n_samples]
        }
    
    def evaluate(self, predictions, ground_truth_df):
        """
        评估预测结果
        :param predictions: predict()方法的输出
        :param ground_truth_df: 包含真实标签的DataFrame
        """
        print("\n" + "="*60)
        print("📊 评估预测结果")
        print("="*60)
        
        # 1. 物理场重构评估
        self.evaluate_physics_reconstruction(predictions, ground_truth_df)
        
        # 2. 故障分类评估
        self.evaluate_fault_classification(predictions, ground_truth_df)
        
        # 3. RUL预测评估
        self.evaluate_rul_prediction(predictions, ground_truth_df)
        
        # 4. 生成综合报告
        self.generate_comprehensive_report(predictions, ground_truth_df)
    
    def evaluate_physics_reconstruction(self, predictions, ground_truth_df):
        """评估物理场重构性能"""
        print("\n🔧 1. 物理场重构评估")
        
        # 提取真实值
        start_idx = self.config.seq_len
        end_idx = start_idx + len(predictions['physics_preds'])
        ground_truth = ground_truth_df[self.state_cols].values[start_idx:end_idx]
        
        # 反归一化预测值
        physics_preds = predictions['physics_preds']
        # 只对状态列进行反归一化
        state_start_idx = len(self.ctrl_cols)
        state_end_idx = state_start_idx + len(self.state_cols)
        state_mean = self.mean_X[state_start_idx:state_end_idx]
        state_std = self.std_X[state_start_idx:state_end_idx]
        
        # 反归一化
        physics_preds_denorm = physics_preds * state_std + state_mean
        
        # 评估每个物理量
        metrics = {}
        for i, col in enumerate(self.state_cols):
            true_vals = ground_truth[:, i]
            pred_vals = physics_preds_denorm[:, -1, i]  # 使用最后一步的预测
            
            mse = mean_squared_error(true_vals, pred_vals)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(true_vals - pred_vals))
            r2 = r2_score(true_vals, pred_vals)
            
            metrics[col] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            print(f"   {col:20s} | MSE: {mse:.6f} | RMSE: {rmse:.6f} | MAE: {mae:.6f} | R2: {r2:.4f}")
        
        # 可视化主要物理量
        self.plot_physics_reconstruction(physics_preds_denorm, ground_truth, metrics)
        
        return metrics
    
    def evaluate_fault_classification(self, predictions, ground_truth_df):
        """评估故障分类性能"""
        print("\n🚨 2. 故障分类评估")
        
        # 提取真实标签
        start_idx = self.config.seq_len
        end_idx = start_idx + len(predictions['class_preds'])
        true_fault_labels = ground_truth_df['fault_label'].values[start_idx:end_idx]
        true_fault_types = ground_truth_df['fault_type'].values[start_idx:end_idx]
        
        # 转换预测结果
        pred_probs = predictions['class_preds']
        pred_classes = np.argmax(pred_probs, axis=1)
        
        # 将真实故障类型转换为分类标签
        true_classes = np.zeros_like(true_fault_labels, dtype=int)
        fault_mask = (true_fault_labels == 1)
        true_classes[fault_mask] = true_fault_types[fault_mask].astype(int)
        true_classes = np.clip(true_classes, 0, self.config.class_dim-1)
        
        # 计算指标
        accuracy = np.mean(pred_classes == true_classes)
        
        print(f"   准确率: {accuracy:.4f}")
        print("\n" + classification_report(true_classes, pred_classes, 
                                          target_names=['Normal', 'Nozzle Clog', 'Mechanical Loose', 'Motor Fault']))
        
        # 混淆矩阵
        cm = confusion_matrix(true_classes, pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Nozzle Clog', 'Mechanical Loose', 'Motor Fault'],
                    yticklabels=['Normal', 'Nozzle Clog', 'Mechanical Loose', 'Motor Fault'])
        plt.title('故障分类混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.savefig(os.path.join(self.config.checkpoint_dir, 'inference_confusion_matrix.png'))
        plt.close()
        
        print(f"   混淆矩阵已保存: {os.path.join(self.config.checkpoint_dir, 'inference_confusion_matrix.png')}")
        
        return accuracy, cm
    
    def evaluate_rul_prediction(self, predictions, ground_truth_df):
        """评估RUL预测性能"""
        print("\n⏳ 3. RUL预测评估")
        
        # 提取真实RUL
        start_idx = self.config.seq_len
        end_idx = start_idx + len(predictions['rul_preds'])
        
        # 计算真实RUL（简化版）
        true_rul = np.zeros(end_idx - start_idx)
        machine_ids = ground_truth_df['machine_id'].values[start_idx:end_idx]
        
        unique_machines = np.unique(machine_ids)
        for mid in unique_machines:
            mask = (machine_ids == mid)
            fault_labels = ground_truth_df['fault_label'].values[start_idx:end_idx][mask]
            
            if np.any(fault_labels == 1):
                fault_indices = np.where(fault_labels == 1)[0]
                first_fault_idx = fault_indices[0]
                
                # 计算每个时间点到故障的时间
                for i in range(len(mask)):
                    if i < first_fault_idx:
                        steps_to_fault = first_fault_idx - i
                        true_rul[mask][i] = steps_to_fault * 0.001  # 转换为秒
                    else:
                        true_rul[mask][i] = 0
            else:
                # 无故障机器，设置为最大RUL
                true_rul[mask] = 3600  # 1小时
        
        # 反归一化预测的RUL
        pred_rul = predictions['rul_preds'].flatten() * 3600  # 从[0,1]映射回[0,3600]秒
        
        # 评估
        mse = mean_squared_error(true_rul, pred_rul)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_rul - pred_rul))
        r2 = r2_score(true_rul, pred_rul)
        
        print(f"   RUL预测性能:")
        print(f"      MSE: {mse:.2f} 秒²")
        print(f"      RMSE: {rmse:.2f} 秒")
        print(f"      MAE: {mae:.2f} 秒")
        print(f"      R2: {r2:.4f}")
        
        # 可视化
        plt.figure(figsize=(12, 6))
        plt.plot(true_rul[:1000], 'b-', label='真实RUL', alpha=0.7)
        plt.plot(pred_rul[:1000], 'r--', label='预测RUL', alpha=0.7)
        plt.xlabel('样本索引')
        plt.ylabel('RUL (秒)')
        plt.title('RUL预测对比')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config.checkpoint_dir, 'rul_prediction_comparison.png'))
        plt.close()
        
        print(f"   RUL对比图已保存: {os.path.join(self.config.checkpoint_dir, 'rul_prediction_comparison.png')}")
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    def plot_physics_reconstruction(self, physics_preds, ground_truth, metrics):
        """绘制物理场重构结果"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('物理场重构结果对比', fontsize=16)
        
        plot_cols = ['temperature_C', 'vibration_disp_x_m', 'vibration_disp_y_m', 
                    'motor_current_x_A', 'pressure_bar', 'print_quality']
        
        for i, col in enumerate(plot_cols):
            if col not in self.state_cols:
                continue
            
            col_idx = self.state_cols.index(col)
            ax = axes[i//2, i%2]
            
            # 取前1000个样本进行可视化
            n_plot = min(1000, len(ground_truth))
            true_vals = ground_truth[:n_plot, col_idx]
            pred_vals = physics_preds[:n_plot, -1, col_idx]  # 使用最后一步的预测
            
            ax.plot(true_vals, 'b-', label='真实值', alpha=0.7)
            ax.plot(pred_vals, 'r--', label='预测值', alpha=0.7)
            ax.set_title(f'{col}\nRMSE: {metrics[col]["RMSE"]:.6f}')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.checkpoint_dir, 'physics_reconstruction.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"   物理场重构图已保存: {plot_path}")
    
    def generate_comprehensive_report(self, predictions, ground_truth_df):
        """生成综合评估报告"""
        print("\n" + "="*60)
        print("📋 生成综合评估报告")
        print("="*60)
        
        report_path = os.path.join(self.config.checkpoint_dir, 'inference_evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("3D打印机PINN-Transformer模型推理评估报告\n")
            f.write("="*60 + "\n")
            f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型路径: {os.path.abspath(self.config.checkpoint_dir)}/best_multitask_model.pth\n")
            f.write(f"数据样本数: {len(predictions['timestamps'])}\n\n")
            
            # 关键指标摘要
            f.write("关键性能指标摘要:\n")
            f.write("-"*40 + "\n")
            
            # 物理场重构摘要
            f.write("1. 物理场重构性能:\n")
            for col in ['temperature_C', 'vibration_disp_x_m', 'vibration_disp_y_m']:
                if col in self.state_cols:
                    col_idx = self.state_cols.index(col)
                    ground_truth = ground_truth_df[col].values[self.config.seq_len:self.config.seq_len + len(predictions['physics_preds'])]
                    pred_vals = predictions['physics_preds'][:, -1, col_idx] * self.std_X[len(self.ctrl_cols)+col_idx] + self.mean_X[len(self.ctrl_cols)+col_idx]
                    
                    rmse = np.sqrt(mean_squared_error(ground_truth, pred_vals))
                    f.write(f"   {col:20s}: RMSE = {rmse:.6f}\n")
            
            # 故障分类摘要
            f.write("\n2. 故障分类性能:\n")
            pred_classes = np.argmax(predictions['class_preds'], axis=1)
            true_classes = np.zeros_like(pred_classes)
            true_fault_labels = ground_truth_df['fault_label'].values[self.config.seq_len:self.config.seq_len + len(pred_classes)]
            true_fault_types = ground_truth_df['fault_type'].values[self.config.seq_len:self.config.seq_len + len(pred_classes)]
            
            fault_mask = (true_fault_labels == 1)
            true_classes[fault_mask] = true_fault_types[fault_mask].astype(int)
            true_classes = np.clip(true_classes, 0, self.config.class_dim-1)
            
            accuracy = np.mean(pred_classes == true_classes)
            f.write(f"   准确率: {accuracy:.4f}\n")
            
            # RUL预测摘要
            f.write("\n3. RUL预测性能:\n")
            true_rul = np.zeros(len(predictions['rul_preds']))
            pred_rul = predictions['rul_preds'].flatten() * 3600
            
            # 简化的RUL计算
            for i in range(len(true_rul)):
                if true_fault_labels[i] == 0:  # 无故障
                    true_rul[i] = 3600
                else:
                    true_rul[i] = 0
            
            rmse_rul = np.sqrt(mean_squared_error(true_rul, pred_rul))
            f.write(f"   RMSE: {rmse_rul:.2f} 秒\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("详细结果请查看生成的图像文件:\n")
            f.write(f"   - physics_reconstruction.png\n")
            f.write(f"   - inference_confusion_matrix.png\n")
            f.write(f"   - rul_prediction_comparison.png\n")
        
        print(f"✅ 综合评估报告已生成: {report_path}")

# ==================== 主函数 ====================
def main():
    """主函数：演示推理评估流程"""
    # 配置参数
    model_path = './checkpoints_multitask/best_multitask_model.pth'
    data_path = 'printer_dataset/nozzle_simulation_gear_print.csv'
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    print("🚀 3D打印机PINN-Transformer推理评估系统")
    print("="*70)
    
    # 加载评估器
    evaluator = NozzleInferenceEvaluator(model_path)
    
    # 加载测试数据
    print(f"\n📥 加载测试数据: {data_path}")
    test_df = pd.read_csv(data_path)
    
    # 选择特定机器进行测试（例如机器1）
    test_machine_id = 1
    test_machine_df = test_df[test_df['machine_id'] == test_machine_id].copy()
    print(f"   选择机器 {test_machine_id} | 样本数: {len(test_machine_df)}")
    
    # 选择时间范围（例如前10秒）
    max_time = 10.0  # 秒
    test_machine_df = test_machine_df[test_machine_df['timestamp'] <= max_time]
    print(f"   选择时间范围 [0, {max_time}] 秒 | 样本数: {len(test_machine_df)}")
    
    # 进行预测
    predictions = evaluator.predict(test_machine_df, batch_size=64)
    
    # 评估结果
    evaluator.evaluate(predictions, test_machine_df)
    
    print("\n" + "="*70)
    print("🎉 推理评估完成！")
    print("="*70)

if __name__ == "__main__":
    main()