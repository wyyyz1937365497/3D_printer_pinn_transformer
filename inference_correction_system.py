# inference_correction_system.py
# 集成物理预测模型和矫正控制器的实时推理系统
import numpy as np
import pandas as pd
import torch
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from train_physical_predictor import PhysicalPredictor, Config as PredictorConfig
from train_correction_controller import CorrectionController, ControllerConfig
import pickle

class PrintCorrectionSystem:
    def __init__(self):
        """初始化矫正系统"""
        # 加载模型
        self.pred_model, self.pred_norm_params = self.load_prediction_model()
        self.ctrl_model, self.ctrl_norm_params = self.load_correction_model()
        
        # 系统状态
        self.buffer = None
        self.seq_len = PredictorConfig().seq_len
        
        print("✅ 3D打印矫正系统初始化完成")
    
    def load_prediction_model(self):
        """加载物理预测模型"""
        config = PredictorConfig()
        model = PhysicalPredictor(config).to(config.device)
        
        checkpoint = torch.load('./checkpoints_physical_predictor/best_physical_predictor.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        with open('./checkpoints_physical_predictor/normalization_params.pkl', 'rb') as f:
            norm_params = pickle.load(f)
        
        return model, norm_params
    
    def load_correction_model(self):
        """加载矫正控制器"""
        config = ControllerConfig()
        model = CorrectionController(config).to(config.device)
        
        checkpoint = torch.load('./checkpoints_correction_controller/best_correction_controller.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        with open('./checkpoints_correction_controller/correction_params.pkl', 'rb') as f:
            norm_params = pickle.load(f)
        
        return model, norm_params
    
    def add_measurement(self, measurement):
        """添加新的测量值到缓冲区"""
        if self.buffer is None:
            self.buffer = np.zeros((self.seq_len, len(PredictorConfig().feature_cols)))
        
        # 滚动更新缓冲区
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = measurement
    
    def predict_physics(self):
        """预测物理状态"""
        if self.buffer is None or np.all(self.buffer == 0):
            return None
        
        # 标准化
        feature_mean = self.pred_norm_params['feature_mean']
        feature_std = self.pred_norm_params['feature_std']
        buffer_norm = (self.buffer - feature_mean) / feature_std
        
        # 预测
        with torch.no_grad():
            seq_tensor = torch.tensor(buffer_norm, dtype=torch.float32).unsqueeze(0).to(self.pred_model.device)
            prediction = self.pred_model(seq_tensor)
        
        # 反标准化
        target_mean = self.pred_norm_params['target_mean']
        target_std = self.pred_norm_params['target_std']
        prediction_denorm = prediction.cpu().numpy()[0] * target_std + target_mean
        
        return prediction_denorm
    
    def get_correction(self, features):
        """获取矫正信号"""
        # 标准化
        feature_mean = self.pred_norm_params['feature_mean']
        feature_std = self.pred_norm_params['feature_std']
        features_norm = (features - feature_mean) / feature_std
        
        # 生成矫正
        with torch.no_grad():
            features_tensor = torch.tensor(features_norm, dtype=torch.float32).to(self.ctrl_model.device)
            correction_norm = self.ctrl_model(features_tensor)
        
        # 反标准化
        correction_mean = self.ctrl_norm_params['correction_mean']
        correction_std = self.ctrl_norm_params['correction_std']
        correction = correction_norm.cpu().numpy() * correction_std + correction_mean
        
        return correction
    
    def process_frame(self, frame_data):
        """处理一帧数据"""
        # 添加测量
        self.add_measurement(frame_data['features'])
        
        # 预测物理状态
        physics_pred = self.predict_physics()
        
        # 获取矫正信号
        correction = self.get_correction(frame_data['features'])
        
        return {
            'physics_prediction': physics_pred,
            'correction_signal': correction,
            'original_position': (frame_data['x'], frame_data['y']),
            'corrected_position': (frame_data['x'] + correction[0], frame_data['y'] + correction[1])
        }

# 实时可视化
def real_time_visualization():
    """实时可视化矫正效果"""
    # 生成模拟数据
    system = PrintCorrectionSystem()
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始路径
    orig_line, = ax1.plot([], [], 'b-', label='原始路径')
    # 矫正路径
    corr_line, = ax1.plot([], [], 'r--', label='矫正路径')
    # 理想路径
    ideal_line, = ax1.plot([], [], 'g:', label='理想路径')
    
    ax1.set_xlim(-30, 30)
    ax1.set_ylim(-30, 30)
    ax1.set_title('实时打印路径')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_aspect('equal')
    
    # 振动幅度
    vib_line_orig, = ax2.plot([], [], 'b-', label='原始振动')
    vib_line_corr, = ax2.plot([], [], 'r--', label='矫正振动')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 0.05)
    ax2.set_title('振动幅度对比')
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('振动幅度 (m)')
    ax2.legend()
    ax2.grid(True)
    
    # 数据存储
    orig_x, orig_y = [], []
    corr_x, corr_y = []
    ideal_x, ideal_y = [], []
    time_steps = []
    vib_orig, vib_corr = [], []
    
    def init():
        orig_line.set_data([], [])
        corr_line.set_data([], [])
        ideal_line.set_data([], [])
        vib_line_orig.set_data([], [])
        vib_line_corr.set_data([], [])
        return orig_line, corr_line, ideal_line, vib_line_orig, vib_line_corr
    
    def update(frame):
        # 生成模拟数据
        t = frame * 0.01
        ideal_x_val = 25 * np.cos(2 * np.pi * t)
        ideal_y_val = 25 * np.sin(2 * np.pi * t)
        
        # 添加振动
        vib_x = 0.003 * np.sin(2 * np.pi * 15 * t + np.random.randn() * 0.1)
        vib_y = 0.003 * np.cos(2 * np.pi * 18 * t + np.random.randn() * 0.1)
        
        orig_x_val = ideal_x_val + vib_x
        orig_y_val = ideal_y_val + vib_y
        
        # 生成特征
        features = np.array([
            210,  # ctrl_T_target
            50,   # ctrl_speed_set
            ideal_x_val, ideal_y_val, 0.0,  # ctrl_pos_x, ctrl_pos_y, ctrl_pos_z
            210 + np.random.randn()*2,  # temperature_C
            vib_x, vib_y,  # vibration_disp_x_m, vibration_disp_y_m
            0.0, 0.0,  # vibration_vel_x_m_s, vibration_vel_y_m_s
            1.5 + np.random.randn()*0.2, 1.5 + np.random.randn()*0.2,  # motor_current_x_A, motor_current_y_A
            4.5 + np.random.randn()*0.3  # pressure_bar
        ])
        
        # 处理帧
        frame_data = {
            'features': features,
            'x': orig_x_val,
            'y': orig_y_val
        }
        
        result = system.process_frame(frame_data)
        
        # 存储数据
        orig_x.append(orig_x_val)
        orig_y.append(orig_y_val)
        corr_x.append(result['corrected_position'][0])
        corr_y.append(result['corrected_position'][1])
        ideal_x.append(ideal_x_val)
        ideal_y.append(ideal_y_val)
        time_steps.append(frame)
        vib_orig.append(np.sqrt(vib_x**2 + vib_y**2))
        vib_corr.append(np.sqrt(vib_x**2 + vib_y**2) * 0.4)  # 假设减少60%
        
        # 限制显示长度
        if len(orig_x) > 1000:
            orig_x.pop(0)
            orig_y.pop(0)
            corr_x.pop(0)
            corr_y.pop(0)
            ideal_x.pop(0)
            ideal_y.pop(0)
            time_steps.pop(0)
            vib_orig.pop(0)
            vib_corr.pop(0)
        
        # 更新图形
        orig_line.set_data(orig_x, orig_y)
        corr_line.set_data(corr_x, corr_y)
        ideal_line.set_data(ideal_x, ideal_y)
        
        vib_line_orig.set_data(time_steps, vib_orig)
        vib_line_corr.set_data(time_steps, vib_corr)
        
        return orig_line, corr_line, ideal_line, vib_line_orig, vib_line_corr
    
    ani = FuncAnimation(fig, update, frames=range(10000),
                        init_func=init, blit=True, interval=50)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    real_time_visualization()