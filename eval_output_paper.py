# enhanced_3d_visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import os
from datetime import datetime

# ==================== 配置参数 ====================
class Config:
    def __init__(self):
        self.data_path = 'pretrained_nn_results_data.pkl'  # 保存数据的pkl文件
        self.output_dir = '3d_visualization_results'
        self.fig_size = (14, 10)
        self.dpi = 300
        self.animation_fps = 30
        self.view_elev = 20  # 3D视角仰角
        self.view_azim = 45  # 3D视角方位角
        self.color_map = 'viridis'
        self.smooth_factor = 0.9  # 平滑因子（0-1），值越小越平滑

# ==================== 3D可视化类 ====================
class Enhanced3DVisualizer:
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 创建颜色映射
        self.cmap = plt.get_cmap(self.config.color_map)
        self.scalar_map = cmx.ScalarMappable(
            norm=colors.Normalize(vmin=0, vmax=1), 
            cmap=self.cmap
        )
        
        # 初始化数据
        self.data = None
        self.fig = None
        self.ax = None
        self.scatter = None
        self.lines = []
        self.annotations = []
        self.slider = None
        self.time_index = 0
        self.max_time = 0
    
    def load_data(self):
        """加载评估结果数据"""
        print("🔄 加载评估结果数据...")
        if not os.path.exists(self.config.data_path):
            print("⚠️ 评估数据文件不存在，使用模拟数据生成示例")
            self._generate_sample_data()
        else:
            try:
                with open(self.config.data_path, 'rb') as f:
                    self.data = pickle.load(f)
                print(f"✅ 数据加载成功 | 样本数: {len(self.data['x_ideal'])}")
            except Exception as e:
                print(f"❌ 数据加载失败: {e} | 使用模拟数据")
                self._generate_sample_data()
        
        # 确保数据长度一致
        self.max_time = min(len(self.data['x_ideal']), 
                           len(self.data['x_original']),
                           len(self.data['x_corrected']))
        
        # 创建时间索引
        self.time_index = min(500, self.max_time // 2)
    
    def _generate_sample_data(self):
        """生成模拟数据用于演示"""
        print("💡 生成模拟数据...")
        n_points = 2000
        t = np.linspace(0, 2*np.pi, n_points)
        
        # 理想轨迹（齿轮）
        radius = 10
        teeth = 16
        tooth_profile = radius * (1 + 0.08 * np.sin(teeth * t))
        x_ideal = tooth_profile * np.cos(t)
        y_ideal = tooth_profile * np.sin(t)
        z_ideal = np.linspace(0, 5, n_points)
        
        # 原始轨迹（带振动）
        vibration_amp = 0.5
        x_original = x_ideal + vibration_amp * np.sin(5*t) * np.exp(-0.2*t)
        y_original = y_ideal + vibration_amp * np.cos(5*t) * np.exp(-0.2*t)
        z_original = z_ideal + 0.1 * np.sin(10*t)
        
        # 矫正轨迹（模拟神经网络矫正效果）
        x_corrected = x_original * 0.98 + x_ideal * 0.02
        y_corrected = y_original * 0.98 + y_ideal * 0.02
        z_corrected = z_original * 0.98 + z_ideal * 0.02
        
        # 打印质量指标（振动幅度）
        vibration_magnitude = np.sqrt(
            (x_original - x_ideal)**2 + 
            (y_original - y_ideal)**2
        )
        
        # 矫正后的振动幅度
        corrected_vibration = np.sqrt(
            (x_corrected - x_ideal)**2 + 
            (y_corrected - y_ideal)**2
        )
        
        self.data = {
            'x_ideal': x_ideal,
            'y_ideal': y_ideal,
            'z_ideal': z_ideal,
            'x_original': x_original,
            'y_original': y_original,
            'z_original': z_original,
            'x_corrected': x_corrected,
            'y_corrected': y_corrected,
            'z_corrected': z_corrected,
            'vibration_magnitude': vibration_magnitude,
            'corrected_vibration': corrected_vibration,
            'time': np.arange(n_points)
        }
    
    def create_3d_visualization(self):
        """创建3D可视化"""
        print("🎨 创建3D可视化...")
        self.fig = plt.figure(figsize=self.config.fig_size, dpi=self.config.dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 设置坐标轴标签
        self.ax.set_xlabel('X (mm)', fontsize=12)
        self.ax.set_ylabel('Y (mm)', fontsize=12)
        self.ax.set_zlabel('Z (mm)', fontsize=12)
        
        # 设置标题
        self.ax.set_title('3D打印轨迹对比与质量评估', fontsize=14, fontweight='bold')
        
        # 设置坐标轴范围
        all_x = np.concatenate([
            self.data['x_ideal'], 
            self.data['x_original'], 
            self.data['x_corrected']
        ])
        all_y = np.concatenate([
            self.data['y_ideal'], 
            self.data['y_original'], 
            self.data['y_corrected']
        ])
        all_z = np.concatenate([
            self.data['z_ideal'], 
            self.data['z_original'], 
            self.data['z_corrected']
        ])
        
        self.ax.set_xlim([np.min(all_x)-1, np.max(all_x)+1])
        self.ax.set_ylim([np.min(all_y)-1, np.max(all_y)+1])
        self.ax.set_zlim([np.min(all_z)-0.5, np.max(all_z)+0.5])
        
        # 绘制理想轨迹
        self.ax.plot(
            self.data['x_ideal'], 
            self.data['y_ideal'], 
            self.data['z_ideal'],
            'g-', linewidth=2.5, alpha=0.8, label='理想轨迹'
        )
        
        # 绘制原始轨迹
        self.ax.plot(
            self.data['x_original'], 
            self.data['y_original'], 
            self.data['z_original'],
            'r--', linewidth=1.5, alpha=0.7, label='原始轨迹'
        )
        
        # 绘制矫正轨迹
        self.ax.plot(
            self.data['x_corrected'], 
            self.data['y_corrected'], 
            self.data['z_corrected'],
            'b-', linewidth=1.5, alpha=0.7, label='矫正轨迹'
        )
        
        # 绘制振动幅度
        normalized_vib = self._smooth_data(self.data['vibration_magnitude'])
        self.scatter = self.ax.scatter(
            self.data['x_original'], 
            self.data['y_original'], 
            self.data['z_original'],
            c=normalized_vib, 
            cmap=self.cmap,
            s=15,
            alpha=0.6,
            label='振动幅度'
        )
        
        # 添加颜色条
        cbar = self.fig.colorbar(self.scatter, ax=self.ax, pad=0.02)
        cbar.set_label('振动幅度 (mm)', fontsize=10)
        
        # 添加图例
        self.ax.legend(loc='upper right', fontsize=10)
        
        # 添加网格
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置视角
        self.ax.view_init(elev=self.config.view_elev, azim=self.config.view_azim)
        
        # 添加时间指示器
        self.time_indicator = self.ax.text2D(
            0.05, 0.95, f'时间: {self.time_index}/{self.max_time}',
            transform=self.ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )
        
        # 添加质量评估指标
        self.quality_text = self.ax.text2D(
            0.05, 0.9, f'打印质量: {self._calculate_quality(self.time_index):.2f}',
            transform=self.ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )
        
        # 添加交互式时间滑块
        self._add_time_slider()
        
        plt.tight_layout()
        self._save_image('3d_trajectory_comparison.png')
        
        print(f"✅ 3D可视化创建完成 | 保存至: {os.path.join(self.config.output_dir, '3d_trajectory_comparison.png')}")
    
    def _smooth_data(self, data):
        """平滑数据以减少噪声"""
        smoothed = np.copy(data)
        for i in range(1, len(data)):
            smoothed[i] = self.config.smooth_factor * smoothed[i-1] + \
                          (1 - self.config.smooth_factor) * data[i]
        return smoothed
    
    def _calculate_quality(self, time_idx):
        """计算当前时间点的打印质量"""
        if time_idx >= len(self.data['vibration_magnitude']):
            return 0.5
        
        vib = self.data['vibration_magnitude'][time_idx]
        return max(0, min(1, 1 - vib * 2))  # 振动越小，质量越高
    
    def _add_time_slider(self):
        """添加时间滑块"""
        axcolor = 'lightgoldenrodyellow'
        ax_time = plt.axes([0.2, 0.02, 0.6, 0.03], facecolor=axcolor)
        
        self.slider = Slider(
            ax_time, 
            '时间索引', 
            0, 
            self.max_time-1, 
            valinit=self.time_index,
            valstep=1
        )
        
        self.slider.on_changed(self._update_time_index)
    
    def _update_time_index(self, val):
        """更新时间索引"""
        self.time_index = int(val)
        self.time_indicator.set_text(f'时间: {self.time_index}/{self.max_time}')
        self.quality_text.set_text(f'打印质量: {self._calculate_quality(self.time_index):.2f}')
        self.fig.canvas.draw_idle()
    
    def create_3d_animation(self):
        """创建3D轨迹动画"""
        print("🎬 生成3D轨迹动画...")
        self.fig = plt.figure(figsize=self.config.fig_size, dpi=self.config.dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 设置坐标轴标签
        self.ax.set_xlabel('X (mm)', fontsize=12)
        self.ax.set_ylabel('Y (mm)', fontsize=12)
        self.ax.set_zlabel('Z (mm)', fontsize=12)
        self.ax.set_title('3D打印轨迹动画', fontsize=14, fontweight='bold')
        
        # 设置坐标轴范围
        all_x = np.concatenate([
            self.data['x_ideal'], 
            self.data['x_original'], 
            self.data['x_corrected']
        ])
        all_y = np.concatenate([
            self.data['y_ideal'], 
            self.data['y_original'], 
            self.data['y_corrected']
        ])
        all_z = np.concatenate([
            self.data['z_ideal'], 
            self.data['z_original'], 
            self.data['z_corrected']
        ])
        
        self.ax.set_xlim([np.min(all_x)-1, np.max(all_x)+1])
        self.ax.set_ylim([np.min(all_y)-1, np.max(all_y)+1])
        self.ax.set_zlim([np.min(all_z)-0.5, np.max(all_z)+0.5])
        
        # 创建轨迹线
        self.ideal_line, = self.ax.plot([], [], [], 'g-', linewidth=2.5, alpha=0.8, label='理想轨迹')
        self.original_line, = self.ax.plot([], [], [], 'r--', linewidth=1.5, alpha=0.7, label='原始轨迹')
        self.corrected_line, = self.ax.plot([], [], [], 'b-', linewidth=1.5, alpha=0.7, label='矫正轨迹')
        
        # 创建当前点标记
        self.ideal_point, = self.ax.plot([], [], [], 'go', markersize=8, alpha=0.9)
        self.original_point, = self.ax.plot([], [], [], 'ro', markersize=8, alpha=0.9)
        self.corrected_point, = self.ax.plot([], [], [], 'bo', markersize=8, alpha=0.9)
        
        # 添加振动幅度信息
        self.vib_text = self.ax.text2D(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=10)
        
        # 添加质量评估
        self.quality_text = self.ax.text2D(0.05, 0.9, '', transform=self.ax.transAxes, fontsize=10)
        
        # 添加图例
        self.ax.legend(loc='upper right', fontsize=10)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.view_init(elev=self.config.view_elev, azim=self.config.view_azim)
        
        # 创建动画
        anim = animation.FuncAnimation(
            self.fig, 
            self._update_animation,
            frames=range(0, self.max_time, max(1, self.max_time//200)),
            interval=1000//self.config.animation_fps
        )
        
        # 保存动画
        anim.save(
            os.path.join(self.config.output_dir, '3d_trajectory_animation.mp4'),
            writer='ffmpeg',
            dpi=self.config.dpi,
            fps=self.config.animation_fps
        )
        
        print(f"✅ 3D动画创建完成 | 保存至: {os.path.join(self.config.output_dir, '3d_trajectory_animation.mp4')}")
    
    def _update_animation(self, frame):
        """更新动画帧"""
        # 更新轨迹线
        self.ideal_line.set_data_3d(
            self.data['x_ideal'][:frame+1],
            self.data['y_ideal'][:frame+1],
            self.data['z_ideal'][:frame+1]
        )
        self.original_line.set_data_3d(
            self.data['x_original'][:frame+1],
            self.data['y_original'][:frame+1],
            self.data['z_original'][:frame+1]
        )
        self.corrected_line.set_data_3d(
            self.data['x_corrected'][:frame+1],
            self.data['y_corrected'][:frame+1],
            self.data['z_corrected'][:frame+1]
        )
        
        # 更新当前点
        self.ideal_point.set_data_3d(
            [self.data['x_ideal'][frame]],
            [self.data['y_ideal'][frame]],
            [self.data['z_ideal'][frame]]
        )
        self.original_point.set_data_3d(
            [self.data['x_original'][frame]],
            [self.data['y_original'][frame]],
            [self.data['z_original'][frame]]
        )
        self.corrected_point.set_data_3d(
            [self.data['x_corrected'][frame]],
            [self.data['y_corrected'][frame]],
            [self.data['z_corrected'][frame]]
        )
        
        # 更新振动信息
        vib = self.data['vibration_magnitude'][frame]
        self.vib_text.set_text(f'振动幅度: {vib:.4f} mm')
        
        # 更新质量评估
        quality = self._calculate_quality(frame)
        self.quality_text.set_text(f'打印质量: {quality:.2f}')
        
        return self.ideal_line, self.original_line, self.corrected_line, \
               self.ideal_point, self.original_point, self.corrected_point, \
               self.vib_text, self.quality_text
    
    def create_quality_comparison(self):
        """创建打印质量对比可视化"""
        print("📊 创建打印质量对比可视化...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=self.config.dpi)
        
        # 计算质量指标
        original_quality = 1 - self.data['vibration_magnitude'] * 2
        corrected_quality = 1 - self.data['corrected_vibration'] * 2
        
        # 限制在0-1范围内
        original_quality = np.clip(original_quality, 0, 1)
        corrected_quality = np.clip(corrected_quality, 0, 1)
        
        # 绘制质量对比
        ax.plot(original_quality, 'r-', alpha=0.7, linewidth=2, label='原始质量')
        ax.plot(corrected_quality, 'b-', alpha=0.7, linewidth=2, label='矫正后质量')
        
        # 添加理想质量线
        ax.axhline(y=1, color='g', linestyle='--', alpha=0.5, label='理想质量')
        
        # 添加质量阈值
        ax.axhline(y=0.8, color='y', linestyle=':', alpha=0.5, label='良好质量阈值')
        ax.axhline(y=0.6, color='orange', linestyle=':', alpha=0.5, label='可接受质量阈值')
        
        # 设置标签
        ax.set_xlabel('样本索引', fontsize=12)
        ax.set_ylabel('打印质量 (0-1)', fontsize=12)
        ax.set_title('打印质量对比', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='lower right', fontsize=10)
        
        plt.tight_layout()
        self._save_image('quality_comparison.png')
        
        print(f"✅ 质量对比可视化创建完成 | 保存至: {os.path.join(self.config.output_dir, 'quality_comparison.png')}")
    
    def create_vibration_comparison(self):
        """创建振动幅度对比可视化"""
        print("🔍 创建振动幅度对比可视化...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=self.config.dpi)
        
        # 计算振动幅度
        vibration_magnitude = self.data['vibration_magnitude']
        corrected_vibration = self.data['corrected_vibration']
        
        # 绘制振动对比
        ax.plot(vibration_magnitude, 'r-', alpha=0.7, linewidth=2, label='原始振动')
        ax.plot(corrected_vibration, 'b-', alpha=0.7, linewidth=2, label='矫正后振动')
        
        # 添加阈值线
        ax.axhline(y=0.05, color='g', linestyle='--', alpha=0.5, label='理想振动阈值')
        ax.axhline(y=0.1, color='y', linestyle=':', alpha=0.5, label='良好振动阈值')
        
        # 设置标签
        ax.set_xlabel('样本索引', fontsize=12)
        ax.set_ylabel('振动幅度 (mm)', fontsize=12)
        ax.set_title('振动幅度对比', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        self._save_image('vibration_comparison.png')
        
        print(f"✅ 振动对比可视化创建完成 | 保存至: {os.path.join(self.config.output_dir, 'vibration_comparison.png')}")
    
    def create_error_3d(self):
        """创建3D误差可视化"""
        print("🔍 创建3D误差可视化...")
        fig = plt.figure(figsize=self.config.fig_size, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # 计算误差向量
        error_x = self.data['x_original'] - self.data['x_ideal']
        error_y = self.data['y_original'] - self.data['y_ideal']
        error_z = self.data['z_original'] - self.data['z_ideal']
        error_magnitude = np.sqrt(error_x**2 + error_y**2 + error_z**2)
        
        # 创建颜色映射
        normalized_error = error_magnitude / np.max(error_magnitude)
        
        # 绘制误差向量
        for i in range(0, len(self.data['x_ideal']), 10):
            ax.quiver(
                self.data['x_ideal'][i], 
                self.data['y_ideal'][i], 
                self.data['z_ideal'][i],
                error_x[i], 
                error_y[i], 
                error_z[i],
                length=0.5,
                color=self.scalar_map.to_rgba(normalized_error[i]),
                alpha=0.6
            )
        
        # 绘制理想轨迹
        ax.plot(
            self.data['x_ideal'], 
            self.data['y_ideal'], 
            self.data['z_ideal'],
            'g-', linewidth=2, alpha=0.8
        )
        
        # 设置标签
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_zlabel('Z (mm)', fontsize=12)
        ax.set_title('3D打印误差可视化', fontsize=14, fontweight='bold')
        
        # 添加颜色条
        cbar = fig.colorbar(self.scalar_map, ax=ax, pad=0.02)
        cbar.set_label('归一化误差', fontsize=10)
        
        plt.tight_layout()
        self._save_image('3d_error_visualization.png')
        
        print(f"✅ 3D误差可视化创建完成 | 保存至: {os.path.join(self.config.output_dir, '3d_error_visualization.png')}")
    
    def _save_image(self, filename):
        """保存图像"""
        plt.savefig(
            os.path.join(self.config.output_dir, filename),
            dpi=self.config.dpi,
            bbox_inches='tight'
        )
        plt.close()
    
    def create_all_visualizations(self):
        """创建所有可视化"""
        self.load_data()
        self.create_3d_visualization()
        self.create_quality_comparison()
        self.create_vibration_comparison()
        self.create_error_3d()
        self.create_3d_animation()
        print(f"\n🎉 所有可视化创建完成！结果保存在: {self.config.output_dir}")

# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("3D打印质量优化可视化系统")
    print("=" * 80)
    
    config = Config()
    visualizer = Enhanced3DVisualizer(config)
    visualizer.create_all_visualizations()
    
    print("\n" + "=" * 80)
    print("可视化系统使用说明:")
    print("1. 3D轨迹对比图 (3d_trajectory_comparison.png):")
    print("   - 绿色: 理想轨迹")
    print("   - 红色: 原始轨迹")
    print("   - 蓝色: 神经网络矫正轨迹")
    print("   - 颜色: 振动幅度（越红振动越大）")
    print("   - 滑块: 交互式查看不同时间点")
    print("\n2. 3D轨迹动画 (3d_trajectory_animation.mp4):")
    print("   - 动态展示打印过程")
    print("   - 红点: 原始轨迹当前位置")
    print("   - 蓝点: 矫正轨迹当前位置")
    print("   - 绿点: 理想轨迹当前位置")
    print("\n3. 打印质量对比 (quality_comparison.png):")
    print("   - 红线: 原始打印质量")
    print("   - 蓝线: 矫正后打印质量")
    print("   - 绿线: 理想质量 (1.0)")
    print("\n4. 振动幅度对比 (vibration_comparison.png):")
    print("   - 红线: 原始振动幅度")
    print("   - 蓝线: 矫正后振动幅度")
    print("   - 绿线: 理想振动阈值 (0.05mm)")
    print("\n5. 3D误差可视化 (3d_error_visualization.png):")
    print("   - 绿线: 理想轨迹")
    print("   - 箭头: 打印误差向量（长度和颜色表示误差大小）")
    print("=" * 80)

if __name__ == "__main__":
    main()