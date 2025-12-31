%% =========================================================================
%  文件名: nozzle_simulation_with_correction_targets.m
%  功能: 仿真3D打印喷头行为，同时生成用于矫正控制的目标信号
%  特点: 生成"原始"和"理想"轨迹对，用于训练矫正控制器
% =========================================================================
clear; clc; close all;
rng(2025); % 设置随机种子确保可重复性

%% ==================== 1. 配置参数 ==========================
config = struct();
config.simulation_hours = 0.2;   % 仿真时长 12分钟（典型打印任务）
config.dt = 0.001;              % 时间步长 1ms（高精度振动捕捉）
config.T_total = config.simulation_hours * 3600;
config.n_machines = 25;         % 25台打印机（20正常+5故障）
config.output_dir = 'printer_dataset_correction';
config.shape_type = 'gear_optimized'; % 优化的齿轮形状

% 创建输出目录
if ~exist(config.output_dir, 'dir')
    mkdir(config.output_dir);
end

fprintf('=== 3D打印喷头仿真与矫正目标生成 ===\n');
fprintf('形状: %s | 时长: %.1f小时 | 步长: %.3fs | 机器数: %d\n', ...
    config.shape_type, config.simulation_hours, config.dt, config.n_machines);

%% ==================== 2. 生成打印路径 ==========================
% 生成更平滑的齿轮路径
function [x_path, y_path, z_path, x_ideal, y_ideal] = generate_optimized_gear_path(radius, teeth, layers, layer_height)
    t = linspace(0, 2*pi, 2000);
    tooth_angle = 2*pi/teeth;
    
    % 基础齿轮轮廓（平滑处理）
    tooth_profile = radius * (1 + 0.08*(sin(teeth*t) + 0.3*sin(2*teeth*t)));
    
    % 生成单层路径
    x_layer = tooth_profile .* cos(t);
    y_layer = tooth_profile .* sin(t);
    
    % 生成理想路径（无振动）
    smoothing_factor = 0.95;
    x_ideal_layer = smooth(x_layer, smoothing_factor);
    y_ideal_layer = smooth(y_layer, smoothing_factor);
    
    % 生成多层路径
    n_points = length(x_layer);
    total_points = n_points * layers;
    x_path = zeros(total_points, 1);
    y_path = zeros(total_points, 1);
    z_path = zeros(total_points, 1);
    x_ideal = zeros(total_points, 1);
    y_ideal = zeros(total_points, 1);
    
    for i = 1:layers
        idx_start = (i-1)*n_points + 1;
        idx_end = i*n_points;
        x_path(idx_start:idx_end) = x_layer;
        y_path(idx_start:idx_end) = y_layer;
        z_path(idx_start:idx_end) = (i-1) * layer_height;
        x_ideal(idx_start:idx_end) = x_ideal_layer;
        y_ideal(idx_start:idx_end) = y_ideal_layer;
    end
end

% 生成齿轮路径
gear_radius = 25;      % mm
gear_teeth = 16;
n_layers = 30;
layer_height = 0.2;    % mm

[x_path, y_path, z_path, x_ideal, y_ideal] = generate_optimized_gear_path(gear_radius, gear_teeth, n_layers, layer_height);

% 计算路径总长度和打印时间
path_length = 0;
for i = 2:length(x_path)
    dx = x_path(i) - x_path(i-1);
    dy = y_path(i) - y_path(i-1);
    dz = z_path(i) - z_path(i-1);
    path_length = path_length + sqrt(dx^2 + dy^2 + dz^2);
end

% 计算打印速度
print_speed = 50; % mm/s
total_print_time = path_length / print_speed; % 秒
fprintf('⚙️ 齿轮路径生成完成 | 总长度: %.2f mm | 预计打印时间: %.2f 秒\n', path_length, total_print_time);

% 将路径映射到时间轴
N_steps = ceil(config.T_total / config.dt);
time_vector = (0:config.dt:(N_steps-1)*config.dt)';
path_time_ratio = min(1, total_print_time / config.T_total);
path_steps = round(N_steps * path_time_ratio);
path_indices = round(linspace(1, length(x_path), path_steps));

%% ==================== 3. 物理参数生成 ==========================
% 机器特定参数
thermal_model = struct();
thermal_model.T_ambient = 22 + 1.5*rand(config.n_machines, 1);
thermal_model.T_target = 215 + 2*rand(config.n_machines, 1);
thermal_model.mass = 0.035 + 0.003*rand(config.n_machines, 1);
thermal_model.specific_heat = 1750 + 80*rand(config.n_machines, 1);
thermal_model.convection_coeff = 16 + 1.5*rand(config.n_machines, 1);
thermal_model.heater_power_base = 32 + 2*rand(config.n_machines, 1);

% 振动参数（关键：喷头动态特性）
vibration_model = struct();
vibration_model.mass = 0.045 + 0.008*rand(config.n_machines, 1);
vibration_model.stiffness_x = 1600 + 250*rand(config.n_machines, 1);
vibration_model.stiffness_y = 1500 + 220*rand(config.n_machines, 1);
vibration_model.damping_x = 0.48 + 0.08*rand(config.n_machines, 1);
vibration_model.damping_y = 0.42 + 0.07*rand(config.n_machines, 1);
vibration_model.natural_freq_x = sqrt(vibration_model.stiffness_x ./ vibration_model.mass)/(2*pi);
vibration_model.natural_freq_y = sqrt(vibration_model.stiffness_y ./ vibration_model.mass)/(2*pi);

% 电机参数
motor_model = struct();
motor_model.rated_current = 1.6 + 0.1*rand(config.n_machines, 1);
motor_model.resistance = 1.25 + 0.08*rand(config.n_machines, 1);
motor_model.inductance = 0.0042 + 0.0004*rand(config.n_machines, 1);
motor_model.back_emf_constant = 0.042 + 0.004*rand(config.n_machines, 1);

% 打印质量参数
print_quality = struct();
print_quality.filament_diameter = 1.75 + 0.03*rand(config.n_machines, 1);
print_quality.extrusion_multiplier = 1.0 + 0.03*randn(config.n_machines, 1);

%% ==================== 4. 选择故障机器 ==========================
faulty_machines = randperm(config.n_machines, 5);
fault_types = randi([1, 3], 1, length(faulty_machines)); % 3种故障类型
fault_start_step = zeros(1, length(faulty_machines));

fprintf('🔧 故障机器配置:\n');
for i = 1:length(faulty_machines)
    mid = faulty_machines(i);
    fault_start_ratio = 0.4 + 0.3*rand(); % 故障在40%-70%打印过程中发生
    fault_start_step(i) = round(fault_start_ratio * N_steps);
    
    switch fault_types(i)
        case 1 % 喷嘴部分堵塞
            fprintf('  机器 %d: 喷嘴部分堵塞 (步 %d)\n', mid, fault_start_step(i));
        case 2 % 机械松动
            fprintf('  机器 %d: 机械松动故障 (步 %d)\n', mid, fault_start_step(i));
        case 3 % 电机性能下降
            fprintf('  机器 %d: 电机性能下降 (步 %d)\n', mid, fault_start_step(i));
    end
end

%% ==================== 5. 预分配数组 ==========================
% 原始（未矫正）变量
temperature = zeros(N_steps, config.n_machines);
vibration_disp_x = zeros(N_steps, config.n_machines);
vibration_disp_y = zeros(N_steps, config.n_machines);
vibration_vel_x = zeros(N_steps, config.n_machines);
vibration_vel_y = zeros(N_steps, config.n_machines);
motor_current_x = zeros(N_steps, config.n_machines);
motor_current_y = zeros(N_steps, config.n_machines);
nozzle_position_x = zeros(N_steps, config.n_machines);
nozzle_position_y = zeros(N_steps, config.n_machines);
nozzle_position_z = zeros(N_steps, config.n_machines);
extrusion_pressure = zeros(N_steps, config.n_machines);
print_quality_metric = zeros(N_steps, config.n_machines);

% 理想（矫正后）变量
ideal_position_x = zeros(N_steps, config.n_machines);
ideal_position_y = zeros(N_steps, config.n_machines);
ideal_temperature = zeros(N_steps, config.n_machines);
ideal_vibration_disp_x = zeros(N_steps, config.n_machines);
ideal_vibration_disp_y = zeros(N_steps, config.n_machines);

% 矫正控制信号
correction_signal_x = zeros(N_steps, config.n_machines);
correction_signal_y = zeros(N_steps, config.n_machines);
correction_signal_temp = zeros(N_steps, config.n_machines);

% 初始条件
temperature(1, :) = thermal_model.T_ambient';
ideal_temperature(1, :) = thermal_model.T_ambient';
vibration_disp_x(1, :) = 0.0005*randn(1, config.n_machines);
vibration_disp_y(1, :) = 0.0005*randn(1, config.n_machines);
ideal_vibration_disp_x(1, :) = zeros(1, config.n_machines);
ideal_vibration_disp_y(1, :) = zeros(1, config.n_machines);
nozzle_position_x(1, :) = x_path(1);
nozzle_position_y(1, :) = y_path(1);
nozzle_position_z(1, :) = z_path(1);
ideal_position_x(1, :) = x_ideal(1);
ideal_position_y(1, :) = y_ideal(1);

%% ==================== 6. 仿真主循环 ==========================
fprintf('🚀 开始高精度喷头仿真...\n');
progress_interval = round(N_steps/10);

for t = 2:N_steps
    current_time = time_vector(t);
    
    % 显示进度
    if mod(t, progress_interval) == 0
        fprintf('📊 仿真进度: %.1f%% (%d/%d steps)\n', t/N_steps*100, t, N_steps);
    end
    
    for mid = 1:config.n_machines
        % ========= 路径跟随控制 =========
        is_faulty = false;
        fault_idx = 0;
        
        if any(faulty_machines == mid)
            fault_idx = find(faulty_machines == mid, 1);
            if fault_idx <= length(fault_start_step) && t > fault_start_step(fault_idx)
                is_faulty = true;
            end
        end
        
        % 获取目标位置
        if t <= path_steps
            path_idx = min(t, length(path_indices));
            target_x = x_path(path_indices(path_idx));
            target_y = y_path(path_indices(path_idx));
            target_z = z_path(path_indices(path_idx));
            ideal_target_x = x_ideal(path_indices(path_idx));
            ideal_target_y = y_ideal(path_indices(path_idx));
        else
            target_x = nozzle_position_x(t-1, mid);
            target_y = nozzle_position_y(t-1, mid);
            target_z = nozzle_position_z(t-1, mid);
            ideal_target_x = ideal_position_x(t-1, mid);
            ideal_target_y = ideal_position_y(t-1, mid);
        end
        
        % ========= 理想系统（无振动） =========
        % 温度控制（理想）
        prev_temp_ideal = ideal_temperature(t-1, mid);
        temp_error_ideal = thermal_model.T_target(mid) - prev_temp_ideal;
        heater_power_ideal = thermal_model.heater_power_base(mid) * (1 + 0.3*tanh(temp_error_ideal));
        heat_loss_ideal = thermal_model.convection_coeff(mid) * (prev_temp_ideal - thermal_model.T_ambient(mid));
        dTdt_ideal = (heater_power_ideal - heat_loss_ideal) / (thermal_model.mass(mid) * thermal_model.specific_heat(mid));
        ideal_temperature(t, mid) = prev_temp_ideal + dTdt_ideal * config.dt;
        
        % 位置控制（理想，无振动）
        ideal_position_x(t, mid) = ideal_target_x;
        ideal_position_y(t, mid) = ideal_target_y;
        ideal_position_z = target_z;
        
        % ========= 实际系统（有振动） =========
        % 温度模型
        prev_temp = temperature(t-1, mid);
        temp_error = thermal_model.T_target(mid) - prev_temp;
        
        % 温度传感器故障
        is_temp_sensor_fault = (fault_types(fault_idx) == 3 && is_faulty);
        measured_temp = prev_temp;
        if is_temp_sensor_fault
            measured_temp = prev_temp * (0.85 + 0.1*rand());
        end
        
        heater_power = thermal_model.heater_power_base(mid) * (1 + 0.5*tanh(temp_error));
        heat_loss = thermal_model.convection_coeff(mid) * (measured_temp - thermal_model.T_ambient(mid));
        dTdt = (heater_power - heat_loss) / (thermal_model.mass(mid) * thermal_model.specific_heat(mid));
        temperature(t, mid) = prev_temp + dTdt * config.dt;
        
        % 振动模型
        prev_disp_x = vibration_disp_x(t-1, mid);
        prev_vel_x = vibration_vel_x(t-1, mid);
        prev_disp_y = vibration_disp_y(t-1, mid);
        prev_vel_y = vibration_vel_y(t-1, mid);
        
        % 机械故障（刚度降低）
        is_mech_fault = (fault_types(fault_idx) == 2 && is_faulty);
        if is_mech_fault
            kx = vibration_model.stiffness_x(mid) * 0.6;  % 刚度降低40%
            ky = vibration_model.stiffness_y(mid) * 0.6;
            cx = vibration_model.damping_x(mid) * 0.7;
            cy = vibration_model.damping_y(mid) * 0.7;
        else
            kx = vibration_model.stiffness_x(mid);
            ky = vibration_model.stiffness_y(mid);
            cx = vibration_model.damping_x(mid);
            cy = vibration_model.damping_y(mid);
        end
        
        % 位置误差（控制目标与当前位置的差异）
        pos_error_x = target_x - nozzle_position_x(t-1, mid) - prev_disp_x;
        pos_error_y = target_y - nozzle_position_y(t-1, mid) - prev_disp_y;
        
        % 电机性能故障
        motor_factor = 1.0;
        is_motor_fault = (fault_types(fault_idx) == 3 && is_faulty);
        if is_motor_fault
            motor_factor = 0.7;  % 电机输出力降低30%
        end
        
        % 计算加速度
        accel_x = motor_factor * (kx * pos_error_x - cx * prev_vel_x) / vibration_model.mass(mid);
        accel_y = motor_factor * (ky * pos_error_y - cy * prev_vel_y) / vibration_model.mass(mid);
        
        % 更新速度和位移
        new_vel_x = prev_vel_x + accel_x * config.dt;
        new_disp_x = prev_disp_x + new_vel_x * config.dt;
        new_vel_y = prev_vel_y + accel_y * config.dt;
        new_disp_y = prev_disp_y + new_vel_y * config.dt;
        
        vibration_vel_x(t, mid) = new_vel_x;
        vibration_disp_x(t, mid) = new_disp_x;
        vibration_vel_y(t, mid) = new_vel_y;
        vibration_disp_y(t, mid) = new_disp_y;
        
        % 电机电流
        prev_current_x = motor_current_x(t-1, mid);
        voltage_x = 12 * sign(pos_error_x);
        back_emf_x = motor_model.back_emf_constant(mid) * abs(new_vel_x);
        motor_current_x(t, mid) = prev_current_x + (voltage_x - back_emf_x - motor_model.resistance(mid)*prev_current_x) * config.dt / motor_model.inductance(mid);
        
        prev_current_y = motor_current_y(t-1, mid);
        voltage_y = 12 * sign(pos_error_y);
        back_emf_y = motor_model.back_emf_constant(mid) * abs(new_vel_y);
        motor_current_y(t, mid) = prev_current_y + (voltage_y - back_emf_y - motor_model.resistance(mid)*prev_current_y) * config.dt / motor_model.inductance(mid);
        
        % 位置更新
        nozzle_position_x(t, mid) = target_x + new_disp_x;
        nozzle_position_y(t, mid) = target_y + new_disp_y;
        nozzle_position_z(t, mid) = target_z;
        
        % 挤出压力
        is_nozzle_fault = (fault_types(fault_idx) == 1 && is_faulty);
        if is_nozzle_fault
            pressure_multiplier = 1.8 + 0.4*rand();
        else
            pressure_multiplier = 1.0;
        end
        
        movement_speed = sqrt((target_x - nozzle_position_x(t-1, mid))^2 + ...
                             (target_y - nozzle_position_y(t-1, mid))^2) / config.dt;
        
        % 压力与运动速度和温度相关
        speed_factor = min(1, movement_speed/80);
        temp_factor = (temperature(t, mid) - 180) / 50;
        base_pressure = 4.5 * (1 + 0.15*randn());
        extrusion_pressure(t, mid) = base_pressure * pressure_multiplier * ...
            (0.6 + 0.25*speed_factor + 0.15*temp_factor) * ...
            print_quality.extrusion_multiplier(mid);
        
        % ========= 计算矫正信号 =========
        % 位置矫正信号
        correction_signal_x(t, mid) = ideal_target_x - target_x;
        correction_signal_y(t, mid) = ideal_target_y - target_y;
        
        % 温度矫正信号
        correction_signal_temp(t, mid) = thermal_model.T_target(mid) - temperature(t, mid);
        
        % ========= 打印质量评估 =========
        vibration_magnitude = sqrt(new_disp_x^2 + new_disp_y^2);
        temp_stability = abs(temperature(t, mid) - thermal_model.T_target(mid));
        
        base_quality = 1.0;
        vibration_penalty = min(0.6, 12*vibration_magnitude);
        temp_penalty = min(0.25, temp_stability/15);
        if is_faulty
            fault_penalty = 0.4 + 0.25*rand();
        else
            fault_penalty = 0;
        end
        
        quality_score = max(0.1, base_quality - vibration_penalty - temp_penalty - fault_penalty);
        print_quality_metric(t, mid) = quality_score * (0.97 + 0.06*randn());
    end
end

%% ==================== 7. 生成故障标签 ==========================
fprintf('🏷️  生成故障标签...\n');
fault_label = zeros(N_steps, config.n_machines);
fault_type_label = zeros(N_steps, config.n_machines);

for i = 1:length(faulty_machines)
    mid = faulty_machines(i);
    fault_start = fault_start_step(i);
    fault_label(fault_start:end, mid) = 1;
    fault_type_label(fault_start:end, mid) = fault_types(i);
end

%% ==================== 8. 导出数据集 ==========================
fprintf('💾 导出数据集...\n');

% 控制信号
ctrl_T_target = repmat(thermal_model.T_target', N_steps, 1);
ctrl_speed_set = 50 * ones(N_steps, config.n_machines); % 50mm/s
ctrl_position_target_x = zeros(N_steps, config.n_machines);
ctrl_position_target_y = zeros(N_steps, config.n_machines);
ctrl_position_target_z = zeros(N_steps, config.n_machines);

for t = 1:min(path_steps, N_steps)
    path_idx = min(t, length(path_indices));
    ctrl_position_target_x(t, :) = x_path(path_indices(path_idx));
    ctrl_position_target_y(t, :) = y_path(path_indices(path_idx));
    ctrl_position_target_z(t, :) = z_path(path_indices(path_idx));
end

% 组合所有数据
[time_grid, machine_grid] = ndgrid(time_vector, 1:config.n_machines);
data_matrix = [time_grid(:), machine_grid(:), ...
    ctrl_T_target(:), ctrl_speed_set(:), ...
    ctrl_position_target_x(:), ctrl_position_target_y(:), ctrl_position_target_z(:), ...
    temperature(:), vibration_disp_x(:), vibration_disp_y(:), ...
    vibration_vel_x(:), vibration_vel_y(:), ...
    motor_current_x(:), motor_current_y(:), ...
    extrusion_pressure(:), nozzle_position_x(:), nozzle_position_y(:), nozzle_position_z(:), ...
    ideal_position_x(:), ideal_position_y(:), ...
    ideal_temperature(:), ideal_vibration_disp_x(:), ideal_vibration_disp_y(:), ...
    correction_signal_x(:), correction_signal_y(:), correction_signal_temp(:), ...
    print_quality_metric(:), fault_label(:), fault_type_label(:)];

column_names = {'timestamp', 'machine_id', ...
    'ctrl_T_target', 'ctrl_speed_set', ...
    'ctrl_pos_x', 'ctrl_pos_y', 'ctrl_pos_z', ...
    'temperature_C', 'vibration_disp_x_m', 'vibration_disp_y_m', ...
    'vibration_vel_x_m_s', 'vibration_vel_y_m_s', ...
    'motor_current_x_A', 'motor_current_y_A', ...
    'pressure_bar', 'nozzle_pos_x_mm', 'nozzle_pos_y_mm', 'nozzle_pos_z_mm', ...
    'ideal_pos_x_mm', 'ideal_pos_y_mm', ...
    'ideal_temperature_C', 'ideal_vib_disp_x_m', 'ideal_vib_disp_y_m', ...
    'correction_x_mm', 'correction_y_mm', 'correction_temp_C', ...
    'print_quality', 'fault_label', 'fault_type'};

T = array2table(data_matrix, 'VariableNames', column_names);

% 保存为CSV
csv_path = fullfile(config.output_dir, 'printer_gear_correction_dataset.csv');
writetable(T, csv_path);

% 保存元数据
metadata = struct();
metadata.physical_models.thermal = thermal_model;
metadata.physical_models.vibration = vibration_model;
metadata.physical_models.motor = motor_model;
metadata.print_quality = print_quality;
metadata.faulty_machines = faulty_machines;
metadata.fault_types = fault_types;
metadata.shape_type = config.shape_type;
metadata.path_length = path_length;
metadata.total_print_time = total_print_time;
save(fullfile(config.output_dir, 'simulation_metadata.mat'), 'metadata');

fprintf('✅ 仿真完成！数据已保存至: %s\n', csv_path);
fprintf('📊 总样本数: %d\n', size(data_matrix, 1));
fprintf('🔧 故障机器数: %d (类型: %s)\n', length(faulty_machines), mat2str(unique(fault_types)));
fprintf('🎯 喷头振动幅度范围: [%.6f, %.6f] m\n', ...
    min(min(vibration_disp_x)), max(max(vibration_disp_x)));
fprintf('🔥 温度范围: [%.1f, %.1f] °C\n', ...
    min(min(temperature)), max(max(temperature)));

%% ==================== 9. 可视化结果 ==========================
fprintf('📊 生成可视化结果...\n');

figure('Position', [100, 100, 1400, 900], 'Color', 'white');

% 1. 选择一台正常机器
normal_machines = setdiff(1:config.n_machines, faulty_machines);
normal_machine = normal_machines(1);

% 2. 实际位置 vs 理想位置对比 (X轴)
subplot(2, 4, 1);
plot(time_vector(1:5000), nozzle_position_x(1:5000, normal_machine), 'b', 'LineWidth', 1.5);
hold on;
plot(time_vector(1:5000), ideal_position_x(1:5000, normal_machine), 'r--', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('X位置 (mm)');
title('X轴位置: 实际 vs 理想');
legend('实际位置', '理想位置');
grid on;

% 3. 实际位置 vs 理想位置对比 (Y轴)
subplot(2, 4, 2);
plot(time_vector(1:5000), nozzle_position_y(1:5000, normal_machine), 'b', 'LineWidth', 1.5);
hold on;
plot(time_vector(1:5000), ideal_position_y(1:5000, normal_machine), 'r--', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('Y位置 (mm)');
title('Y轴位置: 实际 vs 理想');
legend('实际位置', '理想位置');
grid on;

% 4. 振动位移对比
subplot(2, 4, 3);
plot(time_vector(1:5000), vibration_disp_x(1:5000, normal_machine)*1000, 'b', 'LineWidth', 1.5);
hold on;
plot(time_vector(1:5000), vibration_disp_y(1:5000, normal_machine)*1000, 'g', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('振动位移 (mm)');
title('喷头振动位移');
legend('X方向', 'Y方向');
grid on;

% 5. 温度对比
subplot(2, 4, 4);
plot(time_vector(1:5000), temperature(1:5000, normal_machine), 'b', 'LineWidth', 1.5);
hold on;
plot(time_vector(1:5000), ideal_temperature(1:5000, normal_machine), 'r--', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('温度 (°C)');
title('喷嘴温度: 实际 vs 理想');
legend('实际温度', '理想温度');
grid on;

% 6. 3D打印路径与振动
subplot(2, 4, 5);
scatter3(nozzle_position_x(1:2000:end, normal_machine), ...
         nozzle_position_y(1:2000:end, normal_machine), ...
         nozzle_position_z(1:2000:end, normal_machine), ...
         10, vibration_disp_x(1:2000:end, normal_machine)*1000, 'filled');
colormap jet;
colorbar;
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
title('3D打印路径 (颜色: X振动幅度)');
grid on;
axis equal;

% 7. 矫正信号
subplot(2, 4, 6);
plot(time_vector(1:5000), correction_signal_x(1:5000, normal_machine), 'b', 'LineWidth', 1.5);
hold on;
plot(time_vector(1:5000), correction_signal_y(1:5000, normal_machine), 'g', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('矫正信号 (mm)');
title('位置矫正信号');
legend('X方向', 'Y方向');
grid on;

% 8. 打印质量对比
subplot(2, 4, 7);
plot(time_vector(1:5000), print_quality_metric(1:5000, normal_machine), 'b', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('打印质量 (0-1)');
title('打印质量指标');
ylim([0.2, 1.0]);
grid on;

% 9. 3D形状可视化
subplot(2, 4, 8);
plot3(x_path, y_path, z_path, 'b-', 'LineWidth', 1);
hold on;
plot3(x_ideal, y_ideal, z_path, 'r--', 'LineWidth', 1);
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
title('打印路径: 原始 vs 优化');
legend('原始路径', '优化路径');
grid on;
axis equal;

% 保存可视化结果
vis_path = fullfile(config.output_dir, 'correction_simulation_results.png');
exportgraphics(gcf, vis_path, 'Resolution', 300);
fprintf('✅ 可视化结果已保存至: %s\n', vis_path);

fprintf('🎉 全部任务完成！\n');