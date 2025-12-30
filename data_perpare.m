%% =========================================================================
%  文件名: nozzle_simulation_specific_shape.m
%  功能: 仿真3D打印喷头在打印特定几何形状过程中的物理行为
%  包含: 喷头振动、温度场、电机负载等影响打印质量的关键因素
% =========================================================================
clear; clc; close all;
rng(2025); % 设置随机种子确保可重复性

%% ==================== 1. 配置参数 ==========================
config = struct();
config.simulation_hours = 0.5;   % 仿真时长 30分钟（典型打印任务）
config.dt = 0.001;              % 时间步长 1ms（高精度振动捕捉）
config.T_total = config.simulation_hours * 3600;
config.n_machines = 20;         % 20台打印机（15正常+5故障）
config.output_dir = 'printer_dataset';
config.shape_type = 'gear';     % 打印特定形状：齿轮（复杂几何，高振动）

% 创建输出目录
if ~exist(config.output_dir, 'dir')
    mkdir(config.output_dir);
end

fprintf('=== 3D打印喷头仿真 - 特定形状: %s ===\n', config.shape_type);
fprintf('时长: %.1f小时 | 步长: %.3fs | 机器数: %d\n', ...
    config.simulation_hours, config.dt, config.n_machines);

%% ==================== 2. 生成特定形状的打印路径 ==========================
% 生成齿轮形状的打印路径（简化版）
function [x_path, y_path, z_path] = generate_gear_path(radius, teeth, layers, layer_height)
    t = linspace(0, 2*pi, 1000);
    tooth_angle = 2*pi/teeth;
    tooth_profile = radius * (1 + 0.1*sin(teeth*t)); % 齿轮轮廓
    
    % 生成单层路径
    x_layer = tooth_profile .* cos(t);
    y_layer = tooth_profile .* sin(t);
    
    % 生成多层路径
    n_points = length(x_layer);
    total_points = n_points * layers;
    x_path = zeros(total_points, 1);
    y_path = zeros(total_points, 1);
    z_path = zeros(total_points, 1);
    
    for i = 1:layers
        idx_start = (i-1)*n_points + 1;
        idx_end = i*n_points;
        x_path(idx_start:idx_end) = x_layer;
        y_path(idx_start:idx_end) = y_layer;
        z_path(idx_start:idx_end) = (i-1) * layer_height;
    end
end

% 生成齿轮路径
gear_radius = 20;      % mm
gear_teeth = 12;
n_layers = 50;
layer_height = 0.2;    % mm

[x_path, y_path, z_path] = generate_gear_path(gear_radius, gear_teeth, n_layers, layer_height);

% 计算路径总长度和打印时间
path_length = 0;
for i = 2:length(x_path)
    dx = x_path(i) - x_path(i-1);
    dy = y_path(i) - y_path(i-1);
    dz = z_path(i) - z_path(i-1);
    path_length = path_length + sqrt(dx^2 + dy^2 + dz^2);
end

% 计算打印速度（基于典型FDM打印机）
print_speed = 60; % mm/s
total_print_time = path_length / print_speed; % 秒
fprintf('⚙️ 齿轮路径生成完成 | 总长度: %.2f mm | 预计打印时间: %.2f 秒\n', path_length, total_print_time);

% 将路径映射到时间轴
N_steps = ceil(config.T_total / config.dt);
time_vector = (0:config.dt:(N_steps-1)*config.dt)';
path_time_ratio = min(1, total_print_time / config.T_total);
path_steps = round(N_steps * path_time_ratio);
path_indices = round(linspace(1, length(x_path), path_steps));

%% ==================== 3. 物理参数生成 ==========================
% 机器特定参数（每台打印机略有差异）
thermal_model = struct();
thermal_model.T_ambient = 22 + 2*rand(config.n_machines, 1);
thermal_model.T_target = 210 + 3*rand(config.n_machines, 1);
thermal_model.mass = 0.03 + 0.005*rand(config.n_machines, 1); % 喷头质量
thermal_model.specific_heat = 1800 + 100*rand(config.n_machines, 1);
thermal_model.convection_coeff = 15 + 2*rand(config.n_machines, 1);
thermal_model.heater_power_base = 30 + 3*rand(config.n_machines, 1);

% 振动参数（关键：喷头动态特性）
vibration_model = struct();
vibration_model.mass = 0.05 + 0.01*rand(config.n_machines, 1); % 喷头+挤出机质量
vibration_model.stiffness_x = 1500 + 300*rand(config.n_machines, 1); % X轴刚度
vibration_model.stiffness_y = 1400 + 250*rand(config.n_machines, 1); % Y轴刚度
vibration_model.damping_x = 0.5 + 0.1*rand(config.n_machines, 1);    % X轴阻尼
vibration_model.damping_y = 0.45 + 0.08*rand(config.n_machines, 1);  % Y轴阻尼
vibration_model.natural_freq_x = sqrt(vibration_model.stiffness_x ./ vibration_model.mass)/(2*pi);
vibration_model.natural_freq_y = sqrt(vibration_model.stiffness_y ./ vibration_model.mass)/(2*pi);

% 电机和控制系统参数
motor_model = struct();
motor_model.rated_current = 1.5 + 0.15*rand(config.n_machines, 1);
motor_model.resistance = 1.2 + 0.1*rand(config.n_machines, 1);
motor_model.inductance = 0.004 + 0.0005*rand(config.n_machines, 1);
motor_model.back_emf_constant = 0.04 + 0.005*rand(config.n_machines, 1);
motor_model.step_angle = 1.8 * pi/180; % 步进电机步距角（弧度）

% 打印质量相关参数
print_quality = struct();
print_quality.bed_adhesion = 0.8 + 0.1*rand(config.n_machines, 1); % 床面附着力
print_quality.filament_diameter = 1.75 + 0.05*rand(config.n_machines, 1); % 耗材直径
print_quality.extrusion_multiplier = 1.0 + 0.05*randn(config.n_machines, 1); % 挤出系数

%% ==================== 4. 仿真主循环 ==========================
fprintf('🚀 开始高精度喷头仿真...\n');

% 预分配数组
N_steps = ceil(config.T_total / config.dt);
time_vector = (0:config.dt:(N_steps-1)*config.dt)';
temperature = zeros(N_steps, config.n_machines);
vibration_disp_x = zeros(N_steps, config.n_machines);
vibration_disp_y = zeros(N_steps, config.n_machines);
vibration_vel_x = zeros(N_steps, config.n_machines);
vibration_vel_y = zeros(N_steps, config.n_machines);
motor_current_x = zeros(N_steps, config.n_machines);
motor_current_y = zeros(N_steps, config.n_machines);
motor_current_z = zeros(N_steps, config.n_machines);
extrusion_pressure = zeros(N_steps, config.n_machines);
nozzle_position_x = zeros(N_steps, config.n_machines);
nozzle_position_y = zeros(N_steps, config.n_machines);
nozzle_position_z = zeros(N_steps, config.n_machines);
print_quality_metric = zeros(N_steps, config.n_machines); % 打印质量指标

% 初始条件
temperature(1, :) = thermal_model.T_ambient';
vibration_disp_x(1, :) = 0.001*randn(1, config.n_machines); % 初始微小振动
vibration_disp_y(1, :) = 0.001*randn(1, config.n_machines);
nozzle_position_x(1, :) = x_path(1);
nozzle_position_y(1, :) = y_path(1);
nozzle_position_z(1, :) = z_path(1);

% 预计算电机隐式求解系数
motor_inv_dt = 1 / config.dt;

% 选择故障机器（25%的机器会出现故障）
faulty_machines = randperm(config.n_machines, 5);
fault_types = randi([1, 4], 1, length(faulty_machines)); % 4种故障类型

fprintf('🔧 故障机器配置:\n');
fault_start_step = zeros(1, length(faulty_machines)); % 为每个故障机器记录故障开始时间
for i = 1:length(faulty_machines)
    mid = faulty_machines(i);
    fault_start_ratio = 0.3 + 0.4*rand(); % 故障在30%-70%打印过程中发生
    fault_start_step(i) = round(fault_start_ratio * N_steps);
    
    switch fault_types(i)
        case 1 % 喷嘴部分堵塞
            fprintf('  机器 %d: 喷嘴部分堵塞 (步 %d)\n', mid, fault_start_step(i));
        case 2 % 机械松动（刚度降低）
            fprintf('  机器 %d: 机械松动故障 (步 %d)\n', mid, fault_start_step(i));
        case 3 % 热敏电阻漂移（温度读数错误）
            fprintf('  机器 %d: 温度传感器故障 (步 %d)\n', mid, fault_start_step(i));
        case 4 % 电机失步
            fprintf('  机器 %d: 电机失步故障 (步 %d)\n', mid, fault_start_step(i));
    end
end

% 主仿真循环
progress_interval = round(N_steps/10);
for t = 2:N_steps
    current_time = time_vector(t);
    
    % 显示进度
    if mod(t, progress_interval) == 0
        fprintf('📊 仿真进度: %.1f%% (%d/%d steps)\n', t/N_steps*100, t, N_steps);
    end
    
    for mid = 1:config.n_machines
        % ========== 路径跟随控制 ==========
        is_faulty_machine = false;
        fault_idx = 0;
        if any(faulty_machines == mid)
            fault_idx = find(faulty_machines == mid, 1);
            if fault_idx <= length(fault_start_step) && exist('fault_start_step', 'var')
                fault_occurred = t > fault_start_step(fault_idx);
                is_faulty_machine = fault_occurred;
            end
        end
        
        if t <= path_steps && ~is_faulty_machine
            % 正常机器跟随路径
            path_idx = min(t, length(path_indices));
            target_x = x_path(path_indices(path_idx));
            target_y = y_path(path_indices(path_idx));
            target_z = z_path(path_indices(path_idx));
        else
            % 故障机器或完成打印后保持当前位置
            target_x = nozzle_position_x(t-1, mid);
            target_y = nozzle_position_y(t-1, mid);
            target_z = nozzle_position_z(t-1, mid);
        end
        
        % ========== 热力学模型 ==========
        prev_temp = temperature(t-1, mid);
        temp_error = thermal_model.T_target(mid) - prev_temp;
        
        % 检查是否为温度传感器故障
        is_temp_fault = false;
        if any(faulty_machines == mid & fault_types == 3)
            fault_idx = find(faulty_machines == mid & fault_types == 3, 1);
            if ~isempty(fault_idx) && t > fault_start_step(fault_idx)
                is_temp_fault = true;
            end
        end
        if is_temp_fault
            % 温度传感器漂移故障
            measured_temp = prev_temp * (0.9 + 0.05*rand());
        else
            measured_temp = prev_temp;
        end
        
        heater_power = thermal_model.heater_power_base(mid) * (1 + 0.5*tanh(temp_error));
        heat_loss = thermal_model.convection_coeff(mid) * (measured_temp - thermal_model.T_ambient(mid));
        dTdt = (heater_power - heat_loss) / (thermal_model.mass(mid) * thermal_model.specific_heat(mid));
        temperature(t, mid) = prev_temp + dTdt * config.dt;
        
        % ========== 振动模型 (关键：喷头动态响应) ==========
        prev_disp_x = vibration_disp_x(t-1, mid);
        prev_vel_x = vibration_vel_x(t-1, mid);
        prev_disp_y = vibration_disp_y(t-1, mid);
        prev_vel_y = vibration_vel_y(t-1, mid);
        
        % 检查是否为机械松动故障
        is_mech_fault = false;
        if any(faulty_machines == mid & fault_types == 2)
            fault_idx = find(faulty_machines == mid & fault_types == 2, 1);
            if ~isempty(fault_idx) && t > fault_start_step(fault_idx)
                is_mech_fault = true;
            end
        end
        if is_mech_fault
            % 机械松动：刚度降低50%
            kx = vibration_model.stiffness_x(mid) * 0.5;
            ky = vibration_model.stiffness_y(mid) * 0.5;
            cx = vibration_model.damping_x(mid) * 0.8; % 阻尼也略有降低
            cy = vibration_model.damping_y(mid) * 0.8;
        else
            kx = vibration_model.stiffness_x(mid);
            ky = vibration_model.stiffness_y(mid);
            cx = vibration_model.damping_x(mid);
            cy = vibration_model.damping_y(mid);
        end
        
        % 位置误差（控制目标与当前位置的差异）
        pos_error_x = target_x - nozzle_position_x(t-1, mid) - prev_disp_x;
        pos_error_y = target_y - nozzle_position_y(t-1, mid) - prev_disp_y;
        
        % 计算加速度（考虑刚度和阻尼）
        accel_x = (kx * pos_error_x - cx * prev_vel_x) / vibration_model.mass(mid);
        accel_y = (ky * pos_error_y - cy * prev_vel_y) / vibration_model.mass(mid);
        
        % 检查是否为电机失步故障
        is_motor_fault = false;
        if any(faulty_machines == mid & fault_types == 4)
            fault_idx = find(faulty_machines == mid & fault_types == 4, 1);
            if ~isempty(fault_idx) && t > fault_start_step(fault_idx)
                is_motor_fault = true;
            end
        end
        if is_motor_fault && rand() < 0.1 % 10%概率失步
            accel_x = accel_x * 0.3; % 电机输出力降低
            accel_y = accel_y * 0.3;
        end
        
        % 更新速度和位移
        new_vel_x = prev_vel_x + accel_x * config.dt;
        new_disp_x = prev_disp_x + new_vel_x * config.dt;
        new_vel_y = prev_vel_y + accel_y * config.dt;
        new_disp_y = prev_disp_y + new_vel_y * config.dt;
        
        vibration_vel_x(t, mid) = new_vel_x;
        vibration_disp_x(t, mid) = new_disp_x;
        vibration_vel_y(t, mid) = new_vel_y;
        vibration_disp_y(t, mid) = new_disp_y;
        
        % ========== 电机电流模型 ==========
        % X轴电机
        prev_current_x = motor_current_x(t-1, mid);
        voltage_x = 12 * sign(pos_error_x); % 简化的电压控制
        back_emf_x = motor_model.back_emf_constant(mid) * abs(new_vel_x);
        motor_current_x(t, mid) = prev_current_x + (voltage_x - back_emf_x - motor_model.resistance(mid)*prev_current_x) * config.dt / motor_model.inductance(mid);
        
        % Y轴电机（类似）
        prev_current_y = motor_current_y(t-1, mid);
        voltage_y = 12 * sign(pos_error_y);
        back_emf_y = motor_model.back_emf_constant(mid) * abs(new_vel_y);
        motor_current_y(t, mid) = prev_current_y + (voltage_y - back_emf_y - motor_model.resistance(mid)*prev_current_y) * config.dt / motor_model.inductance(mid);
        
        % Z轴电机（层切换）
        if mod(t, round(1/(config.dt*10))) == 0 % 每10ms检查一次层切换
            target_layer = min(floor((t/path_steps)*n_layers), n_layers);
            current_layer = round(nozzle_position_z(t-1, mid)/layer_height);
            if target_layer > current_layer
                pos_error_z = layer_height;
            else
                pos_error_z = 0;
            end
        else
            pos_error_z = 0;
        end
        
        prev_current_z = motor_current_z(t-1, mid);
        voltage_z = 12 * sign(pos_error_z);
        motor_current_z(t, mid) = prev_current_z + (voltage_z - motor_model.resistance(mid)*prev_current_z) * config.dt / motor_model.inductance(mid);
        
        % ========== 挤出压力模型 ==========
        % 检查是否为喷嘴堵塞故障
        is_nozzle_fault = false;
        if any(faulty_machines == mid & fault_types == 1)
            fault_idx = find(faulty_machines == mid & fault_types == 1, 1);
            if ~isempty(fault_idx) && t > fault_start_step(fault_idx)
                is_nozzle_fault = true;
            end
        end
        if is_nozzle_fault
            % 喷嘴部分堵塞：压力增加，挤出量减少
            pressure_multiplier = 1.5 + 0.3*rand();
            extrusion_multiplier = 0.7;
        else
            pressure_multiplier = 1.0;
            extrusion_multiplier = print_quality.extrusion_multiplier(mid);
        end
        
        base_pressure = 4 * (1 + 0.2*randn());
        movement_speed = sqrt((target_x - nozzle_position_x(t-1, mid))^2 + ...
                             (target_y - nozzle_position_y(t-1, mid))^2) / config.dt;
        
        % 压力与运动速度和温度相关
        speed_factor = min(1, movement_speed/100); % 归一化到0-1
        temp_factor = (temperature(t, mid) - 180) / 50; % 温度影响因子
        extrusion_pressure(t, mid) = base_pressure * pressure_multiplier * ...
            (0.5 + 0.3*speed_factor + 0.2*temp_factor) * ...
            extrusion_multiplier;
        
        % ========== 位置更新 ==========
        nozzle_position_x(t, mid) = target_x + new_disp_x;
        nozzle_position_y(t, mid) = target_y + new_disp_y;
        nozzle_position_z(t, mid) = nozzle_position_z(t-1, mid) + (pos_error_z > 0) * layer_height * config.dt * 10;
        
        % ========== 打印质量评估 ==========
        % 基于振动幅度、温度稳定性、挤出压力等综合评估
        vibration_magnitude = sqrt(new_disp_x^2 + new_disp_y^2);
        temp_stability = abs(temperature(t, mid) - thermal_model.T_target(mid));
        
        % 基础质量分数（0-1）
        base_quality = 1.0;
        
        % 振动惩罚（振动越大，质量越差）
        vibration_penalty = min(0.5, 10*vibration_magnitude);
        
        % 温度惩罚
        temp_penalty = min(0.3, temp_stability/20);
        
        % 故障惩罚
        fault_penalty = 0;
        if any(faulty_machines == mid)
            fault_idx = find(faulty_machines == mid, 1);
            if ~isempty(fault_idx) && t > fault_start_step(fault_idx)
                fault_penalty = 0.3 + 0.2*rand(); % 故障导致质量显著下降
            end
        end
        
        % 综合质量指标
        print_quality_metric(t, mid) = max(0.1, base_quality - vibration_penalty - temp_penalty - fault_penalty);
        
        % 添加随机噪声
        print_quality_metric(t, mid) = print_quality_metric(t, mid) * (0.95 + 0.1*randn());
    end
end

%% ==================== 5. 生成故障标签 ==========================
fprintf('🏷️  生成故障标签...\n');
fault_label = zeros(N_steps, config.n_machines);
fault_type_label = zeros(N_steps, config.n_machines); % 具体故障类型

for i = 1:length(faulty_machines)
    mid = faulty_machines(i);
    fault_start_ratio = 0.3 + 0.4*rand();
    fault_start_step = round(fault_start_ratio * N_steps);
    
    fault_label(fault_start_step:end, mid) = 1;
    fault_type_label(fault_start_step:end, mid) = fault_types(i);
end

%% ==================== 6. 导出数据集 ==========================
fprintf('💾 导出数据集...\n');

% 创建控制信号（目标值）
ctrl_T_target = repmat(thermal_model.T_target', N_steps, 1);
ctrl_speed_set = 60 * ones(N_steps, config.n_machines); % 60mm/s
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
    motor_current_x(:), motor_current_y(:), motor_current_z(:), ...
    extrusion_pressure(:), nozzle_position_x(:), nozzle_position_y(:), nozzle_position_z(:), ...
    print_quality_metric(:), fault_label(:), fault_type_label(:)];

column_names = {'timestamp', 'machine_id', ...
    'ctrl_T_target', 'ctrl_speed_set', ...
    'ctrl_pos_x', 'ctrl_pos_y', 'ctrl_pos_z', ...
    'temperature_C', 'vibration_disp_x_m', 'vibration_disp_y_m', ...
    'vibration_vel_x_m_s', 'vibration_vel_y_m_s', ...
    'motor_current_x_A', 'motor_current_y_A', 'motor_current_z_A', ...
    'pressure_bar', 'nozzle_pos_x_mm', 'nozzle_pos_y_mm', 'nozzle_pos_z_mm', ...
    'print_quality', 'fault_label', 'fault_type'};

T = array2table(data_matrix, 'VariableNames', column_names);

% 保存为CSV
csv_path = fullfile(config.output_dir, 'nozzle_simulation_gear_print.csv');
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

%% ==================== 7. 可视化结果 ==========================
fprintf('📊 生成可视化结果...\n');

figure('Position', [100, 100, 1200, 800]);

% 1. 选择一台正常机器和一台故障机器进行对比
normal_machines = setdiff(1:config.n_machines, faulty_machines);
normal_machine = normal_machines(1);
faulty_machine = faulty_machines(1);

% 2. 绘制X轴振动对比
subplot(2, 3, 1);
plot(time_vector(1:10000), vibration_disp_x(1:10000, normal_machine)*1000, 'b', 'LineWidth', 1.5);
hold on;
plot(time_vector(1:10000), vibration_disp_x(1:10000, faulty_machine)*1000, 'r', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('X轴振动位移 (mm)');
title('X轴振动对比：正常 vs 故障');
legend('正常机器', '故障机器');
grid on;

% 3. 绘制温度对比
subplot(2, 3, 2);
plot(time_vector(1:10000), temperature(1:10000, normal_machine), 'b', 'LineWidth', 1.5);
hold on;
plot(time_vector(1:10000), temperature(1:10000, faulty_machine), 'r', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('温度 (°C)');
title('喷嘴温度对比');
legend('正常机器', '故障机器');
grid on;

% 4. 绘制打印质量对比
subplot(2, 3, 3);
plot(time_vector(1:10000), print_quality_metric(1:10000, normal_machine), 'b', 'LineWidth', 1.5);
hold on;
plot(time_vector(1:10000), print_quality_metric(1:10000, faulty_machine), 'r', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('打印质量 (0-1)');
title('打印质量指标对比');
legend('正常机器', '故障机器');
grid on;

% 5. 绘制3D打印路径
subplot(2, 3, 4);
plot3(x_path, y_path, z_path, 'b-', 'LineWidth', 2);
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
title('齿轮打印路径');
grid on;
axis equal;

% 6. 频谱分析（振动特性）
subplot(2, 3, 5);
sample_rate = 1/config.dt;
[freq, Pxx] = pwelch(vibration_disp_x(1:50000, normal_machine), [], [], [], sample_rate);
plot(freq(1:100), Pxx(1:100));
xlabel('频率 (Hz)');
ylabel('功率谱密度');
title('正常机器振动频谱');
grid on;

% 7. 机器学习特征相关性
subplot(2, 3, 6);
features = [squeeze(vibration_disp_x(1:10000, normal_machine)), ...
           squeeze(vibration_disp_y(1:10000, normal_machine)), ...
           squeeze(temperature(1:10000, normal_machine)), ...
           squeeze(extrusion_pressure(1:10000, normal_machine))];
[corr_matrix, p_values] = corrcoef(features);
imagesc(corr_matrix);
colorbar;
title('特征相关性矩阵');
xticklabels({'VibX', 'VibY', 'Temp', 'Pressure'});
yticklabels({'VibX', 'VibY', 'Temp', 'Pressure'});

% 保存可视化结果
vis_path = fullfile(config.output_dir, 'simulation_visualization.png');
saveas(gcf, vis_path);
fprintf('✅ 可视化结果已保存至: %s\n', vis_path);

fprintf('🎉 全部任务完成！\n');