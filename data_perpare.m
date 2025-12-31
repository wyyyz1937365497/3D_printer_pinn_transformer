%% =========================================================================
%  文件名: nozzle_simulation_with_correction_targets.m
%  功能: 仿真3D打印喷头行为，同时生成用于矫正控制的目标信号
%  特点: 生成"原始"和"理想"轨迹对，用于训练矫正控制器
%  新特性: 以完成一次齿轮打印作为仿真结束标志，不依赖时间限制，完善CPU并行加速
% =========================================================================
clear; clc; close all;
rng(2025); % 设置随机种子确保可重复性

% 检查是否有并行计算工具箱和GPU支持
hasParallel = license('test', 'Distrib_Computing_Toolbox');
hasGPU = gpuDeviceCount > 0;
if hasGPU
    gpuDevice; % 使用第一个GPU
end

fprintf('并行和GPU支持检查:\n');
if hasParallel
    fprintf('  并行计算工具箱: %s\n', "可用");
else
    fprintf('  并行计算工具箱: %s\n', "不可用");
end
if hasGPU
    fprintf('  GPU支持: %s\n', "可用");
else
    fprintf('  GPU支持: %s\n', "不可用");
end

%% ==================== 1. 配置参数 ==========================
config = struct();
config.dt = 0.001;              % 时间步长 1ms（高精度振动捕捉）
config.n_machines = 100;        % 100台打印机（80正常+20故障）
config.output_dir = 'printer_dataset_correction';
config.shape_type = 'gear_optimized'; % 优化的齿轮形状
config.use_gpu = hasGPU && false;  % 默认禁用GPU，因为并行处理时可能效率不高
config.use_parallel = hasParallel && true;  % 可以设置为false来禁用并行计算

% 创建输出目录
if ~exist(config.output_dir, 'dir')
    mkdir(config.output_dir);
end
fprintf('=== 3D打印喷头仿真与矫正目标生成 ===\n');
fprintf('形状: %s | 步长: %.3fs | 机器数: %d\n', ...
    config.shape_type, config.dt, config.n_machines);
if config.use_gpu
    gpuStr = "是";
else
    gpuStr = "否";
end
if config.use_parallel
    parallelStr = "是";
else
    parallelStr = "否";
end
fprintf('使用GPU: %s | 使用并行: %s\n', gpuStr, parallelStr);

%% ==================== 2. 生成打印路径 ==========================
% 生成更平滑的齿轮路径
function [x_path, y_path, z_path, x_ideal, y_ideal] = generate_optimized_gear_path(radius, teeth, layers, layer_height)
    t = linspace(0, 2*pi, 2000);
    tooth_angle = 2*pi/teeth;
    % 基础齿轮轮廓（单位：mm）
    tooth_profile = radius * (1 + 0.08*(sin(teeth*t) + 0.3*sin(2*teeth*t)));
    % 生成单层路径
    x_layer = tooth_profile .* cos(t);
    y_layer = tooth_profile .* sin(t);
    % 生成理想路径 - 显著减少振动
    [b, a] = butter(2, 0.1);  % 二阶低通滤波器
    x_ideal_layer = filtfilt(b, a, x_layer);
    y_ideal_layer = filtfilt(b, a, y_layer);
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
gear_radius = 10;      % mm，直径约20mm
gear_teeth = 16;
n_layers = 30;         % 减少层数以适应单次打印
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
print_speed = 50; % mm/s
total_print_time = path_length / print_speed; % 秒

fprintf('⚙️ 齿轮路径生成完成 | 总长度: %.2f mm | 预计打印时间: %.2f 秒\n', path_length, total_print_time);

% 计算仿真步数以确保完整打印整个路径
N_steps = round(total_print_time / config.dt) + length(x_path);  % 确保有足够的时间完成路径
fprintf('仿真步数: %d (约 %.2f 秒)\n', N_steps, N_steps*config.dt);

% 将路径映射到时间轴
time_vector = (0:config.dt:(N_steps-1)*config.dt)';
path_indices = round(linspace(1, length(x_path), min(N_steps, length(x_path))));

%% ==================== 3. 物理参数生成 ==========================
thermal_model = struct();
thermal_model.T_ambient = 22 + 1.5*rand(config.n_machines, 1);
thermal_model.T_target = 215 + 2*rand(config.n_machines, 1);
thermal_model.mass = 0.035 + 0.003*rand(config.n_machines, 1);
thermal_model.specific_heat = 1750 + 80*rand(config.n_machines, 1);
thermal_model.convection_coeff = 16 + 1.5*rand(config.n_machines, 1);
thermal_model.heater_power_base = 32 + 2*rand(config.n_machines, 1);

vibration_model = struct();
vibration_model.mass = 0.45 + 0.05*rand(config.n_machines, 1);
vibration_model.stiffness_x = 800 + 150*rand(config.n_machines, 1);
vibration_model.stiffness_y = 750 + 120*rand(config.n_machines, 1);
vibration_model.damping_x = 0.24 + 0.04*rand(config.n_machines, 1);
vibration_model.damping_y = 0.21 + 0.035*rand(config.n_machines, 1);
vibration_model.natural_freq_x = sqrt(vibration_model.stiffness_x ./ vibration_model.mass)/(2*pi);
vibration_model.natural_freq_y = sqrt(vibration_model.stiffness_y ./ vibration_model.mass)/(2*pi);

motor_model = struct();
motor_model.rated_current = 2.0 + 0.2*rand(config.n_machines, 1);
motor_model.resistance = 1.25 + 0.08*rand(config.n_machines, 1);
motor_model.inductance = 0.0042 + 0.0004*rand(config.n_machines, 1);
motor_model.back_emf_constant = 0.035 + 0.003*rand(config.n_machines, 1);

print_quality = struct();
print_quality.filament_diameter = 1.75 + 0.03*rand(config.n_machines, 1);
print_quality.extrusion_multiplier = 1.0 + 0.03*randn(config.n_machines, 1);

%% ==================== 4. 选择故障机器 ==========================
faulty_machines = randperm(config.n_machines, 20);
fault_types = randi([1, 3], 1, length(faulty_machines));
fault_start_step = zeros(1, length(faulty_machines));
fprintf('🔧 故障机器配置:\n');
for i = 1:length(faulty_machines)
    mid = faulty_machines(i);
    fault_start_ratio = 0.3 + 0.5*rand();  % 在打印过程的30%-80%之间发生故障
    fault_start_step(i) = round(fault_start_ratio * N_steps);
    switch fault_types(i)
        case 1
            fprintf('  机器 %d: 喷嘴部分堵塞 (步 %d)\n', mid, fault_start_step(i));
        case 2
            fprintf('  机器 %d: 机械松动故障 (步 %d)\n', mid, fault_start_step(i));
        case 3
            fprintf('  机器 %d: 电机性能下降 (步 %d)\n', mid, fault_start_step(i));
    end
end

%% ==================== 5. 仿真主循环 + 实时写入（逐机） ==========================
fprintf('🚀 开始高精度喷头仿真...\n');
progress_interval = round(N_steps/10);

% 准备用于并行处理的参数
if config.use_parallel
    % 检查并行池状态并进行优化
    try
        current_pool = gcp('nocreate');
        if isempty(current_pool)
            % 没有活动池，创建新池，限制工作进程数
            current_pool_size = min(config.n_machines, 12);  % 限制最大工作进程数
            parpool('local', current_pool_size);
            fprintf('并行池已创建，工作进程数: %d\n', current_pool_size);
        else
            % 使用现有池
            fprintf('检测到现有并行池，工作进程数: %d\n', current_pool.NumWorkers);
        end
    catch ME
        fprintf('警告: 并行池创建失败，错误: %s\n', ME.message);
        fprintf('将使用串行处理模式\n');
        config.use_parallel = false;
    end
end

% 准备仿真函数参数
sim_params = struct();
sim_params.N_steps = N_steps;
sim_params.path_indices = path_indices;
sim_params.x_path = x_path;
sim_params.y_path = y_path;
sim_params.z_path = z_path;
sim_params.x_ideal = x_ideal;
sim_params.y_ideal = y_ideal;
sim_params.thermal_model = thermal_model;
sim_params.vibration_model = vibration_model;
sim_params.print_quality = print_quality;
sim_params.faulty_machines = faulty_machines;
sim_params.fault_types = fault_types;
sim_params.fault_start_step = fault_start_step;
sim_params.dt = config.dt;
sim_params.use_gpu = config.use_gpu;  % 添加GPU使用配置
sim_params.path_length = length(x_path);  % 添加路径长度信息

% 使用并行或串行方式处理每台机器
start_time = tic;  % 开始计时
if config.use_parallel
    % 并行处理每台机器
    machine_ids = 1:config.n_machines;
    [results] = cell(1, config.n_machines);
    fprintf('开始并行仿真，共%d台机器...\n', config.n_machines);
    
    % 先处理前几台机器以提供反馈
    fprintf('正在启动并行池和仿真进程...\n');
    parfor mid = 1:config.n_machines
        if config.use_gpu
            % 将数据移到GPU上进行计算
            result = simulate_single_machine_gpu(mid, sim_params);
        else
            % CPU计算
            result = simulate_single_machine_cpu(mid, sim_params);
        end
        results{mid} = result;
    end
    
    % 输出并行处理完成信息
    elapsed_time = toc(start_time);
    fprintf('并行仿真完成，总用时: %.1f秒\n', elapsed_time);
else
    % 串行处理每台机器
    results = cell(1, config.n_machines);
    fprintf('开始串行仿真，共%d台机器...\n', config.n_machines);
    for mid = 1:config.n_machines
        if config.use_gpu
            % 将数据移到GPU上进行计算
            result = simulate_single_machine_gpu(mid, sim_params);
        else
            % CPU计算
            result = simulate_single_machine_cpu(mid, sim_params);
        end
        results{mid} = result;
        if mod(mid, 10) == 0 || mid == config.n_machines
            elapsed_time = toc(start_time);
            estimated_total_time = elapsed_time / mid * config.n_machines;
            remaining_time = estimated_total_time - elapsed_time;
            fprintf('串行仿真进度: %d/%d (%.1f%%), 已用时: %.1fs, 预计剩余: %.1fs\n', ...
                mid, config.n_machines, mid/config.n_machines*100, elapsed_time, remaining_time);
        end
    end
end

% 保存结果
for mid = 1:config.n_machines
    result = results{mid};
    machine_df = table(...
        repmat(mid, N_steps, 1), ...
        time_vector, ...
        repmat(config.shape_type, N_steps, 1), ...
        result.nozzle_position_x, ...
        result.nozzle_position_y, ...
        result.nozzle_position_z, ...
        result.temperature, ...
        result.vibration_disp_x, ...
        result.vibration_disp_y, ...
        result.vibration_vel_x, ...
        result.vibration_vel_y, ...
        result.motor_current_x, ...
        result.motor_current_y, ...
        result.extrusion_pressure, ...
        result.print_quality_metric, ...
        result.ideal_position_x, ...
        result.ideal_position_y, ...
        result.correction_signal_x, ...
        result.correction_signal_y, ...
        result.correction_signal_temp, ...
        repmat(result.is_faulty, N_steps, 1), ...
        'VariableNames', {'machine_id', 'time_s', 'shape', 'nozzle_x', 'nozzle_y', 'nozzle_z', ...
        'temperature_C', 'vibration_disp_x_m', 'vibration_disp_y_m', ...
        'vibration_vel_x_m_s', 'vibration_vel_y_m_s', ...
        'motor_current_x_A', 'motor_current_y_A', 'pressure_bar', 'quality_score', ...
        'ideal_x', 'ideal_y', 'correction_x_mm', 'correction_y_mm', 'correction_temp_C', 'fault_label'});

    filename = fullfile(config.output_dir, sprintf('machine_%03d.csv', mid));
    writetable(machine_df, filename);
end

if config.use_parallel
    delete(gcp); % 关闭并行池
end

%% ==================== 7. 可视化（仅绘制前几台） ==========================
fprintf('📊 生成可视化结果...\n');
load_example = 1:min(3, config.n_machines);
all_data = [];
for i = load_example
    tmp = readtable(fullfile(config.output_dir, sprintf('machine_%03d.csv', i)));
    all_data = [all_data; tmp];
end

% 提取数据
time_vector = all_data.time_s(all_data.machine_id == load_example(1));
n_points_per_machine = height(all_data) / length(load_example);

% 检查是否有足够的数据点进行可视化
if n_points_per_machine > 0
    nozzle_position_x = zeros(n_points_per_machine, length(load_example));
    nozzle_position_y = zeros(n_points_per_machine, length(load_example));
    ideal_position_x = zeros(n_points_per_machine, length(load_example));
    ideal_position_y = zeros(n_points_per_machine, length(load_example));
    
    % 按机器拆分数据
    for j = 1:length(load_example)
        mask = all_data.machine_id == load_example(j);
        nozzle_position_x(:,j) = all_data.nozzle_x(mask);
        nozzle_position_y(:,j) = all_data.nozzle_y(mask);
        ideal_position_x(:,j) = all_data.ideal_x(mask);
        ideal_position_y(:,j) = all_data.ideal_y(mask);
    end

    % 绘图
    figure('Position', [100, 100, 1200, 800], 'Name', '喷头轨迹与矫正信号对比');
    subplot(2,3,1);
    plot(nozzle_position_x, nozzle_position_y, 'r-', 'LineWidth', 0.8);
    hold on;
    plot(ideal_position_x, ideal_position_y, 'g--', 'LineWidth', 1.2);
    title('喷头XY平面轨迹对比 (红:原始, 绿:理想)');
    xlabel('X Position (mm)'); ylabel('Y Position (mm)');
    legend('原始轨迹', '理想轨迹', 'Location', 'best');
    grid on; axis equal;
    xlim([-12,12]); ylim([-12,12]);

    % 矫正信号（示例）
    correction_signal_x = all_data.correction_x_mm(all_data.machine_id == load_example(1));
    correction_signal_y = all_data.correction_y_mm(all_data.machine_id == load_example(1));
    subplot(2,3,2);
    n_plot_points = min(2000, length(correction_signal_x));
    plot(time_vector(1:n_plot_points), correction_signal_x(1:n_plot_points)*1000, 'b-');
    hold on;
    plot(time_vector(1:n_plot_points), correction_signal_y(1:n_plot_points)*1000, 'r--');
    title('矫正信号 (机器1)');
    xlabel('时间 (s)'); ylabel('矫正量 (mm)');
    legend('X方向', 'Y方向'); grid on;

    % 温度变化
    temp_data = all_data.temperature_C(all_data.machine_id == load_example(1));
    subplot(2,3,3);
    plot(time_vector(1:n_plot_points), temp_data(1:n_plot_points), 'm-');
    title('温度变化 (机器1)');
    xlabel('时间 (s)'); ylabel('温度 (°C)');
    grid on;

    % 振动变化
    vibration_x = all_data.vibration_disp_x_m(all_data.machine_id == load_example(1));
    vibration_y = all_data.vibration_disp_y_m(all_data.machine_id == load_example(1));
    subplot(2,3,4);
    plot(time_vector(1:n_plot_points), vibration_x(1:n_plot_points)*1000, 'b-');
    hold on;
    plot(time_vector(1:n_plot_points), vibration_y(1:n_plot_points)*1000, 'r--');
    title('振动位移 (机器1)');
    xlabel('时间 (s)'); ylabel('位移 (mm)');
    legend('X方向', 'Y方向'); grid on;

    % 质量指标
    quality_data = all_data.quality_score(all_data.machine_id == load_example(1));
    subplot(2,3,5);
    plot(time_vector(1:n_plot_points), quality_data(1:n_plot_points), 'c-');
    title('打印质量指标 (机器1)');
    xlabel('时间 (s)'); ylabel('质量评分');
    grid on;

    % Z轴变化
    z_data = all_data.nozzle_z(all_data.machine_id == load_example(1));
    subplot(2,3,6);
    plot(time_vector(1:n_plot_points), z_data(1:n_plot_points), 'k-');
    title('Z轴位置变化 (机器1)');
    xlabel('时间 (s)'); ylabel('Z位置 (mm)');
    grid on;

    vis_path = fullfile(config.output_dir, 'correction_simulation_results.png');
    exportgraphics(gcf, vis_path, 'Resolution', 300);
    fprintf('✅ 可视化结果已保存至: %s\n', vis_path);
else
    fprintf('⚠️ 无法生成可视化结果，数据不足\n');
end
fprintf('🎉 全部任务完成！\n');

%% ==================== 8. 辅助函数 ==========================
% CPU版本的单机仿真函数
function result = simulate_single_machine_cpu(mid, sim_params)
    % 初始化每台机器状态
    N_steps = sim_params.N_steps;
    path_indices = sim_params.path_indices;
    
    temperature = zeros(N_steps, 1);
    vibration_disp_x = zeros(N_steps, 1);
    vibration_disp_y = zeros(N_steps, 1);
    vibration_vel_x = zeros(N_steps, 1);
    vibration_vel_y = zeros(N_steps, 1);
    motor_current_x = zeros(N_steps, 1);
    motor_current_y = zeros(N_steps, 1);
    nozzle_position_x = zeros(N_steps, 1);
    nozzle_position_y = zeros(N_steps, 1);
    nozzle_position_z = zeros(N_steps, 1);
    extrusion_pressure = zeros(N_steps, 1);
    print_quality_metric = zeros(N_steps, 1);
    
    ideal_position_x = zeros(N_steps, 1);
    ideal_position_y = zeros(N_steps, 1);
    ideal_temperature = zeros(N_steps, 1);
    ideal_vibration_disp_x = zeros(N_steps, 1);
    ideal_vibration_disp_y = zeros(N_steps, 1);
    
    correction_signal_x = zeros(N_steps, 1);
    correction_signal_y = zeros(N_steps, 1);
    correction_signal_temp = zeros(N_steps, 1);
    
    % 初始化
    temperature(1) = sim_params.thermal_model.T_ambient(mid);
    ideal_temperature(1) = sim_params.thermal_model.T_ambient(mid);
    vibration_disp_x(1) = 0.0005*randn();
    vibration_disp_y(1) = 0.0005*randn();
    ideal_vibration_disp_x(1) = 0;
    ideal_vibration_disp_y(1) = 0;
    nozzle_position_x(1) = sim_params.x_path(1);
    nozzle_position_y(1) = sim_params.y_path(1);
    nozzle_position_z(1) = sim_params.z_path(1);
    ideal_position_x(1) = sim_params.x_ideal(1);
    ideal_position_y(1) = sim_params.y_ideal(1);
    
    % 检查当前机器是否为故障机器
    is_faulty = any(sim_params.faulty_machines == mid);
    fault_idx = 0;
    if is_faulty
        fault_idx = find(sim_params.faulty_machines == mid, 1);
    end
    
    % 仿真主循环
    for t = 2:N_steps
        % 检查是否到达故障开始时间
        if is_faulty && fault_idx > 0 && t > sim_params.fault_start_step(fault_idx)
            is_faulty = true;
        else
            is_faulty = false;  % 在故障发生前保持正常状态
        end
        
        % 获取当前路径点，如果路径已结束则保持在最后位置
        if t <= length(path_indices)
            path_idx = path_indices(t);
            target_x = sim_params.x_path(path_idx);
            target_y = sim_params.y_path(path_idx);
            target_z = sim_params.z_path(path_idx);
            ideal_target_x = sim_params.x_ideal(path_idx);
            ideal_target_y = sim_params.y_ideal(path_idx);
        else
            % 如果路径已结束，保持在最后位置
            target_x = nozzle_position_x(t-1);
            target_y = nozzle_position_y(t-1);
            target_z = nozzle_position_z(t-1);
            ideal_target_x = ideal_position_x(t-1);
            ideal_target_y = ideal_position_y(t-1);
        end
        
        % ========= 理想系统 =========
        prev_temp_ideal = ideal_temperature(t-1);
        temp_error_ideal = sim_params.thermal_model.T_target(mid) - prev_temp_ideal;
        heater_power_ideal = sim_params.thermal_model.heater_power_base(mid) * (1 + 0.3*tanh(temp_error_ideal));
        heat_loss_ideal = sim_params.thermal_model.convection_coeff(mid) * (prev_temp_ideal - sim_params.thermal_model.T_ambient(mid));
        dTdt_ideal = (heater_power_ideal - heat_loss_ideal) / (sim_params.thermal_model.mass(mid) * sim_params.thermal_model.specific_heat(mid));
        ideal_temperature(t) = prev_temp_ideal + dTdt_ideal * sim_params.dt;
        ideal_position_x(t) = ideal_target_x;
        ideal_position_y(t) = ideal_target_y;
        
        % ========= 实际系统 =========
        prev_temp = temperature(t-1);
        temp_error = sim_params.thermal_model.T_target(mid) - prev_temp;
        measured_temp = prev_temp;
        if is_faulty && sim_params.fault_types(fault_idx) == 3  % 传感器故障
            measured_temp = prev_temp * (0.85 + 0.1*rand());
        end
        heater_power = sim_params.thermal_model.heater_power_base(mid) * (1 + 0.5*tanh(temp_error));
        heat_loss = sim_params.thermal_model.convection_coeff(mid) * (measured_temp - sim_params.thermal_model.T_ambient(mid));
        dTdt = (heater_power - heat_loss) / (sim_params.thermal_model.mass(mid) * sim_params.thermal_model.specific_heat(mid));
        temperature(t) = prev_temp + dTdt * sim_params.dt;
        
        % 振动模型
        prev_disp_x = vibration_disp_x(t-1);
        prev_vel_x = vibration_vel_x(t-1);
        prev_disp_y = vibration_disp_y(t-1);
        prev_vel_y = vibration_vel_y(t-1);
        kx = sim_params.vibration_model.stiffness_x(mid);
        ky = sim_params.vibration_model.stiffness_y(mid);
        cx = sim_params.vibration_model.damping_x(mid);
        cy = sim_params.vibration_model.damping_y(mid);
        if is_faulty && sim_params.fault_types(fault_idx) == 2  % 机械松动故障
            kx = kx * 0.6; ky = ky * 0.6; cx = cx * 0.7; cy = cy * 0.7;
        end
        pos_error_x = target_x - nozzle_position_x(t-1) - prev_disp_x;
        pos_error_y = target_y - nozzle_position_y(t-1) - prev_disp_y;
        motor_factor = 1.0;
        if is_faulty && sim_params.fault_types(fault_idx) == 3  % 电机性能下降
            motor_factor = 0.7;
        end
        accel_x = motor_factor * (kx * pos_error_x - cx * prev_vel_x) / sim_params.vibration_model.mass(mid);
        accel_y = motor_factor * (ky * pos_error_y - cy * prev_vel_y) / sim_params.vibration_model.mass(mid);
        new_vel_x = prev_vel_x + accel_x * sim_params.dt;
        new_vel_y = prev_vel_y + accel_y * sim_params.dt;
        new_disp_x = prev_disp_x + new_vel_x * sim_params.dt;
        new_disp_y = prev_disp_y + new_vel_y * sim_params.dt;
        
        nozzle_position_x(t) = target_x + new_disp_x;
        nozzle_position_y(t) = target_y + new_disp_y;
        nozzle_position_z(t) = target_z;  % 直接使用目标Z值
        
        % 挤出压力
        pressure_multiplier = 1.0;
        if is_faulty && sim_params.fault_types(fault_idx) == 1  % 喷嘴堵塞
            pressure_multiplier = 1.8 + 0.4*rand();
        end
        movement_speed = sqrt((target_x - nozzle_position_x(t-1))^2 + (target_y - nozzle_position_y(t-1))^2) / sim_params.dt;
        speed_factor = min(1, movement_speed/80);
        temp_factor = (temperature(t) - 180) / 50;
        base_pressure = 4.5 * (1 + 0.15*randn());
        extrusion_pressure(t) = base_pressure * pressure_multiplier * ...
            (0.6 + 0.25*speed_factor + 0.15*temp_factor) * sim_params.print_quality.extrusion_multiplier(mid);
        
        % 矫正信号（理想 - 实际）
        correction_signal_x(t) = ideal_target_x - target_x;
        correction_signal_y(t) = ideal_target_y - target_y;
        correction_signal_temp(t) = sim_params.thermal_model.T_target(mid) - temperature(t);
        
        % 打印质量
        vibration_magnitude = sqrt(new_disp_x^2 + new_disp_y^2);
        temp_stability = abs(temperature(t) - sim_params.thermal_model.T_target(mid));
        base_quality = 1.0;
        vibration_penalty = min(0.8, 20*vibration_magnitude);
        temp_penalty = min(0.25, temp_stability/15);
        if is_faulty
            fault_penalty = 0.4 + 0.25*rand();
        else
            fault_penalty = 0;
        end
        quality_score = max(0.1, base_quality - vibration_penalty - temp_penalty - fault_penalty);
        print_quality_metric(t) = quality_score * (0.97 + 0.06*randn());
        
        % 检查是否已完成打印路径（作为仿真结束标志）
        if t == length(path_indices) && t < N_steps
            % 扩展剩余的仿真数据为最后的值
            for remaining_t = t+1:N_steps
                temperature(remaining_t) = temperature(t);
                vibration_disp_x(remaining_t) = vibration_disp_x(t);
                vibration_disp_y(remaining_t) = vibration_disp_y(t);
                vibration_vel_x(remaining_t) = vibration_vel_x(t);
                vibration_vel_y(remaining_t) = vibration_vel_y(t);
                nozzle_position_x(remaining_t) = nozzle_position_x(t);
                nozzle_position_y(remaining_t) = nozzle_position_y(t);
                nozzle_position_z(remaining_t) = nozzle_position_z(t);
                extrusion_pressure(remaining_t) = extrusion_pressure(t);
                print_quality_metric(remaining_t) = print_quality_metric(t);
                ideal_position_x(remaining_t) = ideal_position_x(t);
                ideal_position_y(remaining_t) = ideal_position_y(t);
                correction_signal_x(remaining_t) = correction_signal_x(t);
                correction_signal_y(remaining_t) = correction_signal_y(t);
                correction_signal_temp(remaining_t) = correction_signal_temp(t);
            end
            break;
        end
    end
    
    % 构建结果结构体
    result = struct();
    result.nozzle_position_x = nozzle_position_x;
    result.nozzle_position_y = nozzle_position_y;
    result.nozzle_position_z = nozzle_position_z;
    result.temperature = temperature;
    result.vibration_disp_x = vibration_disp_x;
    result.vibration_disp_y = vibration_disp_y;
    result.vibration_vel_x = vibration_vel_x;
    result.vibration_vel_y = vibration_vel_y;
    result.motor_current_x = motor_current_x;
    result.motor_current_y = motor_current_y;
    result.extrusion_pressure = extrusion_pressure;
    result.print_quality_metric = print_quality_metric;
    result.ideal_position_x = ideal_position_x;
    result.ideal_position_y = ideal_position_y;
    result.correction_signal_x = correction_signal_x;
    result.correction_signal_y = correction_signal_y;
    result.correction_signal_temp = correction_signal_temp;
    result.is_faulty = is_faulty;
end

% GPU版本的单机仿真函数
function result = simulate_single_machine_gpu(mid, sim_params)
    % 初始化每台机器状态
    N_steps = sim_params.N_steps;
    path_indices = sim_params.path_indices;
    
    % 将数组移到GPU上
    if sim_params.use_gpu
        temperature = gpuArray.zeros(N_steps, 1);
        vibration_disp_x = gpuArray.zeros(N_steps, 1);
        vibration_disp_y = gpuArray.zeros(N_steps, 1);
        vibration_vel_x = gpuArray.zeros(N_steps, 1);
        vibration_vel_y = gpuArray.zeros(N_steps, 1);
        motor_current_x = gpuArray.zeros(N_steps, 1);
        motor_current_y = gpuArray.zeros(N_steps, 1);
        nozzle_position_x = gpuArray.zeros(N_steps, 1);
        nozzle_position_y = gpuArray.zeros(N_steps, 1);
        nozzle_position_z = gpuArray.zeros(N_steps, 1);
        extrusion_pressure = gpuArray.zeros(N_steps, 1);
        print_quality_metric = gpuArray.zeros(N_steps, 1);
        
        ideal_position_x = gpuArray.zeros(N_steps, 1);
        ideal_position_y = gpuArray.zeros(N_steps, 1);
        ideal_temperature = gpuArray.zeros(N_steps, 1);
        ideal_vibration_disp_x = gpuArray.zeros(N_steps, 1);
        ideal_vibration_disp_y = gpuArray.zeros(N_steps, 1);
        
        correction_signal_x = gpuArray.zeros(N_steps, 1);
        correction_signal_y = gpuArray.zeros(N_steps, 1);
        correction_signal_temp = gpuArray.zeros(N_steps, 1);
    else
        temperature = zeros(N_steps, 1);
        vibration_disp_x = zeros(N_steps, 1);
        vibration_disp_y = zeros(N_steps, 1);
        vibration_vel_x = zeros(N_steps, 1);
        vibration_vel_y = zeros(N_steps, 1);
        motor_current_x = zeros(N_steps, 1);
        motor_current_y = zeros(N_steps, 1);
        nozzle_position_x = zeros(N_steps, 1);
        nozzle_position_y = zeros(N_steps, 1);
        nozzle_position_z = zeros(N_steps, 1);
        extrusion_pressure = zeros(N_steps, 1);
        print_quality_metric = zeros(N_steps, 1);
        
        ideal_position_x = zeros(N_steps, 1);
        ideal_position_y = zeros(N_steps, 1);
        ideal_temperature = zeros(N_steps, 1);
        ideal_vibration_disp_x = zeros(N_steps, 1);
        ideal_vibration_disp_y = zeros(N_steps, 1);
        
        correction_signal_x = zeros(N_steps, 1);
        correction_signal_y = zeros(N_steps, 1);
        correction_signal_temp = zeros(N_steps, 1);
    end
    
    % 初始化
    temperature(1) = sim_params.thermal_model.T_ambient(mid);
    ideal_temperature(1) = sim_params.thermal_model.T_ambient(mid);
    vibration_disp_x(1) = 0.0005*randn();
    vibration_disp_y(1) = 0.0005*randn();
    ideal_vibration_disp_x(1) = 0;
    ideal_vibration_disp_y(1) = 0;
    nozzle_position_x(1) = sim_params.x_path(1);
    nozzle_position_y(1) = sim_params.y_path(1);
    nozzle_position_z(1) = sim_params.z_path(1);
    ideal_position_x(1) = sim_params.x_ideal(1);
    ideal_position_y(1) = sim_params.y_ideal(1);
    
    % 检查当前机器是否为故障机器
    is_faulty = any(sim_params.faulty_machines == mid);
    fault_idx = 0;
    if is_faulty
        fault_idx = find(sim_params.faulty_machines == mid, 1);
    end
    
    % 仿真主循环
    for t = 2:N_steps
        % 检查是否到达故障开始时间
        if is_faulty && fault_idx > 0 && t > sim_params.fault_start_step(fault_idx)
            is_faulty = true;
        else
            is_faulty = false;  % 在故障发生前保持正常状态
        end
        
        % 获取当前路径点，如果路径已结束则保持在最后位置
        if t <= length(path_indices)
            path_idx = path_indices(t);
            target_x = sim_params.x_path(path_idx);
            target_y = sim_params.y_path(path_idx);
            target_z = sim_params.z_path(path_idx);
            ideal_target_x = sim_params.x_ideal(path_idx);
            ideal_target_y = sim_params.y_ideal(path_idx);
        else
            % 如果路径已结束，保持在最后位置
            target_x = nozzle_position_x(t-1);
            target_y = nozzle_position_y(t-1);
            target_z = nozzle_position_z(t-1);
            ideal_target_x = ideal_position_x(t-1);
            ideal_target_y = ideal_position_y(t-1);
        end
        
        % ========= 理想系统 =========
        prev_temp_ideal = ideal_temperature(t-1);
        temp_error_ideal = sim_params.thermal_model.T_target(mid) - prev_temp_ideal;
        heater_power_ideal = sim_params.thermal_model.heater_power_base(mid) * (1 + 0.3*tanh(temp_error_ideal));
        heat_loss_ideal = sim_params.thermal_model.convection_coeff(mid) * (prev_temp_ideal - sim_params.thermal_model.T_ambient(mid));
        dTdt_ideal = (heater_power_ideal - heat_loss_ideal) / (sim_params.thermal_model.mass(mid) * sim_params.thermal_model.specific_heat(mid));
        ideal_temperature(t) = prev_temp_ideal + dTdt_ideal * sim_params.dt;
        ideal_position_x(t) = ideal_target_x;
        ideal_position_y(t) = ideal_target_y;
        
        % ========= 实际系统 =========
        prev_temp = temperature(t-1);
        temp_error = sim_params.thermal_model.T_target(mid) - prev_temp;
        measured_temp = prev_temp;
        if is_faulty && sim_params.fault_types(fault_idx) == 3
            measured_temp = prev_temp * (0.85 + 0.1*rand());
        end
        heater_power = sim_params.thermal_model.heater_power_base(mid) * (1 + 0.5*tanh(temp_error));
        heat_loss = sim_params.thermal_model.convection_coeff(mid) * (measured_temp - sim_params.thermal_model.T_ambient(mid));
        dTdt = (heater_power - heat_loss) / (sim_params.thermal_model.mass(mid) * sim_params.thermal_model.specific_heat(mid));
        temperature(t) = prev_temp + dTdt * sim_params.dt;
        
        % 振动模型
        prev_disp_x = vibration_disp_x(t-1);
        prev_vel_x = vibration_vel_x(t-1);
        prev_disp_y = vibration_disp_y(t-1);
        prev_vel_y = vibration_vel_y(t-1);
        kx = sim_params.vibration_model.stiffness_x(mid);
        ky = sim_params.vibration_model.stiffness_y(mid);
        cx = sim_params.vibration_model.damping_x(mid);
        cy = sim_params.vibration_model.damping_y(mid);
        if is_faulty && sim_params.fault_types(fault_idx) == 2
            kx = kx * 0.6; ky = ky * 0.6; cx = cx * 0.7; cy = cy * 0.7;
        end
        pos_error_x = target_x - nozzle_position_x(t-1) - prev_disp_x;
        pos_error_y = target_y - nozzle_position_y(t-1) - prev_disp_y;
        motor_factor = 1.0;
        if is_faulty && sim_params.fault_types(fault_idx) == 3
            motor_factor = 0.7;
        end
        accel_x = motor_factor * (kx * pos_error_x - cx * prev_vel_x) / sim_params.vibration_model.mass(mid);
        accel_y = motor_factor * (ky * pos_error_y - cy * prev_vel_y) / sim_params.vibration_model.mass(mid);
        new_vel_x = prev_vel_x + accel_x * sim_params.dt;
        new_vel_y = prev_vel_y + accel_y * sim_params.dt;
        new_disp_x = prev_disp_x + new_vel_x * sim_params.dt;
        new_disp_y = prev_disp_y + new_vel_y * sim_params.dt;
        
        nozzle_position_x(t) = target_x + new_disp_x;
        nozzle_position_y(t) = target_y + new_disp_y;
        nozzle_position_z(t) = target_z;  % 直接使用目标Z值，保持与理想路径一致
        
        % 挤出压力
        pressure_multiplier = 1.0;
        if is_faulty && sim_params.fault_types(fault_idx) == 1
            pressure_multiplier = 1.8 + 0.4*rand();
        end
        movement_speed = sqrt((target_x - nozzle_position_x(t-1))^2 + (target_y - nozzle_position_y(t-1))^2) / sim_params.dt;
        speed_factor = min(1, movement_speed/80);
        temp_factor = (temperature(t) - 180) / 50;
        base_pressure = 4.5 * (1 + 0.15*randn());
        extrusion_pressure(t) = base_pressure * pressure_multiplier * ...
            (0.6 + 0.25*speed_factor + 0.15*temp_factor) * sim_params.print_quality.extrusion_multiplier(mid);
        
        % 矫正信号（理想 - 实际）
        correction_signal_x(t) = ideal_target_x - target_x;
        correction_signal_y(t) = ideal_target_y - target_y;
        correction_signal_temp(t) = sim_params.thermal_model.T_target(mid) - temperature(t);
        
        % 打印质量
        vibration_magnitude = sqrt(new_disp_x^2 + new_disp_y^2);
        temp_stability = abs(temperature(t) - sim_params.thermal_model.T_target(mid));
        base_quality = 1.0;
        vibration_penalty = min(0.8, 20*vibration_magnitude);
        temp_penalty = min(0.25, temp_stability/15);
        if is_faulty
            fault_penalty = 0.4 + 0.25*rand();
        else
            fault_penalty = 0;
        end
        quality_score = max(0.1, base_quality - vibration_penalty - temp_penalty - fault_penalty);
        print_quality_metric(t) = quality_score * (0.97 + 0.06*randn());
        
        % 检查是否已完成打印路径（作为仿真结束标志）
        if t == length(path_indices) && t < N_steps
            % 扩展剩余的仿真数据为最后的值
            for remaining_t = t+1:N_steps
                temperature(remaining_t) = temperature(t);
                vibration_disp_x(remaining_t) = vibration_disp_x(t);
                vibration_disp_y(remaining_t) = vibration_disp_y(t);
                vibration_vel_x(remaining_t) = vibration_vel_x(t);
                vibration_vel_y(remaining_t) = vibration_vel_y(t);
                nozzle_position_x(remaining_t) = nozzle_position_x(t);
                nozzle_position_y(remaining_t) = nozzle_position_y(t);
                nozzle_position_z(remaining_t) = nozzle_position_z(t);
                extrusion_pressure(remaining_t) = extrusion_pressure(t);
                print_quality_metric(remaining_t) = print_quality_metric(t);
                ideal_position_x(remaining_t) = ideal_position_x(t);
                ideal_position_y(remaining_t) = ideal_position_y(t);
                correction_signal_x(remaining_t) = correction_signal_x(t);
                correction_signal_y(remaining_t) = correction_signal_y(t);
                correction_signal_temp(remaining_t) = correction_signal_temp(t);
            end
            break;
        end
    end
    
    % 将GPU数组转回CPU
    if sim_params.use_gpu
        temperature = gather(temperature);
        vibration_disp_x = gather(vibration_disp_x);
        vibration_disp_y = gather(vibration_disp_y);
        vibration_vel_x = gather(vibration_vel_x);
        vibration_vel_y = gather(vibration_vel_y);
        nozzle_position_x = gather(nozzle_position_x);
        nozzle_position_y = gather(nozzle_position_y);
        nozzle_position_z = gather(nozzle_position_z);
        extrusion_pressure = gather(extrusion_pressure);
        print_quality_metric = gather(print_quality_metric);
        ideal_position_x = gather(ideal_position_x);
        ideal_position_y = gather(ideal_position_y);
        correction_signal_x = gather(correction_signal_x);
        correction_signal_y = gather(correction_signal_y);
        correction_signal_temp = gather(correction_signal_temp);
    end
    
    % 构建结果结构体
    result = struct();
    result.nozzle_position_x = nozzle_position_x;
    result.nozzle_position_y = nozzle_position_y;
    result.nozzle_position_z = nozzle_position_z;
    result.temperature = temperature;
    result.vibration_disp_x = vibration_disp_x;
    result.vibration_disp_y = vibration_disp_y;
    result.vibration_vel_x = vibration_vel_x;
    result.vibration_vel_y = vibration_vel_y;
    result.motor_current_x = motor_current_x;
    result.motor_current_y = motor_current_y;
    result.extrusion_pressure = extrusion_pressure;
    result.print_quality_metric = print_quality_metric;
    result.ideal_position_x = ideal_position_x;
    result.ideal_position_y = ideal_position_y;
    result.correction_signal_x = correction_signal_x;
    result.correction_signal_y = correction_signal_y;
    result.correction_signal_temp = correction_signal_temp;
    result.is_faulty = is_faulty;
end