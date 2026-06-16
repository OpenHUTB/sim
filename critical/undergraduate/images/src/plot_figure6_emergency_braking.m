% ============================================================
% 复现 figure_6.png — 前车紧急制动场景训练结果对比
% 左图：训练奖励曲线    右图：评估指标柱状图
% ============================================================
clear; clc; close all;

set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultTextFontName', 'SimHei');

%% ==========================================================
%  左图数据：从原图提取的奖励曲线坐标
% ==========================================================

% --- Attention-DQN (蓝色) 像素坐标 ---
dqn_x_pix = [55, 62, 69, 76, 83, 90, 97, 104, 112, 119, 126, 133, 140, ...
    147, 154, 161, 169, 176, 183, 190, 197, 204, 211, 218, 226, 233, 240, ...
    247, 254, 261, 268, 275, 283, 290, 297, 304, 311, 318, 325, 332, 340, ...
    347, 354, 361, 368, 375, 382, 389, 397, 404, 411, 418, 425, 432, 439];

dqn_y_pix = [440, 440, 440, 440, 78, 78, 78, 78, 78, 78, 78, 77, 85, 97, ...
    107, 119, 133, 144, 156, 166, 177, 182, 182, 181, 181, 181, 181, 181, ...
    181, 180, 187, 197, 209, 219, 229, 239, 250, 260, 269, 277, 285, 292, ...
    298, 305, 312, 319, 325, 332, 337, 340, 343, 346, 349, 352, 355];

% --- Smooth-PPO (橙色) 像素坐标 ---
ppo_x_pix = [56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 127, 134, 141, ...
    148, 155, 162, 169, 176, 183, 191, 198, 205, 212, 219, 226, 233, 240, ...
    247, 255, 262, 269, 276, 283, 290, 297, 304, 311, 319, 326, 333, 340, ...
    347, 354, 361, 368, 375, 383, 390, 397, 404, 411, 418, 425, 432, 439];

ppo_y_pix = [422, 422, 422, 422, 73, 73, 73, 73, 73, 72, 72, 72, 73, 73, ...
    73, 73, 73, 73, 73, 73, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, ...
    74, 74, 73, 73, 73, 73, 73, 73, 73, 73, 74, 74, 74, 74, 74, 74, 74, ...
    74, 75, 75, 75, 75, 75, 75];

% --- 像素坐标 -> 实际数值映射 ---
x_axis_y_pix = 55;
y_top_pix = 440;
y_range_pix = y_top_pix - x_axis_y_pix;

episodes_dqn = (dqn_x_pix - min(dqn_x_pix)) / (max(dqn_x_pix) - min(dqn_x_pix)) * 500;
rewards_dqn = (dqn_y_pix - x_axis_y_pix) / y_range_pix * 100;

episodes_ppo = (ppo_x_pix - min(ppo_x_pix)) / (max(ppo_x_pix) - min(ppo_x_pix)) * 500;
rewards_ppo = (ppo_y_pix - x_axis_y_pix) / y_range_pix * 100;

%% ==========================================================
%  右图数据：评估指标柱状图（请替换为真实数据）
% ==========================================================
metrics = {'碰撞率', '安全完成率', '平均奖励(\div10)', '平均TTC', '危险等级'};

dqn_values = [100.0, 0, 0.3, 1.1, 3.0];
ppo_values = [0, 100.0, 11.5, 2.3, 2.0];

%% ==========================================================
%  绘图
% ==========================================================
figure('Position', [30, 50, 1380, 500]);

% --- 左图：训练奖励曲线 ---
subplot(1, 2, 1);
hold on;

plot(episodes_dqn, rewards_dqn, '-', 'Color', [0.18 0.55 0.85], 'LineWidth', 2.2);
plot(episodes_ppo, rewards_ppo, '-', 'Color', [0.93 0.45 0.13], 'LineWidth', 2.2);

xlabel('训练轮次 (Episode)', 'FontSize', 10, 'FontName', 'SimHei');
ylabel('平均奖励', 'FontSize', 10, 'FontName', 'SimHei');
title('训练奖励曲线', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'SimHei');
set(gca, 'FontName', 'SimHei');
legend({'Attention-DQN', 'Smooth-PPO'}, 'Location', 'southeast', 'FontSize', 9);
xlim([0, 500]);
ylim([0, 100]);
grid on; box on;
set(gca, 'GridAlpha', 0.3);

% --- 右图：评估结果柱状图 ---
subplot(1, 2, 2);
hold on;

x = 1:length(metrics);
bar_width = 0.33;

b1 = bar(x - bar_width/2, dqn_values, bar_width, ...
    'FaceColor', [0.18 0.55 0.85], 'EdgeColor', 'none');
b2 = bar(x + bar_width/2, ppo_values, bar_width, ...
    'FaceColor', [0.93 0.45 0.13], 'EdgeColor', 'none');

ylabel('数值', 'FontSize', 10, 'FontName', 'SimHei');
title('评估指标对比', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'SimHei');
set(gca, 'XTickLabel', metrics, 'FontSize', 9, 'FontName', 'SimHei');
	xtickangle(40);  % 旋转标签避免截断
legend([b1, b2], {'Attention-DQN', 'Smooth-PPO'}, ...
    'Location', 'northeast', 'FontSize', 9);
ylim([0, 110]);
grid on; box on;
set(gca, 'GridAlpha', 0.3);

for i = 1:length(x)
    text(x(i) - bar_width/2, dqn_values(i) + 1.8, num2str(dqn_values(i), '%.1f'), ...
        'HorizontalAlignment', 'center', 'FontSize', 7.5, ...
        'Color', [0.18 0.55 0.85], 'FontName', 'SimHei');
    text(x(i) + bar_width/2, ppo_values(i) + 1.8, num2str(ppo_values(i), '%.1f'), ...
        'HorizontalAlignment', 'center', 'FontSize', 7.5, ...
        'Color', [0.93 0.45 0.13], 'FontName', 'SimHei');
end

sgtitle('前车紧急制动场景训练结果对比', ...
    'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'SimHei');
