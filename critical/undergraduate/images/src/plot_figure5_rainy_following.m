% ============================================================
% 复现 figure_5.png — 暴雨天跟车场景训练结果对比
% 左图：训练奖励曲线    右图：评估指标柱状图
% 曲线数据从原图像素提取，尽量还原原图样式
% ============================================================
clear; clc; close all;

set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultTextFontName', 'SimHei');

%% ==========================================================
%  左图数据：从原图提取的奖励曲线坐标
% ==========================================================

% 原图左子图区域: 447x494 像素
% x轴位于 y_pix ~ 55 (距底部)
% y轴范围: y_pix 55~440 -> 对应奖励值范围 0~100

% --- Attention-DQN (蓝色) 像素坐标 ---
dqn_x_pix = [62, 68, 74, 81, 87, 93, 100, 106, 113, 119, 125, 132, 138, ...
    144, 151, 157, 164, 170, 176, 183, 189, 196, 202, 208, 215, 221, 227, ...
    234, 240, 247, 253, 259, 266, 272, 279, 285, 291, 298, 306, 312, 319, ...
    325, 332, 338, 344, 351, 357, 364, 370, 376, 383, 389, 395, 402, 408, ...
    415, 421, 427, 434, 440];

dqn_y_pix = [440, 440, 440, 440, 200, 204, 209, 213, 218, 222, 226, 231, ...
    235, 240, 250, 258, 267, 275, 283, 293, 301, 310, 317, 324, 328, 331, ...
    334, 338, 342, 346, 349, 352, 356, 360, 363, 367, 370, 374, 378, 381, ...
    385, 388, 392, 394, 395, 397, 399, 401, 403, 404, 406, 408, 410, 395, ...
    380, 364, 348, 334, 315, 300];

% --- Smooth-PPO (橙色) 像素坐标 ---
ppo_x_pix = [62, 68, 74, 81, 87, 94, 100, 106, 113, 119, 126, 132, 139, ...
    145, 151, 158, 164, 171, 177, 183, 190, 196, 203, 209, 216, 222, 228, ...
    235, 241, 248, 254, 260, 267, 273, 280, 286, 293, 299, 305, 312, 318, ...
    325, 331, 337, 344, 350, 357, 363, 370, 376, 382, 389, 395, 402, 408, ...
    414, 421, 427, 434, 440];

ppo_y_pix = [422, 422, 422, 284, 422, 296, 302, 308, 315, 321, 328, 334, ...
    340, 345, 350, 355, 359, 364, 368, 373, 378, 382, 387, 391, 395, 398, ...
    402, 406, 409, 413, 417, 420, 424, 419, 408, 398, 387, 378, 368, 357, ...
    347, 336, 327, 328, 332, 336, 341, 345, 349, 353, 357, 362, 366, 370, ...
    373, 376, 380, 383, 387, 390];

% --- 像素坐标 -> 实际数值映射 ---
x_axis_y_pix = 55;
y_top_pix = 440;
y_range_pix = y_top_pix - x_axis_y_pix;

episodes_dqn = (dqn_x_pix - 62) / (440 - 62) * 500;
rewards_dqn = (dqn_y_pix - x_axis_y_pix) / y_range_pix * 100;

episodes_ppo = (ppo_x_pix - 62) / (440 - 62) * 500;
rewards_ppo = (ppo_y_pix - x_axis_y_pix) / y_range_pix * 100;

%% ==========================================================
%  右图数据：评估指标柱状图（请替换为真实数据）
% ==========================================================
metrics = {'碰撞率', '安全完成率',  '平均奖励(\div10)',  '平均TTC', '危险等级'};

dqn_values = [100, 0, 5.6, 0.8, 3.0];
ppo_values = [0, 100, 10.8, 3.2, 1.0];

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

sgtitle('暴雨天跟车场景训练结果对比', ...
    'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'SimHei');
