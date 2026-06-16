% ============================================================
% 复现 figure_7.png — 行人横穿马路场景训练结果对比
% 左图：训练奖励曲线    右图：评估指标柱状图
% ============================================================
clear; clc; close all;

% 设置中文字体（确保 PDF 中正确显示）
set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultTextFontName', 'SimHei');

%% ==========================================================
%  左图数据：从原图提取的奖励曲线坐标
%  PPO 保持稳定，DQN 与之基本重合
% ==========================================================

% --- Smooth-PPO (橙色) 像素坐标：平坦直线 ---
ppo_x_pix = [72, 78, 85, 92, 99, 106, 112, 119, 126, 133, 140, 147, 153, ...
    160, 167, 174, 181, 187, 194, 201, 208, 215, 222, 228, 235, 242, 249, ...
    256, 262, 269, 276, 283, 290, 297, 303, 310, 317, 324, 331, 337, 344, ...
    351, 358, 365, 372, 378, 385, 392, 399, 406, 412, 419, 426, 433, 440];

ppo_y_pix = [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, ...
    256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, ...
    256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, ...
    256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256];

% --- 像素坐标 -> 实际数值映射 ---
x_axis_y_pix = 55;
y_top_pix = 440;
y_range_pix = y_top_pix - x_axis_y_pix;

episodes_ppo = (ppo_x_pix - min(ppo_x_pix)) / (max(ppo_x_pix) - min(ppo_x_pix)) * 500;
rewards_ppo = (ppo_y_pix - x_axis_y_pix) / y_range_pix * 100;

% --- DQN 与 PPO 重合，加微小噪声 ---
rng(43);
rewards_dqn = rewards_ppo + randn(size(rewards_ppo)) * 2.5;
episodes_dqn = episodes_ppo;

%% ==========================================================
%  右图数据：评估指标柱状图（请替换为真实数据）
% ==========================================================
metrics = {'碰撞率', '安全完成率', '平均奖励(\div10)', '平均TTC', '危险等级'};

dqn_values = [6.5, 88.2, 6.8, 3.2, 1.8];
ppo_values = [3.8, 95.5, 6.0, 4.9, 1.2];

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

sgtitle('行人横穿马路场景训练结果对比', ...
    'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'SimHei');

