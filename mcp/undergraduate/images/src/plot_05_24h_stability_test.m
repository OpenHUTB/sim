%% plot_05_24h_stability_test.m
% 24小时连续运行稳定性测试
% 运行后导出：fig_05_24h_stability_test.png / .svg / .pdf

clear; clc; close all;

set(groot, 'defaultAxesFontName', 'Microsoft YaHei');
set(groot, 'defaultTextFontName', 'Microsoft YaHei');

time = 0:2:24;
mem = [450, 455, 462, 470, 478, 482, 486, 490, 488, 492, 496, 499, 501];     % MB
cpu = [33.0, 32.2, 31.6, 30.8, 30.0, 29.2, 28.4, 27.6, 27.2, 26.7, 26.2, 25.7, 25.3]; % %
latency = [45, 46, 48, 50, 52, 55, 58, 62, 65, 68, 72, 75, 78];             % ms
threshold = 100 * ones(size(time));

fig = figure('Color','w','Position',[100 100 1150 650]);
tiledlayout(2, 1, 'TileSpacing','compact', 'Padding','compact');

% ========= 上图：内存与CPU =========
nexttile;
yyaxis left;
p1 = plot(time, mem, '-o', 'LineWidth', 2.0, 'MarkerSize', 5.5, ...
    'Color',[0.10 0.20 0.95], 'MarkerFaceColor',[0.10 0.20 0.95]);
ylabel('内存占用 (MB)','FontSize',12,'FontWeight','bold');
ylim([400 540]);
gca().YColor = [0.10 0.20 0.95];

yyaxis right;
p2 = plot(time, cpu, '-s', 'LineWidth', 2.0, 'MarkerSize', 5.5, ...
    'Color',[0.10 0.55 0.10], 'MarkerFaceColor',[0.10 0.55 0.10]);
ylabel('CPU使用率 (%)','FontSize',12,'FontWeight','bold');
ylim([20 34]);
gca().YColor = [0.10 0.55 0.10];

title('24小时连续运行稳定性测试','FontSize',14,'FontWeight','bold');
legend([p1 p2], {'内存占用(MB)','CPU使用率(%)'}, 'Location','northwest','Box','on');
grid on; box off;
ax = gca;
ax.XLim = [0 24];
ax.XTick = 0:4:24;
ax.GridAlpha = 0.22;

% ========= 下图：响应延迟 =========
nexttile;
hold on;
area(time, latency, 40, 'FaceColor',[1.0 0.45 0.45], 'FaceAlpha',0.28, 'EdgeColor','none');
p3 = plot(time, latency, '-o', 'LineWidth', 2.0, 'MarkerSize', 5.5, ...
    'Color',[1.0 0.14 0.12], 'MarkerFaceColor',[1.0 0.14 0.12]);
p4 = plot(time, threshold, '--', 'LineWidth', 1.8, 'Color',[1.0 0.50 0.10]);

for i = 1:numel(time)
    text(time(i), latency(i)+2.0, sprintf('%dms', latency(i)), ...
        'HorizontalAlignment','center', 'FontSize',8.5, ...
        'Color',[1.0 0.14 0.12], 'FontWeight','bold');
end

ylabel('响应延迟 (ms)','FontSize',12,'FontWeight','bold');
xlabel('运行时间（小时）','FontSize',12);
ylim([40 105]);
xlim([0 24]);
xticks(0:4:24);
legend([p3 p4], {'平均响应延迟','可接受延迟阈值(100ms)'}, 'Location','northwest','Box','on');
grid on; box off;
ax = gca;
ax.GridAlpha = 0.22;

exportgraphics(fig, 'fig_05_24h_stability_test.png', 'Resolution', 300);
exportgraphics(fig, 'fig_05_24h_stability_test.pdf', 'ContentType','vector');
exportgraphics(fig, 'fig_05_24h_stability_test.svg', 'ContentType','vector');
