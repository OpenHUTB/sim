%% plot_04_concurrency_latency_success.m
% 系统并发性能与响应延迟测试
% 运行后导出：fig_04_concurrency_latency_success.png / .svg / .pdf

clear; clc; close all;

set(groot, 'defaultAxesFontName', 'Microsoft YaHei');
set(groot, 'defaultTextFontName', 'Microsoft YaHei');

x = [1, 5, 10, 20, 50, 100];
latency = [45, 52, 68, 95, 180, 320];         % ms
success = [100.0, 100.0, 98.5, 96.2, 92.8, 85.3];  % %

fig = figure('Color','w','Position',[100 100 1150 560]);

yyaxis left;
p1 = plot(x, latency, '-o', 'LineWidth', 2.0, 'MarkerSize', 7, ...
    'Color',[0.17 0.55 0.75], 'MarkerFaceColor',[0.17 0.55 0.75]);
ylabel('平均响应延迟 (ms)','FontSize',13,'FontWeight','bold');
ylim([0 400]);
ax = gca;
ax.YColor = [0.17 0.55 0.75];

for i = 1:numel(x)
    text(x(i), latency(i)+10, sprintf('%dms', latency(i)), ...
        'HorizontalAlignment','center', 'FontSize',9, ...
        'Color',[0.17 0.55 0.75], 'FontWeight','bold');
end

yyaxis right;
p2 = plot(x, success, '-s', 'LineWidth', 2.0, 'MarkerSize', 7, ...
    'Color',[0.95 0.29 0.34], 'MarkerFaceColor',[0.95 0.29 0.34]);
ylabel('指令执行成功率 (%)','FontSize',13,'FontWeight','bold');
ylim([80 105]);
ax.YColor = [0.95 0.29 0.34];

for i = 1:numel(x)
    text(x(i), success(i)-1.3, sprintf('%.1f%%', success(i)), ...
        'HorizontalAlignment','center', 'FontSize',9, ...
        'Color',[0.95 0.29 0.34], 'FontWeight','bold');
end

xlabel('并发指令数量','FontSize',12);
title('系统并发性能与响应延迟测试','FontSize',14,'FontWeight','bold');
grid on; box off;
ax = gca;
ax.XLim = [0 105];
ax.XTick = [0 20 40 60 80 100];
ax.GridAlpha = 0.22;
legend([p1 p2], {'响应延迟','成功率'}, 'Location','east','Box','on');

exportgraphics(fig, 'fig_04_concurrency_latency_success.png', 'Resolution', 300);
exportgraphics(fig, 'fig_04_concurrency_latency_success.pdf', 'ContentType','vector');
exportgraphics(fig, 'fig_04_concurrency_latency_success.svg', 'ContentType','vector');
