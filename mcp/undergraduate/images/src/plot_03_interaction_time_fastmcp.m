%% plot_03_interaction_time_fastmcp.m
% 不同交互方式耗时对比 + FastMCP效率提升倍数
% 运行后导出：fig_03_interaction_time_fastmcp.png / .svg / .pdf

clear; clc; close all;

set(groot, 'defaultAxesFontName', 'Microsoft YaHei');
set(groot, 'defaultTextFontName', 'Microsoft YaHei');

categories = {'终端交互','天气切换','传感配置','多车协同调度','复杂交通场景'};

% 左图：平均操作耗时（秒）
fastmcp = [2.1, 1.8, 3.2, 4.5, 6.8];
gui     = [15.3, 12.6, 28.4, 45.2, 62.5];
api     = [8.5, 6.2, 14.3, 22.1, 35.6];

% 右图：效率提升倍数
speed_gui = gui ./ fastmcp;
speed_api = api ./ fastmcp;

fig = figure('Color','w','Position',[80 100 1300 560]);
tiledlayout(1, 2, 'TileSpacing','compact', 'Padding','compact');

% ========= 左子图 =========
nexttile;
data1 = [fastmcp; gui; api]';
b1 = bar(data1, 'grouped', 'BarWidth', 0.72);
b1(1).FaceColor = [0.21 0.62 0.78];
b1(2).FaceColor = [0.93 0.36 0.42];
b1(3).FaceColor = [0.45 0.67 0.36];

grid on; box off;
ax = gca;
ax.XTick = 1:numel(categories);
ax.XTickLabel = categories;
ax.FontSize = 10;
ax.GridAlpha = 0.22;
ylabel('平均操作时间（秒）','FontSize',12,'FontWeight','bold');
title('不同交互方式耗时对比','FontSize',13,'FontWeight','bold');
legend({'FastMCP自然语言','传统GUI操作','命令行调用'}, 'Location','northwest','Box','on');
ylim([0 70]);

for i = 1:numel(b1)
    xtips = b1(i).XEndPoints;
    ytips = b1(i).YEndPoints;
    labels = string(sprintfc('%.1fs', b1(i).YData));
    text(xtips, ytips + 1.0, labels, 'HorizontalAlignment','center', ...
        'FontSize',8.5,'FontWeight','bold');
end

% ========= 右子图 =========
nexttile;
data2 = [speed_gui; speed_api]';
b2 = bar(data2, 'grouped', 'BarWidth', 0.70);
b2(1).FaceColor = [0.93 0.36 0.42];
b2(2).FaceColor = [0.45 0.67 0.36];

grid on; box off;
ax = gca;
ax.XTick = 1:numel(categories);
ax.XTickLabel = categories;
ax.FontSize = 10;
ax.GridAlpha = 0.22;
ylabel('效率提升倍数','FontSize',12,'FontWeight','bold');
title('FastMCP效率提升倍数','FontSize',13,'FontWeight','bold');
legend({'相对GUI提升倍数','相对API提升倍数'}, 'Location','northwest','Box','on');
ylim([0 11]);

for i = 1:numel(b2)
    xtips = b2(i).XEndPoints;
    ytips = b2(i).YEndPoints;
    labels = string(sprintfc('%.1fx', b2(i).YData));
    text(xtips, ytips + 0.20, labels, 'HorizontalAlignment','center', ...
        'FontSize',8.5,'FontWeight','bold');
end

exportgraphics(fig, 'fig_03_interaction_time_fastmcp.png', 'Resolution', 300);
exportgraphics(fig, 'fig_03_interaction_time_fastmcp.pdf', 'ContentType','vector');
exportgraphics(fig, 'fig_03_interaction_time_fastmcp.svg', 'ContentType','vector');
