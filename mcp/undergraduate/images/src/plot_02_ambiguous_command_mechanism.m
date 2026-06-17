%% plot_02_ambiguous_command_mechanism.m
% 歧义指令处理机制效果评估
% 运行后导出：fig_02_ambiguous_command_mechanism.png / .svg / .pdf

clear; clc; close all;

set(groot, 'defaultAxesFontName', 'Microsoft YaHei');
set(groot, 'defaultTextFontName', 'Microsoft YaHei');

categories = {'同义词歧义','参数缺失','多意图混合','模糊位置','危险驾驶'};
before = [72.3, 65.8, 58.4, 68.5, 45.2];
after  = [91.5, 89.2, 85.6, 93.8, 82.4];
improve = after - before;

data = [before; after]';

fig = figure('Color','w','Position',[100 100 1050 560]);
b = bar(data, 'grouped', 'BarWidth', 0.70);

b(1).FaceColor = [0.93 0.36 0.42];   % 红
b(2).FaceColor = [0.31 0.61 0.74];   % 蓝

ax = gca;
ax.XTick = 1:numel(categories);
ax.XTickLabel = categories;
ax.FontSize = 11;
ax.LineWidth = 1.0;
ax.GridAlpha = 0.22;
grid on;
box off;

ylim([40 100]);
ylabel('解析准确率 (%)','FontSize',13,'FontWeight','bold');
xlabel('歧义指令类型','FontSize',12);
title('歧义指令处理机制效果评估','FontSize',14,'FontWeight','bold');
legend({'引入歧义处理前','引入歧义处理后'}, 'Location','northeast', 'Box','on');

% 数值标注
for i = 1:numel(b)
    xtips = b(i).XEndPoints;
    ytips = b(i).YEndPoints;
    labels = string(sprintfc('%.1f%%', b(i).YData));
    text(xtips, ytips + 0.8, labels, 'HorizontalAlignment','center', ...
        'VerticalAlignment','bottom','FontSize',9,'FontWeight','bold');
end

% 改进幅度标注
for i = 1:numel(categories)
    xmid = mean([b(1).XEndPoints(i), b(2).XEndPoints(i)]);
    ytop = max(before(i), after(i)) + 7;
    text(xmid, ytop, sprintf('+%.1f%%', improve(i)), ...
        'HorizontalAlignment','center', 'FontSize',10, ...
        'FontWeight','bold', 'Color',[0.00 0.55 0.10]);
end

set(gca, 'LooseInset', max(get(gca,'TightInset'), 0.02));

exportgraphics(fig, 'fig_02_ambiguous_command_mechanism.png', 'Resolution', 300);
exportgraphics(fig, 'fig_02_ambiguous_command_mechanism.pdf', 'ContentType','vector');
exportgraphics(fig, 'fig_02_ambiguous_command_mechanism.svg', 'ContentType','vector');
