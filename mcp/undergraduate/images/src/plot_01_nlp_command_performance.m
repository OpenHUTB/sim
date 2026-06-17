%% plot_01_nlp_command_performance.m
% 自然语言指令解析性能评估
% 运行后导出：fig_01_nlp_command_performance.png / .svg / .pdf

clear; clc; close all;

% 字体设置：Windows 优先使用 Microsoft YaHei，论文排版较稳
set(groot, 'defaultAxesFontName', 'Microsoft YaHei');
set(groot, 'defaultTextFontName', 'Microsoft YaHei');

categories = {'车辆控制','天气设置','场景编辑','传感器配置','交通控制','综合指令'};
accuracy = [96.2, 98.5, 94.3, 97.8, 95.6, 91.4];
recall   = [94.8, 97.2, 92.1, 96.5, 93.8, 89.2];
f1score  = [95.5, 97.8, 93.2, 97.1, 94.7, 90.3];

data = [accuracy; recall; f1score]';

fig = figure('Color','w','Position',[100 100 1100 560]);
b = bar(data, 'grouped', 'BarWidth', 0.76);

% 配色接近原图
b(1).FaceColor = [0.31 0.61 0.74];   % 蓝
b(2).FaceColor = [0.74 0.31 0.56];   % 紫红
b(3).FaceColor = [1.00 0.63 0.13];   % 橙

ax = gca;
ax.XTick = 1:numel(categories);
ax.XTickLabel = categories;
ax.FontSize = 11;
ax.LineWidth = 1.0;
ax.GridAlpha = 0.22;
grid on;
box off;

ylim([86 100]);
ylabel('百分比 (%)','FontSize',13,'FontWeight','bold');
xlabel('指令类别','FontSize',12);
title('自然语言指令解析性能评估','FontSize',14,'FontWeight','bold');
legend({'准确率(Accuracy)','召回率(Recall)','F1分数'}, 'Location','northeast', 'Box','on');

% 数值标注
for i = 1:numel(b)
    xtips = b(i).XEndPoints;
    ytips = b(i).YEndPoints;
    labels = string(sprintfc('%.1f', b(i).YData));
    text(xtips, ytips + 0.18, labels, 'HorizontalAlignment','center', ...
        'VerticalAlignment','bottom','FontSize',9,'FontWeight','bold');
end

set(gca, 'LooseInset', max(get(gca,'TightInset'), 0.02));

exportgraphics(fig, 'fig_01_nlp_command_performance.png', 'Resolution', 300);
exportgraphics(fig, 'fig_01_nlp_command_performance.pdf', 'ContentType','vector');
exportgraphics(fig, 'fig_01_nlp_command_performance.svg', 'ContentType','vector');
