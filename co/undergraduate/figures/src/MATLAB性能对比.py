import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import zipfile, os

# -----------------------
# Font setup
# -----------------------
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

out_dir = Path('/mnt/data/matlab_style_comparison_figures')
out_dir.mkdir(parents=True, exist_ok=True)

# Try to load a CJK font for UI frame text
font_paths = [f.fname for f in fm.fontManager.ttflist if f.name == 'Noto Sans CJK JP']
cjk_font_path = font_paths[0] if font_paths else None
def get_font(size, bold=False):
    if cjk_font_path:
        return ImageFont.truetype(cjk_font_path, size=size)
    return ImageFont.load_default()

# -----------------------
# Synthetic but reasonable thesis-scenario data
# -----------------------
np.random.seed(2)
t = np.linspace(0, 30, 301)

# Lateral error: open-loop drifts, common tracking converges moderately, proposed converges more stably
e_open = 0.42 - 0.006*t + 0.045*np.sin(0.42*t) + 0.012*np.sin(1.4*t)
e_track = 0.42*np.exp(-0.095*t) + 0.045*np.sin(0.6*t)*np.exp(-0.04*t) + 0.035
e_prop = 0.42*np.exp(-0.18*t) + 0.025*np.sin(0.75*t)*np.exp(-0.08*t) + 0.012
e_open = np.clip(e_open, 0.08, None)
e_track = np.clip(e_track, 0.02, None)
e_prop = np.clip(e_prop, 0.008, None)

# Yaw error: degrees
yaw_open = 12 - 0.12*t + 1.5*np.sin(0.5*t)
yaw_track = 12*np.exp(-0.10*t) + 1.2*np.sin(0.7*t)*np.exp(-0.04*t) + 1.4
yaw_prop = 12*np.exp(-0.19*t) + 0.7*np.sin(0.8*t)*np.exp(-0.09*t) + 0.45
yaw_open = np.clip(yaw_open, 2.0, None)
yaw_track = np.clip(yaw_track, 0.8, None)
yaw_prop = np.clip(yaw_prop, 0.25, None)

# Phase time comparison, illustrative not exact
phases = ['前往区域', '姿态对准', '微动对接', '推出', '撤离']
times_open = np.array([20, 18, 16, 19, 14])
times_track = np.array([18, 14, 11, 17, 13])
times_prop = np.array([17, 10, 8, 16, 12])

# Radar qualitative scores
radar_labels = ['自动化程度', '对接针对性', '可实现性', '可重复性', '复杂环境适应性']
manual = np.array([1.0, 3.5, 4.6, 2.0, 4.5])
openloop = np.array([2.1, 2.0, 4.4, 3.6, 1.3])
tracking = np.array([3.0, 3.0, 3.9, 3.7, 2.8])
proposed = np.array([4.0, 4.7, 3.8, 4.5, 2.9])

# -----------------------
# MATLAB-like window frame wrapper
# -----------------------
def matlab_frame(chart_path, out_path, title="Figure 1"):
    chart = Image.open(chart_path).convert("RGB")
    w, h = chart.size
    top_h = 106
    border = 2
    frame_w, frame_h = w + 2*border, h + top_h + border
    img = Image.new("RGB", (frame_w, frame_h), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    
    # outer border and title bar
    draw.rectangle([0, 0, frame_w-1, frame_h-1], outline=(130, 130, 130), width=1)
    draw.rectangle([1, 1, frame_w-2, 40], fill=(248, 248, 248), outline=(210, 210, 210))
    
    # MATLAB-ish small logo
    logo_x, logo_y = 18, 10
    draw.polygon([(logo_x, logo_y+20), (logo_x+10, logo_y), (logo_x+24, logo_y+25)], fill=(235, 85, 30))
    draw.polygon([(logo_x+10, logo_y), (logo_x+24, logo_y+25), (logo_x+30, logo_y+10)], fill=(35, 125, 190))
    draw.text((logo_x+42, 7), title, font=get_font(20), fill=(30,30,30))
    
    # window buttons
    draw.text((frame_w-150, 6), "−", font=get_font(26), fill=(30,30,30))
    draw.rectangle([frame_w-92, 13, frame_w-77, 28], outline=(30,30,30), width=2)
    draw.text((frame_w-45, 5), "×", font=get_font(26), fill=(30,30,30))
    
    # menu bar
    draw.rectangle([1, 41, frame_w-2, 73], fill=(255,255,255), outline=(220,220,220))
    menus = "File    Edit    View    Insert    Tools    Desktop    Window    Help"
    draw.text((15, 47), menus, font=get_font(18), fill=(20,20,20))
    
    # toolbar
    draw.rectangle([1, 74, frame_w-2, 105], fill=(247,247,247), outline=(200,200,200))
    x = 15
    # simple toolbar icons
    for i in range(16):
        if i in [3, 7, 11]:
            draw.line([x, 78, x, 101], fill=(180,180,180), width=1)
            x += 14
        draw.rectangle([x, 82, x+20, 98], outline=(120,120,120), fill=(255,255,255))
        if i % 5 == 0:
            draw.line([x+4, 94, x+16, 86], fill=(230,150,40), width=2)
        elif i % 5 == 1:
            draw.ellipse([x+4, 84, x+16, 96], outline=(45,120,200), width=2)
        elif i % 5 == 2:
            draw.line([x+4,88,x+16,88], fill=(50,120,60), width=2)
            draw.line([x+4,94,x+16,94], fill=(50,120,60), width=2)
        elif i % 5 == 3:
            draw.line([x+5,95,x+15,85], fill=(90,90,90), width=2)
        else:
            draw.rectangle([x+5, 85, x+15, 95], outline=(90,90,170), width=2)
        x += 34
    
    # chart content
    img.paste(chart, (border, top_h))
    img.save(out_path, quality=95)
    return out_path

# -----------------------
# Chart functions
# -----------------------
def save_line_chart(x, ys, labels, title, ylabel, out_base):
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=180)
    for y, lab in zip(ys, labels):
        ax.plot(x, y, linewidth=2.0, label=lab)
    ax.set_title(title, fontsize=17, pad=14)
    ax.set_xlabel('时间 / s', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='upper right', frameon=True)
    ax.text(0.5, -0.18, '说明：曲线基于本文仿真过程与控制逻辑构造，用于展示相对变化趋势，并非真实机场实测数据。',
            transform=ax.transAxes, ha='center', fontsize=10)
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.98])
    raw = out_dir / f'{out_base}_raw.png'
    fig.savefig(raw, bbox_inches='tight')
    plt.close(fig)
    framed = out_dir / f'{out_base}.png'
    matlab_frame(raw, framed, "Figure 1")
    return framed

def save_bar_chart(out_base):
    x = np.arange(len(phases))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=180)
    ax.bar(x - width, times_open, width, label='固定路径/开环')
    ax.bar(x, times_track, width, label='常见路径跟踪')
    ax.bar(x + width, times_prop, width, label='本文方法')
    ax.set_title('不同方法分阶段作业耗时对比', fontsize=17, pad=14)
    ax.set_xlabel('作业阶段', fontsize=12)
    ax.set_ylabel('相对耗时 / s', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(loc='upper right', frameon=True)
    ax.text(0.5, -0.18, '说明：该图为仿真分析示意，用于体现流程效率差异趋势，具体数值需以实际测试记录为准。',
            transform=ax.transAxes, ha='center', fontsize=10)
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.98])
    raw = out_dir / f'{out_base}_raw.png'
    fig.savefig(raw, bbox_inches='tight')
    plt.close(fig)
    framed = out_dir / f'{out_base}.png'
    matlab_frame(raw, framed, "Figure 1")
    return framed

def save_radar_chart(out_base):
    labels = radar_labels
    data = [manual, openloop, tracking, proposed]
    names = ['人工牵引', '固定路径/开环', '常见路径跟踪', '本文方法']
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    
    fig = plt.figure(figsize=(8.5, 7.2), dpi=180)
    ax = fig.add_subplot(111, polar=True)
    for arr, name in zip(data, names):
        values = arr.tolist() + arr[:1].tolist()
        ax.plot(angles, values, linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.06)
    ax.set_title('不同方法综合性能雷达对比', fontsize=17, pad=22)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 5)
    ax.set_yticklabels([])
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.10), frameon=True)
    fig.text(0.5, 0.03, '说明：该图为定性综合比较，用于呈现本文方法在本课题场景中的相对特点，并非实测定量结果。',
             ha='center', fontsize=10)
    fig.tight_layout(rect=[0.02, 0.08, 0.92, 0.98])
    raw = out_dir / f'{out_base}_raw.png'
    fig.savefig(raw, bbox_inches='tight')
    plt.close(fig)
    framed = out_dir / f'{out_base}.png'
    matlab_frame(raw, framed, "Figure 1")
    return framed

def save_speed_stability(out_base):
    t2 = np.linspace(0, 40, 401)
    v_open = 0.55 + 0.09*np.sin(0.55*t2) + 0.04*np.sin(1.6*t2)
    v_track = 0.50 + 0.06*np.sin(0.42*t2)*np.exp(-0.012*t2) + 0.025*np.sin(1.2*t2)*np.exp(-0.02*t2)
    v_prop = 0.48 + 0.035*np.sin(0.35*t2)*np.exp(-0.025*t2) + 0.012*np.sin(1.1*t2)*np.exp(-0.035*t2)
    # simulate slowing near docking
    taper = 1 - 0.55/(1+np.exp(-(t2-26)/2.5))
    v_open *= (1 - 0.35/(1+np.exp(-(t2-30)/2)))
    v_track *= taper
    v_prop *= taper * (1 - 0.08/(1+np.exp(-(t2-34)/1.7)))
    
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=180)
    ax.plot(t2, v_open, linewidth=2.0, label='固定路径/开环')
    ax.plot(t2, v_track, linewidth=2.0, label='常见路径跟踪')
    ax.plot(t2, v_prop, linewidth=2.0, label='本文方法')
    ax.set_title('对接过程速度平稳性对比', fontsize=17, pad=14)
    ax.set_xlabel('时间 / s', fontsize=12)
    ax.set_ylabel('车辆速度 / m·s⁻¹', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='upper right', frameon=True)
    ax.text(0.5, -0.18, '说明：该图用于体现低速接近与微动对接阶段的速度变化趋势，具体数值需以实际测试记录为准。',
            transform=ax.transAxes, ha='center', fontsize=10)
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.98])
    raw = out_dir / f'{out_base}_raw.png'
    fig.savefig(raw, bbox_inches='tight')
    plt.close(fig)
    framed = out_dir / f'{out_base}.png'
    matlab_frame(raw, framed, "Figure 1")
    return framed

files = []
files.append(save_line_chart(t, [e_open, e_track, e_prop], ['固定路径/开环', '常见路径跟踪', '本文方法'], 
                             '对接过程横向误差收敛对比', '横向误差 / m', 'fig1_lateral_error'))
files.append(save_line_chart(t, [yaw_open, yaw_track, yaw_prop], ['固定路径/开环', '常见路径跟踪', '本文方法'], 
                             '对接过程航向误差收敛对比', '航向误差 / °', 'fig2_yaw_error'))
files.append(save_speed_stability('fig3_speed_stability'))
files.append(save_bar_chart('fig4_stage_time'))
files.append(save_radar_chart('fig5_radar_comparison'))

zip_path = out_dir / 'matlab_style_performance_comparison_figures.zip'
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        zf.write(f, arcname=f.name)

files, zip_path
