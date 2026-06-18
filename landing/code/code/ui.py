from __future__ import annotations

import csv
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    tk = None
    ttk = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from controller import ControllerConfig, LandingController
from sim_env import LandingSim, SimConfig


class ControlPanel:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.log_dir = self.base_dir / "logs"
        self.fig_dir = self.base_dir / "figs"
        self.log_dir.mkdir(exist_ok=True)
        self.fig_dir.mkdir(exist_ok=True)

        self.enabled = tk is not None and ttk is not None
        self.root: Optional[tk.Tk] = None
        self.preview_canvas = None
        self.trend_canvas = None
        self.left_frame = None

        self.sim: Optional[LandingSim] = None
        self.controller: Optional[LandingController] = None
        self.running = False
        self.log_rows: list[dict] = []
        self.recent_history: deque[dict] = deque(maxlen=1200)
        self.current_csv: Optional[Path] = None
        self.current_png: Optional[Path] = None

        if not self.enabled:
            return

        self.root = tk.Tk()
        self.root.title("固定翼着陆 HMI 完整仿真")
        self.root.geometry("1360x920")
        self._init_vars()
        self._build_layout()
        self.reset_simulation()
        self.root.after(20, self._tick)

    def _init_vars(self) -> None:
        self.init_agl_var = tk.StringVar(value="120")
        self.init_speed_var = tk.StringVar(value="72")
        self.init_roc_var = tk.StringVar(value="-250")
        self.heading_var = tk.StringVar(value="0")
        self.init_xtrack_var = tk.StringVar(value="0")
        self.runway_distance_var = tk.StringVar(value="2200")
        self.runway_heading_var = tk.StringVar(value="0")

        self.terrain_type_var = tk.StringVar(value="flat")
        self.terrain_base_var = tk.StringVar(value="0")
        self.terrain_amp_var = tk.StringVar(value="25")
        self.terrain_wave_var = tk.StringVar(value="1400")
        self.terrain_bump_var = tk.StringVar(value="40")
        self.terrain_center_var = tk.StringVar(value="1800")
        self.terrain_width_var = tk.StringVar(value="350")

        self.glide_slope_var = tk.StringVar(value="3.0")
        self.normal_speed_var = tk.StringVar(value="72")
        self.cons_speed_var = tk.StringVar(value="69")
        self.min_speed_var = tk.StringVar(value="62")
        self.max_speed_var = tk.StringVar(value="88")
        self.flare_height_var = tk.StringVar(value="8")
        self.flare_distance_var = tk.StringVar(value="110")
        self.touchdown_height_var = tk.StringVar(value="2.0")
        self.max_sink_var = tk.StringVar(value="850")
        self.max_roll_var = tk.StringVar(value="20")
        self.max_pitch_var = tk.StringVar(value="16")
        self.min_pitch_var = tk.StringVar(value="-12")
        self.max_xtrack_var = tk.StringVar(value="120")
        self.abort_min_height_var = tk.StringVar(value="8")
        self.target_touchdown_var = tk.StringVar(value="120")
        self.capture_distance_var = tk.StringVar(value="320")
        self.commit_distance_var = tk.StringVar(value="90")
        self.commit_height_var = tk.StringVar(value="22")
        self.flare_xt_var = tk.StringVar(value="28")
        self.flare_hdg_var = tk.StringVar(value="6")
        self.flare_roll_var = tk.StringVar(value="10")
        self.td_xt_var = tk.StringVar(value="45")
        self.td_hdg_var = tk.StringVar(value="10")
        self.td_roll_var = tk.StringVar(value="12")
        self.enforce_constraints_var = tk.BooleanVar(value=True)

        self.mode_var = tk.StringVar(value="normal")
        self.status_var = tk.StringVar(value="")
        self.info_var = tk.StringVar(value="waiting...")
        self.suggestion_var = tk.StringVar(value="建议：等待仿真开始")
        self.alert_var = tk.StringVar(value="告警：无")
        self.trigger_var = tk.StringVar(value="触发原因：无")
        self.output_var = tk.StringVar(value="尚未导出")

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        left_container = ttk.Frame(outer, padding=8)
        right = ttk.Frame(outer, padding=8)
        left_container.pack(side="left", fill="y")
        right.pack(side="right", fill="both", expand=True)

        canvas = tk.Canvas(left_container, width=360)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        self.left_frame = ttk.Frame(canvas)
        self.left_frame.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.left_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="y")
        scrollbar.pack(side="right", fill="y")

        self._build_param_panel(self.left_frame)
        self._build_control_panel(self.left_frame)
        self._build_status_panel(right)

    def _build_param_panel(self, parent) -> None:
        init_frame = ttk.LabelFrame(parent, text="初始条件与跑道", padding=8)
        init_frame.pack(fill="x", pady=4)
        self._add_row(init_frame, "初始离地高度 AGL(ft)", self.init_agl_var, 0)
        self._add_row(init_frame, "初始速度(kts)", self.init_speed_var, 1)
        self._add_row(init_frame, "初始下降率(fpm)", self.init_roc_var, 2)
        self._add_row(init_frame, "初始航向(deg)", self.heading_var, 3)
        self._add_row(init_frame, "初始侧偏(ft)", self.init_xtrack_var, 4)
        self._add_row(init_frame, "跑道距离(ft)", self.runway_distance_var, 5)
        self._add_row(init_frame, "跑道航向(deg)", self.runway_heading_var, 6)

        terrain_frame = ttk.LabelFrame(parent, text="复杂地形参数", padding=8)
        terrain_frame.pack(fill="x", pady=4)
        ttk.Label(terrain_frame, text="地形类型").grid(row=0, column=0, sticky="w", pady=2)
        box = ttk.Combobox(
            terrain_frame,
            textvariable=self.terrain_type_var,
            values=["flat", "rolling", "ridge", "valley", "steps", "mixed"],
            width=16,
            state="readonly",
        )
        box.grid(row=0, column=1, sticky="ew", pady=2)
        box.bind("<<ComboboxSelected>>", lambda _e: self._draw_realtime_views())
        self._add_row(terrain_frame, "基准海拔(ft)", self.terrain_base_var, 1)
        self._add_row(terrain_frame, "起伏幅值(ft)", self.terrain_amp_var, 2)
        self._add_row(terrain_frame, "地形波长(ft)", self.terrain_wave_var, 3)
        self._add_row(terrain_frame, "局部突起(ft)", self.terrain_bump_var, 4)
        self._add_row(terrain_frame, "突起中心(ft)", self.terrain_center_var, 5)
        self._add_row(terrain_frame, "突起宽度(ft)", self.terrain_width_var, 6)

        ctrl_frame = ttk.LabelFrame(parent, text="控制与保护参数", padding=8)
        ctrl_frame.pack(fill="x", pady=4)
        self._add_row(ctrl_frame, "下滑角(deg)", self.glide_slope_var, 0)
        self._add_row(ctrl_frame, "正常目标速度", self.normal_speed_var, 1)
        self._add_row(ctrl_frame, "保守目标速度", self.cons_speed_var, 2)
        self._add_row(ctrl_frame, "最小速度", self.min_speed_var, 3)
        self._add_row(ctrl_frame, "最大速度", self.max_speed_var, 4)
        self._add_row(ctrl_frame, "拉平高度(ft)", self.flare_height_var, 5)
        self._add_row(ctrl_frame, "拉平距离(ft)", self.flare_distance_var, 6)
        self._add_row(ctrl_frame, "接地高度(ft)", self.touchdown_height_var, 7)
        self._add_row(ctrl_frame, "最大下沉率(fpm)", self.max_sink_var, 8)
        self._add_row(ctrl_frame, "最大滚转(deg)", self.max_roll_var, 9)
        self._add_row(ctrl_frame, "最大俯仰(deg)", self.max_pitch_var, 10)
        self._add_row(ctrl_frame, "最小俯仰(deg)", self.min_pitch_var, 11)
        self._add_row(ctrl_frame, "最大侧偏(ft)", self.max_xtrack_var, 12)
        self._add_row(ctrl_frame, "最低复飞高度(ft)", self.abort_min_height_var, 13)
        self._add_row(ctrl_frame, "目标接地点(阈值后 ft)", self.target_touchdown_var, 14)
        self._add_row(ctrl_frame, "跑道捕获距离(ft)", self.capture_distance_var, 15)
        self._add_row(ctrl_frame, "跑道提交距离(ft)", self.commit_distance_var, 16)
        self._add_row(ctrl_frame, "跑道提交高度(ft)", self.commit_height_var, 17)
        self._add_row(ctrl_frame, "flare 最大侧偏(ft)", self.flare_xt_var, 18)
        self._add_row(ctrl_frame, "flare 最大航向误差(deg)", self.flare_hdg_var, 19)
        self._add_row(ctrl_frame, "flare 最大滚转(deg)", self.flare_roll_var, 20)
        self._add_row(ctrl_frame, "接地提交侧偏(ft)", self.td_xt_var, 21)
        self._add_row(ctrl_frame, "接地提交航向误差(deg)", self.td_hdg_var, 22)
        self._add_row(ctrl_frame, "接地提交滚转(deg)", self.td_roll_var, 23)
        ttk.Checkbutton(ctrl_frame, text="启用约束保护", variable=self.enforce_constraints_var).grid(
            row=24, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        ttk.Button(parent, text="重置 / 应用参数", command=self.reset_simulation).pack(fill="x", pady=6)

    def _build_control_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="运行控制与人机交互", padding=8)
        frame.pack(fill="x", pady=4)
        ttk.Button(frame, text="开始", command=self.start).pack(fill="x", pady=2)
        ttk.Button(frame, text="暂停", command=self.pause).pack(fill="x", pady=2)
        ttk.Button(frame, text="继续", command=self.resume).pack(fill="x", pady=2)
        ttk.Button(frame, text="停止", command=self.stop).pack(fill="x", pady=2)
        ttk.Button(frame, text="复飞 / 中止", command=self.trigger_abort).pack(fill="x", pady=2)
        ttk.Button(frame, text="保存 CSV + 曲线图", command=self.save_outputs).pack(fill="x", pady=2)
        ttk.Label(frame, text="人工模式").pack(anchor="w", pady=(8, 2))
        mode_box = ttk.Combobox(frame, textvariable=self.mode_var, values=["normal", "conservative"], state="readonly")
        mode_box.pack(fill="x")
        mode_box.bind("<<ComboboxSelected>>", lambda _e: self._apply_mode())
        ttk.Label(frame, textvariable=self.output_var, wraplength=320, justify="left").pack(anchor="w", pady=(8, 0))

    def _build_status_panel(self, parent) -> None:
        top = ttk.LabelFrame(parent, text="实时状态", padding=8)
        top.pack(fill="x", pady=4)
        ttk.Label(top, textvariable=self.info_var, justify="left", font=("Consolas", 11)).pack(fill="x")
        ttk.Label(top, textvariable=self.suggestion_var, foreground="#0b5394", wraplength=900, justify="left").pack(fill="x", pady=(4, 0))
        ttk.Label(top, textvariable=self.alert_var, foreground="#990000", wraplength=900, justify="left").pack(fill="x", pady=(4, 0))
        ttk.Label(top, textvariable=self.trigger_var, foreground="#7a3e00", wraplength=900, justify="left").pack(fill="x", pady=(4, 0))

        sim_frame = ttk.LabelFrame(parent, text="实时仿真视图（地形 + 飞机轨迹）", padding=8)
        sim_frame.pack(fill="both", expand=True, pady=4)
        self.preview_canvas = tk.Canvas(sim_frame, width=900, height=340, bg="white")
        self.preview_canvas.pack(fill="both", expand=True)

        trend_frame = ttk.LabelFrame(parent, text="实时趋势（最近 20 s）", padding=8)
        trend_frame.pack(fill="x", pady=4)
        self.trend_canvas = tk.Canvas(trend_frame, width=900, height=170, bg="white")
        self.trend_canvas.pack(fill="x")

        ttk.Label(parent, textvariable=self.status_var, foreground="#1f4b99", wraplength=900, justify="left").pack(fill="x", pady=(4, 0))

    def _add_row(self, frame, label: str, var, row: int) -> None:
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        ttk.Entry(frame, textvariable=var, width=18).grid(row=row, column=1, sticky="ew", pady=2)

    def _parse_sim_config(self) -> SimConfig:
        return SimConfig(
            init_agl_ft=float(self.init_agl_var.get()),
            init_vc_kts=float(self.init_speed_var.get()),
            init_roc_fpm=float(self.init_roc_var.get()),
            init_heading_deg=float(self.heading_var.get()),
            init_cross_track_ft=float(self.init_xtrack_var.get()),
            runway_distance_ft=float(self.runway_distance_var.get()),
            runway_heading_deg=float(self.runway_heading_var.get()),
            terrain_type=self.terrain_type_var.get(),
            terrain_base_ft=float(self.terrain_base_var.get()),
            terrain_amp_ft=float(self.terrain_amp_var.get()),
            terrain_wavelength_ft=float(self.terrain_wave_var.get()),
            terrain_bump_ft=float(self.terrain_bump_var.get()),
            terrain_bump_center_ft=float(self.terrain_center_var.get()),
            terrain_bump_width_ft=float(self.terrain_width_var.get()),
        )

    def _parse_controller_config(self) -> ControllerConfig:
        return ControllerConfig(
            glide_slope_deg=float(self.glide_slope_var.get()),
            normal_target_speed_kts=float(self.normal_speed_var.get()),
            conservative_target_speed_kts=float(self.cons_speed_var.get()),
            min_speed_kts=float(self.min_speed_var.get()),
            max_speed_kts=float(self.max_speed_var.get()),
            flare_height_ft=float(self.flare_height_var.get()),
            flare_distance_ft=float(self.flare_distance_var.get()),
            touchdown_height_ft=float(self.touchdown_height_var.get()),
            max_sink_rate_fpm=float(self.max_sink_var.get()),
            max_abs_roll_deg=float(self.max_roll_var.get()),
            max_pitch_deg=float(self.max_pitch_var.get()),
            min_pitch_deg=float(self.min_pitch_var.get()),
            max_cross_track_ft=float(self.max_xtrack_var.get()),
            abort_min_height_ft=float(self.abort_min_height_var.get()),
            target_touchdown_past_threshold_ft=float(self.target_touchdown_var.get()),
            runway_capture_distance_ft=float(self.capture_distance_var.get()),
            runway_commit_distance_ft=float(self.commit_distance_var.get()),
            runway_commit_height_ft=float(self.commit_height_var.get()),
            flare_max_cross_track_ft=float(self.flare_xt_var.get()),
            flare_max_heading_error_deg=float(self.flare_hdg_var.get()),
            flare_max_roll_deg=float(self.flare_roll_var.get()),
            touchdown_commit_cross_track_ft=float(self.td_xt_var.get()),
            touchdown_commit_heading_error_deg=float(self.td_hdg_var.get()),
            touchdown_commit_roll_deg=float(self.td_roll_var.get()),
            enforce_constraints=bool(self.enforce_constraints_var.get()),
        )

    def reset_simulation(self) -> None:
        try:
            sim_cfg = self._parse_sim_config()
            ctl_cfg = self._parse_controller_config()
        except ValueError:
            self._show_error("参数格式错误，请输入数字。")
            return

        self.sim = LandingSim(sim_cfg)
        self.controller = LandingController(ctl_cfg)
        self.controller.set_mode(self.mode_var.get())
        self.log_rows = []
        self.recent_history.clear()
        self.running = False
        self.current_csv = None
        self.current_png = None
        self.output_var.set("尚未导出")
        self._set_status("已应用参数。点击开始运行。")
        self._refresh_status_once()
        self._draw_realtime_views()

    def start(self) -> None:
        if self.sim is None or self.controller is None:
            self.reset_simulation()
        self.running = True
        self._set_status("仿真运行中。")

    def pause(self) -> None:
        self.running = False
        self._set_status("已暂停。")

    def resume(self) -> None:
        if self.sim is None or self.controller is None:
            self.reset_simulation()
        self.running = True
        self._set_status("继续运行。")

    def stop(self) -> None:
        self.running = False
        self._set_status("已停止。")

    def trigger_abort(self) -> None:
        if self.controller is not None:
            self.controller.trigger_abort()
            self._set_status("已触发复飞 / 中止。")

    def _apply_mode(self) -> None:
        if self.controller is not None:
            self.controller.set_mode(self.mode_var.get())
            self._set_status(f"人工模式切换为：{self.mode_var.get()}")

    def _refresh_status_once(self) -> None:
        if self.sim is None or self.controller is None:
            return
        state = self.sim.state()
        self._update_status_text(state, self.controller.last_phase, "waiting", "无", "")

    def _update_status_text(self, state: dict, phase: str, note: str, alerts: str, trigger_reason: str = "") -> None:
        target_td = self.controller.config.target_touchdown_past_threshold_ft if self.controller else 0.0
        dist_to_td = float(state["distance_to_runway_ft"]) - target_td
        self.info_var.set(
            f"t        : {state['t']:.2f} s\n"
            f"distance : {state['distance_ft']:.1f} ft\n"
            f"to rw    : {state['distance_to_runway_ft']:.1f} ft\n"
            f"to td    : {dist_to_td:.1f} ft\n"
            f"terrain  : {state['terrain_elev_ft']:.2f} ft\n"
            f"h_agl    : {state['h_agl_ft']:.2f} ft\n"
            f"h_sl     : {state['h_sl_ft']:.2f} ft\n"
            f"speed    : {state['vc_kts']:.2f} kts\n"
            f"vs       : {state['vs_fpm']:.2f} fpm\n"
            f"pitch    : {state['pitch_deg']:.2f} deg\n"
            f"roll     : {state['roll_deg']:.2f} deg\n"
            f"yaw      : {state['yaw_deg']:.2f} deg\n"
            f"xtrack   : {state['cross_track_ft']:.2f} ft\n"
            f"phase    : {phase}\n"
            f"mode     : {self.controller.mode if self.controller else '-'}\n"
            f"note     : {note}\n"
            f"cmd_th   : {state['cmd_throttle']:.2f}\n"
            f"cmd_el   : {state['cmd_elevator']:.2f}\n"
            f"cmd_ai   : {state['cmd_aileron']:.2f}\n"
            f"cmd_ru   : {state['cmd_rudder']:.2f}\n"
            f"cmd_flap : {state['cmd_flaps']:.2f}\n"
        )
        self.alert_var.set("告警：" + (alerts if alerts else "无"))
        self.trigger_var.set("触发原因：" + (trigger_reason if trigger_reason else "无"))

    def _tick(self) -> None:
        try:
            if self.running and self.sim is not None and self.controller is not None:
                state = self.sim.state()
                self.controller.set_mode(self.mode_var.get())
                cmd = self.controller.compute(state)

                self.sim.set_controls(
                    cmd.throttle,
                    cmd.elevator,
                    cmd.aileron,
                    cmd.rudder,
                    cmd.flaps,
                )

                row = {
                    **state,
                    "phase": cmd.phase,
                    "mode": self.controller.mode,
                    "note": cmd.note,
                    "suggestion": cmd.suggestion,
                    "alerts": " | ".join(cmd.alerts),
                    "throttle": cmd.throttle,
                    "elevator": cmd.elevator,
                    "aileron": cmd.aileron,
                    "rudder": cmd.rudder,
                    "flaps": cmd.flaps,
                    "target_td_past_threshold_ft": self.controller.config.target_touchdown_past_threshold_ft,
                    "distance_to_td_ft": float(state["distance_to_runway_ft"]) - self.controller.config.target_touchdown_past_threshold_ft,
                }
                self.log_rows.append(row)
                self.recent_history.append(row)

                self._update_status_text(state, cmd.phase, cmd.note, "；".join(cmd.alerts), self.controller.last_reason)
                self.suggestion_var.set(f"建议：{cmd.suggestion}")
                self._draw_realtime_views()

                if cmd.phase == "rollout":
                    self.running = False
                    self._set_status("已接地，仿真自动停止。")
                    try:
                        self.save_outputs()
                    except Exception:
                        pass
                    self.root.after(20, self._tick)
                    return

                if not self.sim.step():
                    self.running = False
                    self._set_status("JSBSim 返回停止，仿真已暂停。")

            self.root.after(20, self._tick)

        except Exception as exc:
            self.running = False
            self._set_status(f"运行报错：{exc}")
            self.root.after(100, self._tick)

    def _draw_realtime_views(self) -> None:
        self._draw_terrain_preview()
        self._draw_trend_preview()

    def _draw_terrain_preview(self) -> None:
        if self.preview_canvas is None:
            return
        try:
            sim_cfg = self._parse_sim_config() if self.sim is None else self.sim.config
            x_end = max(sim_cfg.runway_distance_ft + 1200.0, 6000.0)
            if self.log_rows:
                x_end = max(x_end, self.log_rows[-1]["distance_ft"] + 2000.0)
            temp_sim = LandingSim(sim_cfg)
            points = temp_sim.terrain.sample_profile(x_end, 40.0)
        except Exception:
            return

        self.preview_canvas.delete("all")
        width = int(self.preview_canvas.winfo_width() or 900)
        height = int(self.preview_canvas.winfo_height() or 340)
        pad = 30
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        aircraft_line = []
        current_x = 0.0
        if self.log_rows:
            aircraft_line = [(r["distance_ft"], r["h_sl_ft"]) for r in self.log_rows]
            ys.extend([y for _, y in aircraft_line])
            current_x = float(self.log_rows[-1]["distance_ft"])

        full_x_min = min(xs)
        full_x_max = max(xs)
        if aircraft_line and current_x > 0.35 * max(sim_cfg.runway_distance_ft, 1.0):
            x_min_view = max(full_x_min, current_x - 650.0)
            x_max_view = min(full_x_max, current_x + 1800.0)
            if x_max_view <= x_min_view:
                x_min_view = max(full_x_min, current_x - 400.0)
                x_max_view = min(full_x_max, current_x + 1200.0)
        else:
            x_min_view = full_x_min
            x_max_view = min(full_x_max, max(sim_cfg.runway_distance_ft + 800.0, 3000.0))

        if x_max_view <= x_min_view:
            x_min_view = full_x_min
            x_max_view = full_x_min + 2000.0

        y_min = min(ys) - 15.0
        y_max = max(ys) + 15.0
        x_span = x_max_view - x_min_view or 1.0
        y_span = y_max - y_min or 1.0

        def map_x(x: float) -> float:
            return pad + (x - x_min_view) / x_span * (width - 2 * pad)

        def map_y(y: float) -> float:
            return height - pad - (y - y_min) / y_span * (height - 2 * pad)

        self.preview_canvas.create_line(pad, height - pad, width - pad, height - pad, fill="#444")
        self.preview_canvas.create_line(pad, pad, pad, height - pad, fill="#444")
        self.preview_canvas.create_text(width - 150, 16, text=f"terrain={sim_cfg.terrain_type}", fill="#333")
        self.preview_canvas.create_text(pad + 165, 14, text="蓝: 地形  红: 飞机轨迹  虚线: 跑道 / 目标接地点", fill="#333")
        self.preview_canvas.create_text(width - 180, height - 12, text=f"视野: {x_min_view:.0f} ~ {x_max_view:.0f} ft", fill="#666")

        poly = []
        for x, y in points:
            if x_min_view - 40.0 <= x <= x_max_view + 40.0:
                poly.extend([map_x(x), map_y(y)])
        if len(poly) >= 4:
            self.preview_canvas.create_line(*poly, fill="#1f77b4", width=2)

        runway_x = sim_cfg.runway_distance_ft
        if x_min_view <= runway_x <= x_max_view:
            self.preview_canvas.create_line(map_x(runway_x), pad, map_x(runway_x), height - pad, fill="#999", dash=(4, 3))
            self.preview_canvas.create_text(map_x(runway_x), pad + 10, text="跑道阈值", fill="#666")

        if self.controller is not None:
            td_x = sim_cfg.runway_distance_ft + self.controller.config.target_touchdown_past_threshold_ft
            if x_min_view <= td_x <= x_max_view:
                self.preview_canvas.create_line(map_x(td_x), pad, map_x(td_x), height - pad, fill="#cc8800", dash=(3, 3))
                self.preview_canvas.create_text(map_x(td_x), pad + 24, text="目标接地点", fill="#aa6600")

        if aircraft_line:
            flight_poly = []
            visible_aircraft = [(x, y) for x, y in aircraft_line if x_min_view - 100.0 <= x <= x_max_view + 100.0]
            for x, y in visible_aircraft:
                flight_poly.extend([map_x(x), map_y(y)])
            if len(flight_poly) >= 4:
                self.preview_canvas.create_line(*flight_poly, fill="#d62728", width=2)
            x_last, y_last = aircraft_line[-1]
            if x_min_view <= x_last <= x_max_view:
                cx, cy = map_x(x_last), map_y(y_last)
                self.preview_canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill="#d62728", outline="#d62728")
                self.preview_canvas.create_text(cx + 64, cy - 10, text=f"AGL={self.log_rows[-1]['h_agl_ft']:.1f} ft", fill="#d62728")

    def _draw_trend_preview(self) -> None:
        if self.trend_canvas is None:
            return
        self.trend_canvas.delete("all")
        width = int(self.trend_canvas.winfo_width() or 900)
        height = int(self.trend_canvas.winfo_height() or 170)
        pad = 30
        self.trend_canvas.create_line(pad, height - pad, width - pad, height - pad, fill="#444")
        self.trend_canvas.create_line(pad, pad, pad, height - pad, fill="#444")
        if len(self.recent_history) < 2:
            return
        hist = list(self.recent_history)
        t1 = hist[-1]["t"]
        hist = [h for h in hist if h["t"] >= t1 - 20.0]
        times = [h["t"] - hist[0]["t"] for h in hist]
        agls = [h["h_agl_ft"] for h in hist]
        xtracks = [h["cross_track_ft"] for h in hist]
        all_y = agls + xtracks
        y_min = min(all_y) - 5.0
        y_max = max(all_y) + 5.0
        x_span = (times[-1] - times[0]) or 1.0
        y_span = (y_max - y_min) or 1.0

        def map_x(x: float) -> float:
            return pad + (x - times[0]) / x_span * (width - 2 * pad)

        def map_y(y: float) -> float:
            return height - pad - (y - y_min) / y_span * (height - 2 * pad)

        agl_poly = []
        xt_poly = []
        for x, y1, y2 in zip(times, agls, xtracks):
            agl_poly.extend([map_x(x), map_y(y1)])
            xt_poly.extend([map_x(x), map_y(y2)])
        self.trend_canvas.create_line(*agl_poly, fill="#d62728", width=2)
        self.trend_canvas.create_line(*xt_poly, fill="#2ca02c", width=2)
        self.trend_canvas.create_text(pad + 52, 12, text="红: AGL  绿: 侧偏", fill="#333")
        self.trend_canvas.create_text(width - 48, map_y(agls[-1]), text=f"{agls[-1]:.1f}", fill="#d62728")
        self.trend_canvas.create_text(width - 48, map_y(xtracks[-1]), text=f"{xtracks[-1]:.1f}", fill="#2ca02c")

    def save_outputs(self) -> None:
        if not self.log_rows:
            self._set_status("当前没有可保存的数据。")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.log_dir / f"run_{ts}.csv"
        png_path = self.fig_dir / f"run_{ts}.png"

        fieldnames = list(self.log_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.log_rows)

        self._make_figure(png_path)
        self.current_csv = csv_path
        self.current_png = png_path
        self.output_var.set(f"CSV: {csv_path.name}\nPNG: {png_path.name}")
        self._set_status("已保存 CSV 和曲线图。")

    def _make_figure(self, fig_path: Path) -> None:
        rows = self.log_rows
        t = [r["t"] for r in rows]
        agl = [r["h_agl_ft"] for r in rows]
        speed = [r["vc_kts"] for r in rows]
        pitch = [r["pitch_deg"] for r in rows]
        roll = [r["roll_deg"] for r in rows]
        x = [r["distance_ft"] for r in rows]
        terrain = [r["terrain_elev_ft"] for r in rows]
        aircraft = [r["h_sl_ft"] for r in rows]

        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(4, 1, 1)
        ax1.plot(t, agl)
        ax1.set_ylabel("AGL (ft)")
        ax1.grid(True)

        ax2 = fig.add_subplot(4, 1, 2)
        ax2.plot(t, speed)
        ax2.set_ylabel("Speed (kts)")
        ax2.grid(True)

        ax3 = fig.add_subplot(4, 1, 3)
        ax3.plot(t, pitch, label="pitch")
        ax3.plot(t, roll, label="roll")
        ax3.set_ylabel("Attitude (deg)")
        ax3.grid(True)
        ax3.legend()

        ax4 = fig.add_subplot(4, 1, 4)
        ax4.plot(x, terrain, label="terrain")
        ax4.plot(x, aircraft, label="aircraft")
        if self.controller is not None and self.sim is not None:
            td_x = self.sim.config.runway_distance_ft + self.controller.config.target_touchdown_past_threshold_ft
            ax4.axvline(td_x, linestyle="--", label="target touchdown")
        ax4.set_xlabel("Distance (ft)")
        ax4.set_ylabel("Elevation (ft)")
        ax4.grid(True)
        ax4.legend()

        fig.tight_layout()
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)

    def _show_error(self, text: str) -> None:
        self.status_var.set(text)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def run(self) -> None:
        if self.enabled and self.root is not None:
            self.root.mainloop()