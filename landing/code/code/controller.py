from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import math

KTS_TO_FPS = 1.6878098571


@dataclass
class ControllerConfig:
    glide_slope_deg: float = 3.0
    flare_height_ft: float = 8.0
    flare_distance_ft: float = 95.0
    touchdown_height_ft: float = 2.0

    # 目标接地点：跑道阈值后多少英尺
    target_touchdown_past_threshold_ft: float = 120.0

    normal_target_speed_kts: float = 72.0
    conservative_target_speed_kts: float = 69.0
    abort_target_speed_kts: float = 76.0

    min_speed_kts: float = 62.0
    max_speed_kts: float = 88.0
    max_sink_rate_fpm: float = 900.0
    max_abs_roll_deg: float = 22.0
    max_pitch_deg: float = 18.0
    min_pitch_deg: float = -12.0

    # 为了兼容 ui.py 保留这两个字段
    max_cross_track_ft: float = 120.0
    abort_min_height_ft: float = 8.0

    enforce_constraints: bool = True
    startup_grace_s: float = 3.0
    violation_persist_s: float = 0.60

    # 只有非常极端时才自动复飞
    auto_abort_cross_track_ft: float = 220.0
    auto_abort_heading_error_deg: float = 28.0
    auto_abort_roll_deg: float = 35.0
    auto_abort_sink_rate_fpm: float = 1300.0

    # 跑道附近优先落地
    runway_capture_distance_ft: float = 320.0
    runway_commit_distance_ft: float = 100.0
    runway_commit_height_ft: float = 22.0

    # flare / touchdown 条件
    flare_max_cross_track_ft: float = 35.0
    flare_max_heading_error_deg: float = 8.0
    flare_max_roll_deg: float = 10.0

    touchdown_commit_cross_track_ft: float = 55.0
    touchdown_commit_heading_error_deg: float = 12.0
    touchdown_commit_roll_deg: float = 12.0


@dataclass
class Command:
    throttle: float
    elevator: float
    aileron: float
    rudder: float
    flaps: float
    phase: str
    note: str
    suggestion: str = ""
    alerts: List[str] = field(default_factory=list)


class LandingController:
    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        self.mode = "normal"
        self.abort_requested = False
        self.abort_latch = False
        self.flare_latch = False
        self.landing_commit_latch = False
        self.last_phase = "approach"
        self.last_reason = ""
        self._violation_time_s = 0.0

    def set_mode(self, mode: str) -> None:
        if mode in {"normal", "conservative"}:
            self.mode = mode

    def set_config(self, config: ControllerConfig) -> None:
        self.config = config

    def trigger_abort(self) -> None:
        self.abort_requested = True
        self.abort_latch = True
        self.flare_latch = False
        self.landing_commit_latch = False
        self.last_reason = "manual abort"

    def clear_abort(self) -> None:
        self.abort_requested = False
        self.abort_latch = False
        self.flare_latch = False
        self.landing_commit_latch = False
        self.last_reason = ""
        self._violation_time_s = 0.0

    def reset(self) -> None:
        self.mode = "normal"
        self.clear_abort()
        self.last_phase = "approach"

    def compute(self, s: Dict[str, float | bool | str]) -> Command:
        cfg = self.config

        h_agl = float(s["h_agl_ft"])
        h_sl = float(s["h_sl_ft"])
        v_kts = float(s["vc_kts"])
        vs_fpm = float(s["vs_fpm"])
        pitch_deg = float(s["pitch_deg"])
        roll_deg = float(s["roll_deg"])
        yaw_deg = float(s["yaw_deg"])
        wow = bool(s["wow"])
        distance_to_runway_ft = float(s["distance_to_runway_ft"])
        terrain_elev_ft = float(s["terrain_elev_ft"])
        cross_track_ft = float(s["cross_track_ft"])
        runway_heading_deg = float(s["runway_heading_deg"])
        sim_time_s = float(s.get("t", 0.0))
        dt_s = float(s.get("dt", 0.02))

        alerts: List[str] = []
        in_startup_grace = sim_time_s < cfg.startup_grace_s
        heading_error_deg = self._wrap_deg(runway_heading_deg - yaw_deg)

        distance_to_touchdown_ft = (
            distance_to_runway_ft - cfg.target_touchdown_past_threshold_ft
        )

        in_runway_capture_zone = distance_to_touchdown_ft <= cfg.runway_capture_distance_ft
        in_runway_commit_zone = (
            distance_to_touchdown_ft <= cfg.runway_commit_distance_ft
            and h_agl <= cfg.runway_commit_height_ft
        )
        passed_threshold = distance_to_runway_ft < 0.0

        flare_alignment_ok = (
            abs(cross_track_ft) <= cfg.flare_max_cross_track_ft
            and abs(heading_error_deg) <= cfg.flare_max_heading_error_deg
            and abs(roll_deg) <= cfg.flare_max_roll_deg
        )
        touchdown_alignment_ok = (
            abs(cross_track_ft) <= cfg.touchdown_commit_cross_track_ft
            and abs(heading_error_deg) <= cfg.touchdown_commit_heading_error_deg
            and abs(roll_deg) <= cfg.touchdown_commit_roll_deg
        )

        # 接近跑道后，停止新触发自动复飞
        no_new_abort_zone = (
            in_runway_commit_zone
            or passed_threshold
            or (distance_to_touchdown_ft <= 120.0 and h_agl <= 25.0)
        )
        allow_new_auto_abort = (
            h_agl > cfg.abort_min_height_ft and not no_new_abort_zone
        )

        if wow:
            self.last_phase = "rollout"
            self.flare_latch = False
            self.landing_commit_latch = False
            self.abort_latch = False
            self.abort_requested = False
            self._violation_time_s = 0.0
            return Command(
                throttle=0.0,
                elevator=0.02,
                aileron=self._clamp(-0.02 * roll_deg, -0.10, 0.10),
                rudder=self._clamp(-0.012 * cross_track_ft, -0.10, 0.10),
                flaps=1.0,
                phase="rollout",
                note="on ground rollout",
                suggestion="已接地，保持滑跑",
                alerts=alerts,
            )

        # 已复飞时，只有明显重新建立进近才解除复飞锁存
        if self.abort_latch:
            recovered = (
                h_agl > 100.0
                and 66.0 < v_kts < 84.0
                and abs(pitch_deg) < 8.0
                and abs(roll_deg) < 8.0
                and distance_to_runway_ft > 500.0
                and abs(cross_track_ft) < 60.0
                and abs(heading_error_deg) < 8.0
            )
            if recovered:
                self.abort_latch = False
                self.abort_requested = False
                self.flare_latch = False
                self.landing_commit_latch = False
                self.last_reason = ""
                self._violation_time_s = 0.0
                alerts.append("复飞恢复：已重新建立进近")

        # 自动复飞只保留给极端情况
        violations: List[str] = []
        if (
            cfg.enforce_constraints
            and not in_startup_grace
            and not self.abort_latch
            and not self.landing_commit_latch
        ):
            if abs(roll_deg) > cfg.auto_abort_roll_deg:
                violations.append(f"滚转严重超限 {roll_deg:.1f}°")
            if pitch_deg > cfg.max_pitch_deg + 4.0 or pitch_deg < cfg.min_pitch_deg - 4.0:
                violations.append(f"俯仰严重超限 {pitch_deg:.1f}°")
            if v_kts < cfg.min_speed_kts - 5.0:
                violations.append(f"速度严重过低 {v_kts:.1f}kt")
            if v_kts > cfg.max_speed_kts + 5.0:
                violations.append(f"速度严重过高 {v_kts:.1f}kt")
            if h_agl < 80.0 and vs_fpm < -cfg.auto_abort_sink_rate_fpm:
                violations.append(f"下沉率严重过大 {vs_fpm:.0f}fpm")
            if abs(cross_track_ft) > cfg.auto_abort_cross_track_ft and h_agl > cfg.abort_min_height_ft:
                violations.append(f"侧偏严重超限 {cross_track_ft:.1f}ft")
            if abs(heading_error_deg) > cfg.auto_abort_heading_error_deg and h_agl > cfg.abort_min_height_ft:
                violations.append(f"航向严重超限 {heading_error_deg:.1f}°")

        if violations:
            self._violation_time_s += dt_s
        else:
            self._violation_time_s = 0.0

        if (
            violations
            and allow_new_auto_abort
            and self._violation_time_s >= cfg.violation_persist_s
            and not self.landing_commit_latch
        ):
            self.abort_requested = True
            self.abort_latch = True
            self.flare_latch = False
            self.landing_commit_latch = False
            self.last_reason = "; ".join(violations)
            alerts.extend([f"约束触发：{msg}" for msg in violations])

        # 复飞优先级最高
        if self.abort_latch or self.abort_requested:
            self.last_phase = "abort"
            self.flare_latch = False
            self.landing_commit_latch = False

            desired_bank_deg = self._clamp(
                0.14 * heading_error_deg - 0.030 * cross_track_ft,
                -10.0,
                10.0,
            )
            aileron, rudder = self._lateral_control_abort(
                desired_bank_deg, roll_deg, heading_error_deg, cross_track_ft
            )

            return Command(
                throttle=0.72,
                elevator=-0.025,
                aileron=aileron,
                rudder=rudder,
                flaps=0.00,
                phase="abort",
                note=f"go-around(triggered by: {self.last_reason or 'operator request'})",
                suggestion="执行复飞，离开跑道后重新建立进近",
                alerts=alerts,
            )

        # 接近跑道后，优先落地
        if in_runway_commit_zone and touchdown_alignment_ok:
            self.landing_commit_latch = True
            self.flare_latch = True

        enter_flare = (
            distance_to_touchdown_ft <= cfg.flare_distance_ft
            and h_agl <= max(1.5 * cfg.flare_height_ft, 13.0)
            and flare_alignment_ok
        )
        if enter_flare:
            self.flare_latch = True

        if self.landing_commit_latch and h_agl <= max(cfg.runway_commit_height_ft, 18.0):
            self.flare_latch = True

        if h_agl <= cfg.touchdown_height_ft and vs_fpm > -450.0:
            phase = "touchdown"
        elif self.flare_latch or self.landing_commit_latch:
            phase = "flare"
        else:
            phase = "approach"
        self.last_phase = phase

        target_speed_kts = (
            cfg.normal_target_speed_kts
            if self.mode == "normal"
            else cfg.conservative_target_speed_kts
        )

        effective_distance_ft = max(distance_to_touchdown_ft, 0.0)
        target_agl_ft = math.tan(math.radians(cfg.glide_slope_deg)) * effective_distance_ft
        target_alt_ft = terrain_elev_ft + target_agl_ft
        altitude_error_ft = h_sl - target_alt_ft

        ground_speed_fps = max(
            v_kts * KTS_TO_FPS * max(math.cos(math.radians(pitch_deg)), 0.2),
            1.0,
        )
        desired_vs_fpm = -math.tan(math.radians(cfg.glide_slope_deg)) * ground_speed_fps * 60.0

        if phase == "approach":
            throttle_base = 0.23 if self.mode == "normal" else 0.27
            local_target_speed_kts = target_speed_kts

            if in_runway_capture_zone:
                local_target_speed_kts = min(local_target_speed_kts, 69.0)
            if distance_to_touchdown_ft < 220.0:
                local_target_speed_kts = min(local_target_speed_kts, 68.0)

            throttle = (
                throttle_base
                + 0.012 * (local_target_speed_kts - v_kts)
                - 0.00018 * altitude_error_ft
            )
            elevator = (
                0.022
                + 0.00072 * altitude_error_ft
                + 0.000032 * (desired_vs_fpm - vs_fpm)
            )
            flaps = 0.12 if self.mode == "normal" else 0.18
            note = "approach guidance tracking"
            suggestion = "检测到侧偏后立即修正，并沿目标接地点下滑"

        elif phase == "flare":
            flare_ratio = self._clamp(h_agl / max(cfg.flare_height_ft, 1.0), 0.0, 1.0)
            throttle = max(0.0, 0.0007 * (66.0 - v_kts))
            elevator = (
                -0.010
                - 0.028 * (1.0 - flare_ratio)
                + 0.000016 * (-90.0 - vs_fpm)
            )
            flaps = 0.95 if self.mode == "normal" else 1.00
            note = "flare control"
            suggestion = "保持中心线，柔和拉平，在跑道附近接地"
            if vs_fpm < -650.0:
                alerts.append("提示：拉平段下沉率偏大")
        else:
            throttle = 0.00
            elevator = 0.030
            flaps = 1.0
            note = "touchdown settle"
            suggestion = "已进入接地区域，继续保持方向并准备滑跑"

        desired_bank_deg = self._desired_bank(
            phase=phase,
            cross_track_ft=cross_track_ft,
            heading_error_deg=heading_error_deg,
            distance_to_touchdown_ft=distance_to_touchdown_ft,
        )
        aileron, rudder = self._lateral_control(
            desired_bank_deg=desired_bank_deg,
            roll_deg=roll_deg,
            heading_error_deg=heading_error_deg,
            cross_track_ft=cross_track_ft,
            phase=phase,
            distance_to_touchdown_ft=distance_to_touchdown_ft,
        )

        if cfg.enforce_constraints:
            if h_agl < 50.0:
                aileron = self._clamp(aileron, -0.16, 0.16)
                rudder = self._clamp(rudder, -0.13, 0.13)

            if phase != "touchdown":
                if v_kts > cfg.max_speed_kts - 2.0:
                    throttle = min(throttle, 0.08)
                if v_kts < cfg.min_speed_kts + 2.0:
                    throttle = max(throttle, 0.35)

        if abs(cross_track_ft) > 8.0:
            alerts.append(f"提示：检测到侧偏 {cross_track_ft:.1f}ft，正在立即修正")
        if abs(cross_track_ft) > cfg.max_cross_track_ft:
            alerts.append(f"提示：侧偏超过设定阈值 {cfg.max_cross_track_ft:.1f}ft")
        if abs(heading_error_deg) > 4.0:
            alerts.append(f"提示：航向偏差 {heading_error_deg:.1f}°")
        if h_agl < 60.0 and vs_fpm < -650.0:
            alerts.append("提示：近地阶段下沉率偏大")
        if in_runway_capture_zone:
            alerts.append("提示：已进入跑道捕获区，优先在跑道附近接地")

        return Command(
            throttle=self._clamp(throttle, 0.0, 1.0),
            elevator=self._clamp(elevator, -0.16, 0.16),
            aileron=self._clamp(aileron, -0.20, 0.20),
            rudder=self._clamp(rudder, -0.16, 0.16),
            flaps=self._clamp(flaps, 0.0, 1.0),
            phase=phase,
            note=note,
            suggestion=suggestion,
            alerts=alerts,
        )

    def _desired_bank(
        self,
        phase: str,
        cross_track_ft: float,
        heading_error_deg: float,
        distance_to_touchdown_ft: float,
    ) -> float:
        abs_xt = abs(cross_track_ft)

        if phase == "approach":
            if distance_to_touchdown_ft > 500.0:
                kx = 0.10
                kh = 0.36
                limit = 18.0
            elif distance_to_touchdown_ft > 220.0:
                kx = 0.13
                kh = 0.32
                limit = 16.0
            else:
                kx = 0.15
                kh = 0.24
                limit = 12.0

            if abs_xt > 60.0:
                kx *= 1.20
            elif abs_xt > 30.0:
                kx *= 1.10

        elif phase == "flare":
            kx = 0.08
            kh = 0.16
            limit = 7.0
        else:
            kx = 0.05
            kh = 0.10
            limit = 5.0

        desired_bank_deg = kh * heading_error_deg - kx * cross_track_ft
        return self._clamp(desired_bank_deg, -limit, limit)

    @staticmethod
    def _lateral_control(
        desired_bank_deg: float,
        roll_deg: float,
        heading_error_deg: float,
        cross_track_ft: float,
        phase: str,
        distance_to_touchdown_ft: float,
    ) -> tuple[float, float]:
        bank_error = desired_bank_deg - roll_deg

        if phase == "approach":
            if distance_to_touchdown_ft > 250.0:
                aileron = 0.048 * bank_error
                rudder = 0.015 * heading_error_deg - 0.0038 * roll_deg - 0.0010 * cross_track_ft
            else:
                aileron = 0.055 * bank_error
                rudder = 0.017 * heading_error_deg - 0.0040 * roll_deg - 0.0012 * cross_track_ft
        elif phase == "flare":
            aileron = 0.040 * bank_error
            rudder = 0.012 * heading_error_deg - 0.0035 * roll_deg - 0.0008 * cross_track_ft
        else:
            aileron = 0.030 * bank_error
            rudder = 0.010 * heading_error_deg - 0.0030 * roll_deg

        return aileron, rudder

    @staticmethod
    def _lateral_control_abort(
        desired_bank_deg: float,
        roll_deg: float,
        heading_error_deg: float,
        cross_track_ft: float,
    ) -> tuple[float, float]:
        bank_error = desired_bank_deg - roll_deg
        aileron = 0.050 * bank_error
        rudder = 0.016 * heading_error_deg - 0.004 * roll_deg - 0.0012 * cross_track_ft
        return aileron, rudder

    @staticmethod
    def _wrap_deg(angle: float) -> float:
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))