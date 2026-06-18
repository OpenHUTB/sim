from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import jsbsim

KTS_TO_FPS = 1.6878098571


@dataclass
class SimConfig:
    model_name: str = "c172x"
    init_agl_ft: float = 120.0
    init_vc_kts: float = 72.0
    init_roc_fpm: float = -250.0
    init_heading_deg: float = 0.0
    init_cross_track_ft: float = 0.0
    runway_distance_ft: float = 2200.0
    runway_heading_deg: float = 0.0

    terrain_type: str = "flat"
    terrain_base_ft: float = 0.0
    terrain_amp_ft: float = 25.0
    terrain_wavelength_ft: float = 1400.0
    terrain_bump_ft: float = 40.0
    terrain_bump_center_ft: float = 1800.0
    terrain_bump_width_ft: float = 350.0

    dt_s: float = 1.0 / 60.0


class TerrainProfile:
    def __init__(self, config: SimConfig):
        self.config = config

    def elevation_ft(self, x_ft: float) -> float:
        c = self.config
        base = c.terrain_base_ft
        amp = max(c.terrain_amp_ft, 0.0)
        wl = max(c.terrain_wavelength_ft, 1.0)

        if c.terrain_type == "flat":
            return base
        if c.terrain_type == "rolling":
            return base + amp * math.sin(2.0 * math.pi * x_ft / wl)
        if c.terrain_type == "ridge":
            z = (x_ft - c.terrain_bump_center_ft) / max(c.terrain_bump_width_ft, 1.0)
            return base + c.terrain_bump_ft * math.exp(-(z * z))
        if c.terrain_type == "valley":
            z = (x_ft - c.terrain_bump_center_ft) / max(c.terrain_bump_width_ft, 1.0)
            return base - c.terrain_bump_ft * math.exp(-(z * z))
        if c.terrain_type == "steps":
            step = 250.0
            return base + amp * (0.5 if int(x_ft // step) % 2 else -0.5)

        rolling = 0.55 * amp * math.sin(2.0 * math.pi * x_ft / wl)
        short_wave = 0.25 * amp * math.sin(2.0 * math.pi * x_ft / max(wl * 0.35, 1.0) + 0.6)
        z = (x_ft - c.terrain_bump_center_ft) / max(c.terrain_bump_width_ft, 1.0)
        bump = c.terrain_bump_ft * math.exp(-(z * z))
        return base + rolling + short_wave + bump

    def sample_profile(self, x_end_ft: float = 4000.0, step_ft: float = 40.0) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        x = 0.0
        while x <= x_end_ft:
            pts.append((x, self.elevation_ft(x)))
            x += step_ft
        return pts


class LandingSim:
    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config or SimConfig()
        self.terrain = TerrainProfile(self.config)
        self.distance_ft = 0.0
        self.cross_track_ft = self.config.init_cross_track_ft
        self.commanded = {
            "throttle": 0.0,
            "elevator": 0.0,
            "aileron": 0.0,
            "rudder": 0.0,
            "flaps": 0.0,
        }
        self._build_fdm()

    def _build_fdm(self) -> None:
        c = self.config
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_debug_level(0)
        self.fdm.set_dt(c.dt_s)

        ok = self.fdm.load_model(c.model_name)
        if not ok:
            raise RuntimeError(f"Failed to load JSBSim model: {c.model_name}")

        terrain0 = self.terrain.elevation_ft(0.0)
        init_h_sl_ft = terrain0 + c.init_agl_ft

        self.fdm["ic/h-sl-ft"] = init_h_sl_ft
        self.fdm["ic/vc-kts"] = c.init_vc_kts
        self.fdm["ic/roc-fpm"] = c.init_roc_fpm
        self.fdm["ic/psi-true-deg"] = c.init_heading_deg
        self.fdm["ic/lat-gc-deg"] = 0.0
        self.fdm["ic/long-gc-deg"] = 0.0
        self.fdm["position/terrain-elevation-asl-ft"] = terrain0

        if not self.fdm.run_ic():
            raise RuntimeError("JSBSim run_ic() failed")

        self.fdm["propulsion/set-running"] = -1
        try:
            self.fdm["simulation/do_simple_trim"] = 1
        except Exception:
            pass

        self.distance_ft = 0.0
        self.cross_track_ft = c.init_cross_track_ft

    def reset(self, config: SimConfig | None = None) -> None:
        if config is not None:
            self.config = config
            self.terrain = TerrainProfile(self.config)
        self.distance_ft = 0.0
        self.cross_track_ft = self.config.init_cross_track_ft
        self._build_fdm()

    def set_controls(
        self,
        throttle: float,
        elevator: float,
        aileron: float = 0.0,
        rudder: float = 0.0,
        flaps: float = 0.0,
    ) -> None:
        throttle = self._clamp(throttle, 0.0, 1.0)
        elevator = self._clamp(elevator, -1.0, 1.0)
        aileron = self._clamp(aileron, -1.0, 1.0)
        rudder = self._clamp(rudder, -1.0, 1.0)
        flaps = self._clamp(flaps, 0.0, 1.0)

        self.commanded.update(
            {
                "throttle": throttle,
                "elevator": elevator,
                "aileron": aileron,
                "rudder": rudder,
                "flaps": flaps,
            }
        )

        self.fdm["fcs/throttle-cmd-norm[0]"] = throttle
        self.fdm["fcs/elevator-cmd-norm"] = elevator
        self.fdm["fcs/aileron-cmd-norm"] = aileron
        self.fdm["fcs/rudder-cmd-norm"] = rudder
        self.fdm["fcs/flap-cmd-norm"] = flaps

    def step(self) -> bool:
        dt = float(self.fdm.get_delta_t())
        pitch_rad = float(self.fdm["attitude/pitch-rad"])
        yaw_deg = math.degrees(float(self.fdm["attitude/psi-rad"]))
        speed_fps = max(0.0, float(self.fdm["velocities/vc-kts"]) * KTS_TO_FPS * math.cos(pitch_rad))
        self.distance_ft += speed_fps * dt

        track_err_rad = math.radians(self._wrap_deg(yaw_deg - self.config.runway_heading_deg))
        self.cross_track_ft += speed_fps * math.sin(track_err_rad) * dt

        terrain = self.terrain.elevation_ft(self.distance_ft)
        self.fdm["position/terrain-elevation-asl-ft"] = terrain
        return bool(self.fdm.run())

    def state(self) -> Dict[str, float | bool | str]:
        terrain_ft = float(self.fdm["position/terrain-elevation-asl-ft"])
        h_sl_ft = float(self.fdm["position/h-sl-ft"])
        h_agl_ft = h_sl_ft - terrain_ft
        runway_distance_ft = self.config.runway_distance_ft - self.distance_ft

        return {
            "t": float(self.fdm.get_sim_time()),
            "dt": float(self.fdm.get_delta_t()),
            "distance_ft": float(self.distance_ft),
            "distance_to_runway_ft": float(runway_distance_ft),
            "cross_track_ft": float(self.cross_track_ft),
            "terrain_type": self.config.terrain_type,
            "terrain_elev_ft": terrain_ft,
            "h_agl_ft": h_agl_ft,
            "h_sl_ft": h_sl_ft,
            "vc_kts": float(self.fdm["velocities/vc-kts"]),
            "vs_fpm": float(self.fdm["velocities/h-dot-fps"]) * 60.0,
            "roll_deg": math.degrees(float(self.fdm["attitude/roll-rad"])),
            "pitch_deg": math.degrees(float(self.fdm["attitude/pitch-rad"])),
            "yaw_deg": math.degrees(float(self.fdm["attitude/psi-rad"])),
            "wow": bool(self.fdm["gear/wow"] > 0.5),
            "runway_heading_deg": float(self.config.runway_heading_deg),
            "cmd_throttle": self.commanded["throttle"],
            "cmd_elevator": self.commanded["elevator"],
            "cmd_aileron": self.commanded["aileron"],
            "cmd_rudder": self.commanded["rudder"],
            "cmd_flaps": self.commanded["flaps"],
        }

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