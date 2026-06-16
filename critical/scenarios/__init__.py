# scenarios/__init__.py
# 场景模块统一入口 —— 10 种独立危险场景

from .base_scenario import BaseScenario

# 极端天气类 (3)
from .rain_storm import RainStormScenario
from .heavy_fog import HeavyFogScenario
from .tunnel_night import TunnelNightScenario

# 车辆对抗类 (2)
from .emergency_brake import EmergencyBrakeScenario
from .cut_in_scenario import CutInScenario

# 行人危险类 (3)
from .pedestrian_cross import PedestrianCrossScenario
from .ghost_peek import GhostPeekScenario
from .jaywalking import JaywalkingScenario

# 多因素耦合类 (2)
from .combined_night_pedestrian import NightPedestrianScenario
from .combined_fog_ghost import FogGhostScenario

SCENARIO_REGISTRY = {
    "rain_storm": RainStormScenario,
    "heavy_fog": HeavyFogScenario,
    "tunnel_night": TunnelNightScenario,
    "emergency_brake": EmergencyBrakeScenario,
    "cut_in": CutInScenario,
    "pedestrian_cross": PedestrianCrossScenario,
    "ghost_peek": GhostPeekScenario,
    "jaywalking": JaywalkingScenario,
    "night_pedestrian": NightPedestrianScenario,
    "fog_ghost": FogGhostScenario,
}


def create_scenario(name):
    cls = SCENARIO_REGISTRY.get(name)
    if cls is None:
        raise KeyError("未知场景: %s，可用: %s" % (name, list(SCENARIO_REGISTRY.keys())))
    return cls()
