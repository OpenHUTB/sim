# tests/__init__.py
# 测试模块入口

from .test_carla_connection import run_all as test_carla
from .test_scenario import run_all as test_scenarios
from .test_agent import run_all as test_agents
