# tests/test_carla_connection.py
# 测试 CARLA 连接、地图加载、车辆生成

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.carla_config import CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT
from utils.carla_utils import connect_to_carla, enable_sync_mode, disable_sync_mode, spawn_ego_vehicle, destroy_actors


def test_connection():
    print(">>> 测试 CARLA 连接...")
    try:
        client, world = connect_to_carla()
        print("  PASS: %s:%s map=%s" % (CARLA_HOST, CARLA_PORT, world.get_map().name))
        return client, world
    except Exception as e:
        print("  FAIL: %s" % e)
        return None, None


def test_sync_mode(world):
    print(">>> 测试同步模式...")
    try:
        enable_sync_mode(world, fps=20)
        assert world.get_settings().synchronous_mode
        print("  PASS: 同步开启")
        disable_sync_mode(world)
        print("  PASS: 同步关闭")
        return True
    except Exception as e:
        print("  FAIL: %s" % e); return False


def test_spawn(world):
    print(">>> 测试车辆生成...")
    try:
        enable_sync_mode(world, 20)
        v = spawn_ego_vehicle(world)
        for _ in range(5): world.tick()
        print("  PASS: %s" % v.type_id)
        destroy_actors([v]); disable_sync_mode(world)
        return True
    except Exception as e:
        print("  FAIL: %s" % e); return False


def run_all():
    print("=" * 50 + "\n  CARLA 连接测试\n" + "=" * 50)
    results = {}
    client, world = test_connection()
    results["connection"] = client is not None
    if world:
        results["sync"] = test_sync_mode(world)
        results["spawn"] = test_spawn(world)
    passed = sum(1 for v in results.values() if v)
    print("\n结果: %d/%d 通过" % (passed, len(results)))
    for k, v in results.items(): print("  %s: %s" % (k, "PASS" if v else "FAIL"))
    return passed == len(results)


if __name__ == "__main__":
    import sys; sys.exit(0 if run_all() else 1)
