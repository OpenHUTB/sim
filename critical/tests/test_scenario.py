# tests/test_scenario.py
# 测试 10 种场景加载和初始化

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios import SCENARIO_REGISTRY


def test_load(name):
    print("  >>> %s" % name)
    try:
        cls = SCENARIO_REGISTRY[name]; s = cls()
        assert s.name and s.weather and s.ego_speed_ms > 0
        assert s.category in ("extreme_weather", "vehicle_adversarial", "pedestrian_danger", "multi_factor_coupled")
        print("    PASS: %s category=%s speed=%.1f km/h" % (s.name, s.category, s.ego_speed_ms * 3.6))
        return True
    except Exception as e:
        print("    FAIL: %s" % e); return False


def run_all(quick=False):
    print("=" * 50 + "\n  场景测试 (%d)\n" % len(SCENARIO_REGISTRY) + "=" * 50)
    results = {}
    for name in SCENARIO_REGISTRY:
        results[name] = test_load(name)
        if not quick:
            try:
                s = SCENARIO_REGISTRY[name]()
                s.setup()
                assert s.ego_vehicle is not None
                s.cleanup()
                results["%s_lifecycle" % name] = True
            except Exception as e:
                print("    FAIL lifecycle: %s" % e)
                results["%s_lifecycle" % name] = False
    passed = sum(1 for v in results.values() if v)
    print("\n结果: %d/%d 通过" % (passed, len(results)))
    for k, v in results.items():
        if not v: print("  失败: %s" % k)
    return passed == len(results)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=str)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.scenario:
        test_load(args.scenario)
    else:
        run_all(quick=args.quick)
