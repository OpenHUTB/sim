# main.py
# 毕设项目统一入口：菜单式启动训练 / 评估 / 生成场景 / 对比 / 测试
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scenarios import SCENARIO_REGISTRY, create_scenario
from rl_algorithms import ALGORITHM_REGISTRY
from osc_exporter import export_scenario


def menu():
    while True:
        print("\n" + "=" * 55)
        print("   基于 CARLA 的极端驾驶仿真场景生成系统")
        print("=" * 55)
        print("  [1] 训练模型    [2] 评估模型    [3] 算法对比")
        print("  [4] 手动运行    [5] 导出.xosc   [6] 图表")
        print("  [7] 运行测试    [0] 退出")
        print("-" * 55)
        c = input("请输入选项: ").strip()
        if c == "0": print("退出"); break
        elif c == "1": _menu_train()
        elif c == "2": _menu_evaluate()
        elif c == "3": _menu_compare()
        elif c == "4": _menu_run_scenario()
        elif c == "5": _menu_export_xosc()
        elif c == "6": _menu_plot()
        elif c == "7": _menu_test()
        else: print("无效选项")


def _menu_train():
    print("\n--- 训练 ---")
    for i, n in enumerate(ALGORITHM_REGISTRY, 1): print("  %d. %s" % (i, n))
    try: algo_n = list(ALGORITHM_REGISTRY.keys())[int(input("算法: ").strip()) - 1]
    except: print("无效"); return
    for i, n in enumerate(SCENARIO_REGISTRY, 1): print("  %d. %s" % (i, n))
    sc = input("场景 (1-%d): " % len(SCENARIO_REGISTRY)).strip()
    ep = int(input("集数 (默认500): ") or "500")
    scenarios = list(SCENARIO_REGISTRY.keys()) if sc.lower() == "a" else \
        [list(SCENARIO_REGISTRY.keys())[int(sc) - 1]]
    runners = {"dqn": "experiments.run_dqn", "attention_dqn": "experiments.run_attention_dqn",
               "ppo": "experiments.run_ppo", "smooth_ppo": "experiments.run_smooth_ppo"}
    import importlib
    m = importlib.import_module(runners[algo_n])
    for s in scenarios: print("\n>>> %s x %s" % (algo_n, s)); m.run(s, ep)


def _menu_evaluate():
    from experiments.evaluate import evaluate
    evaluate(input("算法: ").strip(), input("场景: ").strip(), input("模型路径: ").strip(),
             int(input("集数 (10): ") or "10"))


def _menu_compare():
    from experiments.compare import compare
    p = input("对比对 (dqn/ppo): ").strip()
    pair = ("DQN", "Attention-DQN") if p == "dqn" else ("PPO", "Smooth-PPO")
    sc = input("场景: ").strip()
    compare(pair, sc, {pair[0]: input("%s CSV: " % pair[0]).strip(),
                       pair[1]: input("%s CSV: " % pair[1]).strip()},
            {pair[0]: input("%s JSON: " % pair[0]).strip(),
             pair[1]: input("%s JSON: " % pair[1]).strip()})


def _menu_run_scenario():
    for i, n in enumerate(SCENARIO_REGISTRY, 1): print("  %d. %s" % (i, n))
    try: sc = list(SCENARIO_REGISTRY.keys())[int(input("场景: ").strip()) - 1]
    except: print("无效"); return
    s = create_scenario(sc)
    s.setup()
    try: s.run()
    finally: s.cleanup()


def _menu_export_xosc():
    for i, n in enumerate(SCENARIO_REGISTRY, 1): print("  %d. %s" % (i, n))
    try: sc = list(SCENARIO_REGISTRY.keys())[int(input("场景: ").strip()) - 1]
    except: print("无效"); return
    s = create_scenario(sc)
    s.setup()
    try: print("导出: %s" % export_scenario(s, input("输出目录 (results/scenarios): ") or "results/scenarios"))
    finally: s.cleanup()


def _menu_plot():
    print("\n  [1]训练曲线 [2]对比图 [3]俯视图")
    c = input("选择: ").strip()
    if c == "1":
        from visualization import plot_reward_curve
        plot_reward_curve([input("CSV路径: ").strip()], title=input("标题: ").strip(),
                          save_path=input("保存路径: ").strip() or None)
    elif c == "2":
        import json
        p = input("JSON路径: ").strip()
        if os.path.exists(p):
            from visualization import plot_bar_comparison, plot_radar_chart
            d = json.load(open(p)); plot_bar_comparison(d.get("metrics", {}), save_path="results/plots/bar.png")
            plot_radar_chart(d.get("radar", {}), save_path="results/plots/radar.png")
    elif c == "3":
        sc = create_scenario(input("场景: ").strip())
        sc.setup()
        from visualization import plot_scenario_topview
        try: plot_scenario_topview(sc, "results/plots/%s_topview.png" % sc.name)
        finally: sc.cleanup()


def _menu_test():
    print("\n  [1]连接 [2]场景 [3]智能体 [4]全部")
    c = input(": ").strip()
    if c == "1": from tests.test_carla_connection import run_all; run_all()
    elif c == "2": from tests.test_scenario import run_all; run_all(quick=True)
    elif c == "3": from tests.test_agent import run_all; run_all()
    elif c == "4":
        from tests.test_carla_connection import run_all as t1
        from tests.test_scenario import run_all as t2
        from tests.test_agent import run_all as t3
        t1(); t2(quick=True); t3()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CARLA 极端驾驶仿真系统")
    sp = p.add_subparsers(dest="cmd")
    pt = sp.add_parser("train"); pt.add_argument("--algo", required=True); pt.add_argument("--scenario", required=True); pt.add_argument("--episodes", type=int, default=500)
    pe = sp.add_parser("evaluate"); pe.add_argument("--algo", required=True); pe.add_argument("--scenario", required=True); pe.add_argument("--model", required=True); pe.add_argument("--episodes", type=int, default=10)
    pc = sp.add_parser("compare"); pc.add_argument("--pair", required=True, choices=["dqn","ppo"]); pc.add_argument("--scenario", required=True)
    px = sp.add_parser("export"); px.add_argument("--scenario", required=True); px.add_argument("--output", default="results/scenarios")
    pt2 = sp.add_parser("test"); pt2.add_argument("--suite", choices=["carla","scenario","agent","all"], default="all")
    args = p.parse_args()
    if args.cmd is None: menu()
    elif args.cmd == "train":
        import importlib
        mods = {"dqn":"experiments.run_dqn","attention_dqn":"experiments.run_attention_dqn",
                "ppo":"experiments.run_ppo","smooth_ppo":"experiments.run_smooth_ppo"}
        importlib.import_module(mods[args.algo]).run(args.scenario, args.episodes)
    elif args.cmd == "evaluate": from experiments.evaluate import evaluate; evaluate(args.algo, args.scenario, args.model, args.episodes)
    elif args.cmd == "compare": from experiments.compare import compare; pair = ("DQN","Attention-DQN") if args.pair=="dqn" else ("PPO","Smooth-PPO"); compare(pair, args.scenario, {}, {})
    elif args.cmd == "export": s = create_scenario(args.scenario); s.setup(); print("导出: %s" % export_scenario(s, args.output)); s.cleanup()
    elif args.cmd == "test":
        if args.suite == "carla": from tests.test_carla_connection import run_all as t; t()
        elif args.suite == "scenario": from tests.test_scenario import run_all as t; t(quick=True)
        elif args.suite == "agent": from tests.test_agent import run_all as t; t()
        else:
            from tests.test_carla_connection import run_all as t1
            from tests.test_scenario import run_all as t2
            from tests.test_agent import run_all as t3
            t1(); t2(quick=True); t3()