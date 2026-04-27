# local code
from interfaces import *
from mujoco_connector import MujocoConnector, MujocoSimpleVisualizer
# 注意：OpenXR 依赖（xr/pyopenxr）可能未安装，需在用 openxr 可视化器时再按需导入。
# from mujoco_xr import MujocoXRVisualizer
from weart import WeartConnector, HAPTIC_FINGERS
from guis import TUI
from benchmarking import Benchmarker, Plotter
from hand import Hand

# libraries
from contextlib import nullcontext
from threading import Thread
from colorama import just_fix_windows_console, Fore, Style

just_fix_windows_console()

def simulation(engine: Engine,
                weart: WeartConnector | None,
                visualizer: Visualizer,
                hand_provider: HandPoseProvider | None,
                gui: GUI,
                hands: tuple[Hand, Hand]):
    print("正在启动仿真...")
    engine.start_simulation()

    # frame_bench = Benchmarker(title="Performance profiler")
    # perf_bench = Benchmarker(title="Performance profiler")
    # force_plot = Plotter(title="Applied force graph")

    def loop():
        print("正在启动可视化...")
        visualizer.start_visualization()
        gui.start_gui(engine, visualizer)
        # if isinstance(visualizer, MujocoXRVisualizer):
        #     visualizer.add_perf_counters(perf_bench, frame_bench)

        print(Style.BRIGHT + Fore.GREEN, f"完成！{Style.NORMAL} 系统已启动并运行中。{Fore.RESET}\n")
        try:
            while not visualizer.should_exit() and not gui.should_exit():
                # frame_bench.new_iteration()
                frame_continue, frame_duration = visualizer.wait_frame()
                # frame_bench.mark("Wait frame")
                # frame_bench.end_iteration()
                if visualizer.should_exit():
                    break
                if not frame_continue:
                    continue
                # perf_bench.new_iteration()
                # force_plot.new_iteration()

                if hand_provider is not None:
                    for hand in filter(lambda h: h.tracking, hands):
                        hand_pose = hand_provider.get_hand_pose(hand.id)
                        if hand_pose is not None:
                            engine.move_hand(hand.id, *hand_pose)

                if weart is not None:
                    for hand in filter(lambda h: h.haptics, hands):
                        for finger in HAPTIC_FINGERS:
                            closure, abduction = weart.get_finger_data(hand.id, finger)
                            engine.move_finger(hand.id, finger, closure, abduction)

                engine.step_simulation(frame_duration)
                # perf_bench.mark("Step simulation")

                visualizer.render_frame()
                # perf_bench.mark("Render")

                # 只有在启用 WEART 时才查询/打印接触力，否则会导致大量 stdout 输出让仿真看起来“卡住”。
                if weart is not None:
                    for hand in filter(lambda h: h.haptics, hands):
                        for finger in HAPTIC_FINGERS:
                            force, texture = engine.get_contact(hand.id, finger)
                            # perf_bench.mark("Contact force")
                            # force_plot.plot(force, f"{finger} hand {hand}")

                            print(force, texture)
                            weart.apply_finger(hand.id, finger, force, texture)
                            # perf_bench.mark("Apply force to finger")

                # force_plot.end_iteration()
                # perf_bench.end_iteration()
        except KeyboardInterrupt:
            pass # To exit gracefully. Even though we swallow the error, we still exit the loop.
        finally:
            # force_plot.stop()
            # perf_bench.stop()
            #perf_bench.export_csv("benchmark.csv", include_time=True)

            print("\n正在停止可视化...")
            gui.stop_gui()
            visualizer.stop_visualization()

            print("正在停止仿真...")
            engine.stop_simulation()

            print("\n再见！\n")

    threaded = False
    if threaded:
        t = Thread(target=loop)
        t.start()
        # we must run the loop in another thread because the graph can only be visualized in the main thread...
        #perf_bench.graph_viz(max_points=1000, use_time=True)
        #force_plot.graph_viz(max_points=10000, y_axis="Force")
        t.join()
    else:
        loop()

if __name__ == "__main__":
    # CHANGEABLE PARAMETERS

    used_engine = "mujoco"
    used_viz = "simple"
    use_weart = False
    used_gui = "tui"

    # 可用的场景文件：
    # scene_path = "assets/deformable_jelly_cube.xml"
    # scene_path = "assets/deformable_cloth.xml"
    # scene_path = "assets/deformable_sponge.xml"
    # scene_path = "assets/MuJoCo hands.xml"  # 双手模型
    # scene_path = "assets/MuJoCo phantom.xml"  # 器官模型（肝脏、肾脏等）
    scene_path = "assets/MuJoCo hands and phantom.xml"  # 双手+器官组合

    # It seems like we can keep tracking and haptics enabled
    # even if not both hands are connected (WEART and Oculus).
    # Disabling tracking and haptics for an unused hand is
    # still useful as it hides the hand in MuJoCo.
    # 
    # 手配置：
    # - simple 可视化器下默认没有 VR tracking 数据，仍可用键盘（TUI）通过 mocap 推手
    # - 未启用 WEART 时关闭 haptics，避免每帧打印大量 0.0 None 让仿真看起来“卡住”
    hands = (
        Hand(id=0, side="left", tracking=(used_viz == "openxr"), haptics=use_weart, controller_rotation=0),
        Hand(id=1, side="right", tracking=(used_viz == "openxr"), haptics=use_weart, controller_rotation=0),
    )

    # 脚本开始
    print("正在启动脚本...\n")

    engine = mujoco = visualizer = weart = hand = gui = None

    match used_engine:
        case "mujoco":
            print("正在加载 MuJoCo...")
            engine = mujoco = MujocoConnector(scene_path, hands)
            print("加载完成。\n")
        case "coppelia":
            from coppelia import CoppeliaConnector
            print("正在连接 CoppeliaSim...")
            engine = CoppeliaConnector()
            print("连接成功。\n")
            used_viz = None
        case _:
            raise RuntimeError("无效的引擎名称")

    match used_viz:
        case None:
            visualizer_ctx = nullcontext(Visualizer())
        case "simple":
            assert mujoco is not None
            visualizer_ctx = nullcontext(MujocoSimpleVisualizer(mujoco))
        case "openxr":
            assert mujoco is not None
            print("正在加载虚拟现实...")
            from mujoco_xr import MujocoXRVisualizer
            visualizer_ctx = hand = MujocoXRVisualizer(mujoco, mirror_window=True, samples=8, fps_counter=False)
        case _:
            raise RuntimeError("无效的可视化器名称")

    match used_gui:
        case "tui":
            gui = TUI()
        # TODO: 添加其他GUI，比如TK窗口
        case _:
            raise RuntimeError("无效的GUI名称")

    with visualizer_ctx as visualizer:
        print("可视化器已创建。\n")

        if use_weart:
            print("正在连接 WEART...")
            enabled_hands_haptic = [hand.id for hand in hands if hand.haptics]
            weart_ctx = WeartConnector(enabled_hands_haptic)
        else:
            weart_ctx = nullcontext()
            print(Fore.RED + Style.BRIGHT, "警告：", Style.NORMAL + "您尚未启用 WEART 设备。\n", Style.RESET_ALL)

        with weart_ctx as weart:
            # everything is initialized at this point
            
            simulation(engine, weart, visualizer, hand, gui, hands)
