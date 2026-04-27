from interfaces import *
from pynput import keyboard
from colorama import just_fix_windows_console, Fore, Style
import numpy as np

just_fix_windows_console()

class TUI(GUI):
    """终端用户界面 - 汉化版"""
    
    # 重置按键
    RESET_KEYS = [keyboard.KeyCode.from_char('R'), keyboard.KeyCode.from_char('r')]
    
    # 视角移动按键
    MOVE_KEYS = {
        keyboard.Key.down: [-1, 0, 0],
        keyboard.Key.up: [1, 0, 0],
        keyboard.Key.left: [0, 0, -1],
        keyboard.Key.right: [0, 0, 1]
    }
    MOVE_SCALE = 0.5
    
    # 手部移动按键
    HAND_MOVE_KEYS = {
        keyboard.KeyCode.from_char('w'): [1, 0, 0],  # 前进
        keyboard.KeyCode.from_char('s'): [-1, 0, 0], # 后退
        keyboard.KeyCode.from_char('a'): [0, 1, 0],  # 左移
        keyboard.KeyCode.from_char('d'): [0, -1, 0], # 右移
        keyboard.KeyCode.from_char('q'): [0, 0, 1],  # 上升
        keyboard.KeyCode.from_char('e'): [0, 0, -1], # 下降
    }
    HAND_MOVE_SCALE = 0.01
    
    def _key_press(self, key):
        """处理按键事件"""
        if key == keyboard.Key.esc:
            self._esc_pressed = True

        elif key in TUI.RESET_KEYS:
            print(f"{Fore.YELLOW}正在重置仿真...{Fore.RESET}")
            self._engine.reset_simulation()

        elif key in TUI.MOVE_KEYS:
            self._offset_origin += TUI.MOVE_KEYS[key]
            self._update_position()

        elif key == keyboard.Key.space:
            self._offset_origin = np.zeros(3)
            self._update_position()

        # 手部控制切换：1=左手(id=0)，2=右手(id=1)
        elif key == keyboard.KeyCode.from_char('1'):
            self._active_hand_id = 0
            print(f"{Fore.CYAN}切换：控制左手 (id=0){Fore.RESET}")

        elif key == keyboard.KeyCode.from_char('2'):
            self._active_hand_id = 1
            print(f"{Fore.CYAN}切换：控制右手 (id=1){Fore.RESET}")

        elif key in TUI.HAND_MOVE_KEYS:
            if hasattr(self._engine, "nudge_hand"):
                delta = np.asarray(TUI.HAND_MOVE_KEYS[key], dtype=float) * TUI.HAND_MOVE_SCALE
                self._engine.nudge_hand(self._active_hand_id, delta.tolist())
    
    def _update_position(self):
        """更新位置显示"""
        print(f"{Fore.CYAN}移动到{Style.BRIGHT}", self._offset_origin, Style.RESET_ALL)
        self._visualizer.offset_origin(self._offset_origin * TUI.MOVE_SCALE)

    def start_gui(self, engine: Engine, visualizer: Visualizer):
        """启动图形界面"""
        self._engine = engine
        self._visualizer = visualizer

        self._esc_pressed = False
        self._kb_listener = keyboard.Listener(on_press=self._key_press)
        self._kb_listener.start()

        self._offset_origin = np.zeros(3)

        # 默认控制右手（id=1）
        self._active_hand_id = 1

        # 打印中文操作说明
        print(Fore.WHITE)
        print("=" * 60)
        print(f"{Style.BRIGHT}           可变形物体触觉仿真系统 - 控制面板{Style.NORMAL}")
        print("=" * 60)
        print()
        print(f"  {Style.BRIGHT}ESC{Style.NORMAL}      - 停止仿真并退出")
        print(f"  {Style.BRIGHT}R{Style.NORMAL}        - 重置仿真")
        print(f"  {Style.BRIGHT}方向键 ↑↓←→{Style.NORMAL} - 移动视角")
        print(f"  {Style.BRIGHT}空格键{Style.NORMAL}   - 重置视角位置")
        print()
        print(f"  {Fore.CYAN}手部控制:{Fore.WHITE} (按 1=左手 / 2=右手，当前默认右手)")
        print(f"    {Style.BRIGHT}W/S{Style.NORMAL} - 前进/后退")
        print(f"    {Style.BRIGHT}A/D{Style.NORMAL} - 左移/右移")
        print(f"    {Style.BRIGHT}Q/E{Style.NORMAL} - 上升/下降")
        print()
        print("=" * 60)
        print(Style.RESET_ALL)

    def should_exit(self) -> bool:
        """检查是否应该退出"""
        return self._esc_pressed
    
    def stop_gui(self):
        """停止图形界面"""
        self._kb_listener.stop()
        print(f"{Fore.GREEN}仿真已结束，感谢使用！{Style.RESET_ALL}")
