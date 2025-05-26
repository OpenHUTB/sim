import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import sys
import time
import importlib.util

sys.path.append(r'A:\carla\WindowsNoEditor_9.15\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg')


class ParameterEditor:
    def __init__(self, parent, parameters_module):
        self.frame = ttk.LabelFrame(parent, text="训练参数设置")
        self.params = {}

        # 动态加载parameters模块
        spec = importlib.util.spec_from_file_location("parameters", parameters_module)
        self.parameters = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.parameters)

        # 创建参数输入控件
        self.create_parameter_controls()

    def create_parameter_controls(self):
        # 公共参数
        self.add_parameter("GAMMA", "折扣因子", float, self.parameters.GAMMA)
        self.add_parameter("BATCH_SIZE", "批次大小", int, self.parameters.BATCH_SIZE)

        # 算法特定参数
        self.add_parameter("PPO_LEARNING_RATE", "PPO学习率", float, self.parameters.PPO_LEARNING_RATE)
        self.add_parameter("DQN_LEARNING_RATE", "DQN学习率", float, self.parameters.DQN_LEARNING_RATE)
        self.add_parameter("EPISODES", "训练回合数", int, self.parameters.EPISODES)
        self.add_parameter("EPSILON", "初始探索率", float, self.parameters.EPSILON)

    def add_parameter(self, name, label, dtype, default):
        row = ttk.Frame(self.frame)
        row.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
        var = tk.StringVar(value=str(default))
        entry = ttk.Entry(row, textvariable=var, width=12)
        entry.pack(side=tk.RIGHT)

        self.params[name] = (var, dtype)

    def get_parameters(self):
        params = {}
        for name, (var, dtype) in self.params.items():
            try:
                params[name] = dtype(var.get())
            except ValueError:
                params[name] = None  # 保持原参数值
        return params


class CarlaRLInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("CARLA 强化学习控制面板")
        self.root.geometry("800x700")

        # 初始化运行状态
        self.is_running = False
        self.worker_thread = None

        # 初始化变量
        self.algorithm_var = tk.StringVar(value="PPO")
        self.model_var = tk.StringVar()
        self.town_var = tk.StringVar(value="Town10HD")
        self.mode_var = tk.StringVar(value="test")

        # 创建主容器
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 算法选择模块
        algo_frame = ttk.LabelFrame(main_frame, text="选择算法")
        algo_frame.pack(fill=tk.X, pady=5)

        algorithms = ["PPO", "DDQN"]
        self.algo_combobox = ttk.Combobox(algo_frame, textvariable=self.algorithm_var,
                                          values=algorithms, state="readonly")
        self.algo_combobox.pack(padx=5, fill=tk.X)
        self.algo_combobox.bind("<<ComboboxSelected>>", self.update_model_directory)

        # 参数编辑器
        self.param_editor = ParameterEditor(main_frame, "parameters.py")
        self.param_editor.frame.pack(fill=tk.X, pady=10)

        # 连接状态指示器
        self.connection_status = ttk.Label(main_frame, text="🔴 未连接", foreground="red")
        self.connection_status.pack(anchor=tk.NE)

        # 模型选择模块
        model_frame = ttk.LabelFrame(main_frame, text="模型选择")
        model_frame.pack(fill=tk.X, pady=5)

        ttk.Button(model_frame, text="浏览模型", command=self.browse_model).pack(side=tk.LEFT, padx=5)
        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_var, width=40)
        self.model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 地图选择模块
        town_frame = ttk.LabelFrame(main_frame, text="选择地图")
        town_frame.pack(fill=tk.X, pady=5)

        towns = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"]
        ttk.Combobox(town_frame, textvariable=self.town_var, values=towns, state="readonly").pack(padx=5, fill=tk.X)

        # 模式选择模块
        mode_frame = ttk.LabelFrame(main_frame, text="运行模式")
        mode_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(mode_frame, text="训练模式", variable=self.mode_var, value="train").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="测试模式", variable=self.mode_var, value="test").pack(anchor=tk.W)

        # 控制按钮
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=20)

        self.start_btn = ttk.Button(control_frame, text="开始运行", command=self.toggle_program)
        self.start_btn.pack(side=tk.LEFT, padx=10)
        ttk.Button(control_frame, text="退出", command=self.safe_exit).pack(side=tk.RIGHT, padx=10)

        # 日志区域
        log_frame = ttk.LabelFrame(main_frame, text="运行日志")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, height=10)
        self.log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def update_model_directory(self, event=None):
        initial_dir = os.path.join("checkpoints", self.algorithm_var.get(), self.town_var.get())
        self.model_var.set("")
        self.log(f"已切换算法到：{self.algorithm_var.get()}")

    def browse_model(self):
        initial_dir = os.path.join("checkpoints", self.algorithm_var.get(), self.town_var.get())
        if not os.path.exists(initial_dir):
            initial_dir = os.getcwd()

        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="选择模型文件",
            filetypes=[("Pickle files", "*.pickle"), ("All files", "*.*")]
        )
        if file_path:
            self.model_var.set(file_path)
            self.log(f"已选择模型：{os.path.basename(file_path)}")

    def toggle_program(self):
        if self.is_running:
            self.stop_program()
        else:
            self.start_program()

    def start_program(self):
        if self.mode_var.get() == "test" and not self.model_var.get():
            messagebox.showerror("错误", "测试模式需要选择模型文件！")
            return

        # 获取用户设置的参数
        params = self.param_editor.get_parameters()
        self.log("正在应用训练参数...")

        # 动态修改parameters模块
        import parameters
        for name, value in params.items():
            if value is not None and hasattr(parameters, name):
                setattr(parameters, name, value)
                self.log(f"参数已更新: {name} = {value}")

        self.is_running = True
        self.safe_ui_update(self.start_btn.config, {'text': "停止运行"})
        self.safe_ui_update(self.connection_status.config, {'text': "🟠 连接中...", 'foreground': "orange"})

        # 构造算法参数
        args = {
            "algorithm": self.algorithm_var.get(),
            "town": self.town_var.get(),
            "train": self.mode_var.get() == "train",
            "load_checkpoint": bool(self.model_var.get()),
            "model_path": self.model_var.get()
        }

        # 启动工作线程
        self.worker_thread = threading.Thread(target=self.run_carla_program, args=(args,))
        self.worker_thread.daemon = True
        self.worker_thread.start()

    # ...（保持其他方法不变，仅在execute_rl_program中添加参数应用）...
    def stop_program(self):
        if self.is_running:
            self.is_running = False
            # 主线程调用时才等待线程结束
            if threading.current_thread() == threading.main_thread():
                if self.worker_thread is not None and self.worker_thread.is_alive():
                    self.worker_thread.join()
            # 更新界面状态
            self.safe_ui_update(self.connection_status.config, {'text': "🔴 未连接", 'foreground': "red"})
            self.safe_ui_update(self.start_btn.config, {'text': "开始运行"})
            self.log("程序已停止")

    def safe_exit(self):
        if self.is_running:
            self.stop_program()
        self.root.destroy()

    def run_carla_program(self, args):
        try:
            # 模拟连接CARLA环境
            self.log("正在连接到CARLA...")
            time.sleep(2)  # 模拟延迟
            self.safe_ui_update(self.connection_status.config, {'text': "🟢 已连接", 'foreground': "green"})
            self.log("成功连接到CARLA环境")

            # 根据模式执行不同任务
            if args["train"]:
                self.log("进入训练模式")
            else:
                self.log("进入测试模式")

            # 示例执行强化学习程序
            self.execute_rl_program(args)

        except Exception as e:
            self.log(f"程序执行错误：{str(e)}")
            self.stop_program()

        finally:
            # 确保最终状态更新
            if self.is_running:
                self.stop_program()

    def execute_rl_program(self, args):
        try:
            # 动态导入对应的算法模块
            if args["algorithm"] == "PPO":
                from continuous_driver_ppo import runner
                sys.argv = [sys.argv[0]]
                sys.argv += ["--exp-name", "ppo"]
                sys.argv += ["--train", str(args["train"])]
                sys.argv += ["--town", args["town"]]
                if args["load_checkpoint"]:
                    sys.argv += ["--load-checkpoint", "True"]

            elif args["algorithm"] == "DDQN":
                from discrete_driver_ddqn import runner
                sys.argv = [sys.argv[0]]
                sys.argv += ["--exp-name", "ddqn"]
                sys.argv += ["--train", str(args["train"])]
                sys.argv += ["--town", args["town"]]
                if args["load_checkpoint"]:
                    sys.argv += ["--load-checkpoint", "True"]

            self.log("正在启动强化学习程序...")
            runner()

        except Exception as e:
            self.log(f"程序执行错误：{str(e)}")
        finally:
            self.log("已清理Carla世界")

    # ...（其余方法保持不变）...
    def log(self, message):
        def _log():
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
        self.root.after(0, _log)

    def safe_ui_update(self, update_method, kwargs):
        """线程安全地更新UI组件"""
        self.root.after(0, lambda: update_method(**kwargs))


if __name__ == "__main__":
    root = tk.Tk()
    app = CarlaRLInterface(root)
    root.protocol("WM_DELETE_WINDOW", app.safe_exit)
    root.mainloop()