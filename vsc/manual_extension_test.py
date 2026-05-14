import hutb
import mcp

def try_completion() -> None:
    # 在下一行手动输入 hutb. 观察补全
    # 在下一行手动输入 mcp. 观察补全
    _ = hutb
    _ = mcp
mcp
def try_diagnostics() -> None:
    # 保存前，可取消下面三行注释来测试诊断能力
     hutb.init_simulater()
     hutb.create_vehicle()
     mcp.unknown_func()
    pass
