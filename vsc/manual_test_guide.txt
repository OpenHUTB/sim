HUTB VS Code 插件手动测试说明
================================

一、先启动扩展开发宿主
1. 在当前工程里按 F5，选择 "Run Extension"。
2. 会打开一个新的 VS Code 窗口，这个窗口就是扩展开发宿主。

二、测试补全
1. 在扩展开发宿主里打开本工程目录。
2. 打开文件 测试/manual_extension_test.py。
3. 把光标放到 "hutb." 或 "mcp." 后面，确认是否弹出函数补全列表。

建议测试代码：

import hutb
import mcp

hutb.
mcp.

三、测试诊断报错
1. 在 测试/manual_extension_test.py 中输入下面代码。
2. 保存文件，确认是否出现红线或告警。

诊断测试代码：

import hutb
import mcp

hutb.init_simulater()
hutb.create_vehicle()
mcp.unknown_func()

预期结果：
- hutb.init_simulater() 应提示函数名可能拼错
- hutb.create_vehicle() 应提示缺少必填参数
- mcp.unknown_func() 应提示未知函数

四、测试命令面板
1. 按 Ctrl+Shift+P。
2. 搜索 HUTB。
3. 依次查看这些命令是否存在：
- HUTB: 一键打包开发环境
- HUTB: 启动调试
- HUTB: 选择调试模板
- HUTB: 运行测试
- HUTB: 打包插件
- HUTB: 发布插件

五、测试调试功能
1. 打开 测试/write_txt_demo.py。
2. 按 Ctrl+Shift+P，执行 "HUTB: 启动调试"。
3. 脚本运行后，应在 测试 目录下生成 manual_test_output.txt。

六、测试语法高亮
1. 在 Python 文件中输入 import hutb 和 import mcp。
2. 确认相关调用和导入没有异常着色问题。

七、如果没看到最新效果
1. 在主工程窗口执行 "Developer: Reload Window"。
2. 在扩展开发宿主窗口重新打开测试文件。
3. 如有需要，执行 "TypeScript: Restart TS Server" 和 "Python: Restart Language Server"。
