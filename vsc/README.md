# HUTB 人车模拟器开发助手

面向人车模拟器的快速开发验证 VSCode 插件系统，为 HUTB 仿真框架和 MCP 通信协议提供一站式开发工具。

## 功能特性

### 1. 代码智能补全
- 输入 `hutb.` 或 `mcp.` 自动弹出 API 函数补全列表
- 显示函数签名、参数说明、返回值类型和使用示例
- 自动插入必填参数占位符
- 已弃用 API 添加删除线标记并提示替代方案

### 2. 语法高亮
- HUTB/MCP 模块导入语句高亮
- 核心函数名称语法着色
- 已弃用函数特殊颜色标记
- 特定常量和枚举值高亮

### 3. 错误检查
- **未知函数检测**：拼写错误自动提示相似函数名
- **参数数量检查**：必填参数缺失或参数过多时给出提示
- **已弃用 API 警告**：使用弃用函数时显示警告并建议替代

### 4. 一键打包开发环境
- 将 VSCode 便携版 + 插件 + Python 环境 + 依赖库打包为 ZIP
- 解压后双击 `start.bat` 即可启动完整开发环境，零配置
- 自动生成启动脚本和 VSCode 配置

### 5. 联动调试
- 一键启动 Python 调试器，自动配置调试环境
- 内置多种调试模板：单脚本调试、多机通信调试、远程附加调试
- 自动生成/更新 `launch.json` 配置

### 6. 自测试与发布
- 基于 `@vscode/test-electron` 的自动化测试
- 支持一键打包 `.vsix` 安装文件
- 支持一键发布到 VSCode 插件商城

## 快速开始

### 安装
1. 从 VSCode 插件商城搜索 "HUTB 人车模拟器" 安装
2. 或下载 `.vsix` 文件手动安装：`code --install-extension hutb-simulator-dev-0.1.0.vsix`

### 使用
1. 打开包含 Python 文件的工作区
2. 在 Python 文件中输入 `hutb.` 或 `mcp.` 即可获得智能补全
3. 按 `Ctrl+Shift+P` 打开命令面板，输入 "HUTB" 查看所有可用命令

### 可用命令
| 命令 | 说明 |
|------|------|
| `HUTB: 一键打包开发环境` | 将完整开发环境打包为 ZIP |
| `HUTB: 启动调试` | 自动配置并启动 Python 调试器 |
| `HUTB: 选择调试模板` | 选择预设的调试配置模板 |
| `HUTB: 运行测试` | 执行插件自动化测试 |
| `HUTB: 打包插件` | 打包为 .vsix 安装文件 |
| `HUTB: 发布插件` | 发布到 VSCode 插件商城 |

## 配置项

在 VSCode 设置中搜索 "hutb" 可配置以下选项：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `hutb.pythonPath` | Python 解释器路径 | 自动检测 |
| `hutb.sdkVersion` | HUTB SDK 版本号 | 空 |
| `hutb.packConfig.vscodePath` | VSCode 便携版路径 | 空 |
| `hutb.packConfig.pythonEnvPath` | Python 环境路径 | 空 |
| `hutb.packConfig.outputDir` | 打包输出目录 | 空 |

## 项目结构

```
├── data/                          # 数据文件
│   ├── api-definitions.json       # HUTB/MCP API 定义
│   ├── debug-templates.json       # 调试配置模板
│   └── pack-config.json           # 打包配置
├── samples/                       # 示例代码
│   ├── example_simulation.py      # 基础仿真示例
│   └── mcp_communication.py       # MCP 通信示例
├── src/                           # 源代码
│   ├── extension.ts               # 插件入口
│   ├── modules/                   # 功能模块
│   │   ├── completion.ts          # 代码补全
│   │   ├── diagnostics.ts         # 错误检查
│   │   ├── packager.ts            # 环境打包
│   │   ├── debugger.ts            # 联动调试
│   │   └── statusBar.ts           # 状态栏
│   ├── utils/                     # 工具类
│   │   └── apiLoader.ts           # API 定义加载器
│   └── test/                      # 测试文件
│       ├── runTest.ts             # 测试入口
│       └── suite/                 # 测试套件
│           ├── index.ts
│           ├── completion.test.ts
│           ├── diagnostics.test.ts
│           ├── debugConfig.test.ts
│           └── packaging.test.ts
├── syntaxes/                      # 语法定义
│   └── hutb.tmLanguage.json       # TextMate 语法高亮规则
├── package.json                   # 插件清单
├── tsconfig.json                  # TypeScript 配置
└── CHANGELOG.md                   # 更新日志
```

## 开发

### 环境要求
- Node.js 18.x 或 20.x
- VSCode 1.80.0+
- Python 3.9+

### 构建与调试
```bash
# 安装依赖
npm install

# 编译
npm run compile

# 监听模式
npm run watch

# 运行测试
npm test

# 打包 .vsix
npm run package

# 发布
npm run publish
```

### 调试插件
1. 在 VSCode 中打开项目
2. 按 `F5` 启动扩展开发宿主
3. 在新窗口中打开 Python 文件测试插件功能

## 技术栈
- **VSCode Extension API** - 插件开发框架
- **TypeScript 5.x** - 核心开发语言
- **Node.js** - 运行环境与工具链
- **TextMate Grammar** - 语法高亮规则
- **@vscode/test-electron** - 自动化测试框架
- **archiver** - ZIP 压缩

## 许可证

MIT

## 仓库地址

[https://github.com/OpenHUTB/vscode-ext](https://github.com/OpenHUTB/vscode-ext)
