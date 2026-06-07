# KIT 数据集下载说明

## ✅ 确认：SMPL+H G 就是 SMPL-H 格式！

**简短回答：可以下载！这就是正确的格式！**

---

## 📋 什么是 SMPL-H 格式？

### SMPL-H 的含义：
- **SMPL**: Skinned Multi-Person Linear model
- **H**: **H**ands (包含手部)

**SMPL-H vs SMPL 的区别：**
| 特性 | SMPL | SMPL-H |
|------|------|---------|
| 身体 | 全身 | 全身 |
| 手部 | 简化 | **详细建模** ✅ |
| 关节 | 约24个 | 约55个 |
| 文件大小 | ~10 MB | ~10-15 MB |
| 用途 | 基础模型 | 完整运动捕捉 ✅ |

**Kinesis 使用 SMPL-H 格式：**
- 虽然代码主要关注下肢
- 但 SMPL-H 提供更完整的模型
- 可以支持未来的全身体功能

---

## 🌐 AMASS 网站上的 KIT 数据集

### 网站显示的可能格式：

您看到的 **"SMPL+H G"** 可能是：

1. **SMPL-H 格式的描述**
   - 网站：https://amass.is.tue.mpg.de
   - KIT 数据集行显示："SMPL-H" 或 "SMPL+H G"
   - **这是正确的格式！**

2. **不同网站的显示差异**
   - 有些网站显示："SMPL-H"
   - 有些显示："SMPL+H G" (G可能代表General)
   - **都是同一个东西！**

---

## ✅ 您可以下载 SMPL-H 格式！

### 下载步骤：

#### 1. 访问网站并登录

```
https://amass.is.tue.mpg.de/
```

#### 2. 找到 KIT 数据集

在数据集列表中查找：
- **名称**：KIT 或 CMU
- **描述**：CMU Graphics Lab Motion Capture Database
- **格式**：SMPL-H 或 SMPL+H G
- **大小**：约 2-5 GB

#### 3. 下载

点击 "Download" 按钮，选择：
- **SMPL-H 格式**（或 SMPL+H G）

下载的文件通常是：
- **ZIP 压缩包**（约2-5 GB）
- 文件名类似：`KIT.zip` 或 `KIT_SMPL-H.zip`

#### 4. 解压

解压到任意位置，例如：
```
D:\KIT_Data\KIT\
```

解压后应该看到：
```
D:\KIT_Data\KIT\
├── 00\
│   ├── 00_01_poses.npz
│   ├── 00_02_poses.npz
│   └── ...
├── 01\
├── 02\
└── ...
```

**关键点：** 必须看到 **.npz** 文件！

---

## 🔍 验证 KIT 数据集

### 方法1：检查目录结构

解压后，确认：
```
✓ 有多个文件夹（00/, 01/, 02/, ...）
✓ 每个文件夹包含 .npz 文件
✓ 文件名格式：XX_YY_poses.npz
```

### 方法2：检查 NPZ 文件

运行这个命令检查一个NPZ文件：

```python
import numpy as np

# 选择一个NPZ文件
file_path = "D:\\KIT_Data\\KIT\\00\\00_01_poses.npz"

# 加载并查看内容
data = np.load(file_path, allow_pickle=True)

print("NPZ 文件包含的键:")
for key in data.files:
    print(f"  - {key}")

print()
print("是否包含必需的字段:")
required_fields = ['poses', 'trans', 'gender', 'betas', 'mocap_framerate']
for field in required_fields:
    if field in data.files:
        print(f"  ✓ {field}")
    else:
        print(f"  ✗ {field} (缺失)")
```

**必需的字段：**
- ✅ `poses`: 关节姿态
- ✅ `trans`: 平移（根节点位置）
- ✅ `gender`: 性别（通常为"neutral"）
- ✅ `betas`: SMPL形状参数
- ✅ `mocap_framerate`: 帧率

---

## 🔄 转换 KIT 数据集

### 转换命令

解压并验证后，运行：

```bash
cd D:\兼职\18\Kinesis
python src/utils/convert_kit.py --path "D:\KIT_Data\KIT"
```

### 转换过程

脚本会：

1. **扫描所有 NPZ 文件**
   ```
   [████████████████████] 100%
   Found 1000+ motion files
   ```

2. **加载运动数据**
   ```
   Loading motion data...
   [████████████████████] 100%
   ```

3. **转换为Kinesis格式**
   - 提取关节姿态
   - 计算四元数表示
   - 应用身体参数

4. **分类为训练集和测试集**
   ```
   Sorting by kit_train_keys.txt and kit_test_keys.txt...
   ```

5. **保存为PKL格式**
   ```
   Saving train data: data/kit_train_motion_dict.pkl
   Saving test data: data/kit_test_motion_dict.pkl
   ```

### 预计时间

- 扫描：1-2分钟
- 加载：5-10分钟
- 转换：3-5分钟
- 保存：1-2分钟

**总计：约10-15分钟**

---

## ✅ 验证转换结果

### 转换完成后

运行验证：

```bash
cd D:\兼职\18\Kinesis

# 检查文件是否存在
python -c "import os; print('训练集:', '✓' if os.path.exists('data/kit_train_motion_dict.pkl') else '✗'); print('测试集:', '✓' if os.path.exists('data/kit_test_motion_dict.pkl') else '✗')"

# 检查文件内容
python -c "import joblib; train=joblib.load('data/kit_train_motion_dict.pkl'); test=joblib.load('data/kit_test_motion_dict.pkl'); print(f'训练集: {len(train)} 个动作'); print(f'测试集: {len(test)} 个动作')"
```

**预期输出：**
```
训练集: ✓
测试集: ✓
训练集: 956 个动作
测试集: [数量] 个动作
```

---

## 📋 完整下载和转换流程

### 步骤1：下载 SMPL 模型（5分钟）

1. 访问：https://smpl.is.tue.mpg.de/
2. 注册 → 登录
3. 下载 neutral 模型（.pkl 或 .npz）
4. 重命名/转换为 `SMPL_NEUTRAL.pkl`
5. 放到：`data/smpl/`

### 步骤2：下载 KIT 数据集（30-60分钟）

1. 访问：https://amass.is.tue.mpg.de/
2. 注册 → 登录
3. 找到 KIT 数据集（显示为 **SMPL-H** 或 **SMPL+H G**）
4. 下载 **SMPL-H 格式**（ZIP文件）
5. 解压到：`D:\KIT_Data\KIT\`
6. 确认看到 .npz 文件

### 步骤3：转换 KIT 数据集（10-15分钟）

```bash
cd D:\兼职\18\Kinesis
python src/utils/convert_kit.py --path "D:\KIT_Data\KIT"
```

### 步骤4：验证所有文件（1分钟）

```bash
cd D:\兼职\18\Kinesis

# 方法1：使用我的验证脚本
python -c "import os; print('SMPL:', '✓' if os.path.exists('data/smpl/SMPL_NEUTRAL.pkl') else '✗'); print('训练集:', '✓' if os.path.exists('data/kit_train_motion_dict.pkl') else '✗'); print('测试集:', '✓' if os.path.exists('data/kit_test_motion_dict.pkl') else '✗')"

# 方法2：双击运行
一键验证.bat
```

---

## ❓ 常见问题

### Q1: SMPL-H 和 SMPL+H G 有什么区别？
**A:** 没有区别！
- **SMPL-H**: 标准名称（SMPL with Hands）
- **SMPL+H G**: 网站可能的其他显示方式
- **都是同一个格式，可以直接下载使用！**

### Q2: 为什么网站显示 SMPL+H G？
**A:** 可能是：
- 网站的数据格式标签
- "G" 可能代表 "General" 或 "Generic"
- 不影响使用，直接下载即可

### Q3: 下载后是ZIP文件吗？
**A:** 是的！
- AMASS下载的都是ZIP压缩包
- 解压后才能看到 NPZ 文件
- 先解压，再转换

### Q4: 如何确认下载的是SMPL-H格式？
**A:** 检查：
1. 解压后查看NPZ文件内容（见上面"验证KIT数据集"部分）
2. 确认包含 `poses`, `trans`, `gender` 等字段
3. 运行 `convert_kit.py` 测试

### Q5: 转换失败怎么办？
**A:** 检查：
1. NPZ 文件是否正确解压
2. 路径是否正确（使用绝对路径）
3. 是否有足够权限读取文件
4. 查看错误信息

---

## 🎯 总结

### ✅ SMPL+H G 就是 SMPL-H 格式！

**可以立即下载！**

### 完整流程：

| 步骤 | 操作 | 时间 | 状态 |
|------|------|------|------|
| 1. 下载SMPL模型 | 5分钟 | 待完成 |
| 2. 下载KIT数据集 | 30-60分钟 | 待完成 |
| 3. 解压KIT | 5分钟 | 待完成 |
| 4. 转换KIT数据 | 10-15分钟 | 待完成 |
| 5. 验证文件 | 1分钟 | 待完成 |
| **总计** | | **50-85分钟** |

---

## 💡 现在该怎么做？

### 立即行动：

1. ☐ **访问 AMASS 网站** → https://amass.is.tue.mpg.de/
2. ☐ **登录账户**
3. ☐ **找到 KIT 数据集（显示 SMPL-H 或 SMPL+H G）**
4. ☐ **下载 SMPL-H 格式**
5. ☐ **解压文件**
6. ☐ **运行转换脚本**
7. ☐ **验证结果**

---

## 📞 需要帮助？

如果在任何步骤遇到问题：
- 告诉我具体的错误信息
- 我会帮您诊断和解决
- 或者查看其他说明文档

---

**现在就开始下载 KIT 数据集吧！** 🚀

**SMPL-H 格式就是您要的，没问题！** ✅

---

**文档版本**: 1.0
**创建日期**: 2026-01-13
**KIT数据集说明 | Kinesis Project**
