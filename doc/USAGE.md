# 使用说明 - 批量注册和回调功能

本文档介绍如何使用批量注册人脸功能和检测回调功能。

## 前提条件

**在使用本功能之前，请确保已经完成模型安装：**

1. 下载 YOLO v8 Face 模型（`yolov8n-face.pt`）
2. 将模型文件放到 `models/` 目录
3. 详细说明请参考 [MODEL_SETUP.md](MODEL_SETUP.md)

## 功能概述

1. **批量注册人脸**：从目录中批量注册多张照片作为一个人脸特征
2. **检测回调**：当检测到指定人脸时，自动触发回调函数

## 1. 批量注册人脸

### 步骤 1：准备照片

将多张同一个人的照片放入一个目录中，例如：
```
models/
└── Alex/
    ├── photo1.jpg
    ├── photo2.jpg
    ├── photo3.png
    └── photo4.jpg
```

### 步骤 2：运行批量注册

```bash
python src/main.py --mode register-dir --input models/Alex --name "Alex"
```

参数说明：
- `--mode register-dir`：批量注册模式
- `--input models/Alex`：照片目录路径
- `--name "Alex"`：人脸名称

### 步骤 3：注册多个人脸

可以为 Bob 也批量注册照片：

```bash
python src/main.py --mode register-dir --input models/Bob --name "Bob"
```

## 2. 设置检测回调

### 步骤 1：创建或修改回调函数

系统已提供示例回调函数 `src/callback_example.py`，你可以：

1. 直接使用示例回调函数
2. 修改示例回调函数
3. 创建自己的回调函数

回调函数必须包含 `on_face_detected` 函数：

```python
def on_face_detected(name: str, similarity: float, bbox: Tuple[int, int, int, int]):
    """
    当检测到人脸时被调用的回调函数

    Args:
        name: 检测到的人脸名称
        similarity: 相似度分数（0-1）
        bbox: 人脸边界框 (x1, y1, x2, y2)
    """
    print(f"检测到人脸: {name}")
    # 在这里添加你的处理逻辑
```

### 步骤 2：创建专属处理程序

可以为 Alex 和 Bob 创建专属的处理程序：

- `src/alex_handler.py`：Alex 的处理程序
- `src/bob_handler.py`：Bob 的处理程序

当检测到对应人脸时，这些处理程序会被自动执行。

### 步骤 3：运行实时监控并启用回调

```bash
python src/main.py --mode monitor --callback src/callback_example.py --targets Alex Bob
```

参数说明：
- `--mode monitor`：实时监控模式
- `--callback src/callback_example.py`：回调函数文件路径
- `--targets Alex Bob`：需要触发回调的人脸名称列表（空格分隔）

## 3. 完整使用流程

### 3.1 准备工作

1. 创建人脸照片目录：
   ```
   models/
   ├── Alex/
   │   ├── photo1.jpg
   │   └── photo2.jpg
   └── Bob/
       ├── photo1.jpg
       └── photo2.jpg
   ```

2. 确保回调函数和专属处理程序已准备好

### 3.2 批量注册人脸

```bash
# 注册 Alex
python src/main.py --mode register-dir --input models/Alex --name "Alex"

# 注册 Bob
python src/main.py --mode register-dir --input models/Bob --name "Bob"
```

### 3.3 启动实时监控

```bash
python src/main.py --mode monitor --callback src/callback_example.py --targets Alex Bob
```

### 3.4 测试

当摄像头检测到 Alex 或 Bob 时：

1. 系统会在控制台显示检测结果
2. 自动触发回调函数
3. 执行对应的专属处理程序（alex_handler.py 或 bob_handler.py）
4. 在视频画面中用绿色框标记识别到的人脸

## 4. 高级配置

### 4.1 调整阈值

```bash
python src/main.py --mode monitor \
    --callback src/callback_example.py \
    --targets Alex Bob \
    --conf 0.6 \
    --sim 0.6
```

- `--conf 0.6`：检测置信度阈值（默认 0.5）
- `--sim 0.6`：识别相似度阈值（默认 0.5）

### 4.2 指定摄像头

```bash
python src/main.py --mode monitor \
    --callback src/callback_example.py \
    --targets Alex Bob \
    --camera 1
```

- `--camera 1`：使用第二个摄像头（默认 0）

### 4.3 自定义数据库路径

```bash
python src/main.py --mode monitor \
    --callback src/callback_example.py \
    --targets Alex Bob \
    --db custom_database.pkl
```

## 5. 回调冷却时间

为了避免频繁触发回调，系统设置了冷却时间（默认 5 秒）。

当检测到同一个人脸后，5 秒内不会再次触发回调。

### 修改冷却时间

在 `src/main.py` 的 `SilentFaceMonitor.__init__` 方法中：

```python
self._trigger_cooldown = 5.0  # 修改这个值（单位：秒）
```

## 6. 示例场景

### 场景 1：智能门禁

当检测到授权人员时：
1. 自动打开门锁
2. 播放欢迎语音
3. 记录访问日志

### 场景 2：智能办公

当检测到特定员工时：
1. 自动登录系统
2. 打开常用应用
3. 调整个人设置

### 场景 3：安全监控

当检测到陌生人时：
1. 发送警报通知
2. 保存监控截图
3. 记录访问日志

## 7. 常见问题

**Q: 如何查看已注册的人脸？**

A: 人脸数据保存在数据库文件中（默认 `data/face_database.pkl`），可以使用以下代码查看：

```python
from src.face_recognizer import FaceRecognizer

recognizer = FaceRecognizer()
recognizer.load_database('data/face_database.pkl')
info = recognizer.get_database_info()
print(info)
```

**Q: 回调函数没有被触发？**

A: 检查以下几点：
1. 回调文件路径是否正确
2. 回调文件中是否有 `on_face_detected` 函数
3. 目标人脸名称是否正确（与注册时的名称一致）
4. 相似度阈值是否过高

**Q: 如何停止监控？**

A: 在监控窗口中按 `q` 键退出。

**Q: 可以同时监控多个人脸吗？**

A: 可以，使用 `--targets` 参数指定多个人脸名称，用空格分隔：

```bash
python src/main.py --mode monitor --callback src/callback_example.py --targets Alex Bob Charlie
```

## 8. 故障排除

### 批量注册失败

- 检查目录路径是否正确
- 确保目录中有图像文件
- 检查图像格式是否支持（jpg, jpeg, png, bmp, webp）

### 检测不到人脸

- 降低检测置信度阈值：`--conf 0.3`
- 检查摄像头是否正常工作
- 确保光照条件良好

### 识别不准确

- 为同一个人注册更多不同角度的照片
- 降低识别相似度阈值：`--sim 0.4`
- 使用更清晰的参考照片

### 回调函数报错

- 检查回调函数语法是否正确
- 确保依赖库已安装
- 查看控制台错误信息

## 9. 安全建议

1. 妥善保管人脸数据库文件
2. 不要将人脸数据上传到云端
3. 定期备份人脸数据库
4. 使用强密码保护系统
5. 限制访问权限