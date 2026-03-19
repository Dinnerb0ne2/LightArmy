# 静默监控定向人脸识别系统（开发者版）

轻量级监控人脸识别系统，针对监控画面小尺寸/远距离人脸优化，完全本地化部署，不依赖云端计算资源。

## 项目特点

- **小目标人脸检测增强**：采用小人脸专用检测模型，优化远距离、低分辨率场景
- **本地化部署**：所有计算在本地设备完成，人脸数据不上云
- **静默监控**：无需用户配合，自动进行人脸检测和识别
- **轻量高效**：基于 YOLO v8 和 InsightFace，资源占用低

## 技术栈

- Python 3.11.0
- YOLO v8 Face（人脸检测）
- InsightFace（人脸识别）
- OpenCV（图像处理）
- NumPy（数值计算）

## 项目结构

```
project/
├── data/               # 数据集和数据库
├── models/             # 模型文件和人脸照片
│   ├── yolov8n-face.pt     # YOLO v8 Face 模型
│   ├── buffalo_l/          # InsightFace 模型目录
│   ├── Alex/               # Alex 的照片目录
│   └── Bob/                # Bob 的照片目录
├── src/                # 源代码
│   ├── face_detector.py    # 人脸检测模块
│   ├── face_recognizer.py  # 人脸识别模块
│   ├── main.py             # 主程序
│   ├── callback_example.py # 回调函数示例
│   ├── alex_handler.py     # Alex 专属处理程序
│   └── bob_handler.py      # Bob 专属处理程序
├── test/               # 测试文件
│   └── test_system.py      # 系统测试
├── requirements.txt    # 依赖列表
├── README.md           # 项目说明
└── USAGE.md            # 详细使用说明（批量注册和回调功能）
```

## 安装依赖

```bash
pip install -r requirements.txt
```

### 模型设置（必需）

**本系统禁止自动下载模型，必须手动下载所有模型文件！**

#### 1. YOLO v8 Face 模型（人脸检测）

1. 下载模型文件：
   - 从 GitHub 下载：https://github.com/deepcam-cn/yolov8-face
   - 推荐下载：`yolov8n-face.pt`（约 6MB，最快）

2. 将模型文件放到 `models/` 目录：
   ```
   models/
   └── yolov8n-face.pt
   ```

#### 2. InsightFace 模型（人脸识别）

1. 使用 git clone 下载完整模型包：
   ```bash
   git clone https://github.com/deepinsight/insightface.git
   ```

2. 将 `insightface/models/buffalo_l/` 目录复制到项目的 `models/` 目录：
   ```
   models/buffalo_l/
   ```

详细下载说明请参考 [MODEL_SETUP.md](MODEL_SETUP.md)

## 快速开始

### 0. 模型设置（必需）

在使用系统之前，必须先下载并安装模型：

```bash
# 1. 下载 YOLO v8 Face 模型
# 从 https://github.com/deepcam-cn/yolov8-face 下载 yolov8n-face.pt
# 将文件放到 models/ 目录下

# 2. 下载 InsightFace 模型
# git clone https://github.com/deepinsight/insightface.git
# 将 insightface/models/buffalo_l/ 目录复制到 models/ 目录

# 3. 验证模型是否正确安装
python check_model.py
```

详细说明请参考 [MODEL_SETUP.md](MODEL_SETUP.md)

### 1. 注册人脸

将已知人脸注册到系统数据库：

```bash
python src/main.py --mode register --input path/to/face_image.jpg --name "张三"
```

可以注册多张同一个人的照片以提高识别准确率。

### 2. 检测并识别单张图像

```bash
python src/main.py --mode detect --input path/to/test_image.jpg --output result.jpg
```

### 3. 处理视频文件

```bash
python src/main.py --mode video --input path/to/video.mp4 --output result.mp4
```

### 4. 实时监控

```bash
python src/main.py --mode monitor --camera 0
```

按 `q` 键退出监控。

### 5. 批量注册人脸（新功能）

从目录中批量注册多张照片作为一个人脸特征：

```bash
python src/main.py --mode register-dir --input models/Alex --name "Alex"
```

将多张 Alex 的照片放在 `models/Alex/` 目录下，系统会自动批量注册。

### 6. 实时监控并触发回调（新功能）

当检测到指定人脸时，自动触发回调函数：

```bash
python src/main.py --mode monitor --callback src/callback_example.py --targets Alex Bob
```

参数说明：
- `--callback`: 回调函数文件路径
- `--targets`: 需要触发回调的人脸名称列表（空格分隔）

详细使用说明请参考 [USAGE.md](USAGE.md)。

## 高级用法

### 自定义阈值

```bash
# 调整检测置信度阈值（默认 0.5）
python src/main.py --mode detect --input image.jpg --conf 0.7

# 调整识别相似度阈值（默认 0.5）
python src/main.py --mode detect --input image.jpg --sim 0.6
```

### 自定义数据库路径

```bash
python src/main.py --mode register --input image.jpg --name "张三" --db custom_db.pkl
```

## API 使用示例

### 人脸检测

```python
from src.face_detector import FaceDetector
import cv2

# 初始化检测器
detector = FaceDetector()

# 读取图像
image = cv2.imread('test_image.jpg')

# 检测人脸
faces = detector.detect(image)

# 可视化结果
vis_image = detector.visualize_detections(image, faces)
cv2.imwrite('result.jpg', vis_image)
```

### 人脸识别

```python
from src.face_recognizer import FaceRecognizer
import cv2

# 初始化识别器
recognizer = FaceRecognizer()

# 读取图像
image = cv2.imread('face_image.jpg')

# 提取特征
embedding = recognizer.extract_embedding(image)

# 注册人脸
recognizer.register_face('张三', embedding)

# 识别
name, similarity = recognizer.identify(embedding)
print(f"识别结果: {name}, 相似度: {similarity}")
```

### 完整流程

```python
from src.main import SilentFaceMonitor

# 初始化系统
monitor = SilentFaceMonitor()

# 注册人脸
monitor.register_from_image('person1.jpg', '张三')
monitor.register_from_image('person2.jpg', '李四')

# 保存数据库
monitor.save_database()

# 处理图像
results = monitor.process_image('test_image.jpg', 'output.jpg')
print(results)
```

## 运行测试

```bash
python test/test_system.py
```

## 性能优化建议

1. **GPU 加速**：如果安装了 CUDA，可以安装 GPU 版本的依赖：
   ```bash
   pip install onnxruntime-gpu
   ```

2. **模型选择**：根据设备性能选择合适的模型：
   - `yolov8n-face.pt`：最快，精度较低
   - `yolov8s-face.pt`：平衡速度和精度
   - `yolov8m-face.pt`：精度最高，速度较慢

3. **图像预处理**：对于小尺寸人脸，系统会自动进行增强处理以提高检测效果。

## 隐私说明

- 本系统所有计算均在本地设备完成
- 人脸特征数据存储在本地数据库文件中
- 不会上传任何数据到云端服务器
- 建议妥善保管数据库文件，避免泄露

## 常见问题

**Q: 模型下载失败怎么办？**

A: 可以手动下载模型文件到 `models/` 目录，或配置代理后重试。

**Q: 小人脸检测效果不好？**

A: 系统已经内置了小人脸增强功能，如果效果仍不理想，可以尝试：
- 提高图像分辨率
- 使用更大的人脸检测模型（如 yolov8s-face.pt）
- 降低检测置信度阈值

**Q: 识别准确率不高？**

A: 可以尝试：
- 为同一人注册多张不同角度、不同光照的照片
- 调整识别相似度阈值
- 使用更清晰的参考图像

**Q: 如何提高处理速度？**

A: 可以尝试：
- 使用 GPU 加速
- 降低输入图像分辨率
- 使用更轻量的模型

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交 Issue 和 Pull Request。

## 联系方式

如有问题或建议，请通过 Issue 联系。