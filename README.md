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
├── models/             # 模型文件
├── src/                # 源代码
│   ├── face_detector.py    # 人脸检测模块
│   ├── face_recognizer.py  # 人脸识别模块
│   └── main.py             # 主程序
├── test/               # 测试文件
│   └── test_system.py      # 系统测试
├── requirements.txt    # 依赖列表
└── README.md           # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

### 注意事项

首次运行时，系统会自动下载以下模型：
- YOLO v8 Face 模型（用于人脸检测）
- InsightFace Buffalo_L 模型（用于人脸识别）

请确保网络连接正常，或手动下载模型到 `models/` 目录。

## 快速开始

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