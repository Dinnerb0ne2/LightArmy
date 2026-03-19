# 模型下载指南

**重要：本系统禁止自动下载模型，必须手动下载模型文件才能使用。**

## 必需模型

### 1. YOLO v8 Face 模型（人脸检测）

**必需的模型文件：** `yolov8n-face.pt`

#### 下载方法

**方法 1：从 GitHub 下载**

1. 访问 YOLO v8 Face GitHub 仓库：
   ```
   https://github.com/deepcam-cn/yolov8-face
   ```

2. 在仓库中找到模型文件（通常在 `weights/` 或 `models/` 目录）

3. 下载以下任一模型：
   - `yolov8n-face.pt` - 推荐（最小最快）
   - `yolov8s-face.pt` - 平衡速度和精度
   - `yolov8m-face.pt` - 精度最高

**方法 2：使用 wget 下载（如果有 Git Bash 或 WSL）**

```bash
# 下载 yolov8n-face.pt
wget https://github.com/deepcam-cn/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt
```

**方法 3：使用 curl 下载**

```bash
# 下载 yolov8n-face.pt
curl -L -o yolov8n-face.pt https://github.com/deepcam-cn/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt
```

#### 安装模型

1. 将下载的模型文件放到项目目录：
   ```
   LightArmy/
   └── models/
       └── yolov8n-face.pt
   ```

2. 确保文件名正确（默认使用 `yolov8n-face.pt`）

3. 如果使用其他名称，运行时指定路径：
   ```bash
   python src/main.py --detector-model models/yolov8s-face.pt --mode monitor
   ```

### 2. InsightFace Buffalo_L 模型（人脸识别）

**必需的模型文件：** Buffalo_L 模型包

#### 下载方法

**方法 1：使用 git clone 下载（推荐）**

```bash
# 克隆 InsightFace 仓库
git clone https://github.com/deepinsight/insightface.git

# 复制模型文件到项目的 models/ 目录
# Windows
mkdir models\buffalo_l
xcopy /E /I insightface\models\buffalo_l models\buffalo_l

# Linux/Mac
mkdir -p models/buffalo_l
cp -r insightface/models/buffalo_l/* models/buffalo_l/
```

**方法 2：手动下载模型文件**

1. 访问 InsightFace GitHub 仓库：
   ```
   https://github.com/deepinsight/insightface
   ```

2. 进入 `models/buffalo_l/` 目录

3. 下载以下文件：
   - `det_10g.onnx` - 人脸检测模型
   - `w600k_r50.onnx` - 人脸识别模型
   - `genderage.onnx` - 性别年龄模型
   - `2d106det.onnx` - 人脸关键点检测模型

#### 安装模型

将下载的模型文件复制到项目的 `models/` 目录：

```bash
# 创建模型目录
mkdir models\buffalo_l

# 复制模型文件
# Windows
xcopy /E /I insightface\models\buffalo_l models\buffalo_l

# Linux/Mac
cp -r insightface/models/buffalo_l/* models/buffalo_l/
```

或者手动复制：
- 将 `insightface/models/buffalo_l/` 目录下的所有文件
- 复制到项目的 `models/buffalo_l/` 目录

#### 验证模型

确保项目目录结构如下：
```
LightArmy/
├── models/
│   ├── yolov8n-face.pt          # YOLO v8 Face 模型
│   ├── buffalo_l/               # InsightFace 模型目录
│   │   ├── det_10g.onnx
│   │   ├── w600k_r50.onnx
│   │   ├── genderage.onnx
│   │   ├── 2d106det.onnx
│   │   └── ...
│   ├── Alex/                     # Alex 的照片
│   └── Bob/                      # Bob 的照片
└── ...
```

## 验证模型安装

运行以下命令验证模型是否正确安装：

```bash
python src/main.py --mode monitor
```

如果看到以下错误，说明模型文件未正确放置：
```
ValueError: 必须提供模型路径！
```

或

```
FileNotFoundError: 模型文件不存在: models/yolov8n-face.pt
```

## 模型选择建议

| 模型 | 大小 | 速度 | 精度 | 推荐场景 |
|------|------|------|------|----------|
| yolov8n-face.pt | ~6MB | 最快 | 中等 | 实时监控、低配设备 |
| yolov8s-face.pt | ~12MB | 快 | 较高 | 一般监控场景 |
| yolov8m-face.pt | ~25MB | 中等 | 最高 | 高精度需求 |

**推荐使用 `yolov8n-face.pt`，适合大多数监控场景。**

## 目录结构

安装模型后的目录结构：

```
LightArmy/
├── data/
├── models/
│   ├── yolov8n-face.pt          # YOLO v8 Face 模型（必需）
│   ├── buffalo_l/               # InsightFace 模型目录（必需）
│   │   ├── det_10g.onnx
│   │   ├── w600k_r50.onnx
│   │   ├── genderage.onnx
│   │   └── ...
│   ├── Alex/                     # Alex 的照片
│   └── Bob/                      # Bob 的照片
├── src/
└── ...
```

## 常见问题

**Q: 为什么禁止自动下载？**

A: 为了确保模型的来源安全可控，避免下载到恶意或错误的模型文件。

**Q: 可以使用其他 YOLO 模型吗？**

A: 只能使用 YOLO v8 Face 模型，普通 YOLO v8 模型不支持人脸检测。

**Q: 模型文件损坏怎么办？**

A: 重新下载模型文件，并覆盖原文件。

**Q: 可以共享模型文件吗？**

A: 可以，但请确保从官方或可信来源下载。

**Q: 模型文件很大，占用磁盘空间？**

A: `yolov8n-face.pt` 只有约 6MB，占用很小。如果空间有限，只下载这一个模型即可。

## 安全提示

1. 只从官方 GitHub 仓库下载模型
2. 不要使用来源不明的模型文件
3. 定期检查模型文件的完整性
4. 妥善保管模型文件，避免泄露

## 联系支持

如果遇到模型下载或安装问题，请：
1. 检查网络连接
2. 确认下载的模型文件完整
3. 查看控制台错误信息
4. 提交 Issue 寻求帮助