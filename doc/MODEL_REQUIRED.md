# ⚠️ 模型文件必需

**本系统需要手动下载模型文件才能使用！**

## 快速开始

### 1. 下载 YOLO v8 Face 模型（人脸检测）

从以下地址下载：

```
https://github.com/deepcam-cn/yolov8-face
```

推荐下载：`yolov8n-face.pt`（约 6MB）

### 2. 下载 InsightFace 模型（人脸识别）

从以下地址下载：

```
https://github.com/deepinsight/insightface
```

推荐使用 git clone 下载完整模型包：
```bash
git clone https://github.com/deepinsight/insightface.git
```

### 3. 安装模型

**YOLO v8 Face 模型：**

将 `yolov8n-face.pt` 放到项目的 `models/` 目录：
```
LightArmy/
└── models/
    └── yolov8n-face.pt
```

**InsightFace 模型：**

将 `insightface/models/buffalo_l/` 目录复制到项目的 `models/` 目录：
```
LightArmy/
└── models/
    └── buffalo_l/
        ├── det_10g.onnx
        ├── w600k_r50.onnx
        ├── genderage.onnx
        └── ...
```

### 4. 验证安装

运行以下命令验证模型是否正确安装：

```bash
python src/main.py --mode monitor
```

如果看到监控窗口启动，说明模型安装成功。

## 详细说明

请查看 [MODEL_SETUP.md](MODEL_SETUP.md) 了解更详细的模型下载和安装说明。

## 常见问题

**Q: 为什么不能自动下载？**

A: 为了确保模型来源安全可控，本系统禁止自动下载。

**Q: 可以跳过模型下载吗？**

A: 不可以，模型是系统运行的核心必需品。

**Q: 下载失败怎么办？**

A: 请检查网络连接，或使用代理下载。也可以从其他可信源获取模型文件。

---

**安装模型后，请参考 [README.md](README.md) 了解如何使用系统。**