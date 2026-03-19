# 静默监控定向人脸识别系统

轻量级监控人脸识别系统，针对监控画面小尺寸、远距离人脸优化，完全本地化部署。

## 快速开始

详细文档请查看 [doc/README.md](doc/README.md)

## 文档

- [README.md](doc/README.md) - 项目完整说明
- [USAGE.md](doc/USAGE.md) - 批量注册和回调功能使用说明
- [MODEL_SETUP.md](doc/MODEL_SETUP.md) - 模型下载和安装指南
- [MODEL_REQUIRED.md](doc/MODEL_REQUIRED.md) - 模型必需说明

## 快速检查

```bash
# 检查模型文件是否已安装
python check_model.py
```

## 项目结构

```
LightArmy/
├── doc/                # 文档目录
│   ├── README.md
│   ├── USAGE.md
│   ├── MODEL_SETUP.md
│   └── MODEL_REQUIRED.md
├── data/               # 数据集和数据库
├── models/             # 模型文件和人脸照片
├── src/                # 源代码
├── test/               # 测试文件
└── check_model.py      # 模型检查脚本
```

请查看 [doc/README.md](doc/README.md) 获取完整的使用说明。