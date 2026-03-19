"""
模型检查脚本
用于验证模型文件是否存在
"""

import os
import sys


def check_model():
    """检查模型文件是否存在"""

    print("=" * 50)
    print("模型文件检查")
    print("=" * 50)

    all_ok = True

    # 检查 YOLO v8 Face 模型
    print("\n1. 检查 YOLO v8 Face 模型...")
    yolo_model_path = 'models/yolov8n-face.pt'
    if os.path.exists(yolo_model_path):
        file_size = os.path.getsize(yolo_model_path) / (1024 * 1024)  # MB
        print(f"   ✓ YOLO v8 Face 模型已找到: {yolo_model_path}")
        print(f"   ✓ 文件大小: {file_size:.2f} MB")
    else:
        print(f"   ✗ YOLO v8 Face 模型未找到: {yolo_model_path}")
        all_ok = False

    # 检查 InsightFace 模型
    print("\n2. 检查 InsightFace 模型...")
    insightface_model_dir = os.path.join('models', 'buffalo_l')

    if os.path.exists(insightface_model_dir):
        print(f"   ✓ InsightFace 模型目录已找到: {insightface_model_dir}")

        # 检查关键文件
        required_files = ['det_10g.onnx', 'w600k_r50.onnx', 'genderage.onnx']
        for file in required_files:
            file_path = os.path.join(insightface_model_dir, file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"   ✓ {file} ({file_size:.2f} MB)")
            else:
                print(f"   ✗ {file} 未找到")
                all_ok = False
    else:
        print(f"   ✗ InsightFace 模型目录未找到: {insightface_model_dir}")
        all_ok = False

    # 输出结果
    print("\n" + "=" * 50)
    if all_ok:
        print("✓ 所有模型文件检查通过！")
        print("\n系统可以正常使用！")
        print("\n运行以下命令开始监控:")
        print("python src/main.py --mode monitor")
    else:
        print("✗ 部分模型文件缺失！")
        print("\n请先下载缺失的模型文件：")
        print("\nYOLO v8 Face 模型:")
        print("1. 访问: https://github.com/deepcam-cn/yolov8-face")
        print("2. 下载: yolov8n-face.pt")
        print("3. 将文件放到: models/ 目录")
        print("\nInsightFace 模型:")
        print("1. 访问: https://github.com/deepinsight/insightface")
        print("2. 使用 git clone 下载完整模型包:")
        print("   git clone https://github.com/deepinsight/insightface.git")
        print("3. 将 insightface/models/buffalo_l/ 目录复制到项目的 models/ 目录:")
        print("   models/buffalo_l/")
        print("\n详细说明请参考: doc/MODEL_SETUP.md")

    print("=" * 50)
    return all_ok


if __name__ == '__main__':
    success = check_model()
    sys.exit(0 if success else 1)