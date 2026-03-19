"""
人脸检测模块 - 基于 YOLO v8
优化小尺寸、远距离人脸检测
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional


class FaceDetector:
    """人脸检测器，使用 YOLO v8 face 模型"""

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.5):
        """
        初始化人脸检测器

        Args:
            model_path: 模型文件路径，如果为 None 则使用默认的 YOLO v8n-face
            conf_threshold: 置信度阈值
        """
        self.conf_threshold = conf_threshold

        if model_path:
            self.model = YOLO(model_path)
        else:
            # 使用 YOLO v8 nano face 模型（轻量级，适合本地部署）
            self.model = YOLO('yolov8n-face.pt')

    def detect(self, image: np.ndarray, enhance_small: bool = True) -> List[Tuple[int, int, int, int]]:
        """
        检测图像中的人脸

        Args:
            image: 输入图像 (BGR 格式)
            enhance_small: 是否增强小人脸检测

        Returns:
            人脸边界框列表，每个边界框为 (x1, y1, x2, y2)
        """
        if enhance_small:
            # 对图像进行增强以提高小人脸检测效果
            enhanced_image = self._enhance_small_faces(image)
        else:
            enhanced_image = image

        # 运行检测
        results = self.model(enhanced_image, conf=self.conf_threshold, verbose=False)

        faces = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    faces.append((int(x1), int(y1), int(x2), int(y2)))

        return faces

    def detect_with_confidence(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        检测图像中的人脸并返回置信度

        Args:
            image: 输入图像 (BGR 格式)

        Returns:
            人脸列表，每个元素为 (x1, y1, x2, y2, confidence)
        """
        results = self.model(image, conf=self.conf_threshold, verbose=False)

        faces = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    faces.append((int(x1), int(y1), int(x2), int(y2), conf))

        return faces

    def _enhance_small_faces(self, image: np.ndarray) -> np.ndarray:
        """
        增强图像以提高小人脸检测效果

        Args:
            image: 输入图像

        Returns:
            增强后的图像
        """
        # 使用 CLAHE 增强对比度
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 应用 CLAHE 到 L 通道
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # 合并通道
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # 轻微锐化
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced

    def visualize_detections(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        在图像上可视化检测结果

        Args:
            image: 输入图像
            faces: 人脸边界框列表

        Returns:
            带有标注的图像
        """
        vis_image = image.copy()
        for face in faces:
            x1, y1, x2, y2 = face
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return vis_image


if __name__ == '__main__':
    # 测试代码
    detector = FaceDetector()

    # 创建测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, 'Test Image', (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    faces = detector.detect(test_image)
    print(f"检测到 {len(faces)} 个人脸")