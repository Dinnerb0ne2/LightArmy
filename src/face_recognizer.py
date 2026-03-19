"""
人脸识别模块 - 基于 InsightFace
用于人脸特征提取和比对
"""

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from typing import List, Tuple, Dict, Optional
import pickle
import os


class FaceRecognizer:
    """人脸识别器，使用 InsightFace 进行人脸特征提取和比对"""

    def __init__(self, model_path: Optional[str] = None, sim_threshold: float = 0.5):
        """
        初始化人脸识别器

        Args:
            model_path: 模型文件路径，如果为 None 则使用默认模型
            sim_threshold: 相似度阈值，用于判断是否为同一人
        """
        self.sim_threshold = sim_threshold
        self.face_db = {}  # 人脸数据库：{name: [embeddings]}

        # 初始化 InsightFace 应用
        providers = ['CPUExecutionProvider']  # 使用 CPU，避免依赖 CUDA
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def extract_embedding(self, image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        提取人脸特征向量

        Args:
            image: 输入图像 (BGR 格式)
            face_bbox: 人脸边界框 (x1, y1, x2, y2)，如果为 None 则自动检测

        Returns:
            人脸特征向量，如果检测失败则返回 None
        """
        if face_bbox is not None:
            # 使用指定的人脸区域
            x1, y1, x2, y2 = face_bbox
            face_img = image[y1:y2, x1:x2]

            # 提取特征
            faces = self.app.get(face_img)
            if len(faces) > 0:
                return faces[0].embedding
            return None
        else:
            # 自动检测人脸并提取特征
            faces = self.app.get(image)
            if len(faces) > 0:
                return faces[0].embedding
            return None

    def extract_embeddings(self, image: np.ndarray) -> List[np.ndarray]:
        """
        提取图像中所有人脸的特征向量

        Args:
            image: 输入图像 (BGR 格式)

        Returns:
            人脸特征向量列表
        """
        faces = self.app.get(image)
        embeddings = []
        for face in faces:
            embeddings.append(face.embedding)
        return embeddings

    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        计算两个人脸特征向量的相似度

        Args:
            embedding1: 第一个人脸特征向量
            embedding2: 第二个人脸特征向量

        Returns:
            相似度分数（0-1之间，越高越相似）
        """
        # 使用余弦相似度
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    def register_face(self, name: str, embedding: np.ndarray):
        """
        注册人脸到数据库

        Args:
            name: 人脸名称/ID
            embedding: 人脸特征向量
        """
        if name not in self.face_db:
            self.face_db[name] = []
        self.face_db[name].append(embedding)

    def register_face_from_image(self, name: str, image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None):
        """
        从图像中注册人脸

        Args:
            name: 人脸名称/ID
            image: 输入图像
            face_bbox: 人脸边界框，如果为 None 则自动检测
        """
        embedding = self.extract_embedding(image, face_bbox)
        if embedding is not None:
            self.register_face(name, embedding)
            return True
        return False

    def identify(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        识别人脸

        Args:
            embedding: 待识别的人脸特征向量

        Returns:
            (名称, 相似度)，如果没有匹配则返回 (None, 0)
        """
        best_match = None
        best_similarity = 0

        for name, embeddings in self.face_db.items():
            # 计算与该名称下所有样本的平均相似度
            similarities = [self.compare(embedding, emb) for emb in embeddings]
            avg_similarity = np.mean(similarities)
            max_similarity = np.max(similarities)

            # 使用最大相似度作为该人脸的匹配分数
            if max_similarity > best_similarity and max_similarity > self.sim_threshold:
                best_similarity = max_similarity
                best_match = name

        return best_match, best_similarity

    def save_database(self, path: str):
        """
        保存人脸数据库到文件

        Args:
            path: 保存路径
        """
        with open(path, 'wb') as f:
            pickle.dump(self.face_db, f)

    def load_database(self, path: str):
        """
        从文件加载人脸数据库

        Args:
            path: 加载路径
        """
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.face_db = pickle.load(f)
            return True
        return False

    def clear_database(self):
        """清空人脸数据库"""
        self.face_db = {}

    def get_database_info(self) -> Dict[str, int]:
        """
        获取数据库信息

        Returns:
            包含数据库统计信息的字典
        """
        total_faces = sum(len(embeddings) for embeddings in self.face_db.values())
        return {
            'total_persons': len(self.face_db),
            'total_faces': total_faces
        }


if __name__ == '__main__':
    # 测试代码
    recognizer = FaceRecognizer()

    # 创建测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, 'Test Image', (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 测试特征提取
    embedding = recognizer.extract_embedding(test_image)
    if embedding is not None:
        print(f"特征向量维度: {embedding.shape}")
    else:
        print("未检测到人脸")

    # 测试数据库操作
    info = recognizer.get_database_info()
    print(f"数据库信息: {info}")