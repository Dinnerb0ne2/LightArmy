"""
静默监控定向人脸识别系统 - 主程序
整合人脸检测和识别功能
"""

import cv2
import numpy as np
import os
import argparse
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict

from face_detector import FaceDetector
from face_recognizer import FaceRecognizer


class SilentFaceMonitor:
    """静默监控定向人脸识别系统"""

    def __init__(self,
                 detector_model: Optional[str] = None,
                 recognizer_model: Optional[str] = None,
                 db_path: str = 'data/face_database.pkl',
                 conf_threshold: float = 0.5,
                 sim_threshold: float = 0.5):
        """
        初始化监控系统

        Args:
            detector_model: 人脸检测模型路径
            recognizer_model: 人脸识别模型路径
            db_path: 人脸数据库路径
            conf_threshold: 检测置信度阈值
            sim_threshold: 识别相似度阈值
        """
        self.detector = FaceDetector(model_path=detector_model, conf_threshold=conf_threshold)
        self.recognizer = FaceRecognizer(model_path=recognizer_model, sim_threshold=sim_threshold)
        self.db_path = db_path

        # 加载已有的人脸数据库
        self._load_database()

    def _load_database(self):
        """加载人脸数据库"""
        if os.path.exists(self.db_path):
            if self.recognizer.load_database(self.db_path):
                info = self.recognizer.get_database_info()
                print(f"已加载人脸数据库: {info['total_persons']} 人, {info['total_faces']} 张人脸")
            else:
                print("人脸数据库加载失败")
        else:
            print("人脸数据库不存在，将创建新数据库")

    def save_database(self):
        """保存人脸数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.recognizer.save_database(self.db_path)
        info = self.recognizer.get_database_info()
        print(f"已保存人脸数据库: {info['total_persons']} 人, {info['total_faces']} 张人脸")

    def register_from_image(self, image_path: str, name: str) -> bool:
        """
        从图像注册人脸

        Args:
            image_path: 图像路径
            name: 人脸名称/ID

        Returns:
            是否注册成功
        """
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            return False

        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return False

        # 检测人脸
        faces = self.detector.detect(image)
        if len(faces) == 0:
            print(f"未在图像中检测到人脸: {image_path}")
            return False

        if len(faces) > 1:
            print(f"警告: 图像中检测到 {len(faces)} 个人脸，将使用第一个")

        # 注册第一张人脸
        success = self.recognizer.register_face_from_image(name, image, faces[0])
        if success:
            print(f"成功注册人脸: {name}")
            self.save_database()
            return True
        else:
            print(f"注册人脸失败: {name}")
            return False

    def process_image(self, image_path: str, output_path: Optional[str] = None) -> List[Dict]:
        """
        处理单张图像，检测并识别人脸

        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径（可选）

        Returns:
            检测结果列表
        """
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            return []

        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return []

        # 检测人脸
        faces = self.detector.detect(image)
        print(f"检测到 {len(faces)} 个人脸")

        results = []

        # 识别每张人脸
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face

            # 提取特征
            embedding = self.recognizer.extract_embedding(image, face)
            if embedding is None:
                results.append({
                    'bbox': face,
                    'name': 'Unknown',
                    'similarity': 0.0
                })
                continue

            # 识别
            name, similarity = self.recognizer.identify(embedding)
            results.append({
                'bbox': face,
                'name': name if name else 'Unknown',
                'similarity': similarity
            })

        # 可视化结果
        vis_image = self._visualize_results(image, results)

        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"结果已保存到: {output_path}")

        return results

    def process_video(self, video_path: str, output_path: Optional[str] = None) -> List[Dict]:
        """
        处理视频文件，检测并识别人脸

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）

        Returns:
            检测结果列表
        """
        if not os.path.exists(video_path):
            print(f"视频文件不存在: {video_path}")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return []

        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 准备视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_results = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"处理帧 {frame_count}")

            # 检测人脸
            faces = self.detector.detect(frame)

            # 识别每张人脸
            frame_results = []
            for face in faces:
                embedding = self.recognizer.extract_embedding(frame, face)
                if embedding is None:
                    frame_results.append({
                        'bbox': face,
                        'name': 'Unknown',
                        'similarity': 0.0
                    })
                    continue

                name, similarity = self.recognizer.identify(embedding)
                frame_results.append({
                    'bbox': face,
                    'name': name if name else 'Unknown',
                    'similarity': similarity
                })

            all_results.extend(frame_results)

            # 可视化结果
            vis_frame = self._visualize_results(frame, frame_results)

            # 写入视频
            if writer:
                writer.write(vis_frame)

            # 显示结果（可选）
            cv2.imshow('Silent Face Monitor', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        if output_path:
            print(f"结果已保存到: {output_path}")

        return all_results

    def _visualize_results(self, image: np.ndarray, results: List[Dict]) -> np.ndarray:
        """
        可视化检测结果

        Args:
            image: 输入图像
            results: 检测结果列表

        Returns:
            可视化后的图像
        """
        vis_image = image.copy()

        for result in results:
            x1, y1, x2, y2 = result['bbox']
            name = result['name']
            similarity = result['similarity']

            # 根据是否识别到选择颜色
            if name == 'Unknown':
                color = (0, 0, 255)  # 红色
            else:
                color = (0, 255, 0)  # 绿色

            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{name} ({similarity:.2f})"
            cv2.putText(vis_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis_image

    def real_time_monitor(self, camera_index: int = 0):
        """
        实时监控摄像头

        Args:
            camera_index: 摄像头索引
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"无法打开摄像头: {camera_index}")
            return

        print("按 'q' 键退出监控")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检测人脸
            faces = self.detector.detect(frame)

            # 识别每张人脸
            results = []
            for face in faces:
                embedding = self.recognizer.extract_embedding(frame, face)
                if embedding is None:
                    results.append({
                        'bbox': face,
                        'name': 'Unknown',
                        'similarity': 0.0
                    })
                    continue

                name, similarity = self.recognizer.identify(embedding)
                results.append({
                    'bbox': face,
                    'name': name if name else 'Unknown',
                    'similarity': similarity
                })

            # 可视化结果
            vis_frame = self._visualize_results(frame, results)

            # 显示结果
            cv2.imshow('Silent Face Monitor', vis_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='静默监控定向人脸识别系统')
    parser.add_argument('--mode', type=str, default='detect',
                        choices=['register', 'detect', 'video', 'monitor'],
                        help='运行模式: register(注册), detect(检测图像), video(处理视频), monitor(实时监控)')
    parser.add_argument('--input', type=str, help='输入文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--name', type=str, help='注册时的人脸名称')
    parser.add_argument('--camera', type=int, default=0, help='摄像头索引')
    parser.add_argument('--db', type=str, default='data/face_database.pkl',
                        help='人脸数据库路径')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='检测置信度阈值')
    parser.add_argument('--sim', type=float, default=0.5,
                        help='识别相似度阈值')

    args = parser.parse_args()

    # 初始化系统
    monitor = SilentFaceMonitor(
        db_path=args.db,
        conf_threshold=args.conf,
        sim_threshold=args.sim
    )

    # 根据模式执行相应操作
    if args.mode == 'register':
        if not args.input or not args.name:
            print("注册模式需要指定 --input 和 --name 参数")
            return
        monitor.register_from_image(args.input, args.name)

    elif args.mode == 'detect':
        if not args.input:
            print("检测模式需要指定 --input 参数")
            return
        results = monitor.process_image(args.input, args.output)
        print(f"检测结果: {json.dumps(results, indent=2, ensure_ascii=False)}")

    elif args.mode == 'video':
        if not args.input:
            print("视频模式需要指定 --input 参数")
            return
        results = monitor.process_video(args.input, args.output)
        print(f"检测到 {len(results)} 个人脸实例")

    elif args.mode == 'monitor':
        monitor.real_time_monitor(args.camera)


if __name__ == '__main__':
    main()