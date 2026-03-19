"""
人脸检测回调函数示例
当检测到指定人脸时，此模块中的函数会被自动调用
"""

from typing import Tuple
from datetime import datetime
import subprocess
import os


def on_face_detected(name: str, similarity: float, bbox: Tuple[int, int, int, int]):
    """
    当检测到人脸时被调用的回调函数

    Args:
        name: 检测到的人脸名称
        similarity: 相似度分数（0-1）
        bbox: 人脸边界框 (x1, y1, x2, y2)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    x1, y1, x2, y2 = bbox

    print(f"[{timestamp}] 检测到人脸: {name}")
    print(f"  相似度: {similarity:.4f}")
    print(f"  位置: ({x1}, {y1}) -> ({x2}, {y2})")

    # 根据不同的人脸执行不同的操作
    if name == "Alex":
        handle_alex_detected(similarity, bbox)
    elif name == "Bob":
        handle_bob_detected(similarity, bbox)
    else:
        handle_other_face_detected(name, similarity, bbox)


def handle_alex_detected(similarity: float, bbox: Tuple[int, int, int, int]):
    """
    处理检测到 Alex 的逻辑
    """
    print(">>> Alex 被检测到！执行 Alex 相关的操作...")

    # 示例1: 运行另一个 Python 文件
    try:
        # 运行 alex_handler.py 文件
        alex_handler_path = os.path.join(os.path.dirname(__file__), 'alex_handler.py')
        if os.path.exists(alex_handler_path):
            print(f"运行 Alex 处理程序: {alex_handler_path}")
            # 使用 subprocess 运行 Python 文件
            subprocess.run(['python', alex_handler_path], check=True)
        else:
            print(f"Alex 处理程序不存在: {alex_handler_path}")
    except Exception as e:
        print(f"运行 Alex 处理程序失败: {e}")

    # 示例2: 调用另一个模块中的函数
    try:
        # 假设有一个 alex_actions.py 模块
        # from alex_actions import welcome_alex
        # welcome_alex(similarity)
        print("可以调用 Alex 专属的函数")
    except Exception as e:
        print(f"调用 Alex 函数失败: {e}")

    # 示例3: 发送通知（需要安装相应的库）
    # send_notification(f"Alex 被检测到，相似度: {similarity:.2f}")


def handle_bob_detected(similarity: float, bbox: Tuple[int, int, int, int]):
    """
    处理检测到 Bob 的逻辑
    """
    print(">>> Bob 被检测到！执行 Bob 相关的操作...")

    # 运行 Bob 的处理程序
    try:
        bob_handler_path = os.path.join(os.path.dirname(__file__), 'bob_handler.py')
        if os.path.exists(bob_handler_path):
            print(f"运行 Bob 处理程序: {bob_handler_path}")
            subprocess.run(['python', bob_handler_path], check=True)
        else:
            print(f"Bob 处理程序不存在: {bob_handler_path}")
    except Exception as e:
        print(f"运行 Bob 处理程序失败: {e}")


def handle_other_face_detected(name: str, similarity: float, bbox: Tuple[int, int, int, int]):
    """
    处理检测到其他人脸的逻辑
    """
    print(f">>> 检测到其他人脸: {name}")
    # 可以记录日志或执行其他操作


# 辅助函数示例
def send_notification(message: str):
    """
    发送通知（示例函数）

    需要安装相应的通知库，例如：
    - plyer: pip install plyer
    - win10toast: pip install win10toast (Windows)
    """
    try:
        from plyer import notification
        notification.notify(
            title='人脸检测通知',
            message=message,
            app_icon=None,  # 可以设置图标路径
            timeout=10
        )
    except ImportError:
        print("通知库未安装，跳过通知发送")
    except Exception as e:
        print(f"发送通知失败: {e}")


def log_to_file(name: str, similarity: float, bbox: Tuple[int, int, int, int]):
    """
    将检测结果记录到日志文件

    Args:
        name: 人脸名称
        similarity: 相似度
        bbox: 边界框
    """
    log_file = 'data/detection_log.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    x1, y1, x2, y2 = bbox

    log_entry = f"[{timestamp}] {name} | 相似度: {similarity:.4f} | 位置: ({x1}, {y1}, {x2}, {y2})\n"

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)

    print(f"检测结果已记录到: {log_file}")


# 如果你想在每次检测时都记录日志，可以在 on_face_detected 函数中调用 log_to_file
# 例如：
# def on_face_detected(name: str, similarity: float, bbox: Tuple[int, int, int, int]):
#     log_to_file(name, similarity, bbox)
#     # ... 其他处理逻辑