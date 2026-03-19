"""
Alex 专属处理程序
当检测到 Alex 时，此文件会被执行
"""

from datetime import datetime


def main():
    """主函数"""
    print("=" * 50)
    print("Alex 处理程序已启动")
    print("=" * 50)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] 检测到 Alex！")

    # 在这里添加 Alex 专属的处理逻辑
    # 例如：
    # 1. 发送欢迎消息
    # 2. 打开特定的应用程序
    # 3. 播放特定的音效
    # 4. 调整系统设置
    # 5. 记录到数据库
    # 6. 发送邮件或短信通知

    print("正在执行 Alex 专属操作...")
    print("1. 欢迎回来，Alex！")
    print("2. 系统已为 Alex 优化")
    print("3. 已记录 Alex 的到达时间")

    # 示例：保存到日志文件
    log_alex_detection()

    print("Alex 处理程序执行完成")
    print("=" * 50)


def log_alex_detection():
    """记录 Alex 的检测信息到日志"""
    import os
    log_file = 'data/alex_log.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Alex 被检测到\n"

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)

    print(f"Alex 的检测信息已记录到: {log_file}")


if __name__ == '__main__':
    main()