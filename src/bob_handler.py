"""
Bob 专属处理程序
当检测到 Bob 时，此文件会被执行
"""

from datetime import datetime


def main():
    """主函数"""
    print("=" * 50)
    print("Bob 处理程序已启动")
    print("=" * 50)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] 检测到 Bob！")

    # 在这里添加 Bob 专属的处理逻辑
    print("正在执行 Bob 专属操作...")
    print("1. 你好，Bob！")
    print("2. 已为 Bob 准备工作环境")
    print("3. 已记录 Bob 的到达时间")

    # 示例：保存到日志文件
    log_bob_detection()

    print("Bob 处理程序执行完成")
    print("=" * 50)


def log_bob_detection():
    """记录 Bob 的检测信息到日志"""
    import os
    log_file = 'data/bob_log.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Bob 被检测到\n"

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)

    print(f"Bob 的检测信息已记录到: {log_file}")


if __name__ == '__main__':
    main()