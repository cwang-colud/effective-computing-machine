# replay_csv_to_tcp_1000hz.py
# 功能（只做两件事）：
# 1) 按 CSV 行顺序读取 (timestamp, hex)，逐行发送对应的原始帧 bytes（不排序、不重排）
# 2) 发送节奏：固定 1000 Hz（每 0.001 s 发送 1 帧）
#
# 说明（事实约束）：
# - 你当前 CSV 的 timestamp 很可能来自接收端 time.time()（不是设备端 1ms 节奏），
#   所以如果严格按 timestamp 差值 sleep，并不能保证 1000Hz。
# - Python 的 time.sleep(0.001) 在不同 OS 上精度不一；此脚本采用“按目标时刻调度”的方式，
#   尽量贴近 1000Hz，但实际精度受系统调度影响。

import argparse
import csv
import socket
import time


def replay_1000hz(csv_path: str, host: str, port: int) -> None:
    period = 0.001  # 1000 Hz
    with socket.create_connection((host, port)) as s:
        next_t = time.perf_counter()

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue

                # 兼容表头：timestamp 不能解析就跳过（不重排顺序，只跳过无效行）
                try:
                    float(row[0])
                except Exception:
                    continue

                hex_str = row[1].strip()
                try:
                    frame = bytes.fromhex(hex_str)
                except Exception as e:
                    raise ValueError(f"Invalid hex: {e}")

                # 按 1000Hz 节奏调度
                now = time.perf_counter()
                wait = next_t - now
                if wait > 0:
                    time.sleep(wait)

                s.sendall(frame)
                next_t += period


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV file: timestamp,hex")
    ap.add_argument("--host", required=True, help="TCP server host/IP")
    ap.add_argument("--port", required=True, type=int, help="TCP server port")
    args = ap.parse_args()

    replay_1000hz(args.input, args.host, args.port)


if __name__ == "__main__":
    main()
