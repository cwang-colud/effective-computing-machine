# -*- coding: utf-8 -*-
"""
server_with_snapshots.py

在“接收数据代码.py”的基础上增加“准实时快照导出”能力：
- TCP 接收线程持续收帧（固定 buf_len 切帧）
- 后台快照线程每隔 snapshot_interval 秒，把“最近 snapshot_window 秒”的帧写成一个独立 CSV（写 tmp 再原子替换）
- 可选：限制内存仅保留最近 max_keep 秒，避免 RAW_FRAMES 无限增长

用途：
1) 作为“采集服务器”：持续运行，不必等采集结束才落盘
2) 结合 detector_watch_snapshots.py，实现“采集同时检测”的准实时流水线
"""
import argparse
import csv
import logging
import os
import socket
import struct
import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple

# ----------------
# 日志
# ----------------
def setup_logger(log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("csi_server")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ----------------
# PCAP（可选）
# ----------------
def save_frames_pcap(pcap_path: str, frames: Deque[bytes], ts_list: Deque[float], buf_len: int, logger: logging.Logger) -> None:
    if not frames:
        logger.warning("PCAP：没有帧可保存：%s", pcap_path)
        return
    os.makedirs(os.path.dirname(pcap_path), exist_ok=True)
    with open(pcap_path, "wb") as f:
        # global header (little-endian, microsecond)
        global_header = struct.pack(
            "<IHHIIII",
            0xA1B2C3D4,  # magic_number
            2, 4,        # version
            0, 0,        # thiszone, sigfigs
            buf_len,     # snaplen
            147          # LINKTYPE_USER0
        )
        f.write(global_header)
        for ts, frame in zip(ts_list, frames):
            sec = int(ts)
            usec = int((ts - sec) * 1_000_000)
            incl_len = len(frame)
            pkt_header = struct.pack("<IIII", sec, usec, incl_len, incl_len)
            f.write(pkt_header)
            f.write(frame)
    logger.info("PCAP：已保存 %d 帧：%s", len(frames), pcap_path)


# ----------------
# CSV 快照（关键）
# ----------------
def write_csv_snapshot_atomic(csv_path: str, frames: Deque[bytes], ts_list: Deque[float], logger: logging.Logger) -> None:
    """
    原子写：先写 .tmp，再 os.replace 到最终文件名，避免检测端读到“半文件”。
    CSV 格式：
        timestamp,hex
        1731829383.123456,a1b2...
    """
    if not frames:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "hex"])
        for ts, fr in zip(ts_list, frames):
            w.writerow([f"{ts:.6f}", fr.hex()])
    os.replace(tmp_path, csv_path)
    logger.debug("快照：写入 %d 帧 -> %s", len(frames), csv_path)


# ----------------
# 服务器类
# ----------------
class CSIServer:
    def __init__(
        self,
        host: str,
        port: int,
        buf_len: int,
        out_dir: str,
        snapshot_interval: float,
        snapshot_window: float,
        max_keep: float,
        enable_snapshot: bool,
        log: logging.Logger,
    ):
        self.host = host
        self.port = port
        self.buf_len = buf_len
        self.out_dir = out_dir
        self.snapshot_interval = snapshot_interval
        self.snapshot_window = snapshot_window
        self.max_keep = max_keep
        self.enable_snapshot = enable_snapshot
        self.log = log

        self.raw_frames: Deque[bytes] = deque()
        self.frame_ts: Deque[float] = deque()

        self.raw_lock = threading.Lock()
        self.recv_buffer = b""
        self.recv_lock = threading.Lock()

        self.connection_alive = threading.Event()
        self.stop_event = threading.Event()

        self._client_thread: Optional[threading.Thread] = None
        self._snapshot_thread: Optional[threading.Thread] = None

    def _client_recv_loop(self, conn: socket.socket) -> None:
        self.log.info("clientThread started")
        try:
            while self.connection_alive.is_set() and (not self.stop_event.is_set()):
                try:
                    chunk = conn.recv(4096)
                    if not chunk:
                        self.log.warning("TCP 连接关闭（对端断开）")
                        break
                    with self.recv_lock:
                        # 防御性：防止 recv_buffer 异常膨胀
                        if len(self.recv_buffer) + len(chunk) > 100 * self.buf_len:
                            self.log.warning("recv_buffer 过大（%d bytes），丢弃旧数据防止 OOM", len(self.recv_buffer) + len(chunk))
                            self.recv_buffer = chunk
                        else:
                            self.recv_buffer += chunk

                    # 尽可能多切帧
                    frames_extracted = 0
                    while True:
                        with self.recv_lock:
                            if len(self.recv_buffer) < self.buf_len:
                                break
                            frame = self.recv_buffer[: self.buf_len]
                            self.recv_buffer = self.recv_buffer[self.buf_len :]

                        ts = time.time()
                        with self.raw_lock:
                            self.raw_frames.append(frame)
                            self.frame_ts.append(ts)
                        frames_extracted += 1

                    if frames_extracted:
                        self.log.debug("提取了 %d 帧", frames_extracted)

                except socket.timeout:
                    continue
                except Exception as e:
                    self.log.error("recv error: %s", e, exc_info=True)
                    break
        finally:
            self.connection_alive.clear()
            self.log.info("clientThread 退出")

    def _snapshot_loop(self) -> None:
        """
        每隔 snapshot_interval 秒：
        - 复制最近 snapshot_window 秒的帧
        - 原子写成一个新 CSV 文件（独立文件，便于检测端“只处理已完成文件”）
        - 可选：裁剪内存仅保留最近 max_keep 秒
        """
        self.log.info("snapshotThread started (interval=%.3fs, window=%.3fs, max_keep=%.3fs)", self.snapshot_interval, self.snapshot_window, self.max_keep)
        last_written = 0
        while (not self.stop_event.is_set()):
            time.sleep(self.snapshot_interval)
            now = time.time()
            with self.raw_lock:
                if not self.raw_frames:
                    continue

                # 选取最近 snapshot_window
                cutoff = now - self.snapshot_window
                # 找到起始索引（deque 无法直接二分，只能线性扫描；窗口秒数较小可接受）
                idx = 0
                for ts in self.frame_ts:
                    if ts >= cutoff:
                        break
                    idx += 1
                # 复制切片
                frames_list = list(self.raw_frames)[idx:]
                ts_list = list(self.frame_ts)[idx:]

                # 内存裁剪：仅保留最近 max_keep
                if self.max_keep > 0:
                    keep_cutoff = now - self.max_keep
                    while self.frame_ts and self.frame_ts[0] < keep_cutoff:
                        self.frame_ts.popleft()
                        self.raw_frames.popleft()

            # 写快照（不持锁，避免阻塞接收）
            if frames_list:
                # 以“快照末尾时间”作为文件名，便于排序
                stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
                csv_name = f"raw_csi_frames2_live_{stamp}.csv"
                csv_path = os.path.join(self.out_dir, csv_name)
                write_csv_snapshot_atomic(csv_path, deque(frames_list), deque(ts_list), self.log)

                # 打印帧数变化（降低刷屏）
                if len(frames_list) != last_written:
                    self.log.info("快照输出：%s（帧数=%d, 窗口=%.1fs）", csv_name, len(frames_list), self.snapshot_window)
                    last_written = len(frames_list)

        self.log.info("snapshotThread 退出")

    def serve_forever(self, stop_seconds: float = 0.0, stop_frames: int = 0) -> None:
        """
        stop_seconds=0 表示不按时间停止（Ctrl+C 停止）
        stop_frames=0 表示不按帧数停止
        """
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(1.0)

        conn = None
        t0 = time.time()

        try:
            srv.bind((self.host, self.port))
            srv.listen(5)
            self.log.info("服务器已在端口 %d 监听，等待客户端连接...", self.port)

            while conn is None and (not self.stop_event.is_set()):
                try:
                    conn, (ip, port) = srv.accept()
                    conn.settimeout(0.1)
                    self.log.info("接收到来自 %s:%d 的连接，开始采集", ip, port)
                except socket.timeout:
                    continue

            if conn is None:
                return

            self.connection_alive.set()
            self._client_thread = threading.Thread(target=self._client_recv_loop, args=(conn,), daemon=True)
            self._client_thread.start()

            if self.enable_snapshot:
                self._snapshot_thread = threading.Thread(target=self._snapshot_loop, daemon=True)
                self._snapshot_thread.start()

            last_frame_count = 0
            stall_count = 0

            while True:
                if self.stop_event.is_set():
                    break
                if not (self._client_thread.is_alive() and self.connection_alive.is_set()):
                    self.log.warning("接收线程已结束或连接已关闭，停止主循环")
                    break

                with self.raw_lock:
                    n = len(self.raw_frames)

                if n != last_frame_count:
                    self.log.info("已收集原始帧: %d", n)
                    last_frame_count = n
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > 50:
                        self.log.warning("数据接收可能停滞（约 5 秒无新帧）")
                        stall_count = 0

                elapsed = time.time() - t0
                if stop_seconds > 0 and elapsed >= stop_seconds:
                    self.log.info("达到 stop_seconds=%.2f，准备退出", stop_seconds)
                    break
                if stop_frames > 0 and n >= stop_frames:
                    self.log.info("达到 stop_frames=%d，准备退出", stop_frames)
                    break

                time.sleep(0.1)

        except KeyboardInterrupt:
            self.log.info("Ctrl+C：停止采集")
        except Exception as e:
            self.log.error("服务器错误: %s", e, exc_info=True)
        finally:
            self.stop_event.set()
            self.connection_alive.clear()

            # 关闭连接
            if conn:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
            try:
                srv.close()
            except Exception:
                pass

            # 等线程退出
            if self._client_thread and self._client_thread.is_alive():
                self._client_thread.join(timeout=2.0)
            if self._snapshot_thread and self._snapshot_thread.is_alive():
                self._snapshot_thread.join(timeout=2.0)

            self.log.info("服务器已关闭")

    def save_final(self, prefix: str = "raw_csi_frames2_final") -> Tuple[Optional[str], Optional[str]]:
        """
        手动保存当前缓存为：
        - CSV（全量）
        - PCAP（全量）
        """
        with self.raw_lock:
            frames = deque(self.raw_frames)
            ts_list = deque(self.frame_ts)

        if not frames:
            return None, None

        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        csv_path = os.path.join(self.out_dir, f"{prefix}_{stamp}.csv")
        pcap_path = os.path.join(self.out_dir, f"{prefix}_{stamp}.pcap")

        write_csv_snapshot_atomic(csv_path, frames, ts_list, self.log)
        save_frames_pcap(pcap_path, frames, ts_list, self.buf_len, self.log)
        return csv_path, pcap_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="", help="监听地址，默认 0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--buf-len", type=int, default=594, help="固定帧长度（bytes）")
    ap.add_argument("--out-dir", default="./data/raw", help="快照输出目录")
    ap.add_argument("--enable-snapshot", action="store_true", help="开启准实时快照输出")
    ap.add_argument("--snapshot-interval", type=float, default=1.0, help="快照输出间隔（秒）")
    ap.add_argument("--snapshot-window", type=float, default=40.0, help="每次快照包含的最近秒数（建议 >= 20+8+冗余）")
    ap.add_argument("--max-keep", type=float, default=120.0, help="内存最多保留的最近秒数（0=不裁剪）")
    ap.add_argument("--stop-seconds", type=float, default=0.0, help="运行多久后自动停止（0=不停止）")
    ap.add_argument("--stop-frames", type=int, default=0, help="收满多少帧后自动停止（0=不停止）")
    ap.add_argument("--log-file", default="server_log.txt")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--save-final", action="store_true", help="退出时保存全量 CSV+PCAP 到 out-dir")
    args = ap.parse_args()

    logger = setup_logger(args.log_file, args.log_level)
    srv = CSIServer(
        host=args.host,
        port=args.port,
        buf_len=args.buf_len,
        out_dir=args.out_dir,
        snapshot_interval=args.snapshot_interval,
        snapshot_window=args.snapshot_window,
        max_keep=args.max_keep,
        enable_snapshot=args.enable_snapshot,
        log=logger,
    )

    srv.serve_forever(stop_seconds=args.stop_seconds, stop_frames=args.stop_frames)

    if args.save_final:
        csv_path, pcap_path = srv.save_final(prefix="raw_csi_frames2_final")
        logger.info("最终保存：CSV=%s, PCAP=%s", csv_path, pcap_path)


if __name__ == "__main__":
    main()
