# -*- coding: utf-8 -*-
"""
detector_watch_snapshots.py

监视目录中的“快照 CSV”（由 server_with_snapshots.py 生成），对每个新文件触发一次检测。

设计要点：
1) 不依赖 watchdog，采用轮询 + 文件稳定性检查，减少外部依赖
2) 默认“只分析最新一个 20s 窗口”（latest-window 模式），避免每秒重复跑全文件导致 CPU 爆炸
3) 可选 subprocess 模式：直接调用离线检测脚本的 CLI（兼容性最好，但更慢）

重要前提：
- 快照文件应“写完后再被检测端看到”。server_with_snapshots.py 采用写 .tmp 再 os.replace 的原子替换，
  所以检测端只需要忽略 .tmp 文件即可。
"""
import argparse
import fnmatch
import importlib.util
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def wait_file_stable(path: str, stable_seconds: float = 0.3, timeout: float = 10.0) -> bool:
    """
    防御性：如果快照端没有原子写，这里能避免读到“正在写入”的文件。
    现在快照端是原子替换，理论上不需要；但保留它以适配其它写法。
    """
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            st = os.stat(path)
            cur = (st.st_size, st.st_mtime_ns)
            if last is not None and cur == last:
                # 再等一小段确认稳定
                time.sleep(stable_seconds)
                st2 = os.stat(path)
                cur2 = (st2.st_size, st2.st_mtime_ns)
                return cur2 == cur
            last = cur
        except FileNotFoundError:
            return False
        time.sleep(stable_seconds)
    return False


def import_detector(detector_py: str):
    spec = importlib.util.spec_from_file_location("fall_detector_mod", detector_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入检测脚本：{detector_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def detect_latest_window(
    mod,
    csv_path: str,
    *,
    # 基础参数
    window_len_s: float,
    fs_target: float,
    fill_gap: str,
    # 检测参数（必须与离线脚本一致）
    hf_band: Tuple[float, float],
    lf_band: Tuple[float, float],
    tot_band: Tuple[float, float],
    active_abs_thr: float,
    active_pctl: float,
    denom_floor_pctl: float,
    score_thr: float,
    seg_dur_min_s: float,
    seg_dur_max_s: float,
    still_abs_thr: float,
    still_window_s: float,
    still_min_frac: float,
    still_min_run_s: float,
    still_grace_s: float,
    smooth_bins: int,
) -> Dict[str, Any]:
    """
    分析最新一个完整窗口，返回结果字典。
    """
    # 获取函数引用
    load_fn = mod.load_csi_all_from_rawcsv
    sim_fn = mod.compute_simB
    resample_fn = mod.resample_1d_with_gaps
    analyze_fn = mod.analyze_window

    # 1. 解码：返回 (csi_all, selected_ants, tx_ts_s_out)
    csi_all, selected_ants, tx_ts_s = load_fn(csv_path)

    # 2. 计算 sim_B
    sim_raw = sim_fn(csi_all)  # 一维数组

    # 3. 准备时间轴
    if tx_ts_s is None or not np.isfinite(tx_ts_s).any():
        # 无有效时间戳，使用索引时间轴（仅相对时间）
        ts_raw = np.arange(len(sim_raw), dtype=float) / fs_target
        base_abs = 0.0
    else:
        ts_raw = tx_ts_s
        base_abs = float(ts_raw[0])

    # 4. 重采样（均匀化），得到 sim_u 和空洞计数
    sim_u, gap_cnt = resample_fn(ts_raw, sim_raw, fs_target=fs_target, fill_gap=fill_gap)

    n = len(sim_u)
    win_n = int(round(window_len_s * fs_target))
    if n < win_n:
        return {
            "state": "NO_WINDOW",
            "reason": "not_enough_samples",
            "n": n,
            "need": win_n,
            "csv": os.path.basename(csv_path),
        }

    # 5. 取最后一个完整窗口
    start = n - win_n
    win = sim_u[start:]
    win_start_s = start / fs_target

    # 6. 调用 analyze_window，传入所有检测参数
    result = analyze_fn(
        win,
        fs=fs_target,
        win_start_s=win_start_s,
        hf_band=hf_band,
        lf_band=lf_band,
        tot_band=tot_band,
        active_abs_thr=active_abs_thr,
        active_pctl=active_pctl,
        denom_floor_pctl=denom_floor_pctl,
        score_thr=score_thr,
        seg_dur_min_s=seg_dur_min_s,
        seg_dur_max_s=seg_dur_max_s,
        still_abs_thr=still_abs_thr,
        still_window_s=still_window_s,
        still_min_frac=still_min_frac,
        still_min_run_s=still_min_run_s,
        still_grace_s=still_grace_s,
        smooth_bins=smooth_bins,
    )

    # 7. 包装结果，添加常用字段
    out = dict(result)
    out["csv"] = os.path.basename(csv_path)
    out["win_start_s"] = win_start_s
    out["n_samples"] = n
    out["gap_cnt"] = gap_cnt

    # 添加绝对时间戳（如果可用）
    if out.get("fall_found") and out.get("fall_time_s") is not None:
        out["fall_time_abs_s"] = base_abs + out["fall_time_s"]

    # 生成简洁的 state 字段（用于显示）
    if out.get("alarm"):
        out["state"] = "ALARM"
    elif out.get("fall_found"):
        out["state"] = "FALL_ONLY"
    else:
        out["state"] = "NO_EVENT"
    # 可选的详细原因
    out["reason"] = out.get("w_reason", "UNKNOWN")

    return out


def detect_by_subprocess(detector_py: str, csv_path: str, out_dir: str, extra_args: Optional[List[str]] = None) -> int:
    """
    兼容模式：每个快照启动一次离线脚本子进程去跑。
    优点：最少耦合；缺点：慢，且每次都会走“全文件滑窗”。
    """
    cmd = [sys.executable, detector_py, "--input", csv_path, "--out-dir", out_dir, "--summary-only", "--no-plots"]
    if extra_args:
        cmd.extend(extra_args)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # 把输出打出来，方便你直接看报警
    print(p.stdout, end="")
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-dir", required=True, help="监视快照目录")
    ap.add_argument("--pattern", default="raw_csi_frames2_live_*.csv", help="匹配快照文件名")
    ap.add_argument("--poll-interval", type=float, default=0.5, help="轮询间隔（秒）")
    ap.add_argument("--detector-py", required=True, help="离线检测脚本路径（解码检测代码.py）")
    ap.add_argument("--out-dir", default="./data/results", help="检测结果输出目录（subprocess 模式用）")
    ap.add_argument("--mode", choices=["latest-window", "subprocess"], default="latest-window")
    ap.add_argument("--window-len-s", type=float, default=20.0, help="窗口长度（秒）")
    ap.add_argument("--fs-target", type=int, default=1000, help="目标采样率（Hz）")
    ap.add_argument("--fill-gap", choices=["ffill", "interp", "zero"], default="ffill", help="空洞填充方式")
    ap.add_argument("--stable-check", action="store_true", help="额外做文件稳定性检查（一般不需要）")
    ap.add_argument("--once", action="store_true", help="处理完当前已有文件就退出（用于测试）")
    ap.add_argument("--log-jsonl", default="", help="把每次检测结果追加写到 jsonl（给实时系统对齐用）")

    # 检测参数（与解码检测代码.py 保持一致）
    ap.add_argument("--hf-band", type=float, nargs=2, default=[10.0, 30.0],
                    help="HF band (Hz) [default: 10 30]")
    ap.add_argument("--lf-band", type=float, nargs=2, default=[2.0, 5.0],
                    help="LF band (Hz) [default: 2 5]")
    ap.add_argument("--tot-band", type=float, nargs=2, default=[0.0, 90.0],
                    help="Total energy band (Hz) [default: 0 90]")
    ap.add_argument("--active-abs-thr", type=float, default=8e-5,
                    help="Absolute energy threshold for active point [default: 8e-5]")
    ap.add_argument("--active-pctl", type=float, default=90.0,
                    help="Percentile threshold for active point [default: 90]")
    ap.add_argument("--denom-floor-pctl", type=float, default=20.0,
                    help="Percentile for denominator floor [default: 20]")
    ap.add_argument("--score-thr", type=float, default=1.4,
                    help="Score threshold for fall decision [default: 1.4]")
    ap.add_argument("--seg-dur-min-s", type=float, default=0.5,
                    help="Minimum duration of suspected fall segment (s) [default: 0.5]")
    ap.add_argument("--seg-dur-max-s", type=float, default=4.0,
                    help="Maximum duration of suspected fall segment (s) [default: 4.0]")
    ap.add_argument("--still-abs-thr", type=float, default=5e-5,
                    help="Energy threshold for stillness [default: 5e-5]")
    ap.add_argument("--still-window-s", type=float, default=8.0,
                    help="Stillness check window length (s) [default: 8.0]")
    ap.add_argument("--still-min-frac", type=float, default=0.95,
                    help="Minimum fraction of below-threshold points in still window [default: 0.95]")
    ap.add_argument("--still-min-run-s", type=float, default=5.0,
                    help="Minimum consecutive stillness duration (s) [default: 5.0]")
    ap.add_argument("--still-grace-s", type=float, default=0.0,
                    help="Grace period after suspected segment before stillness check (s) [default: 0.0]")
    ap.add_argument("--smooth-bins", type=int, default=1,
                    help="Smoothing bins for score curve [default: 1]")

    args = ap.parse_args()

    watch_dir = os.path.abspath(args.watch_dir)
    os.makedirs(watch_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    mod = None
    if args.mode == "latest-window":
        mod = import_detector(args.detector_py)

    processed = set()
    jsonl_fp = None
    if args.log_jsonl:
        jsonl_fp = open(args.log_jsonl, "a", encoding="utf-8")

    try:
        while True:
            names = sorted(os.listdir(watch_dir))
            matched = [n for n in names if fnmatch.fnmatch(n, args.pattern) and (not n.endswith(".tmp"))]
            new_files = [n for n in matched if n not in processed]
            for name in new_files:
                path = os.path.join(watch_dir, name)
                if args.stable_check and (not wait_file_stable(path)):
                    continue

                if args.mode == "subprocess":
                    rc = detect_by_subprocess(args.detector_py, path, args.out_dir)
                    result = {"csv": name, "mode": "subprocess", "returncode": rc, "ts_wall": time.time()}
                else:
                    try:
                        result = detect_latest_window(
                            mod, path,
                            window_len_s=args.window_len_s,
                            fs_target=args.fs_target,
                            fill_gap=args.fill_gap,
                            hf_band=tuple(args.hf_band),
                            lf_band=tuple(args.lf_band),
                            tot_band=tuple(args.tot_band),
                            active_abs_thr=args.active_abs_thr,
                            active_pctl=args.active_pctl,
                            denom_floor_pctl=args.denom_floor_pctl,
                            score_thr=args.score_thr,
                            seg_dur_min_s=args.seg_dur_min_s,
                            seg_dur_max_s=args.seg_dur_max_s,
                            still_abs_thr=args.still_abs_thr,
                            still_window_s=args.still_window_s,
                            still_min_frac=args.still_min_frac,
                            still_min_run_s=args.still_min_run_s,
                            still_grace_s=args.still_grace_s,
                            smooth_bins=args.smooth_bins,
                        )
                        result["ts_wall"] = time.time()
                    except Exception as e:
                        result = {"csv": name, "mode": "latest-window", "error": str(e), "ts_wall": time.time()}

                # 输出到控制台（简短）
                state = result.get("state") or result.get("label") or result.get("result")
                print(f"[DETECT] {name} -> {state}")

                if jsonl_fp:
                    jsonl_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                    jsonl_fp.flush()

                processed.add(name)

            if args.once:
                break
            time.sleep(args.poll_interval)
    finally:
        if jsonl_fp:
            jsonl_fp.close()


if __name__ == "__main__":
    main()