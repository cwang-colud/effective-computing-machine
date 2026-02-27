# -*- coding: utf-8 -*-
"""
Rule-based fall detection (real-time sliding-window semantics) - v3

Implements your design (2026-01-28) with these guarantees:
(1) Preprocess pipeline is consistent with 跌倒检测完整版.py:
    1) raw CSV -> CSI + tx_ts_s (frame-tail 16B or timestamp columns)
    2) CSI -> sim_B (1D activity series)
    3) resample sim_B to uniform fs using real time axis + gap handling (gap_ms / fill_gap)
    4) STFT (same params as baseline)
(2) Fixes highest-priority issue: window start time / fall time / still(8s) are computed on the real-time axis
    (after resampling), not by "index/fs" on raw frames.
(3) Fixes second-priority issue: summary table includes diagnostic fields:
    - why label is NO_EVENT / NO_FALL / FALL_ONLY / ALARM
    - counts and time ranges of action segments and suspected segments (in the chosen window)
    - still_len_s (how long stillness actually lasted)
    - energy statistics (mean/p90/max) in the chosen window

Detection rules (as you specified):
- window_len = 20s, step = 1s, sliding over uniform fs series
- HF band: 10–30 Hz
- LF band: 2–5 Hz
- Active point: (E_tot > 8e-5) AND (E_tot > p90(E_tot))
- Action segments: contiguous active points
- Suspected fall segment: 0.5s < duration < 4s
- Fall decision: score_raw = E_hf / (E_lf + denom_floor) > 1.4
- Stillness: after suspected segment END, within next 8s, all E_tot < 5e-5
- Alarm: fall AND still

Outputs:
- summary_rule_predictions.csv
- plots/*.png (STFT heatmap + segments + ratio curve + still window)

IMPORTANT:
- Absolute thresholds 8e-5 / 5e-5 will drift with gain/position; this script does NOT "solve drift",
  it only implements the rule faithfully.
"""

from __future__ import annotations

# ====== baseline code (kept consistent with 跌倒检测完整版.py) ======
from __future__ import annotations
import argparse
import os
import glob
import re
import fnmatch
import struct
import hashlib
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# ====== 中文字体修复（避免 DejaVu Sans 缺字告警） ======
# 说明：
# - Windows 通常有 "Microsoft YaHei"/"SimHei"；
# - Linux 常见 "Noto Sans CJK SC"/"Source Han Sans SC"/"WenQuanYi Zen Hei"；
# - 若仍出现缺字告警，请安装任一 CJK 字体后重试，或设置环境变量 MPL_CN_FONT 指定字体名。
from matplotlib import font_manager as _font_manager

def _setup_mpl_chinese_font() -> str | None:
    env_font = os.environ.get("MPL_CN_FONT", "").strip()
    candidates = [env_font] if env_font else []
    candidates += [
        # Windows
        "Microsoft YaHei",
        "SimHei",
        "Microsoft JhengHei",
        # macOS
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        # Linux / cross-platform
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    avail = {f.name for f in _font_manager.fontManager.ttflist}
    for name in candidates:
        if name and (name in avail):
            matplotlib.rcParams["font.sans-serif"] = [name] + list(matplotlib.rcParams.get("font.sans-serif", []))
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    # 至少修复负号显示
    matplotlib.rcParams["axes.unicode_minus"] = False
    return None

_CN_FONT_USED = _setup_mpl_chinese_font()

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import stft
from numpy.linalg import norm

EPS = 1e-12
ETOT_UNIT = 1e-5  # 10e-6：以 1e-5 为单位缩放显示 E_tot，避免设置过密刻度导致 MAXTICKS


# ====== 画框颜色（全局统一） ======
BOX_COLOR_ACTION = "#00AA00"   # 绿色：动作段
BOX_COLOR_SUSPECT = "#CC0000"  # 红色：疑似段
BOX_COLOR_CHOSEN = "#FFD400"   # 黄色：选中段/选中窗口
BOX_COLOR_STILL = "#FF8C00"    # 橙色：静止检查窗
BOX_COLOR_SUSPECT_WIN = "#0000FF"  # 蓝色：窗口图中的疑似段


# ==================== 解码相关函数 ====================
def _segs_to_str(segs: list[tuple[int, int]], t: np.ndarray, win_start_s: float) -> str:
    if not segs:
        return ""
    parts = []
    for L, R in segs:
        a = win_start_s + float(t[L])
        b = win_start_s + float(t[R])
        parts.append("[{:.3f},{:.3f}]".format(a, b))
    return ";".join(parts)

def hampel_filter_1d(x: np.ndarray, window_size: int = 11, n_sigmas: float = 3.0) -> np.ndarray:
    """Hampel滤波器去除1D序列中的异常值"""
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1:
        raise ValueError("输入必须为1D数组")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size必须为≥3的奇数")
    if n_sigmas <= 0:
        raise ValueError("n_sigmas必须>0")

    s = pd.Series(x_arr)
    med = s.rolling(window=window_size, center=True).median()
    diff = (s - med).abs()
    mad = diff.rolling(window=window_size, center=True).median()
    sigma = 1.4826 * mad

    out = s.copy()
    mask = diff > (n_sigmas * sigma)
    out[mask] = med[mask]
    out = out.fillna(s)
    return out.to_numpy(dtype=float)

def _find_hex_column(df: pd.DataFrame) -> str:
    """自动识别CSV里存放帧十六进制字符串的列名"""
    for c in ["hex", "hex_data", "data", "payload"]:
        if c in df.columns:
            return c
    raise ValueError(f"CSV缺少帧十六进制列，期望包含: hex / hex_data / data / payload，实际列={list(df.columns)}")

def _decode_tail16_sec_sub(frame: bytes, endian: str = "<", fmt: str = "qq") -> Optional[Tuple[int, int]]:
    """解析帧尾16B -> (sec, sub)。解析失败返回None。"""
    if frame is None or len(frame) < 16:
        return None
    try:
        sec, sub = struct.unpack(endian + fmt, frame[-16:])
        return int(sec), int(sub)
    except Exception:
        return None

def _infer_unit_div(sub_values: np.ndarray) -> Tuple[float, str]:
    """根据sub取值范围推断单位（ms/us/ns），返回(div, unit_name)"""
    sub_values = np.asarray(sub_values, dtype=np.float64)
    sub_values = sub_values[np.isfinite(sub_values)]
    if sub_values.size == 0:
        return 1e6, "us"
    mx = float(np.nanmax(sub_values))
    # 常见：微秒 [0,1e6)，纳秒 [0,1e9)，毫秒 [0,1e3)
    if mx < 2e3:
        return 1e3, "ms"
    if mx < 2e6:
        return 1e6, "us"
    if mx < 2e9:
        return 1e9, "ns"
    return 1e6, "us"

def resample_1d_with_gaps(
    ts_s: np.ndarray,
    x: np.ndarray,
    fs_target: float,
    gap_ms: float = 50.0,
    fill_gap: str = "ffill",
) -> Tuple[np.ndarray, int]:
    """
    用真实时间轴把1D序列重采样到均匀采样。
    - 先根据 gap_ms 找到"大空洞"，把空洞覆盖区间标为 NaN
    - 再按 fill_gap 策略填充 NaN，避免跨空洞硬插值导致的虚假"平滑运动"
    """
    ts_s = np.asarray(ts_s, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if ts_s.ndim != 1 or x.ndim != 1 or ts_s.size != x.size or ts_s.size < 2:
        return x.copy(), 0

    # 相对时间，避免 epoch 大数精度问题
    t = ts_s - ts_s[0]
    valid = np.isfinite(t) & np.isfinite(x)
    if np.sum(valid) < 2:
        return x.copy(), 0

    t = t[valid]
    x = x[valid]

    if fs_target <= 0:
        return x.copy(), 0

    # 保证单调（少量乱序/重复直接排序+去重）
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    # 去重
    keep = np.ones_like(t, dtype=bool)
    keep[1:] = np.diff(t) > 0
    t = t[keep]
    x = x[keep]
    if t.size < 2:
        return x.copy(), 0

    dt = np.diff(t)
    gap_thr = float(gap_ms) / 1000.0
    gaps = np.where(dt > gap_thr)[0]
    gap_cnt = int(gaps.size)

    t0 = float(t[0])
    t1 = float(t[-1])
    n_u = int(np.floor((t1 - t0) * fs_target)) + 1
    if n_u < 2:
        return x.copy(), 0
    t_u = t0 + np.arange(n_u, dtype=np.float64) / fs_target

    # 先插值到均匀网格（不处理空洞）
    x_u = np.interp(t_u, t, x).astype(np.float64)

    # 把空洞覆盖区间置 NaN
    if gap_cnt > 0:
        for gi in gaps:
            left = float(t[gi])
            right = float(t[gi + 1])
            m = (t_u > left) & (t_u < right)
            x_u[m] = np.nan

    # 填充 NaN
    if np.isnan(x_u).any():
        s = pd.Series(x_u)
        if fill_gap == "interp":
            s = s.interpolate(limit_direction="both")
        elif fill_gap == "zero":
            s = s.fillna(0.0)
        else:  # ffill
            s = s.ffill().bfill()
        x_u = s.to_numpy(dtype=np.float64)

    return x_u, gap_cnt

def _extract_timestamps_from_df(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    从CSV中提取真实时间戳（秒）。用于用真实时间轴代替"帧数/采样率"估算。
    支持：timestamp/ts/time/t 及 *_ms
    找不到则返回None。
    """
    cand_cols = ["timestamp", "ts", "time", "t", "local_ts", "timestamp_ms", "time_ms", "ts_ms"]
    for col in cand_cols:
        if col not in df.columns:
            continue
        vals = df[col].to_numpy()
        try:
            vals = vals.astype(float)
        except Exception:
            try:
                vals = np.array([float(x) for x in vals], dtype=float)
            except Exception:
                continue
        if col.endswith("_ms"):
            vals = vals / 1000.0
        if vals.size >= 2 and np.all(np.isfinite(vals)):
            return vals
    return None

def _sanitize_key_part(s: str) -> str:
    s = s.strip().replace(" ", "")
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def _parse_path_fields(csv_path: str, data_root: str) -> Dict[str, str]:
    abs_csv = os.path.abspath(csv_path)
    abs_root = os.path.abspath(data_root)
    rel = os.path.relpath(abs_csv, abs_root)
    parts = [p for p in rel.split(os.sep) if p]

    scene = parts[0] if len(parts) >= 2 else ""
    action = parts[1] if len(parts) >= 3 else ""
    position = parts[2] if len(parts) >= 4 else ""
    trial = parts[3] if len(parts) >= 5 else ""

    known_actions = {"坐下", "摔倒", "摔倒与行走", "行走", "静止", "站立", "躺下", "起身", "翻身"}
    if not action or action not in known_actions:
        for p in parts:
            if p in known_actions:
                action = p
                break
    if not position:
        for p in parts:
            if "号" in p:
                position = p
                break
    if not trial:
        for p in parts:
            if re.fullmatch(r"\d+", p):
                trial = p
                break

    return {"scene": scene, "action": action, "position": position, "trial": trial, "rel_path": rel}

def _make_stable_file_key(fields: Dict[str, str]) -> str:
    """使用原始文件夹名和文件名作为key"""
    rel_path = fields.get("rel_path", "")
    
    # 获取目录部分和文件名部分
    dir_part = os.path.dirname(rel_path)
    file_name = os.path.splitext(os.path.basename(rel_path))[0]
    
    # 清理目录名和文件名中的非法字符
    if dir_part:
        dir_name = os.path.basename(dir_part)
        dir_name_clean = _sanitize_key_part(dir_name)
        file_name_clean = _sanitize_key_part(file_name)
        return f"{dir_name_clean}_{file_name_clean}"
    
    return _sanitize_key_part(file_name)

def _collect_csv_files(root_dir: str, pattern: str) -> List[str]:
    files: List[str] = []
    for root, dirnames, filenames in os.walk(root_dir):
        if "simB_out" in root:
            continue
        for fn in filenames:
            if fnmatch.fnmatch(fn, pattern):
                files.append(os.path.join(root, fn))
    return sorted(set(files))

class DirectMemoryDecoderOffline:
    """Nexmon CSI解码器"""
    
    def __init__(self, device: str = "rtac86u") -> None:
        self.device = device
        self._setup_device_params()
        self.nulls_20 = [0, 1, 2, 3, 4, 11, 25, 32, 39, 53, 60, 61, 62, 63]
        self.nulls_40 = [0, 1, 2, 3, 4, 5, 11, 39, 53, 63, 64, 65, 66, 75, 89, 117, 123, 124, 125, 126, 127]

    def _setup_device_params(self) -> None:
        if self.device in ("rtac86u", "bcm4366c0"):
            self.nman = 12
            self.nexp = 6
        elif self.device in ("nexus6p", "bcm4358"):
            self.nman = 9
            self.nexp = 5
        else:
            self.nman = 12
            self.nexp = 6

    @staticmethod
    def extract_antenna_id(frame: bytes) -> int:
        if len(frame) < 62:
            return -1
        return frame[61] & 0x0F

    def extract_csi_from_frame(self, frame: bytes) -> Optional[np.ndarray]:
        csi_start = 42 + 18 + 6
        trailer_len = 16

        if len(frame) <= csi_start + trailer_len:
            return None

        csi_bytes = frame[csi_start:-trailer_len]
        if len(csi_bytes) < 64 * 4:
            return None

        n_words = len(csi_bytes) // 4
        if n_words not in (64, 128, 256, 512):
            return None

        csi_raw = csi_bytes[: n_words * 4]
        return np.frombuffer(csi_raw, dtype=np.uint32)

    def unpack_csi_direct(self, csi_uint32: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if csi_uint32 is None or csi_uint32.size == 0:
            return None

        csi_flat = csi_uint32.flatten().astype(np.int64)

        nman = self.nman
        nexp = self.nexp

        mask_iq = (1 << (nman - 1)) - 1
        mask_ex = (1 << (nexp - 1)) - 1
        mask_sign_i = 1 << (nexp + 2 * nman - 1)
        mask_sign_q = 1 << (nexp + 1 * nman - 1)

        value_i = np.bitwise_and(csi_flat, mask_iq << (nexp + nman)).astype(np.int64)
        value_q = np.bitwise_and(csi_flat, mask_iq << nexp).astype(np.int64)
        value_e = np.bitwise_and(csi_flat, mask_ex).astype(np.int64)

        value_i = np.right_shift(value_i, nexp + nman)
        value_q = np.right_shift(value_q, nexp)

        sign_i = np.bitwise_and(csi_flat, mask_sign_i).astype(np.int64)
        sign_q = np.bitwise_and(csi_flat, mask_sign_q).astype(np.int64)
        value_i[sign_i != 0] *= -1
        value_q[sign_q != 0] *= -1

        value_e = value_e + 10
        value_e = np.power(2.0, value_e)

        value_i = value_i * value_e
        value_q = value_q * value_e

        stacked = np.stack((value_i, value_q), axis=1).astype(np.float32)
        complex_view = stacked.view(np.complex64).flatten()
        unpacked = complex_view.reshape(csi_uint32.shape)

        unpacked = np.fft.fftshift(unpacked, axes=-1)
        
        nsub = unpacked.shape[-1]
        if nsub == 64:
            unpacked = np.delete(unpacked, self.nulls_20, axis=-1)
        elif nsub == 128:
            unpacked = np.delete(unpacked, self.nulls_40, axis=-1)

        return unpacked

def load_csi_all_from_rawcsv(
    csv_path: str,
    device: str = "rtac86u",
    max_frames: int = 0,
    endian: str = "<",
    fmt: str = "qq",
    max_seconds: float = 20.0,
) -> Tuple[np.ndarray, List[int], Optional[np.ndarray]]:
    """从CSV加载CSI数据并解码，同时优先解析帧尾16B (sec, sub) 作为发送端真实时间轴。

    返回: (csi_all, selected_ants, tx_ts_s)
    - csi_all: [frames, subcarriers, 3]
    - selected_ants: 选中的3根天线ID
    - tx_ts_s: [frames] 发送端时间戳(秒)。若无法解析则返回 None
    """
    df = pd.read_csv(csv_path)
    hex_col = _find_hex_column(df)

    # 旧格式兼容：如果CSV里本身带 timestamp 列，可作为兜底
    ts_fallback = _extract_timestamps_from_df(df)

    decoder = DirectMemoryDecoderOffline(device=device)
    antenna_csi: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: [], 3: []}
    antenna_sec: Dict[int, List[float]] = {0: [], 1: [], 2: [], 3: []}
    antenna_sub: Dict[int, List[float]] = {0: [], 1: [], 2: [], 3: []}
    antenna_ts_fb: Dict[int, List[float]] = {0: [], 1: [], 2: [], 3: []}

    total_rows = len(df)
    use_rows = min(total_rows, max_frames) if max_frames > 0 else total_rows

    for idx in range(use_rows):
        hex_str = str(df.loc[idx, hex_col]).strip()
        if not hex_str:
            continue
        try:
            frame = bytes.fromhex(hex_str)
        except ValueError:
            continue

        ant = decoder.extract_antenna_id(frame)
        if ant not in antenna_csi:
            continue

        csi_uint32 = decoder.extract_csi_from_frame(frame)
        if csi_uint32 is None:
            continue

        csi = decoder.unpack_csi_direct(csi_uint32)
        if csi is None:
            continue

        tail = _decode_tail16_sec_sub(frame, endian=endian, fmt=fmt)
        if tail is None:
            sec, sub = (np.nan, np.nan)
        else:
            sec, sub = tail

        antenna_csi[ant].append(csi)
        antenna_sec[ant].append(float(sec))
        antenna_sub[ant].append(float(sub))

        if ts_fallback is not None and idx < len(ts_fallback):
            antenna_ts_fb[ant].append(float(ts_fallback[idx]))

    active_ants = [a for a, lst in antenna_csi.items() if len(lst) > 0]
    if len(active_ants) < 3:
        raise RuntimeError(f"活跃天线不足: {active_ants}，至少需要3根")

    active_ants_sorted = sorted(active_ants, key=lambda a: len(antenna_csi[a]), reverse=True)
    selected_ants = active_ants_sorted[:3]

    min_len = min(len(antenna_csi[a]) for a in selected_ants)
    if min_len == 0:
        raise RuntimeError("选中天线帧数为0")

    csi_list = [np.array(antenna_csi[a][:min_len]) for a in selected_ants]
    csi_all = np.stack(csi_list, axis=-1)

    # 组装 TX 时间戳（sec + sub/unit_div）
    subs_all = []
    for a in selected_ants:
        subs_all.extend([v for v in antenna_sub[a][:min_len] if np.isfinite(v)])
    unit_div, unit_name = _infer_unit_div(np.asarray(subs_all, dtype=np.float64))

    ts_stack = []
    for a in selected_ants:
        sec_arr = np.asarray(antenna_sec[a][:min_len], dtype=np.float64)
        sub_arr = np.asarray(antenna_sub[a][:min_len], dtype=np.float64)
        ts_stack.append(sec_arr + sub_arr / unit_div)
    ts_stack = np.vstack(ts_stack).T  # [min_len, 3]
    tx_ts_s = np.nanmedian(ts_stack, axis=1).astype(np.float64)

    # 如果帧尾16B基本不可用，兜底用CSV里的timestamp列（如果存在）
    if not np.isfinite(tx_ts_s).any():
        fb_stack = []
        ok = True
        for a in selected_ants:
            arr = np.asarray(antenna_ts_fb[a][:min_len], dtype=np.float64)
            if arr.size < min_len:
                ok = False
                break
            fb_stack.append(arr)
        if ok and fb_stack:
            fb_stack = np.vstack(fb_stack).T
            fb_ts = np.nanmedian(fb_stack, axis=1).astype(np.float64)
            if np.isfinite(fb_ts).any():
                tx_ts_s = fb_ts

    # 时间截取：只保留前 max_seconds 秒的数据
    if tx_ts_s is not None and np.isfinite(tx_ts_s).any():
        # 计算相对时间（从第一个有效时间戳开始）
        tx_ts_s_relative = tx_ts_s - np.nanmin(tx_ts_s)
        
        # 找到小于等于 max_seconds 的索引
        valid_indices = tx_ts_s_relative <= max_seconds
        
        # 应用截取
        csi_all = csi_all[valid_indices]
        tx_ts_s = tx_ts_s[valid_indices]
        
        print(f"  时间截取：保留前 {max_seconds} 秒，{np.sum(valid_indices)}/{len(valid_indices)} 帧")
    else:
        print(f"  时间截取：无有效时间戳，使用所有帧")

    if not np.isfinite(tx_ts_s).any():
        tx_ts_s_out: Optional[np.ndarray] = None
    else:
        tx_ts_s_out = tx_ts_s

    print(
        f"  活跃天线数据量: { {a: len(antenna_csi[a]) for a in antenna_csi} }, "
        f"使用天线: {selected_ants}, 帧数={csi_all.shape[0]}"
    )
    if tx_ts_s_out is not None:
        print(f"  时间轴: 帧尾16B {endian}{fmt}, sub单位推断={unit_name}")
    else:
        print("  时间轴: 未解析到帧尾16B/CSV timestamp，将回退到 index 时间轴")

    return csi_all, selected_ants, tx_ts_s_out

# ==================== 跌倒检测专用函数 ====================

def compute_simB(csi_all: np.ndarray) -> np.ndarray:
    """sim_B：每帧三天线幅度向量与全1向量的cos相似度（取绝对值后平均）"""
    if csi_all.ndim != 3 or csi_all.shape[2] != 3:
        raise ValueError(f"csi_all 形状应为 [N,S,3]，实际 {csi_all.shape}")
    amp = np.abs(csi_all).astype(np.float64)
    ones = np.ones((amp.shape[1],), dtype=np.float64)
    ones_norm = np.linalg.norm(ones) + EPS
    sim = np.zeros((amp.shape[0],), dtype=np.float64)
    for i in range(amp.shape[0]):
        vals = []
        for a in range(3):
            v = amp[i, :, a]
            vals.append(float(np.dot(v, ones) / ((np.linalg.norm(v) + EPS) * ones_norm)))
        sim[i] = float(np.mean(np.abs(vals)))
    return sim

def stft_Z(x: np.ndarray, fs: float):
    f, t, Z = stft(
        x, fs=fs, window="hann",
        nperseg=512, noverlap=384, nfft=2048,
        detrend="constant",
    )
    return f.astype(float), t.astype(float), Z

def contiguous_segment_containing_peak(active: np.ndarray, peak_i: int):
    """返回 active 中包含 peak_i 的连续段 [L,R]（若峰点不在active则退化为单点）"""
    if active.size == 0:
        return 0, 0
    if (not np.any(active)) or (peak_i < 0) or (peak_i >= active.size) or (not active[peak_i]):
        return peak_i, peak_i
    L = peak_i
    while L - 1 >= 0 and active[L - 1]:
        L -= 1
    R = peak_i
    while R + 1 < active.size and active[R + 1]:
        R += 1
    return L, R

def find_segments(active: np.ndarray) -> list[tuple[int, int]]:
    """返回所有连续 True 段的 [L,R]（闭区间）"""
    segs = []
    i = 0
    n = active.size
    while i < n:
        if not active[i]:
            i += 1
            continue
        L = i
        while i + 1 < n and active[i + 1]:
            i += 1
        R = i
        segs.append((L, R))
        i += 1
    return segs

def glob_csv_files(inp: str, pattern: str, recursive: bool) -> list[str]:
    """
    pattern：支持一个 * 的简单通配，比如 raw_csi_frames2_*.csv
    """
    inp = os.path.abspath(inp)
    if os.path.isfile(inp):
        return [inp]
    if not os.path.isdir(inp):
        raise ValueError(f"--input 既不是文件也不是目录：{inp}")

    if "*" in pattern:
        pre, suf = pattern.split("*", 1)
    else:
        pre, suf = pattern, ""

    def match(fn: str) -> bool:
        if "*" not in pattern:
            return fn == pattern
        return fn.startswith(pre) and fn.endswith(suf)

    out: list[str] = []
    if recursive:
        for root, _ds, fs in os.walk(inp):
            for fn in fs:
                if fn.lower().endswith(".csv") and match(fn):
                    out.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(inp):
            if fn.lower().endswith(".csv") and match(fn):
                out.append(os.path.join(inp, fn))
    return sorted(out)

def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    if k <= 1:
        return x.astype(np.float64, copy=True)
    kernel = np.ones(k, dtype=np.float64) / float(k)
    return np.convolve(x.astype(np.float64), kernel, mode="same")



# ====== override: make decode-max-seconds<=0 mean "no truncation" (and print truthful message) ======
# 需求：参考 v1 输出，显示 “时间截取：不截断（保留全量帧）”
# 做法：仍复用 baseline 的解码/时间戳解析/截断逻辑（max_seconds=1e12 => 等效不截断），
# 但对 baseline 的打印进行捕获并替换为真实语义的打印。
import contextlib
import io as _io
import re as _re

_load_csi_all_from_rawcsv__baseline = load_csi_all_from_rawcsv

def load_csi_all_from_rawcsv(
    csv_path: str,
    device: str = "rtac86u",
    max_frames: int = 0,
    endian: str = "<",
    fmt: str = "qq",
    max_seconds: float = 20.0,
):
    if max_seconds is None or float(max_seconds) <= 0.0:
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            ret = _load_csi_all_from_rawcsv__baseline(
                csv_path=csv_path,
                device=device,
                max_frames=max_frames,
                endian=endian,
                fmt=fmt,
                max_seconds=float(1e12),  # effectively no truncation
            )
        cap = buf.getvalue()
        mm = _re.search(r"(\d+)\s*/\s*(\d+)\s*帧", cap)
        if mm:
            kept, total = mm.group(1), mm.group(2)
            print(f"  时间截取：不截断（保留全量帧），{kept}/{total} 帧")
        else:
            print("  时间截取：不截断（保留全量帧）")
        return ret

    # normal truncation path: keep baseline printing
    return _load_csi_all_from_rawcsv__baseline(
        csv_path=csv_path,
        device=device,
        max_frames=max_frames,
        endian=endian,
        fmt=fmt,
        max_seconds=float(max_seconds),
    )

# ====== extra imports for this script ======
import argparse
import fnmatch
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================
# Rule engine (per window)
# ===========================

def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if k is None or int(k) <= 1 or x.size == 0:
        return x.copy()
    k = int(k)
    w = np.ones((k,), dtype=float) / float(k)
    return np.convolve(x, w, mode="same")


class WindowResult(dict):
    pass


def analyze_window(
    sim_u: np.ndarray,
    fs: float,
    win_start_s: float,
    *,
    hf_band: tuple[float, float],
    lf_band: tuple[float, float],
    tot_band: tuple[float, float],
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
) -> WindowResult:
    f, t, Z = stft_Z(sim_u, fs=fs)
    P = np.abs(Z) ** 2

    hf0, hf1 = float(min(hf_band)), float(max(hf_band))
    lf0, lf1 = float(min(lf_band)), float(max(lf_band))
    tot0, tot1 = float(min(tot_band)), float(max(tot_band))

    hf_mask = (f >= hf0) & (f < hf1)
    lf_mask = (f >= lf0) & (f < lf1)
    tot_mask = (f >= tot0) & (f <= tot1)
    if (not np.any(hf_mask)) or (not np.any(lf_mask)) or (not np.any(tot_mask)):
        raise ValueError("HF/LF/总能量 频段不在频率轴范围内，请检查参数与 fs/STFT 设置")

    E_hf = P[hf_mask, :].sum(axis=0)
    E_lf = P[lf_mask, :].sum(axis=0)
    E_tot = P[tot_mask, :].sum(axis=0)

    denom_floor = float(np.percentile(E_lf, float(denom_floor_pctl))) + EPS
    score_raw = E_hf / (E_lf + denom_floor)
    # Gate HF/LF score by total energy: only keep score where E_tot > active_abs_thr, else set to 0
    _gate_thr = float(active_abs_thr)
    if E_tot.size == score_raw.size:
        score_raw = score_raw.astype(np.float64, copy=True)
        score_raw[E_tot <= _gate_thr] = 0.0
    _ = moving_average(score_raw, int(smooth_bins))  # 仅用于调试/可视化

    ratio_bounded = E_hf / (E_hf + E_lf + EPS)  # HF/(HF+LF+ε)

    thr_pctl = float(np.percentile(E_tot, float(active_pctl)))
    active = (E_tot > float(active_abs_thr)) & (E_tot > thr_pctl)

    action_segs = find_segments(active)
    dt = float(np.median(np.diff(t))) if t.size >= 2 else 0.0

    action_durs_s: list[float] = []
    if dt > 0:
        for L, R in action_segs:
            action_durs_s.append(float((R - L + 1) * dt))

    suspected_segs: list[tuple[int, int]] = []
    suspect_durs_s: list[float] = []
    for (L, R), dur in zip(action_segs, action_durs_s):
        if dur > float(seg_dur_min_s) and dur < float(seg_dur_max_s):
            suspected_segs.append((L, R))
            suspect_durs_s.append(float(dur))

    fall_found = False
    best_seg = None
    best_peak_i = None
    best_peak_score = -np.inf

    for L, R in suspected_segs:
        seg_scores = score_raw[L:R + 1]
        if seg_scores.size == 0:
            continue
        local_i = int(np.argmax(seg_scores))
        local_max = float(seg_scores[local_i])
        if local_max > best_peak_score:
            best_peak_score = local_max
            best_seg = (L, R)
            best_peak_i = int(L + local_i)

    if best_seg is not None and best_peak_i is not None and best_peak_score > float(score_thr):
        fall_found = True

    # 静止检查（比例门）：允许少量点偶发超阈值
    # 规则：静止窗内低于阈值的比例 >= still_min_frac
    still_found = False
    still_len_s = 0.0  # 仅用于诊断（最长连续静止时长），不参与判定
    still_start_t = None
    still_end_t = None
    still_exceed_cnt = 0
    still_total_cnt = 0
    still_exceed_frac = float("nan")
    still_below_frac = float("nan")
    still_post_max = float("nan")
    still_post_min = float("nan")

    if best_seg is not None:
        seg_end_t = float(t[best_seg[1]])
        still_start_t = seg_end_t + float(still_grace_s)
        still_end_t = still_start_t + float(still_window_s)

        post_mask = (t > still_start_t) & (t <= still_end_t + 1e-9)
        if np.any(post_mask):
            E_post = E_tot[post_mask]
            still_total_cnt = int(E_post.size)
            still_exceed_cnt = int(np.sum(E_post >= float(still_abs_thr)))
            still_exceed_frac = float(still_exceed_cnt / max(1, still_total_cnt))
            still_below_frac = float(1.0 - still_exceed_frac)
            still_post_max = float(np.max(E_post)) if E_post.size else float("nan")
            still_post_min = float(np.min(E_post)) if E_post.size else float("nan")

            still_found = bool(still_below_frac >= float(still_min_frac))

            # 诊断：最长连续静止时长（不参与 still_found）
            dt = float(np.median(np.diff(t))) if t.size >= 2 else 0.0
            if dt > 0:
                below = (E_post < float(still_abs_thr)).astype(np.int32)
                run = 0
                best_run = 0
                for v in below:
                    if v == 1:
                        run += 1
                        if run > best_run:
                            best_run = run
                    else:
                        run = 0
                still_len_s = float(best_run * dt)

    alarm = bool(fall_found and still_found)
    fall_time_s = float(win_start_s + float(t[best_peak_i])) if (fall_found and best_peak_i is not None) else float("nan")

    # 窗口原因（用于中文标签映射）
    if len(action_segs) == 0:
        w_reason = "NO_ACTIVE"
    elif len(suspected_segs) == 0:
        w_reason = "ACTIVE_NO_SUSPECT"
    elif not fall_found:
        w_reason = "SUSPECT_SCORE_LOW"
    elif fall_found and (not still_found):
        w_reason = "FALL_ONLY_STILL_FAIL"
    else:
        w_reason = "ALARM"

    E_tot_mean = float(np.mean(E_tot)) if E_tot.size else float("nan")
    E_tot_p90 = float(np.percentile(E_tot, 90)) if E_tot.size else float("nan")
    E_tot_max = float(np.max(E_tot)) if E_tot.size else float("nan")

    return WindowResult(
        win_start_s=float(win_start_s),
        alarm=alarm,
        fall_found=fall_found,
        still_found=still_found,
        w_reason=str(w_reason),

        fall_time_s=fall_time_s,
        best_peak_score=float(best_peak_score) if np.isfinite(best_peak_score) else float("nan"),
        best_peak_i=int(best_peak_i) if best_peak_i is not None else -1,
        best_seg=best_seg,

        action_segs=action_segs,
        suspected_segs=suspected_segs,
        action_durs_s=action_durs_s,
        suspect_durs_s=suspect_durs_s,

        f=f, t=t, P=P,
        E_hf=E_hf, E_lf=E_lf, E_tot=E_tot,
        score_raw=score_raw,
        ratio_bounded=ratio_bounded,

        thr_pctl=float(thr_pctl),
        active_abs_thr=float(active_abs_thr),
        active_pctl=float(active_pctl),
        denom_floor=float(denom_floor),
        score_thr=float(score_thr),
        seg_dur_min_s=float(seg_dur_min_s),
        seg_dur_max_s=float(seg_dur_max_s),

        still_abs_thr=float(still_abs_thr),
        still_window_s=float(still_window_s),
        still_min_frac=float(still_min_frac),
        still_grace_s=float(still_grace_s),
        still_start_t=still_start_t,
        still_end_t=still_end_t,
        still_len_s=float(still_len_s),
        still_exceed_cnt=int(still_exceed_cnt),
        still_total_cnt=int(still_total_cnt),
        still_exceed_frac=float(still_exceed_frac),
        still_below_frac=float(still_below_frac),
        still_post_max=float(still_post_max),
        still_post_min=float(still_post_min),

        dt=float(dt),
        E_tot_mean=E_tot_mean,
        E_tot_p90=E_tot_p90,
        E_tot_max=E_tot_max,
    )


# ===========================
# Full-file (global) series for debug plots
# ===========================

def compute_full_series(
    sim_u: np.ndarray,
    fs: float,
    *,
    hf_band: tuple[float, float],
    lf_band: tuple[float, float],
    tot_band: tuple[float, float],
    denom_floor_pctl: float,
    active_abs_thr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''对全量 sim_u 做一次 STFT，输出全局时间轴上的 E_tot(t) 与 score_raw(t)。

    返回：
      - t_g: 全局 STFT time-bin（秒，从文件起点算起）
      - E_tot_g: 总能量序列
      - score_g: score_raw = E_hf / (E_lf + floor)
    '''
    f, t, Z = stft_Z(sim_u, fs=fs)
    P = np.abs(Z) ** 2

    hf0, hf1 = float(min(hf_band)), float(max(hf_band))
    lf0, lf1 = float(min(lf_band)), float(max(lf_band))
    tot0, tot1 = float(min(tot_band)), float(max(tot_band))

    hf_mask = (f >= hf0) & (f < hf1)
    lf_mask = (f >= lf0) & (f < lf1)
    tot_mask = (f >= tot0) & (f <= tot1)
    if (not np.any(hf_mask)) or (not np.any(lf_mask)) or (not np.any(tot_mask)):
        raise ValueError("HF/LF/总能量 频段不在频率轴范围内，请检查参数与 fs/STFT 设置")

    E_hf = P[hf_mask, :].sum(axis=0)
    E_lf = P[lf_mask, :].sum(axis=0)
    E_tot = P[tot_mask, :].sum(axis=0)

    denom_floor = float(np.percentile(E_lf, float(denom_floor_pctl))) + EPS
    score_raw = E_hf / (E_lf + denom_floor)

    # Gate HF/LF score by total energy: only keep score where E_tot > active_abs_thr, else set to 0
    _gate_thr = float(active_abs_thr)
    if E_tot.size == score_raw.size:
        score_raw = score_raw.astype(np.float64, copy=True)
        score_raw[E_tot <= _gate_thr] = 0.0
    return t.astype(float), E_tot.astype(float), score_raw.astype(float)


def overlay_pick_boxes_on_axis(
    ax: plt.Axes,
    pick: WindowResult,
    *,
    window_len_s: float,
    alpha_action: float = 0.10,
    alpha_suspect: float = 0.18,
    alpha_chosen: float = 0.28,
    alpha_still: float = 0.12,
    alpha_win: float = 0.06,
):
    """把 pick(选中窗口) 的动作段/疑似段/选中段/静止窗/峰值线映射到全局时间轴并叠加到 ax 上。

    说明（按 v5 行为）：
      - 动作段：绿色（BOX_COLOR_ACTION）
      - 疑似段：红色（BOX_COLOR_SUSPECT）
      - 选中段：黄色（BOX_COLOR_CHOSEN）
      - 静止检查窗：橙色（BOX_COLOR_STILL）
      - 选中窗口范围：淡黄色（BOX_COLOR_CHOSEN, alpha 很小）
    图例：只给“实际画出来的元素”添加 label（未出现的不写标签）。
    """
    if pick is None:
        return

    win_start = float(pick.get("win_start_s", 0.0))
    t = pick.get("t", None)
    if t is None or len(t) == 0:
        return

    # 选中窗口范围
    ax.axvspan(
        win_start,
        win_start + float(window_len_s),
        color=BOX_COLOR_CHOSEN,
        alpha=float(alpha_win),
        label="选中窗口范围",
    )

    # 动作段
    first_action = True
    for (L, R) in (pick.get("action_segs", []) or []):
        try:
            a = win_start + float(t[int(L)])
            b = win_start + float(t[int(R)])
            ax.axvspan(a, b, color=BOX_COLOR_ACTION, alpha=float(alpha_action),
                       label="动作段" if first_action else None)
            first_action = False
        except Exception:
            continue

    # 疑似段
    first_sus = True
    for (L, R) in (pick.get("suspected_segs", []) or []):
        try:
            a = win_start + float(t[int(L)])
            b = win_start + float(t[int(R)])
            ax.axvspan(a, b, color=BOX_COLOR_SUSPECT, alpha=float(alpha_suspect),
                       label="疑似段" if first_sus else None)
            first_sus = False
        except Exception:
            continue

    # 选中段（best_seg）
    best_seg = pick.get("best_seg", None)
    if best_seg is not None:
        try:
            L = int(best_seg[0]); R = int(best_seg[1])
            a = win_start + float(t[L])
            b = win_start + float(t[R])
            ax.axvspan(a, b, color=BOX_COLOR_CHOSEN, alpha=float(alpha_chosen), label="选中段")
        except Exception:
            pass

    # 静止检查窗（still_start_t/end_t 为窗口内“秒”）
    st0 = pick.get("still_start_t", None)
    st1 = pick.get("still_end_t", None)
    if st0 is not None and st1 is not None:
        try:
            a = win_start + float(st0)
            b = win_start + float(st1)
            ax.axvspan(a, b, color=BOX_COLOR_STILL, alpha=float(alpha_still), label="静止检查窗")
        except Exception:
            pass

    # 峰值线（best_peak_i 是窗口内 time-bin index）
    best_peak_i = pick.get("best_peak_i", None)
    if best_peak_i is not None:
        try:
            tp = win_start + float(t[int(best_peak_i)])
            ax.axvline(tp, linestyle="--", linewidth=1.0, label="峰值时刻")
        except Exception:
            pass


def overlay_pick_fall_still_on_axis(
    ax: plt.Axes,
    pick: WindowResult,
    *,
    window_len_s: float,
    alpha_fall: float = 0.22,
    alpha_still: float = 0.12,
):
    """只叠加“最终摔倒段(选中疑似段)”与“静止检查窗”。

    注意：这里不会再画动作段/全部疑似段/选中窗口范围等，避免框太多。
    图例：只给“实际画出来的元素”添加 label（未出现的不写标签）。
    """
    if pick is None:
        return

    win_start = float(pick.get("win_start_s", 0.0))
    t = pick.get("t", None)
    best_seg = pick.get("best_seg", None)

    # 最终摔倒段（best_seg，映射到全局时间）
    if t is not None and best_seg is not None:
        try:
            L = int(best_seg[0]); R = int(best_seg[1])
            a = win_start + float(t[L])
            b = win_start + float(t[R])
            ax.axvspan(a, b, color=BOX_COLOR_CHOSEN, alpha=float(alpha_fall), label="最终摔倒段")
        except Exception:
            pass

    # 静止检查窗（映射到全局时间）
    st0 = pick.get("still_start_t", None)
    st1 = pick.get("still_end_t", None)
    if st0 is not None and st1 is not None:
        try:
            a = win_start + float(st0)
            b = win_start + float(st1)
            ax.axvspan(a, b, color=BOX_COLOR_STILL, alpha=float(alpha_still), label="静止检查窗")
        except Exception:
            pass


def plot_window_result(
    wr: WindowResult,
    *,
    out_png: str,
    title: str,
    fmax_hz: float = 60.0,
    crop_edge_s: float = 0.5,
):
    """绘制单个窗口的诊断图（三联图：STFT + score_raw + E_tot）。

    本版窗口图绘制规则：
      - 不再标动作段（action_segs）。
      - 疑似段（suspected_segs）用蓝色框，并在段上标注：第x号疑似段（得分）。
      - 选中段不再单独画框；仅在对应疑似段标注追加“选中”。
      - 静止检查窗仍绘制（橙色框）。
      - STFT 与曲线图显示时，首尾各裁掉 crop_edge_s 秒（只影响显示，不改变判定）。
    """

    f = wr["f"]; t = wr["t"]; P = wr["P"]
    score = wr["score_raw"]
    E_tot = wr["E_tot"]
    suspected_segs = wr["suspected_segs"]
    best_seg = wr["best_seg"]
    best_peak_i = wr["best_peak_i"]

    band = (f >= 0) & (f <= float(fmax_hz))
    f2 = f[band]
    P2 = P[band, :]

    # 只裁显示，不改判定
    if t.size >= 2 and float(crop_edge_s) > 0:
        t_min = float(crop_edge_s)
        t_max = float(t[-1]) - float(crop_edge_s)
        if t_max > t_min:
            t_mask = (t >= t_min) & (t <= t_max + 1e-9)
        else:
            t_mask = np.ones_like(t, dtype=bool)
    else:
        t_mask = np.ones_like(t, dtype=bool)

    t_plot = t[t_mask]
    if t_plot.size < 2:
        t_mask = np.ones_like(t, dtype=bool)
        t_plot = t

    P2_plot = P2[:, t_mask]
    score_plot = score[t_mask]
    E_tot_plot = E_tot[t_mask]

    x0 = float(t_plot[0])
    x1 = float(t_plot[-1])

    def _clip_span(a: float, b: float) -> tuple[float, float] | None:
        aa = max(float(a), x0)
        bb = min(float(b), x1)
        if bb <= aa:
            return None
        return aa, bb

    # 预先计算每个疑似段的“得分”（段内 score_raw 最大值），以及是否为选中段
    suspect_infos: list[dict[str, float | int | bool]] = []
    for i, (L, R) in enumerate(suspected_segs, 1):
        seg_scores = score[L:R + 1]
        seg_max = float(np.max(seg_scores)) if seg_scores.size else float("nan")
        is_sel = bool(best_seg is not None and int(L) == int(best_seg[0]) and int(R) == int(best_seg[1]))
        suspect_infos.append({"idx": i, "L": int(L), "R": int(R), "seg_max": seg_max, "is_sel": is_sel})

    # ========== 画图 ==========
    fig = plt.figure(figsize=(12, 8.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 1.05, 1.05], hspace=0.12)

    # (1) STFT 热力图
    ax = fig.add_subplot(gs[0, 0])
    mesh = ax.pcolormesh(t_plot, f2, P2_plot, shading="auto")
    fig.colorbar(mesh, ax=ax, label="STFT能量 |Z|^2")

    # 疑似段（蓝框）+ 文本标注
    first_suspect = True
    for info in suspect_infos:
        L = int(info["L"]); R = int(info["R"])
        span = _clip_span(float(t[L]), float(t[R]))
        if span is None:
            continue
        ax.axvspan(span[0], span[1], color=BOX_COLOR_SUSPECT_WIN, alpha=0.20,
                   label="疑似段" if first_suspect else None)
        first_suspect = False

        txt = "第{}号疑似段（{:.2f}）{}".format(
            int(info["idx"]),
            float(info["seg_max"]) if np.isfinite(info["seg_max"]) else float("nan"),
            "选中" if bool(info["is_sel"]) else "",
        ).strip()
        xm = 0.5 * (span[0] + span[1])
        y_text = float(fmax_hz) * max(0.15, 0.92 - (int(info["idx"]) - 1) * 0.06)
        ax.text(
            xm, y_text, txt,
            ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.25),
        )

    # 峰值线
    if best_peak_i is not None and int(best_peak_i) >= 0:
        tp = float(t[int(best_peak_i)])
        if x0 <= tp <= x1:
            ax.axvline(tp, linestyle="--", linewidth=1.0, label="峰值时刻")

    # 静止检查窗（橙色框）
    if wr.get("still_start_t") is not None and wr.get("still_end_t") is not None:
        span = _clip_span(float(wr["still_start_t"]), float(wr["still_end_t"]))
        if span is not None:
            ax.axvspan(span[0], span[1], color=BOX_COLOR_STILL, alpha=0.12, label="静止检查窗(8s)")

            ax.text(
                span[0] + 0.05,
                float(fmax_hz) * 0.82,
                "静止连续≈{:.2f}s".format(float(wr.get("still_len_s", float("nan")))),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", alpha=0.25),
            )

    ax.set_ylabel("频率 (Hz)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)

    # (2) score_raw 曲线
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    ax2.plot(t_plot, score_plot, linewidth=1.1, label="score_raw=E_hf/(E_lf+floor)")
    ax2.axhline(float(wr.get("score_thr", 1.4)), linestyle="--", linewidth=1.0, label="score阈值")

    # 同步框（只画疑似段 + 静止窗 + 峰值线，不画动作段/选中段）
    def _sync_boxes(axx: plt.Axes, *, with_text: bool = False):
        # 疑似段（蓝框）
        for info in suspect_infos:
            L = int(info["L"]); R = int(info["R"])
            span = _clip_span(float(t[L]), float(t[R]))
            if span is None:
                continue
            axx.axvspan(span[0], span[1], color=BOX_COLOR_SUSPECT_WIN, alpha=0.20)
            if with_text:
                txt = "第{}号疑似段（{:.2f}）{}".format(
                    int(info["idx"]),
                    float(info["seg_max"]) if np.isfinite(info["seg_max"]) else float("nan"),
                    "选中" if bool(info["is_sel"]) else "",
                ).strip()
                xm = 0.5 * (span[0] + span[1])
                y0, y1 = axx.get_ylim()
                frac = max(0.15, 0.85 - (int(info["idx"]) - 1) * 0.08)
                axx.text(
                    xm, y0 + frac * (y1 - y0),
                    txt, ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", alpha=0.18),
                )
        # 峰值线
        if best_peak_i is not None and int(best_peak_i) >= 0:
            tp = float(t[int(best_peak_i)])
            if x0 <= tp <= x1:
                axx.axvline(tp, linestyle="--", linewidth=1.0)
        # 静止窗
        if wr.get("still_start_t") is not None and wr.get("still_end_t") is not None:
            span = _clip_span(float(wr["still_start_t"]), float(wr["still_end_t"]))
            if span is not None:
                axx.axvspan(span[0], span[1], color=BOX_COLOR_STILL, alpha=0.12)

    _sync_boxes(ax2, with_text=True)

    ax2.set_ylabel("score_raw")
    ax2.legend(loc="upper right", fontsize=9)

    # (3) E_tot 曲线（按 1e-5 缩放显示）
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax)
    E_scaled = E_tot_plot / ETOT_UNIT
    ax3.plot(t_plot, E_scaled, linewidth=1.0, label="E_tot (×1e-5)")
    ax3.axhline(float(wr.get("active_abs_thr", 0.0)) / ETOT_UNIT, linestyle="--", linewidth=1.0, label="active_abs_thr")
    ax3.axhline(float(wr.get("still_abs_thr", 0.0)) / ETOT_UNIT, linestyle="--", linewidth=1.0, label="still_abs_thr")
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=6))

    _sync_boxes(ax3, with_text=False)

    ax3.set_xlabel("窗口内时间 (秒)")
    ax3.set_ylabel("E_tot (×1e-5)")
    ax3.legend(loc="upper right", fontsize=9)

    # 文字信息
    info = (
        "窗口原因={}\n"
        "E_tot: 均值={:.3e}, P90={:.3e}, 最大={:.3e}\n"
        "静止窗: 超阈值点数={}/{}, 低于阈值比例={:.3f}, 超阈值比例={:.3f}, post最大={:.3e}"
    ).format(
        wr.get("w_reason", ""),
        float(wr.get("E_tot_mean", float("nan"))),
        float(wr.get("E_tot_p90", float("nan"))),
        float(wr.get("E_tot_max", float("nan"))),
        int(wr.get("still_exceed_cnt", 0)),
        int(wr.get("still_total_cnt", 0)),
        float(wr.get("still_below_frac", float("nan"))),
        float(wr.get("still_exceed_frac", float("nan"))),
        float(wr.get("still_post_max", float("nan"))),
    )
    fig.text(0.01, 0.01, info, fontsize=9, va="bottom", ha="left")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ===========================
# Batch processing
# ===========================

def collect_csv_files(inputs: list[str], pattern: str, recursive: bool) -> list[str]:
    out: list[str] = []
    for p in inputs:
        p = os.path.expanduser(p)
        if os.path.isfile(p):
            if fnmatch.fnmatch(os.path.basename(p), pattern):
                out.append(os.path.abspath(p))
            continue
        if os.path.isdir(p):
            if recursive:
                for root, _dirs, files in os.walk(p):
                    for fn in files:
                        if fnmatch.fnmatch(fn, pattern):
                            out.append(os.path.abspath(os.path.join(root, fn)))
            else:
                for fn in os.listdir(p):
                    if fnmatch.fnmatch(fn, pattern):
                        out.append(os.path.abspath(os.path.join(p, fn)))
            continue
        import glob as _glob
        for g in _glob.glob(p, recursive=recursive):
            if os.path.isfile(g) and fnmatch.fnmatch(os.path.basename(g), pattern):
                out.append(os.path.abspath(g))
    return sorted(list(dict.fromkeys(out)))


def process_file(
    csv_path: str,
    *,
    out_dir: str,
    device: str,
    fs_target: float,
    window_len_s: float,
    step_s: float,
    gap_ms: float,
    fill_gap: str,
    max_seconds_decode: float,
    hf_band: tuple[float, float],
    lf_band: tuple[float, float],
    tot_band: tuple[float, float],
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
    fmax_hz: float,
    max_frames: int,
    summary_only: bool,
    no_plots: bool,
) -> dict[str, Any]:
    """
    每个文件一个文件夹：<out_dir>/<文件basename>/
      - windows_中文.csv：每个滑窗一行（中文表头 + 完整调试信息）
      - file_summary_中文.csv：文件级汇总（中文表头 + 标签计数）
      - plots/*.png：每个窗口一张图（STFT热力图 + 段/静止窗 + 两条曲线）
      - debug_plots/*.png：文件级统计图（用于排查 3 类问题）
    """
    # ======== 添加这里：参数调试打印 ========
    print("\n" + "="*60)
    print(f"开始处理文件: {os.path.basename(csv_path)}")
    print("传入的参数检查:")
    print(f"  1. active_abs_thr = {active_abs_thr:.2e} (默认: 8e-5)")
    print(f"  2. still_abs_thr = {still_abs_thr:.2e} (默认: 5e-5)")
    print(f"  3. still_min_frac = {still_min_frac} (默认: 0.95)")
    print(f"  4. score_thr = {score_thr} (默认: 1.4)")
    print(f"  5. still_min_run_s = {still_min_run_s} (默认: 5.0)")
    print(f"  6. still_grace_s = {still_grace_s} (默认: 0.0)")
    print("="*60 + "\n")
    # ======== 结束添加 ========
    
    base = os.path.splitext(os.path.basename(csv_path))[0]
    if summary_only:
        # summary-only：不创建每文件目录，不输出每窗表/图
        file_dir = out_dir
        plots_dir = None
        dbg_dir = None
    else:
        file_dir = os.path.join(out_dir, base)
        plots_dir = os.path.join(file_dir, "plots")
        dbg_dir = os.path.join(file_dir, "debug_plots")
        os.makedirs(file_dir, exist_ok=True)
        if not no_plots:
            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(dbg_dir, exist_ok=True)

    # 文件级汇总行（中文字段）
    row: dict[str, Any] = dict(
        文件名=os.path.basename(csv_path),
        文件路径=csv_path.replace("\\", "/"),
        判别标签="ERROR",
        原因="",
        摔倒时间_秒=np.nan,
        选中窗口起始_秒=np.nan,
        选中峰值分数=np.nan,
        动作段数=np.nan,
        疑似段数=np.nan,
        动作段区间_秒="",
        疑似段区间_秒="",
        静止连续时长_秒=np.nan,
        E_tot_均值=np.nan,
        E_tot_P90=np.nan,
        E_tot_最大=np.nan,
        空洞数_gap_cnt=np.nan,
        输出文件夹=file_dir.replace("\\", "/"),
        静止窗超阈值点数=0,
        静止窗点数=0,
        静止窗超阈值比例=np.nan,
        静止窗低于阈值比例=np.nan,
        静止窗Etot最大=np.nan,
        静止窗Etot最小=np.nan,
        错误信息="",
    )

    try:
        # (1) raw CSV -> CSI + 真实时间轴（与完整版一致）
        csi_all, _ants, tx_ts_s = load_csi_all_from_rawcsv(
            csv_path,
            device=device,
            max_frames=int(max_frames),
            max_seconds=float(max_seconds_decode),
        )

        # (2) CSI -> sim_B
        sim = compute_simB(csi_all)
        sim = hampel_filter_1d(sim, window_size=11, n_sigmas=3.0).astype(np.float64)

        # (3) 用真实时间轴重采样到均匀 fs，并处理丢包空洞
        if tx_ts_s is None or (not np.isfinite(tx_ts_s).any()):
            ts = np.arange(sim.size, dtype=np.float64) / float(fs_target)
        else:
            ts = np.asarray(tx_ts_s, dtype=np.float64)

        sim_u, gap_cnt = resample_1d_with_gaps(
            ts, sim, fs_target=float(fs_target),
            gap_ms=float(gap_ms), fill_gap=str(fill_gap)
        )
        row["空洞数_gap_cnt"] = int(gap_cnt)

        n = int(sim_u.size)
        win_n = int(round(float(window_len_s) * float(fs_target)))
        step_n = max(1, int(round(float(step_s) * float(fs_target))))

        if n < win_n:
            row["判别标签"] = "TOO_SHORT"
            row["原因"] = "FILE_SHORTER_THAN_WINDOW"
            row["错误信息"] = "重采样后长度不足：{} samples < {}".format(n, win_n)
            if not summary_only:
                pd.DataFrame([]).to_csv(os.path.join(file_dir, "windows_中文.csv"), index=False, encoding="utf-8-sig")
                pd.DataFrame([row]).to_csv(os.path.join(file_dir, "file_summary_中文.csv"), index=False, encoding="utf-8-sig")
            return row

        win_rows: list[dict[str, Any]] = []

        label_counts: dict[str, int] = {}
        any_active = False
        any_suspect = False
        any_fall = False
        alarm_wr = None

        best_plot = None
        best_metric = -np.inf
        best_fall = None
        best_fall_metric = -np.inf

        win_idx = 0
        for start in range(0, n - win_n + 1, step_n):
            win_start_s = float(start) / float(fs_target)
            win_end_s = win_start_s + float(window_len_s)
            win = sim_u[start:start + win_n]

            wr = analyze_window(
                win, fs=float(fs_target), win_start_s=float(win_start_s),
                hf_band=hf_band, lf_band=lf_band, tot_band=tot_band,
                active_abs_thr=float(active_abs_thr),
                active_pctl=float(active_pctl),
                denom_floor_pctl=float(denom_floor_pctl),
                score_thr=float(score_thr),
                seg_dur_min_s=float(seg_dur_min_s),
                seg_dur_max_s=float(seg_dur_max_s),
                still_abs_thr=float(still_abs_thr),
                still_window_s=float(still_window_s),
                still_min_frac=float(still_min_frac),
                still_min_run_s=float(still_min_run_s),
                still_grace_s=float(still_grace_s),
                smooth_bins=int(smooth_bins),
            )

            if len(wr["action_segs"]) > 0:
                any_active = True
            if len(wr["suspected_segs"]) > 0:
                any_suspect = True
            if bool(wr["fall_found"]):
                any_fall = True
            if bool(wr["alarm"]) and alarm_wr is None:
                alarm_wr = wr

            metric = float(wr.get("best_peak_score", -np.inf))
            if np.isfinite(metric) and metric > best_metric:
                best_metric = metric
                best_plot = wr
            if bool(wr["fall_found"]) and np.isfinite(metric) and metric > best_fall_metric:
                best_fall_metric = metric
                best_fall = wr

            w_reason = str(wr.get("w_reason", ""))
            if w_reason == "NO_ACTIVE":
                w_label = "无活动"
            elif w_reason == "ACTIVE_NO_SUSPECT":
                w_label = "有活动但无疑似段"
            elif w_reason == "SUSPECT_SCORE_LOW":
                w_label = "有疑似段但分数不足"
            elif w_reason == "FALL_ONLY_STILL_FAIL":
                w_label = "分数通过但静止失败"
            else:
                w_label = "报警"

            label_counts[w_label] = int(label_counts.get(w_label, 0) + 1)

            action_s = _segs_to_str(wr["action_segs"], wr["t"], float(wr["win_start_s"]))
            suspect_s = _segs_to_str(wr["suspected_segs"], wr["t"], float(wr["win_start_s"]))

            action_durs = wr.get("action_durs_s", [])
            suspect_durs = wr.get("suspect_durs_s", [])
            action_durs_s = ";".join(["{:.3f}".format(float(x)) for x in action_durs]) if action_durs else ""
            suspect_durs_s = ";".join(["{:.3f}".format(float(x)) for x in suspect_durs]) if suspect_durs else ""
            max_action_dur = float(np.max(action_durs)) if action_durs else float("nan")
            max_suspect_dur = float(np.max(suspect_durs)) if suspect_durs else float("nan")

            plot_png = ""  # summary-only/no-plots: keep defined


            if (not summary_only) and (not no_plots):
                plot_png = os.path.join(plots_dir, "窗_{:05d}_t{:08.3f}.png".format(win_idx, win_start_s))
                title = "{} | 窗#{:d} [{:.3f},{:.3f}] | {}".format(base, win_idx, win_start_s, win_end_s, w_label)
                plot_window_result(wr, out_png=plot_png, title=title, fmax_hz=float(fmax_hz))

            # ====== A/B/C: 诊断字段（全窗峰 vs 选中疑似段） ======
            score_raw_arr = wr.get("score_raw", None)
            t_arr = wr.get("t", None)
            E_tot_arr = wr.get("E_tot", None)

            if score_raw_arr is not None and t_arr is not None and len(score_raw_arr) > 0 and len(t_arr) == len(score_raw_arr):
                i_score_peak = int(np.argmax(score_raw_arr))
                score_peak = float(score_raw_arr[i_score_peak])
                t_score_peak = float(win_start_s + float(t_arr[i_score_peak]))
                if E_tot_arr is not None and len(E_tot_arr) == len(score_raw_arr):
                    E_tot_at_score_peak = float(E_tot_arr[i_score_peak])
                else:
                    E_tot_at_score_peak = float("nan")
                # active_at_score_peak：是否属于 active（即是否落在任一 action_segs 连段内）
                active_at_score_peak = any((int(L) <= i_score_peak <= int(R)) for (L, R) in (wr.get("action_segs", []) or []))
            else:
                i_score_peak = -1
                score_peak = float("nan")
                t_score_peak = float("nan")
                E_tot_at_score_peak = float("nan")
                active_at_score_peak = False

            best_seg_local = wr.get("best_seg", None)
            if best_seg_local is not None and t_arr is not None and score_raw_arr is not None and len(score_raw_arr) > 0:
                Ls = int(best_seg_local[0]); Rs = int(best_seg_local[1])
                if 0 <= Ls < len(t_arr) and 0 <= Rs < len(t_arr) and Ls <= Rs:
                    t_start = float(win_start_s + float(t_arr[Ls]))
                    t_end = float(win_start_s + float(t_arr[Rs]))
                    in_suspect = bool(np.isfinite(t_score_peak) and (t_start <= t_score_peak <= t_end))
                    # 疑似段内最大分数（用 score_raw，注意 score_raw 已经做了 Etot gate）
                    seg_scores = np.asarray(score_raw_arr[Ls:Rs+1], dtype=float)
                    score_max_in_suspect = float(np.max(seg_scores)) if seg_scores.size else float("nan")
                    t_center_of_suspect = 0.5 * (t_start + t_end)
                    dt_peak_minus_center = float(t_score_peak - t_center_of_suspect) if np.isfinite(t_score_peak) else float("nan")
                else:
                    t_start = float("nan")
                    t_end = float("nan")
                    in_suspect = False
                    score_max_in_suspect = float("nan")
                    dt_peak_minus_center = float("nan")
            else:
                t_start = float("nan")
                t_end = float("nan")
                in_suspect = False
                score_max_in_suspect = float("nan")
                dt_peak_minus_center = float("nan")

            win_rows.append(dict(
                窗口序号=int(win_idx),
                窗口开始_秒=float(win_start_s),
                窗口结束_秒=float(win_end_s),
                窗口标签=str(w_label),
                窗口原因=str(w_reason),
                是否报警=bool(wr["alarm"]),
                是否判摔倒=bool(wr["fall_found"]),
                是否判静止=bool(wr["still_found"]),
                摔倒时间_秒=float(wr.get("fall_time_s", np.nan)) if wr.get("fall_found", False) else np.nan,
                # 旧：best_peak_* 只在“疑似段”存在时才有值，会导致大量 NaN，画图只剩少数窗
                疑似段峰值分数=float(wr.get("best_peak_score", np.nan)),
                疑似段峰值能量=float(wr["E_tot"][int(wr.get("best_peak_i", -1))]) if (int(wr.get("best_peak_i", -1)) >= 0 and int(wr.get("best_peak_i", -1)) < len(wr["E_tot"])) else np.nan,

                # 新：全窗峰值（所有 time-bin 参与；但 score_raw 已经在 Etot 门槛下置 0）
                峰值分数=float(np.max(wr["score_raw"])) if (wr.get("score_raw") is not None and len(wr["score_raw"]) > 0) else np.nan,
                峰值能量=float(wr["E_tot"][int(np.argmax(wr["score_raw"]))]) if (wr.get("score_raw") is not None and len(wr["score_raw"]) > 0) else np.nan,

                # A. 全窗峰点落在哪个段里
                t_score_peak=float(t_score_peak),
                score_peak=float(score_peak),
                E_tot_at_score_peak=float(E_tot_at_score_peak),
                active_at_score_peak=bool(active_at_score_peak),
                t_start=float(t_start),
                t_end=float(t_end),
                in_suspect=bool(in_suspect),

                # B. 疑似段内的最大分数（用于对比：score_max_in_suspect < 1 但 score_peak > 1）
                score_max_in_suspect=float(score_max_in_suspect),

                # C. 峰与疑似段中心的时间差（秒）
                dt=float(dt_peak_minus_center),
                动作段数=int(len(wr["action_segs"])),
                疑似段数=int(len(wr["suspected_segs"])),
                动作段区间_秒=action_s,
                疑似段区间_秒=suspect_s,
                动作段时长列表_秒=action_durs_s,
                疑似段时长列表_秒=suspect_durs_s,
                最大动作段时长_秒=max_action_dur,
                最大疑似段时长_秒=max_suspect_dur,

                静止连续时长_秒=float(wr.get("still_len_s", np.nan)),
                静止窗超阈值点数=int(wr.get("still_exceed_cnt", 0)),
                静止窗点数=int(wr.get("still_total_cnt", 0)),
                静止窗超阈值比例=float(wr.get("still_exceed_frac", np.nan)),
                静止窗低于阈值比例=float(wr.get("still_below_frac", np.nan)),
                静止窗Etot最大=float(wr.get("still_post_max", np.nan)),
                静止窗Etot最小=float(wr.get("still_post_min", np.nan)),

                E_tot_均值=float(wr.get("E_tot_mean", np.nan)),
                E_tot_P90=float(wr.get("E_tot_p90", np.nan)),
                E_tot_最大=float(wr.get("E_tot_max", np.nan)),

                活跃阈值_abs=float(wr.get("active_abs_thr", np.nan)),
                活跃阈值_Pctl=float(wr.get("active_pctl", np.nan)),
                活跃阈值_p90值=float(wr.get("thr_pctl", np.nan)),
                denom_floor=float(wr.get("denom_floor", np.nan)),
                score阈值=float(wr.get("score_thr", np.nan)),
                段长下限_秒=float(wr.get("seg_dur_min_s", np.nan)),
                段长上限_秒=float(wr.get("seg_dur_max_s", np.nan)),
                静止阈值=float(wr.get("still_abs_thr", np.nan)),
                静止窗长_秒=float(wr.get("still_window_s", np.nan)),

                图路径=plot_png.replace("\\", "/"),
            ))

            win_idx += 1

        win_df = pd.DataFrame(win_rows)
        if not summary_only:
            win_df.to_csv(os.path.join(file_dir, "windows_中文.csv"), index=False, encoding="utf-8-sig")

        # 文件级判定（与之前一致，但补中文输出）
        if alarm_wr is not None and bool(alarm_wr["alarm"]):
            label = "ALARM"
            reason = "FALL_AND_STILL"
            pick = alarm_wr
        else:
            if not any_active:
                label = "NO_EVENT"
                reason = "NO_ACTIVE_ALL_WINDOWS"
                pick = best_plot
            elif any_active and (not any_suspect):
                label = "NO_EVENT"
                reason = "ACTIVE_BUT_NO_SUSPECT_ALL_WINDOWS"
                pick = best_plot
            elif any_suspect and (not any_fall):
                label = "NO_FALL"
                reason = "SUSPECT_BUT_SCORE_BELOW_THR_ALL_WINDOWS"
                pick = best_plot
            else:
                label = "FALL_ONLY"
                reason = "FALL_FOUND_BUT_STILL_FAIL"
                pick = best_fall if best_fall is not None else best_plot

        row["判别标签"] = label
        row["原因"] = reason

        if pick is not None:
            row["选中窗口起始_秒"] = float(pick["win_start_s"])
            row["选中峰值分数"] = float(pick.get("best_peak_score", np.nan))
            row["摔倒时间_秒"] = float(pick.get("fall_time_s", np.nan)) if pick.get("fall_found", False) else np.nan
            row["动作段数"] = int(len(pick["action_segs"]))
            row["疑似段数"] = int(len(pick["suspected_segs"]))
            row["动作段区间_秒"] = _segs_to_str(pick["action_segs"], pick["t"], float(pick["win_start_s"]))
            row["疑似段区间_秒"] = _segs_to_str(pick["suspected_segs"], pick["t"], float(pick["win_start_s"]))
            row["静止连续时长_秒"] = float(pick.get("still_len_s", np.nan))
            row["静止窗超阈值点数"] = int(pick.get("still_exceed_cnt", 0))
            row["静止窗点数"] = int(pick.get("still_total_cnt", 0))
            row["静止窗超阈值比例"] = float(pick.get("still_exceed_frac", np.nan))
            row["静止窗低于阈值比例"] = float(pick.get("still_below_frac", np.nan))
            row["静止窗Etot最大"] = float(pick.get("still_post_max", np.nan))
            row["静止窗Etot最小"] = float(pick.get("still_post_min", np.nan))

            row["E_tot_均值"] = float(pick.get("E_tot_mean", np.nan))
            row["E_tot_P90"] = float(pick.get("E_tot_p90", np.nan))
            row["E_tot_最大"] = float(pick.get("E_tot_max", np.nan))

        # 文件级汇总（包含窗口标签计数）
        file_sum = dict(row)
        file_sum["窗口总数"] = int(len(win_rows))
        for k in ["无活动", "有活动但无疑似段", "有疑似段但分数不足", "分数通过但静止失败", "报警"]:
            file_sum[f"计数_{k}"] = int(label_counts.get(k, 0))
        if not summary_only:
            pd.DataFrame([file_sum]).to_csv(os.path.join(file_dir, "file_summary_中文.csv"), index=False, encoding="utf-8-sig")

        if (not summary_only) and (not no_plots):
            # 文件级统计图（中文）
            def _save_bar(d: dict, out_png: str, title: str, xlabel: str, ylabel: str):
                keys = list(d.keys())
                vals = [d[k] for k in keys]
                fig = plt.figure(figsize=(10, 4))
                ax = fig.add_subplot(1, 1, 1)
                ax.bar(keys, vals)
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                plt.xticks(rotation=30, ha="right")
                fig.tight_layout()
                fig.savefig(out_png, dpi=160, bbox_inches="tight")
                plt.close(fig)

            _save_bar(label_counts, os.path.join(dbg_dir, "窗口标签计数.png"), "窗口标签计数", "标签", "窗口数")

            # 总能量随时间（每窗，不做任何“背景扣除/归一化”，直接画原始窗口统计量）
            # 说明：E_tot_* 来自窗口内 STFT 能量序列 E_tot(t) 的统计值（均值/P90/最大），用于表示“活动强度”随时间的变化
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 1, 1)
            # 两条都画：P90 更稳健，最大值更敏感（不扣背景，只做单位缩放显示）
            y_p90 = win_df["E_tot_P90"].to_numpy(dtype=float) / ETOT_UNIT
            y_max = win_df["E_tot_最大"].to_numpy(dtype=float) / ETOT_UNIT
            ax.plot(win_df["窗口开始_秒"], y_p90, label="E_tot_P90 (×1e-5)")
            ax.plot(win_df["窗口开始_秒"], y_max, label="E_tot_最大 (×1e-5)", alpha=0.8)
            # 阈值也按同一单位缩放显示（仅作参考）
            ax.axhline(float(active_abs_thr) / ETOT_UNIT, linestyle="--", linewidth=1.0, label="active_abs_thr")
            ax.axhline(float(still_abs_thr) / ETOT_UNIT, linestyle="--", linewidth=1.0, label="still_abs_thr")
            ax.set_title("总能量随时间变化（每窗）")
            ax.set_xlabel("窗口开始时间 (秒)")
            ax.set_ylabel("E_tot (×1e-5)")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.legend(loc="upper right", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(dbg_dir, "总能量随时间.png"), dpi=160, bbox_inches="tight")
            plt.close(fig)


            # 峰值分数随时间
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(win_df["窗口开始_秒"], win_df["峰值分数"])
            ax.axhline(float(score_thr), linestyle="--")
            ax.set_title("峰值分数随时间变化（每窗）")
            ax.set_xlabel("窗口开始时间 (秒)")
            ax.set_ylabel("峰值分数")
            fig.tight_layout()
            fig.savefig(os.path.join(dbg_dir, "能量比(峰值分数)随时间.png"), dpi=160, bbox_inches="tight")
            fig.savefig(os.path.join(dbg_dir, "峰值分数随时间.png"), dpi=160, bbox_inches="tight")
            plt.close(fig)

            # 峰值能量随时间（每窗）：取 best_peak_i 对应的 E_tot(time-bin)
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 1, 1)
            y_peakE = win_df.get("峰值能量", pd.Series([np.nan]*len(win_df))).to_numpy(dtype=float) / ETOT_UNIT
            ax.plot(win_df["窗口开始_秒"], y_peakE, linewidth=1.0, label="峰值能量E_tot@peak (×1e-5)")
            # 参考阈值（同单位缩放）
            ax.axhline(float(active_abs_thr) / ETOT_UNIT, linestyle="--", linewidth=1.0, label="active_abs_thr")
            ax.axhline(float(still_abs_thr) / ETOT_UNIT, linestyle="--", linewidth=1.0, label="still_abs_thr")
            ax.set_title("峰值能量随时间变化（每窗）")
            ax.set_xlabel("窗口开始时间 (秒)")
            ax.set_ylabel("E_tot (×1e-5)")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.legend(loc="upper right", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(dbg_dir, "峰值能量随时间.png"), dpi=160, bbox_inches="tight")
            plt.close(fig)

            # 全量文件：score_raw 与 E_tot 随时间（time-bin），并叠加与选中窗口 pick 同步的框
            # 目的：让你直接看到“整段数据流”上两路信号的变化，以及 pick 里的动作段/疑似段/静止窗在全局时间轴的位置
            try:
                t_g, E_tot_g, score_g = compute_full_series(
                    sim_u, fs=float(fs_target),
                    hf_band=hf_band, lf_band=lf_band, tot_band=tot_band,
                    denom_floor_pctl=float(denom_floor_pctl),
                    active_abs_thr=float(active_abs_thr),
                )

                # (A) 全量 score_raw（简化：只画最终摔倒段+静止检查窗，避免框太多）
                fig = plt.figure(figsize=(12, 4))
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(t_g, score_g, linewidth=1.0, label="score_raw=E_hf/(E_lf+floor)")
                ax.axhline(float(score_thr), linestyle="--", linewidth=1.0, label="score阈值")

                # 仅叠加“最终摔倒段(best_seg)”与“静止检查窗”，并且图例只对实际画出来的框加 label
                overlay_pick_fall_still_on_axis(ax, pick, window_len_s=float(window_len_s))

                ax.set_title("全量文件 score_raw 随时间（仅摔倒段+静止窗）")
                ax.set_xlabel("时间 (秒)")
                ax.set_ylabel("score_raw")
                ax.legend(loc="upper right", fontsize=9)
                fig.tight_layout()
                fig.savefig(os.path.join(dbg_dir, "全量文件_score_raw随时间.png"), dpi=160, bbox_inches="tight")
                plt.close(fig)

                # (B) 全量 E_tot
                fig = plt.figure(figsize=(12, 4))
                ax = fig.add_subplot(1, 1, 1)

                # 仅该图：开头/结尾各裁掉 0.5 秒（只影响显示）
                if t_g.size >= 2:
                    t_min = 0.5
                    t_max = float(t_g[-1]) - 0.5
                    if t_max > t_min:
                        m_edge = (t_g >= t_min) & (t_g <= t_max + 1e-9)
                    else:
                        m_edge = np.ones_like(t_g, dtype=bool)
                else:
                    m_edge = np.ones_like(t_g, dtype=bool)

                t_plot_g = t_g[m_edge]
                E_tot_g_scaled = (E_tot_g[m_edge] / ETOT_UNIT)

                ax.plot(t_plot_g, E_tot_g_scaled, linewidth=1.0, label="E_tot (×1e-5)")
                ax.axhline(float(active_abs_thr) / ETOT_UNIT, linestyle="--", linewidth=1.0, label="active_abs_thr")
                ax.axhline(float(still_abs_thr) / ETOT_UNIT, linestyle="--", linewidth=1.0, label="still_abs_thr")

                # 简化框：只画最终摔倒段与静止检查窗（与 pick 同步）
                overlay_pick_fall_still_on_axis(ax, pick, window_len_s=float(window_len_s))

                ax.set_title("全量文件 E_tot 随时间（仅摔倒段+静止窗，首尾各裁0.5s）")
                ax.set_xlabel("时间 (秒)")
                ax.set_ylabel("E_tot (×1e-5)")
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.legend(loc="upper right", fontsize=9)
                fig.tight_layout()
                fig.savefig(os.path.join(dbg_dir, "全量文件_E_tot随时间.png"), dpi=160, bbox_inches="tight")
                plt.close(fig)

            except Exception as _e:
                # 不让 debug 图失败影响主流程
                pass


            # 静止连续时长分布
            vals = win_df["静止连续时长_秒"].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                fig = plt.figure(figsize=(10, 4))
                ax = fig.add_subplot(1, 1, 1)
                ax.hist(vals, bins=30)
                ax.axvline(float(still_window_s), linestyle="--")
                ax.set_title("静止连续时长分布（秒）")
                ax.set_xlabel("静止连续时长 (秒)")
                ax.set_ylabel("窗口数")
                fig.tight_layout()
                fig.savefig(os.path.join(dbg_dir, "静止连续时长分布.png"), dpi=160, bbox_inches="tight")
                plt.close(fig)

            # 段长随时间
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(win_df["窗口开始_秒"], win_df["最大动作段时长_秒"], label="最大动作段时长")
            ax.plot(win_df["窗口开始_秒"], win_df["最大疑似段时长_秒"], label="最大疑似段时长")
            ax.axhline(float(seg_dur_min_s), linestyle="--")
            ax.axhline(float(seg_dur_max_s), linestyle="--")
            ax.set_title("段长随时间变化（每窗）")
            ax.set_xlabel("窗口开始时间 (秒)")
            ax.set_ylabel("时长 (秒)")
            ax.legend(loc="upper right", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(dbg_dir, "段长随时间.png"), dpi=160, bbox_inches="tight")
            plt.close(fig)

            # 静止窗超阈值点数随时间
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(win_df["窗口开始_秒"], win_df["静止窗超阈值点数"])
            ax.set_title("静止窗超阈值点数随时间（每窗）")
            ax.set_xlabel("窗口开始时间 (秒)")
            ax.set_ylabel("超阈值点数")
            fig.tight_layout()
            fig.savefig(os.path.join(dbg_dir, "静止窗超阈值点数.png"), dpi=160, bbox_inches="tight")
            plt.close(fig)

        if not summary_only:
            with open(os.path.join(file_dir, "说明.txt"), "w", encoding="utf-8") as f:
                f.write("输出说明：\n")
                f.write("1) windows_中文.csv：该文件每个滑动窗口的结果与调试信息（中文表头）。\n")
                f.write("2) plots/：每个窗口一张图（STFT热力图 + 动作段/疑似段/选中段/静止窗 + HF/(HF+LF+ε) + score曲线）。\n")
                f.write("3) debug_plots/：文件级统计图（窗口标签计数、峰值分数随时间、静止长度分布、段长随时间、静止窗超阈值点数等）。\n")
                f.write("4) file_summary_中文.csv：文件级汇总（中文表头）。\n")

        return row

    except Exception as e:
        row["判别标签"] = "ERROR"
        row["原因"] = "EXCEPTION"
        row["错误信息"] = str(e)
        if not summary_only:
            os.makedirs(file_dir, exist_ok=True)
            with open(os.path.join(file_dir, "error.txt"), "w", encoding="utf-8") as f:
                f.write(str(e))
            pd.DataFrame([row]).to_csv(os.path.join(file_dir, "file_summary_中文.csv"), index=False, encoding="utf-8-sig")
        return row


def main():


    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="CSV file or directory (supports multiple)")
    ap.add_argument("--pattern", default="raw_csi_frames2_*.csv", help="filename pattern")
    ap.add_argument("--recursive", action="store_true", help="recursively search directories")
    ap.add_argument("--out-dir", required=True, help="output directory")

    ap.add_argument("--summary-only", action="store_true", help="只输出总表_中文.csv：不生成每文件目录/窗口表/图")
    ap.add_argument("--no-plots", action="store_true", help="不生成 plots/*.png 与 debug_plots/*.png（仍输出每文件 csv）")

    # baseline-compatible decoding/time handling
    ap.add_argument("--device", default="rtac86u", help="decoder device profile")
    ap.add_argument("--fs", type=float, default=1000.0, help="target uniform sampling rate for sim_B (Hz)")
    ap.add_argument("--gap-ms", type=float, default=50.0, help="gap threshold (ms) to mark holes as NaN before fill")
    ap.add_argument("--fill-gap", type=str, default="ffill", choices=["ffill", "interp", "zero"], help="how to fill NaN gaps")
    ap.add_argument("--decode-max-seconds", type=float, default=0.0, help=">0 to truncate decoded frames by real time (seconds); 0 keeps full file")
    ap.add_argument("--max-frames", type=int, default=0, help="0=all frames; for debugging only")

    # windowing
    ap.add_argument("--window-len-s", type=float, default=20.0)
    ap.add_argument("--step-s", type=float, default=1.0)

    # frequency bands
    ap.add_argument("--hf-band", type=float, nargs=2, default=[10.0, 30.0], help="HF band (Hz)")
    ap.add_argument("--lf-band", type=float, nargs=2, default=[2.0, 5.0], help="LF band (Hz)")
    ap.add_argument("--tot-band", type=float, nargs=2, default=[0.0, 90.0], help="Total energy band (Hz)")

    # thresholds
    ap.add_argument("--active-abs-thr", type=float, default=8e-5)
    ap.add_argument("--active-pctl", type=float, default=90.0)
    ap.add_argument("--denom-floor-pctl", type=float, default=20.0)
    ap.add_argument("--score-thr", type=float, default=1.4)
    ap.add_argument("--seg-dur-min-s", type=float, default=0.5)
    ap.add_argument("--seg-dur-max-s", type=float, default=4.0)

    ap.add_argument("--still-abs-thr", type=float, default=5e-5)
    ap.add_argument("--still-window-s", type=float, default=8.0)
    ap.add_argument("--still-min-frac", type=float, default=0.95,
                    help="静止窗内低于静止阈值的时间比例下限（例如0.95表示允许5%点超阈值）")
    ap.add_argument("--still-min-run-s", type=float, default=5.0,
                    help="静止窗内低于阈值的最长连续静止时长下限（秒），用于抑制零散低能量误判")
    ap.add_argument("--still-grace-s", type=float, default=0.0,
                    help="静止检查延迟（秒），从疑似段结束后延迟这么久再开始统计静止")

    ap.add_argument("--smooth-bins", type=int, default=1)
    ap.add_argument("--fmax-hz", type=float, default=60.0)

    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    files = collect_csv_files(args.input, pattern=args.pattern, recursive=bool(args.recursive))
    if not files:
        raise SystemExit("No matching CSV files found.")

    rows = []
    for i, fp in enumerate(files, 1):
        print("[{}/{}] {}".format(i, len(files), fp))
        row = process_file(
            fp,
            out_dir=out_dir,
            device=str(args.device),
            fs_target=float(args.fs),
            window_len_s=float(args.window_len_s),
            step_s=float(args.step_s),
            gap_ms=float(args.gap_ms),
            fill_gap=str(args.fill_gap),
            max_seconds_decode=float(args.decode_max_seconds),
            hf_band=(float(args.hf_band[0]), float(args.hf_band[1])),
            lf_band=(float(args.lf_band[0]), float(args.lf_band[1])),
            tot_band=(float(args.tot_band[0]), float(args.tot_band[1])),
            active_abs_thr=float(args.active_abs_thr),
            active_pctl=float(args.active_pctl),
            denom_floor_pctl=float(args.denom_floor_pctl),
            score_thr=float(args.score_thr),
            seg_dur_min_s=float(args.seg_dur_min_s),
            seg_dur_max_s=float(args.seg_dur_max_s),
            still_abs_thr=float(args.still_abs_thr),
            still_window_s=float(args.still_window_s),
            still_min_frac=float(args.still_min_frac),
            still_min_run_s=float(args.still_min_run_s),
            still_grace_s=float(args.still_grace_s),
            smooth_bins=int(args.smooth_bins),
            fmax_hz=float(args.fmax_hz),
            max_frames=int(args.max_frames),
            summary_only=bool(args.summary_only),
            no_plots=bool(args.no_plots),
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "总表_中文.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("\n已保存总表: {}".format(out_csv))
    if (not bool(args.summary_only)) and (not bool(args.no_plots)):
        print("Plots: {}".format(os.path.join(out_dir, "<每文件子目录>/plots")))


if __name__ == "__main__":
    main()
