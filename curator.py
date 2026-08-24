"""
curator.py — 自动美学筛选与连镜表（P0-2）

生成大量随机/语义视频后，用零依赖启发式指标对产物打分、排序、挑优，
并产出连镜表（contact sheet）与 ranked manifest（musickit 风格）。

指标（全部从视频帧统计，无需 CLIP/模型）：
  * brightness_mean / brightness_std  亮度与对比
  * entropy                          亮度直方图熵（丰富度；过平=无聊，过噪=乱）
  * color_var                        跨像素通道标准差（色彩多样性）
  * motion                           相邻帧差分 RMS（动态能量）

评分 = 归一化加权（motion 0.4 / entropy 0.35 / color_var 0.25），
全黑帧（brightness_mean<8）重罚。排名用于"挑 top-K"，绝不直接删除原片。

解码复用 audio_mixer.find_ffmpeg（同样优先 FFMPEG_BIN）。
"""

import json
import os
import subprocess
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

try:
    from audio_mixer import find_ffmpeg
except Exception:  # noqa: BLE001
    def find_ffmpeg() -> str:
        return os.environ.get("FFMPEG_BIN", "ffmpeg")


# ─────────────────────────────────────────────────────────────────────────
# 帧解码（ffmpeg rawvideo，小分辨率、降采样以控内存）
# ─────────────────────────────────────────────────────────────────────────

def decode_frames(path: str, target=(160, 160), max_frames: int = 24
                  ) -> List[np.ndarray]:
    """用 ffmpeg 把视频解码为若干 RGB uint8 帧（已降采样 + 限帧）。"""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return []
    cmd = [
        ffmpeg, "-y", "-i", path,
        "-vf", f"scale={target[0]}:{target[1]}",
        "-r", "10",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception as e:  # noqa: BLE001
        print(f"[Curator] 解码失败 {os.path.basename(path)}: {e}")
        return []
    buf = proc.stdout
    fw, fh = target
    fb = fw * fh * 3
    if len(buf) < fb:
        return []
    n = len(buf) // fb
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, fh, fw, 3)
    if n > max_frames:
        idx = np.linspace(0, n - 1, max_frames).astype(int)
        arr = arr[idx]
    return [arr[i] for i in range(arr.shape[0])]


# ─────────────────────────────────────────────────────────────────────────
# 指标
# ─────────────────────────────────────────────────────────────────────────

def _metrics(frames: List[np.ndarray]) -> Optional[Dict]:
    if not frames:
        return None
    arr = np.stack(frames).astype(np.float32)  # (N,H,W,3)
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    brightness_mean = float(luma.mean())
    brightness_std = float(luma.std())

    hist, _ = np.histogram(luma, bins=64, range=(0, 255))
    p = hist[hist > 0] / hist.sum()
    entropy = float(-(p * np.log2(p)).sum())

    color_var = float(arr.reshape(-1, 3).std(axis=0).mean())

    if arr.shape[0] > 1:
        diff = np.abs(arr[1:] - arr[:-1:]).mean(axis=(1, 2, 3))
        motion = float(diff.mean())
    else:
        motion = 0.0

    return {
        "brightness_mean": round(brightness_mean, 2),
        "brightness_std": round(brightness_std, 2),
        "entropy": round(entropy, 3),
        "color_var": round(color_var, 2),
        "motion": round(motion, 2),
        "n_frames": int(arr.shape[0]),
    }


def _normalize_and_score(records: List[Dict]) -> List[float]:
    """跨批次 min-max 归一化后加权评分。"""
    def col(key):
        return np.array([r["metrics"].get(key, 0.0) for r in records],
                        dtype=np.float64)

    def norm(x):
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-9:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    m_norm = norm(col("motion"))
    e_norm = norm(col("entropy"))
    c_norm = norm(col("color_var"))

    scores = 0.40 * m_norm + 0.35 * e_norm + 0.25 * c_norm

    # 全黑重罚
    bm = col("brightness_mean")
    scores = np.where(bm < 8.0, scores * 0.2, scores)
    return [float(s) for s in scores]


# ─────────────────────────────────────────────────────────────────────────
# 连镜表
# ─────────────────────────────────────────────────────────────────────────

def make_contact_sheet(records: List[Dict], cols: int = 4,
                       thumb_w: int = 240, thumb_h: int = 240) -> Optional[Image.Image]:
    recs = [r for r in records if r.get("thumb") is not None]
    if not recs:
        return None
    n = len(recs)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(recs):
        crow, ccol = divmod(i, cols)
        x0, y0 = ccol * thumb_w, crow * thumb_h
        thumb = Image.fromarray(r["thumb"].astype(np.uint8)).resize(
            (thumb_w, thumb_h))
        sheet.paste(thumb, (x0, y0))
        label = f"#{i+1}  score={r.get('score', 0.0):.3f}"
        draw.text((x0 + 4, y0 + 4), label, fill=(255, 255, 0))
        if not r.get("kept", True):
            draw.text((x0 + 4, y0 + thumb_h - 16),
                      "rejected", fill=(255, 80, 80))
    return sheet


# ─────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────

def curate(paths: List[str], out_dir: Optional[str] = None,
           keep_top: Optional[int] = None,
           contact_sheet: bool = False) -> List[Dict]:
    """对一批视频打分、排序、挑优，写入 manifest / 连镜表。

    返回 ranked manifest 列表（每条含 path/score/kept/metrics）。
    """
    records: List[Dict] = []
    for p in paths:
        if not os.path.isfile(p):
            print(f"[Curator] 跳过不存在文件: {p}")
            continue
        frames = decode_frames(p)
        m = _metrics(frames)
        thumb = frames[len(frames) // 2] if frames else None
        records.append({"path": p, "metrics": m, "thumb": thumb})

    if not records:
        print("[Curator] 无有效视频可评估")
        return []

    scores = _normalize_and_score(records)
    for r, s in zip(records, scores):
        r["score"] = s

    ranked = sorted(records, key=lambda r: r["score"], reverse=True)
    if keep_top and keep_top < len(ranked):
        for rank, r in enumerate(ranked):
            r["kept"] = rank < keep_top
    else:
        for r in ranked:
            r["kept"] = True

    manifest = [
        {
            "rank": i + 1,
            "path": r["path"],
            "score": round(r.get("score", 0.0), 3),
            "kept": r.get("kept", True),
            "metrics": r["metrics"],
        }
        for i, r in enumerate(ranked)
    ]

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        man_path = os.path.join(out_dir, "curate_manifest.json")
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[Curator] manifest → {man_path}")
        if contact_sheet:
            sheet = make_contact_sheet(ranked)
            if sheet:
                sheet_path = os.path.join(out_dir, "contact_sheet.png")
                sheet.save(sheet_path)
                print(f"[Curator] contact sheet → {sheet_path}")

    # 打印排名摘要
    print("\n[Curator] 排名（高→低）：")
    for m in manifest:
        flag = "✓" if m["kept"] else "✗"
        print(f"  {flag} #{m['rank']}  score={m['score']:.3f}  "
              f"{os.path.basename(m['path'])}")
    return manifest


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python curator.py <video1.mp4> [video2.mp4 ...] [--out DIR] [--keep K] [--sheet]")
        raise SystemExit(1)
    paths_, out_dir_, keep_, sheet_ = [], None, None, False
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--out":
            out_dir_ = sys.argv[i + 1]; i += 2
        elif a == "--keep":
            keep_ = int(sys.argv[i + 1]); i += 2
        elif a == "--sheet":
            sheet_ = True; i += 1
        else:
            paths_.append(a); i += 1
    curate(paths_, out_dir=out_dir_, keep_top=keep_, contact_sheet=sheet_)
