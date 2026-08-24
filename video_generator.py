#!/usr/bin/env python3
"""
video_generator.py — 主脚本：随机视频生成器

功能：
  1. 解析命令行参数
  2. 加载两个项目的效果库
  3. 随机（或按关键词）生成视频组合
  4. 渲染帧（使用 draw_renderer / pw_renderer）
  5. 图层合成 + 后处理
  6. 输出 MP4 视频（OpenCV VideoWriter）

用法：
  python video_generator.py                          # 完全随机
  python video_generator.py --seed 42                # 可复现随机
  python video_generator.py --keywords BZReaction particle  # 按关键词
  python video_generator.py --duration 30 --fps 60    # 自定义参数
  python video_generator.py --output my_video.mp4     # 指定输出
"""

# ── 确保脚本所在目录在 sys.path（Git Bash 下 python xx.py 不会自动加入）──
import sys as _sys, os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPT_DIR)
del _sys, _os, _SCRIPT_DIR

import argparse
import math
import os
import json
import random
import subprocess
import time
import threading
import queue
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# ── 项目内部模块 ──────────────────────────────────────────────────────
from effect_model import EffectConfig, EffectSource
from effect_loader import EffectRegistry, get_default_registry
from random_composer import RandomComposer, VideoComposition, LayerConfig
import draw_renderer
import pw_renderer
import audio_mixer
import cloud_generator
import llm_composer
import curator
import audio_stems
import gl_renderer


# ── 版本信息 ───────────────────────────────────────────────────────────
VERSION = "1.0.0"


# ── 缓动辅助（联网调研：smoothstep 转场比线性更顺滑）──────────
def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


# ── 音频色相旋转辅助（RGB uint8 → HSV → 旋转 H → RGB）──────────
def _color_temperature(arr: np.ndarray, warmth: float) -> np.ndarray:
    """按色温微调整帧：warmth>0 偏暖（提红绿压蓝），warmth<0 偏冷（反之）。

    用于音频频谱质心驱动：明亮度高→暖、低→冷（调研：质心=明亮度，
    可映射为色温）。warmth ∈ [-1,1]，幅度保持克制以免过饱和。
    """
    if abs(warmth) < 1e-3:
        return arr
    w = float(np.clip(warmth, -1.0, 1.0))
    a = min(abs(w), 1.0)
    arr = arr.astype(np.float64)
    if w > 0:
        arr[..., 0] = arr[..., 0] + a * 28.0
        arr[..., 1] = arr[..., 1] + a * 10.0
        arr[..., 2] = arr[..., 2] - a * 26.0
    else:
        arr[..., 0] = arr[..., 0] - a * 18.0
        arr[..., 1] = arr[..., 1] + a * 6.0
        arr[..., 2] = arr[..., 2] + a * 34.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _hsv_hue_rotate(arr: np.ndarray, hue_deg: float) -> np.ndarray:
    """对整帧做色相旋转（hue_deg 度，cyclic）。纯 NumPy，无依赖。

    用于音频驱动：beat 时色相被按黄金比推进，帧间缓动，
    形成随音乐律动的"色彩脉动"。"""
    h = hue_deg / 360.0
    if abs(h) < 1e-4:
        return arr
    rgb = arr.astype(np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    # 色相 H ∈ [0,1)
    H = np.zeros_like(mx)
    nz = d > 1e-8
    mask = nz & (mx == r)
    H[mask] = ((g[mask] - b[mask]) / d[mask]) % 1.0
    mask = nz & (mx == g)
    H[mask] = ((b[mask] - r[mask]) / d[mask] + 1.0 / 3.0) % 1.0
    mask = nz & (mx == b)
    H[mask] = ((r[mask] - g[mask]) / d[mask] + 2.0 / 3.0) % 1.0
    # 旋转
    H = (H + h) % 1.0
    # 饱和度 S、明度 V
    S = np.where(mx > 1e-8, d / np.maximum(mx, 1e-8), 0.0)
    V = mx
    # HSV → RGB
    i = np.floor(H * 6.0)
    f = H * 6.0 - i
    p = V * (1.0 - S)
    q = V * (1.0 - S * f)
    t = V * (1.0 - S * (1.0 - f))
    i = i.astype(int) % 6
    rr = np.choose(i, [V, q, p, p, t, V])
    gg = np.choose(i, [t, V, V, q, p, p])
    bb = np.choose(i, [p, p, t, V, V, q])
    out = np.stack([rr, gg, bb], axis=-1)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════
#  主题渲染器（背景层）
# ══════════════════════════════════════════════════════════════════

class ThemeRenderer:
    """主题（背景）渲染器"""

    def __init__(self, theme: str, width: int, height: int):
        self.theme = theme
        self.w = width
        self.h = height
        self.time = 0.0

    def update(self, dt: float):
        self.time += dt

    def render(self) -> Image.Image:
        """渲染主题帧（PIL 版本）"""
        return Image.fromarray(self.render_numpy())

    def render_numpy(self) -> np.ndarray:
        """渲染主题帧（NumPy 版本，避免 PIL 转换）"""
        if self.theme == "starfield":
            return self._render_starfield_numpy()
        elif self.theme == "gradient":
            return self._render_gradient_numpy()
        elif self.theme == "aurora":
            return self._render_aurora_numpy()
        else:
            return self._render_empty_numpy()

    def _render_empty_numpy(self) -> np.ndarray:
        """微光呼吸（极暗中央辉光）— NumPy 直接返回"""
        arr = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        arr[:, :, 0] = 4
        arr[:, :, 2] = 8
        cx, cy = self.w // 2, self.h // 2
        r = min(self.w, self.h) // 4
        y, x = np.ogrid[:self.h, :self.w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        glow = (np.clip(1.0 - dist / r, 0, 1) ** 2 * 30).astype(np.uint8)
        arr[:, :, 0] += glow
        arr[:, :, 1] += (glow * 0.7).astype(np.uint8)
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _render_starfield_numpy(self) -> np.ndarray:
        """星空背景 — NumPy 直接返回"""
        arr = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        arr[:, :, 2] = 8
        rng = random.Random(int(self.time * 10) % 100000)
        n_stars = 200
        for _ in range(n_stars):
            x = rng.randint(0, self.w - 1)
            y = rng.randint(0, self.h - 1)
            brightness = rng.randint(50, 255)
            arr[y, x, 0] = brightness
            arr[y, x, 1] = brightness
            arr[y, x, 2] = min(255, int(brightness * 1.2))
        return arr

    def _render_gradient_numpy(self) -> np.ndarray:
        """渐变流动背景 — 已是 NumPy，直接返回"""
        arr = np.zeros((self.h, self.w, 3), dtype=np.float32)
        t = self.time * 0.02
        y_norm = np.linspace(0, 1, self.h)[:, np.newaxis]
        hue = (y_norm + t) % 1.0
        r = (np.sin(hue * 6.28) * 0.5 + 0.5) * 40
        g = (np.sin(hue * 6.28 + 2.094) * 0.5 + 0.5) * 40
        b = (np.sin(hue * 6.28 + 4.189) * 0.5 + 0.5) * 40
        arr[:, :, 0] = r
        arr[:, :, 1] = g
        arr[:, :, 2] = b
        return arr.astype(np.uint8)

    def _render_aurora_numpy(self) -> np.ndarray:
        """极光效果背景 — NumPy 直接返回"""
        arr = np.zeros((self.h, self.w, 3), dtype=np.float32)
        t = self.time * 0.01
        y, x = np.ogrid[:self.h, :self.w]
        y_norm = y / max(self.h - 1, 1)
        for wave_i in range(3):
            phase = wave_i * 2.094 + t * (wave_i + 1) * 0.5
            wave = np.sin(x * 0.01 + phase) * np.sin(y_norm * 3.14) * 0.5 + 0.5
            hue = (wave_i * 0.33 + t * 0.05) % 1.0
            r, g, b = _hsv_to_rgb_simple(hue, 0.7, wave * 0.5 + 0.1)
            arr[:, :, 0] += r.astype(np.float32)
            arr[:, :, 1] += g.astype(np.float32)
            arr[:, :, 2] += b.astype(np.float32)
        return np.clip(arr, 0, 80).astype(np.uint8)

    # ── 保留原 PIL 方法（兼容）──────────────────────────────────────
    def _render_empty(self) -> Image.Image:
        return Image.fromarray(self._render_empty_numpy())

    def _render_starfield(self) -> Image.Image:
        return Image.fromarray(self._render_starfield_numpy())

    def _render_gradient(self) -> Image.Image:
        return Image.fromarray(self._render_gradient_numpy())

    def _render_aurora(self) -> Image.Image:
        return Image.fromarray(self._render_aurora_numpy())


def _hsv_to_rgb_simple(h, s, v):
    """
    向量化 HSV→RGB 转换。
    h, s, v 可以是标量或数组，会被广播到相同形状。
    返回 (r, g, b) 三个 numpy 数组，形状与广播后的输入相同。
    """
    # 广播到相同形状
    h, s, v = np.broadcast_arrays(h, s, v)

    # 确保 h 在 [0, 1) 范围内
    h = h % 1.0

    # 向量化 HSV 到 RGB 转换
    i = (h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)

    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    # 创建输出数组
    r = np.zeros_like(v)
    g = np.zeros_like(v)
    b = np.zeros_like(v)

    # 根据色扇区分配 RGB 值
    for i_val in range(6):
        mask = i == i_val
        if i_val == 0:
            r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
        elif i_val == 1:
            r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
        elif i_val == 2:
            r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
        elif i_val == 3:
            r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
        elif i_val == 4:
            r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
        elif i_val == 5:
            r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]

    # 转换为 0-255 范围的 uint8
    return (
        (r * 255).clip(0, 255).astype(np.uint8),
        (g * 255).clip(0, 255).astype(np.uint8),
        (b * 255).clip(0, 255).astype(np.uint8),
    )


# ══════════════════════════════════════════════════════════════════
#  图层渲染器工厂
# ══════════════════════════════════════════════════════════════════

def create_layer_renderer(layer: LayerConfig, width: int, height: int):
    """为图层创建渲染器"""
    effect = layer.effect

    if effect.source == EffectSource.DRAW:
        # draw 效果 → DrawEffectRenderer
        renderer = draw_renderer.DrawEffectRenderer(
            width=width, height=height,
            grid_w=layer.params.get("grid_w", 64),
            grid_h=layer.params.get("grid_h", 64),
        )
        # 加载配置（需要从 JSON 文件重新加载）
        _load_draw_config(renderer, effect.file_path, layer.params)
        return renderer

    else:
        # pywallpaper 效果 → PWEffecRenderer
        renderer = pw_renderer.PWEffecRenderer(width=width, height=height)
        _load_pw_config(renderer, effect.file_path)
        return renderer


def _load_draw_config(renderer: draw_renderer.DrawEffectRenderer,
                      filepath: str, params: dict):
    """从 JSON 文件加载 draw 效果配置到渲染器"""
    try:
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
        renderer.load_config(config)
        # 覆盖随机参数
        if "palette" in params:
            renderer.palette = params["palette"]
        if "steps_per_frame" in params:
            renderer._steps_per_frame = params["steps_per_frame"]
    except Exception as e:
        print(f"[Warn] 加载 draw 配置失败 {filepath}: {e}")


def _load_pw_config(renderer: pw_renderer.PWEffecRenderer, filepath: str):
    """从 JSON 文件加载 pywallpaper 效果配置到渲染器"""
    try:
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
        renderer.load_config(config)
    except Exception as e:
        print(f"[Warn] 加载 pw 配置失败 {filepath}: {e}")


# ══════════════════════════════════════════════════════════════════
#  图层合成
# ══════════════════════════════════════════════════════════════════

def composite_frames(bottom: Image.Image, top: Image.Image,
                    opacity: float, blend_mode: str) -> Image.Image:
    """合成两个帧（PIL 版本，保留兼容性）"""
    if opacity <= 0.0:
        return bottom
    if opacity >= 1.0 and blend_mode == "overlay":
        return top

    top_alpha = Image.new("L", top.size, int(opacity * 255))
    top_rgba = top.copy().convert("RGBA")
    top_rgba.putalpha(top_alpha)

    bottom_rgba = bottom.copy().convert("RGBA")

    if blend_mode == "add":
        result = Image.alpha_composite(bottom_rgba, top_rgba)
    elif blend_mode == "multiply":
        result_arr = (np.array(bottom) / 255.0 * np.array(top) / 255.0 * 255).astype(np.uint8)
        return Image.fromarray(result_arr)
    elif blend_mode == "screen":
        b = np.array(bottom) / 255.0
        t = np.array(top) / 255.0
        r = 1.0 - (1.0 - b) * (1.0 - t)
        return Image.fromarray((r * 255).astype(np.uint8))
    else:
        result = Image.alpha_composite(bottom_rgba, top_rgba).convert("RGB")

    return result.convert("RGB")


def composite_frames_numpy(bottom: np.ndarray, top: np.ndarray,
                          opacity: float, blend_mode: str) -> np.ndarray:
    """合成两个帧（NumPy 版本，避免 PIL 转换）

    bottom, top: (H, W, 3) uint8 NumPy 数组
    返回: (H, W, 3) uint8 NumPy 数组
    """
    if opacity <= 0.0:
        return bottom
    op = np.float32(opacity)

    # 转 float32 [0,1] 计算
    b = bottom.astype(np.float32) / 255.0
    t = top.astype(np.float32) / 255.0

    if blend_mode == "add":
        # 加法混合（clamp）
        r = np.clip(b + t * op, 0.0, 1.0)
    elif blend_mode == "multiply":
        # 乘法混合
        r = np.clip(b * t * op, 0.0, 1.0)
    elif blend_mode == "screen":
        # 滤色: 1 - (1-b)*(1-t)
        r = np.clip(1.0 - (1.0 - b) * (1.0 - t * op), 0.0, 1.0)
    else:  # "overlay" or default: alpha 混合
        r = np.clip(b * (1.0 - op) + t * op, 0.0, 1.0)

    return (r * 255).astype(np.uint8)


class FFmpegWriter:
    """ffmpeg 视频编码器（NVENC GPU 加速 + CPU fallback）

    使用 stdin pipe 模式逐帧写入原始 RGB 数据，
    替代 cv2.VideoWriter，支持硬件编码。
    """

    def __init__(self, output_path: str, width: int, height: int,
                 fps: int, prefer_hw: str = "auto", codec: str = "h264",
                 container: str = "mp4", bit_depth: int = 8):
        """FFmpeg 编码器（NVENC GPU 加速 + CPU fallback）。

        container : "mp4" | "webm" | "gif"
        bit_depth : 8（默认）或 10（仅 mp4/webm 的 hevc/vp9 生效，HDR 友好）
        """
        self.output_path = output_path
        self.w = width
        self.h = height
        self.fps = fps
        self.prefer_hw = prefer_hw
        self.codec = codec
        self.container = (container or "mp4").lower()
        self.bit_depth = 10 if int(bit_depth) >= 10 else 8
        self.proc = None
        self.used_hw = None
        self._stderr = None
        self._gif = False
        self._open()

    def _find_ffmpeg(self) -> str:
        # 1) 显式环境变量（最高优先级）：FFMPEG_BIN 可指向任意 ffmpeg 二进制
        env_bin = os.environ.get("FFMPEG_BIN", "").strip()
        if env_bin:
            try:
                subprocess.run([env_bin, "-version"],
                               capture_output=True, timeout=5)
                return env_bin
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass
        # 2) 项目自带 essentials build（与 ffprobe 配套，优先于 PATH 上的散装 ffmpeg）
        bundled = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ffmpeg-2026-06-08-git-6028720d70-essentials_build", "bin", "ffmpeg.exe")
        # 3) 默认候选路径
        for cand in [bundled, "ffmpeg", "C:/ffmpeg/bin/ffmpeg.exe"]:
            if not cand:
                continue
            try:
                subprocess.run([cand, "-version"],
                               capture_output=True, timeout=5)
                return cand
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                continue
        return "ffmpeg"





    def _detect_hw_encoder(self) -> Optional[str]:
        codec = (self.codec or "h264").lower()
        # webm 容器默认走 libvpx（VP9），无稳定硬件编码器，强制 CPU 软编码
        if self.container == "webm":
            return None
        if self.prefer_hw == "cpu":
            return None
        # 硬件编码器名映射（prefer_hw 显式指定时直接返回）
        hw_map = {
            "nvenc": {"h264": "h264_nvenc", "hevc": "hevc_nvenc", "av1": "av1_nvenc"},
            "qsv":   {"h264": "h264_qsv",   "hevc": "hevc_qsv",   "av1": "av1_qsv"},
        }
        if self.prefer_hw in hw_map:
            return hw_map[self.prefer_hw].get(codec)
        # auto: 检测可用硬件编码器
        try:
            result = subprocess.run(
                [self._find_ffmpeg(), "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=10,
            )
            encoders = result.stdout + result.stderr
            for enc in (f"{codec}_nvenc", f"{codec}_qsv"):
                if enc in encoders:
                    return enc
        except Exception:
            pass
        return None

    def _open(self):
        ffmpeg = self._find_ffmpeg()
        hw_enc = self._detect_hw_encoder()
        codec = (self.codec or "h264").lower()
        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.w}x{self.h}",
            "-r", str(self.fps),
            "-i", "pipe:0",
        ]
        is_mp4 = self.container == "mp4"
        is_gif = self.container == "gif"
        is_webm = self.container == "webm"
        # 10-bit 仅对 hevc(vp9 容器)/vp9 有意义；h264 强制 8-bit
        use_10bit = (self.bit_depth >= 10) and (codec in ("hevc", "vp9"))

        if is_gif:
            # 动态 GIF：先写无损中间（palettegen 两遍），这里走单遍 + 全局调色板
            # 用 split + palettegen/paletteuse 保证色彩还原（真实验证可用）
            cmd = [
                ffmpeg, "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{self.w}x{self.h}", "-r", str(self.fps), "-i", "pipe:0",
                "-vf", "split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
                       "[s1][p]paletteuse=dither=bayer:bayer_scale=5",
                "-loop", "0", self.output_path,
            ]
            self.used_hw = None
            self._gif = True
            print(f"[FFmpeg] 动态 GIF: {self.w}x{self.h} @ {self.fps}fps")
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL)
            return

        # ── 视频容器（mp4 / webm）──
        if hw_enc:
            # NVENC/QSV 质量增强（联网调研：spatial/temporal AQ + rc-lookahead + B 帧）
            hw_profiles = {
                "h264": ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq",
                          "-rc", "vbr", "-cq", "23", "-b:v", "8M",
                          "-spatial-aq", "1", "-temporal-aq", "1",
                          "-rc-lookahead", "20", "-bf", "2", "-b_ref_mode", "middle"],
                "hevc": ["-c:v", "hevc_nvenc", "-preset", "p4", "-tune", "hq",
                          "-rc", "vbr", "-b:v", "8M",
                          "-spatial-aq", "1", "-temporal-aq", "1",
                          "-rc-lookahead", "20", "-bf", "2", "-b_ref_mode", "middle",
                          "-tag:v", "hvc1"],
                "av1":  ["-c:v", "av1_nvenc", "-preset", "p4",
                          "-rc", "vbr", "-b:v", "8M", "-rc-lookahead", "20",
                          "-tag:v", "av01"],
            }
            cmd += hw_profiles.get(
                codec, ["-c:v", hw_enc, "-preset", "p4", "-rc", "vbr", "-b:v", "8M"])
            self.used_hw = hw_enc
            print(f"[FFmpeg] 硬件编码: {hw_enc} (codec={codec}, "
                  f"{'10-bit' if use_10bit else '8-bit'})")
        else:
            # CPU 软编码（libx264 / libx265 / libaom-av1 / libvpx-vp9）
            cpu_profiles = {
                "h264": ["-c:v", "libx264", "-preset", "medium", "-crf", "23"],
                "hevc": (["-c:v", "libx265", "-preset", "medium", "-crf", "28",
                          "-pix_fmt", "yuv420p10le", "-x265-params",
                          "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc"]
                         if use_10bit else
                         ["-c:v", "libx265", "-preset", "medium", "-crf", "28"]),
                "vp9":  (["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32",
                          "-pix_fmt", "yuv420p10le",
                          "-metadata:s:v:0", "color_primaries=bt2020",
                          "-metadata:s:v:0", "transfer_characteristics=smpte2084",
                          "-metadata:s:v:0", "matrix_coefficients=bt2020nc"]
                         if use_10bit else
                         ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32"]),
                "av1":  ["-c:v", "libaom-av1", "-crf", "30", "-b:v", "0",
                         "-cpu-used", "4", "-row-mt", "1"],
            }
            cpu_cmd = cpu_profiles.get(
                codec, ["-c:v", "libx264", "-preset", "medium", "-crf", "23"])
            cmd += cpu_cmd
            self.used_hw = None
            print(f"[FFmpeg] CPU 软编码: {codec} "
                  f"({'10-bit HDR' if use_10bit else '8-bit'})")

        # 兼容性 + 网络流式（MP4 前置 faststart，便于边下边播）
        if not use_10bit:
            cmd += ["-pix_fmt", "yuv420p"]
        if is_mp4:
            cmd += ["-movflags", "+faststart"]
        if is_webm and codec == "vp9":
            # webm 容器 VP9 默认即可，无需额外 muxer flag
            pass
        cmd.append(self.output_path)

        # stderr=DEVNULL 避免 ffmpeg 进度输出填满 PIPE 缓冲区导致阻塞
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

    def write(self, frame: np.ndarray):
        if self.proc is None:
            return
        try:
            self.proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            stderr = self.proc.stderr.read()
            raise RuntimeError(
                f"ffmpeg 管道中断:\n{stderr.decode(errors='replace')}"
            )

    def release(self):
        if self.proc is None:
            return
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()
        ret = self.proc.returncode
        self.proc = None
        if ret != 0:
            raise RuntimeError(f"ffmpeg 退出码: {ret}")

    def is_opened(self) -> bool:
        return self.proc is not None and self.proc.poll() is None



# ═════════════════════════════════════════════════════════════════
#  主生成流程
# ═════════════════════════════════════════════════════════════════


def find_ffprobe() -> str:
    """查找 ffprobe 可执行文件，优先与 ffmpeg 同目录（配套 build）。

    返回可用路径；找不到时回退 "ffprobe"（依赖 PATH）。
    """
    # 1) 与 FFMPEG_BIN 指向的 ffmpeg 同目录
    env_bin = os.environ.get("FFMPEG_BIN", "").strip()
    if env_bin:
        d = os.path.dirname(env_bin)
        same_dir = os.path.join(d, "ffprobe.exe" if os.name == "nt" else "ffprobe")
        if os.path.isfile(same_dir):
            return same_dir
    # 2) 项目自带 essentials build（ffmpeg 与 ffprobe 同 bin 目录）
    bundled = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ffmpeg-2026-06-08-git-6028720d70-essentials_build", "bin")
    for name in ("ffprobe.exe", "ffprobe"):
        cand = os.path.join(bundled, name)
        if os.path.isfile(cand):
            return cand
    # 3) 常见位置
    for cand in (
        "ffprobe",
        "C:/ffmpeg/bin/ffprobe.exe",
        os.path.join(os.environ.get("PROGRAMFILES", ""), "ffmpeg", "bin", "ffprobe.exe"),
    ):
        if not cand:
            continue
        if os.path.isfile(cand) or shutil_which(cand):
            return cand
    return "ffprobe"


def shutil_which(cmd: str) -> bool:
    from shutil import which
    return which(cmd) is not None
class VideoGenerator:
    """视频生成器主类"""

    def __init__(self, composition: VideoComposition, output_path: str,
                 prefer_hw: str = "auto", codec: str = "h264",
                 audio_path: str = None, audio_hue_shift: bool = False,
                 audio_balance: bool = False,
                 audio_volume: float = 1.0, audio_mux: bool = True,
                 use_audio_stems: bool = False, use_gl: bool = False,
                 container: str = "mp4", bit_depth: int = 8,
                 aspect: str = None, diffusion_base: object = None):
        self.comp = composition
        self.output_path = output_path
        self.prefer_hw = prefer_hw
        self.codec = codec
        self.container = (container or "mp4").lower()
        self.bit_depth = 10 if int(bit_depth) >= 10 else 8
        # P2-3：竖屏/方形社媒预设（9:16 / 1:1 / 16:9）按长边算分辨率
        if aspect:
            self._apply_aspect(aspect)
        # GIF 无音轨，强制关闭混入
        if self.container == "gif":
            audio_mux = False
        # P2-1：扩散基础层（可选；GL 模式下不与 shader 基底混用，仅 CPU 路径生效）
        self.diffusion_base = diffusion_base
        self.audio_path = audio_path
        self.audio_hue_shift = audio_hue_shift
        self.audio_balance = audio_balance
        self.audio_volume = audio_volume
        self.audio_mux = audio_mux
        self.use_audio_stems = use_audio_stems
        self.use_gl = use_gl
        self.audio_reactive = None  # 由 generate() 在需要时构造（= 整曲 mix）
        self.audio_stems = None     # P0-3：Demucs 分轨（可选增强）
        self.stems_drive_beat = False  # 分轨接管 beat 冲击时置 True

        # 渲染器缓存
        self._layer_renderers: List[Tuple[LayerConfig, object]] = []

        # 主题渲染器
        self.theme_renderer = ThemeRenderer(
            composition.theme, composition.width, composition.height
        )
        self.gl = None  # 由 _prepare_renderers() 在 generate() 内填充

    def _log(self, event_type: str, stage: str = None, **payload):
        """P4 任务事件日志（写失败静默，不阻断生成）。"""
        tasklog = getattr(self, "_tasklog", None)
        if tasklog is not None:
            try:
                tasklog.append(event_type, stage=stage, run_id=self._run_id, **payload)
            except Exception:
                pass

    def _apply_aspect(self, aspect: str):
        """按社媒比例重算分辨率（P2-3）。

        aspect 形如 "9:16" / "1:1" / "16:9" / "4:3"；长边沿用当前较大尺寸，
        短边按比例向下取偶。竖屏 9:16 即竖屏短视频预设。
        """
        try:
            a, b = aspect.lower().split(":")
            a, b = float(a), float(b)
            if a <= 0 or b <= 0:
                raise ValueError
        except Exception:
            print(f"[Aspect] 非法比例 {aspect!r}，忽略")
            return
        w, h = self.comp.width, self.comp.height
        long_edge = max(w, h)
        if a >= b:  # 横向或方形
            nw = long_edge
            nh = int(round(long_edge * b / a / 2.0) * 2)
        else:       # 竖向（如 9:16）
            nh = long_edge
            nw = int(round(long_edge * a / b / 2.0) * 2)
        self.comp.width, self.comp.height = max(2, nw), max(2, nh)
        print(f"[Aspect] {aspect} → {self.comp.width}x{self.comp.height}")

    def _prepare_renderers(self):
        """准备渲染后端：GL 着色器（可选）与 pw 图层渲染器。

        generate() 与单帧测试（如 _verify_gl_layers）共用，避免重复逻辑。
        """
        w, h, fps = self.comp.width, self.comp.height, self.comp.fps
        # P1-1：GPU(GLSL) 渲染后端（可选增强；不可用则降级 CPU）
        self.gl = None
        if self.use_gl:
            try:
                from gl_renderer import GLRenderer, GLUnavailable
                fam = GLRenderer.family_for_theme(self.comp.theme)
                pal = GLRenderer.palette_for_theme(self.comp.theme)
                self.gl = GLRenderer(
                    w, h, fps, family=fam, palette=pal,
                    seed=float(self.comp.duration) + 1.0,
                    speed=1.0, complexity=0.5)
                print(f"[GL] ✓ GPU 渲染后端已启用 (family={fam}, palette={pal})")
                # 主线程建完上下文/着色器后必须释放，否则 worker 线程无法
                # make_context_current 认领 → glUseProgram 解析为 NULL。
                self.gl.release_context()
            except GLUnavailable as e:
                print(f"[GL] 不可用，降级 CPU 渲染: {e}")
            except Exception as e:  # noqa: BLE001
                print(f"[GL] 初始化失败，降级 CPU 渲染: {type(e).__name__}: {e}")

        # 初始化图层渲染器（GL 模式下基底由 GL 着色器产出，跳过 CPU 图层）
        if self.gl is None:
            self._init_layer_renderers()

    def generate(self, preview: bool = False):
        """生成视频"""
        w, h = self.comp.width, self.comp.height
        fps = self.comp.fps
        n_frames = int(self.comp.duration * fps)

        print(f"\n{'='*60}")
        print(f"  视频生成器 v{VERSION}")
        print(f"{'='*60}")
        print(self.comp.summary())
        print(f"\n输出: {self.output_path}")
        print(f"总帧数: {n_frames} ({self.comp.duration}s × {fps}fps)")
        print(f"{'='*60}\n")

        # ── 任务事件日志（P4：日志即事实源，<输出>.pipeline.log） ──
        try:
            from genlog import PipelineLog
            self._tasklog = PipelineLog(str(self.output_path) + ".pipeline.log")
            self._run_id = PipelineLog.new_run_id()
            self._tasklog.append("run_start", run_id=self._run_id,
                                 output=str(self.output_path),
                                 duration_s=float(self.comp.duration), fps=fps,
                                 width=w, height=h)
        except Exception:
            self._tasklog = None

        # 准备渲染后端（GL 着色器 + pw 图层）
        self._log("stage_start", stage="renderers")
        try:
            self._prepare_renderers()
        finally:
            self._log("stage_done", stage="renderers", detail=f"gl={self.gl is not None}")

        # 音频驱动：若有音频，构造逐帧能量包络（可选依赖，失败降级中性调制）
        self._log("stage_start", stage="audio")
        self.audio_reactive = None
        self.audio_stems = None
        self.stems_drive_beat = False
        if self.audio_path:
            try:
                from audio_reactive import AudioReactive
                self.audio_reactive = AudioReactive(
                    self.audio_path, fps, self.comp.duration)
                print(f"[Audio] 已加载音频驱动: {self.audio_path}")
            except Exception as e:  # noqa: BLE001
                print(f"[Audio] 音频驱动初始化失败，降级为中性调制: {e}")

            # P0-3：Demucs 分轨音频驱动（可选增强；缺失/失败则降级整曲混合）
            if self.use_audio_stems:
                try:
                    self.audio_stems = audio_stems.AudioStems(
                        self.audio_path, fps, self.comp.duration)
                    self.stems_drive_beat = True
                    # 复用 stems.mix 作为基础调制，避免二次计算
                    if self.audio_stems.mix is not None:
                        self.audio_reactive = self.audio_stems.mix
                    print(f"[Audio] ✓ {self.audio_stems.summary()}")
                except audio_stems.DemucsUnavailable as e:
                    print(f"[Audio] Demucs 不可用，降级为整曲混合驱动: {e}")
                except Exception as e:  # noqa: BLE001
                    print(f"[Audio] 分轨驱动初始化失败，降级为整曲混合驱动: {e}")

        self._log("stage_done" if self.audio_path else "stage_skip", stage="audio",
                  detail=("reactive" if self.audio_reactive is not None else "none")
                  + (", stems" if self.audio_stems is not None else ""))
        self._log("stage_start", stage="encode")

        # 打开视频编码器（ffmpeg NVENC + CPU fallback；P2-3 支持 webm/gif/10bit）
        writer = FFmpegWriter(
            output_path=self.output_path,
            width=w, height=h,
            fps=fps,
            prefer_hw=self.prefer_hw,
            codec=self.codec,
            container=self.container,
            bit_depth=self.bit_depth,
        )
        if not writer.is_opened():
            raise RuntimeError("无法打开 ffmpeg 视频编码器")

        # 启动渲染线程（生产者）→ 队列 → 主线程写 ffmpeg（消费者）
        frame_queue = queue.Queue(maxsize=4)

        worker = threading.Thread(
            target=self._render_worker,
            args=(frame_queue, n_frames, preview),
            daemon=True,
            name="RenderWorker",
        )
        worker.start()

        # 主线程：从队列取帧，写入 ffmpeg（编码并行）
        start_time = time.time()
        frames_written = 0
        frames_to_write = n_frames
        if preview:
            frames_to_write = min(frames_to_write, 30)

        while True:
            frame_idx, frame_rgb = frame_queue.get(block=True)

            if frame_idx is None:
                # 渲染线程结束（正常或异常）
                if frame_rgb is not None:
                    writer.release()
                    raise RuntimeError(
                        f"渲染线程异常: {frame_rgb}"
                    ) from frame_rgb
                break  # 正常结束

            # 写入 ffmpeg pipe
            writer.write(frame_rgb)
            frames_written += 1

            # 进度
            if (frames_written % (fps * 2) == 0
                    or frames_written == frames_to_write):
                elapsed = time.time() - start_time
                fps_actual = frames_written / max(elapsed, 0.001)
                pct = frames_written / frames_to_write * 100
                print(
                    f"  帧 {frames_written}/{frames_to_write} "
                    f"({pct:.1f}%) "
                    f"速度: {fps_actual:.1f} fps"
                )

        worker.join()
        writer.release()
        self._log("stage_done", stage="encode", frames=frames_written,
                  fps_actual=round(frames_written / max(time.time() - start_time, 0.001), 1))
        elapsed = time.time() - start_time
        print(f"[OK] 视频生成完成！")
        print(f"  输出文件: {self.output_path}")
        print(f"  耗时: {elapsed:.1f}s")
        print(f"  平均速度: {frames_to_write/max(elapsed, 0.001):.1f} fps")

        # ── 音频合成：把 audio_path 真正混入 mp4（FFMPEG_BIN 优先） ──
        # audio_path 同时驱动色相偏移（audio_reactive）与最终音轨，二合一。
        self._log("stage_start", stage="mux")
        if self.audio_path and self.audio_mux:
            try:
                ok = audio_mixer.mux_audio(
                    video_path=self.output_path,
                    audio_arg=self.audio_path,
                    volume=self.audio_volume,
                    script_dir=os.path.dirname(os.path.abspath(__file__)),
                )
                if ok:
                    print(f"[Audio] ✓ 音频已混入 mp4: {self.audio_path}")
            except Exception as e:  # noqa: BLE001
                print(f"[Audio] ⚠ 音频混入失败（视频已生成，仅无声）: {e}")
                self._log("stage_error", stage="mux", error=str(e))

        self._log("stage_done" if (self.audio_path and self.audio_mux) else "stage_skip",
                  stage="mux", detail=str(self.output_path))
        self._log("run_end", ok=True, output=str(self.output_path))

    def _render_worker(self, frame_queue: queue.Queue, n_frames: int, preview: bool):
        """渲染线程：提前渲染帧，放入队列

        主线程从队列取帧写入 ffmpeg，实现渲染/编码并行。
        """
        if self.gl is not None:
            self.gl.make_current()  # 把 GL 上下文绑到本线程
        try:
            for frame_idx in range(n_frames):
                if preview and frame_idx >= 30:
                    break
                t = frame_idx / self.comp.fps
                frame_rgb = self._render_frame(t, frame_idx)
                # put 阻塞直到队列有空位（maxsize=4）
                frame_queue.put((frame_idx, frame_rgb), block=True)
        except Exception as e:
            # 把异常传到主线程
            frame_queue.put((None, e), block=True)
            return
        # sentinel：通知主线程渲染结束
        frame_queue.put((None, None), block=True)

    def _init_layer_renderers(self):
        """初始化所有图层渲染器"""
        print("初始化图层渲染器...")
        for i, layer in enumerate(self.comp.layers):
            print(f"  [{i}] {layer.effect.name}...", end="")
            try:
                renderer = create_layer_renderer(
                    layer, self.comp.width, self.comp.height
                )
                self._layer_renderers.append((layer, renderer))
                print(" OK")
            except Exception as e:
                print(f" 失败: {e}")
                self._layer_renderers.append((layer, None))

    def _render_frame(self, t: float, frame_idx: int) -> np.ndarray:
        """渲染单帧，全程 NumPy 数组，返回 (H,W,3) uint8"""
        dt = 1.0 / self.comp.fps

        # ── GPU(GLSL) 基底帧（P1-1）──
        if self.gl is not None:
            env = (self.audio_reactive.envelope_at(frame_idx)
                   if self.audio_reactive is not None else 0.5)
            frame = self.gl.render_frame(t, frame_idx, env)
            # GL 基底之上叠加既有 pw 图层（circle_map / flow / 等），再统一后处理
            frame = self._composite_layers_numpy(frame, t, dt)
            frame = self._apply_global_postprocess_numpy(frame, frame_idx)
            return frame

        # ── CPU 路径（既有 pw_renderer 全量管线）──
        # 1. 渲染主题（背景）→ NumPy 数组
        #    P2-1：若启用扩散基础层，用其帧替代程序化主题背景（风格化叠加基底）
        if self.diffusion_base is not None:
            try:
                frame = self.diffusion_base.frame_at(frame_idx).copy()
                if frame.shape[:2] != (self.comp.height, self.comp.width):
                    from PIL import Image as _PIL
                    frame = np.array(_PIL.fromarray(frame).resize(
                        (self.comp.width, self.comp.height), _PIL.BILINEAR))
            except Exception as e:  # noqa: BLE001
                print(f"[Diffusion] 基础层取帧失败，降级程序化主题: {e}")
                self.theme_renderer.update(dt)
                frame = self.theme_renderer.render_numpy()
        else:
            self.theme_renderer.update(dt)
            frame = self.theme_renderer.render_numpy()

        # 音频驱动：能量包络调制主题整体亮度（beat 时短暂提亮）
        if self.audio_reactive is not None:
            env = self.audio_reactive.envelope_at(frame_idx)
            beat = self.audio_reactive.beat_at(frame_idx)
            bright = 0.75 + 0.5 * env + (0.2 if beat else 0.0)
            bright = min(max(bright, 0.4), 1.6)
            frame = np.clip(
                frame.astype(np.float32) * bright, 0, 255
            ).astype(np.uint8)

        # 2. 叠加各图层（NumPy 合成，与 GL 分支共享同一实现）
        frame = self._composite_layers_numpy(frame, t, dt)

        # 3. 全局后处理（NumPy 版本）
        frame = self._apply_global_postprocess_numpy(frame, frame_idx)

        return frame

    def _composite_layers_numpy(self, frame: np.ndarray,
                                t: float, dt: float) -> np.ndarray:
        """把已初始化的 pw 图层合成到 `frame` 上（CPU 与 GL 路径共用）。

        处理：图层启停时间、update 推进、获取图层帧（render_frame /
        draw_to_array / draw 三态 fallback）、尺寸对齐、smoothstep 淡入淡出、
        最后 composite_frames_numpy 混合。
        """
        for layer_cfg, renderer in self._layer_renderers:
            if renderer is None:
                continue
            if layer_cfg.start_time > t:
                continue
            if layer_cfg.duration is not None:
                if t > layer_cfg.start_time + layer_cfg.duration:
                    continue

            if hasattr(renderer, "update"):
                renderer.update(dt)

            # 获取图层帧（优先 NumPy 路径）
            layer_frame = None
            if hasattr(renderer, "render_frame"):
                # draw_renderer: 返回 NumPy 数组
                layer_frame = renderer.render_frame()
            elif hasattr(renderer, "draw_to_array"):
                # pw_renderer: 直接写 NumPy 数组
                canvas = np.zeros((self.comp.height, self.comp.width, 3), dtype=np.uint8)
                layer_frame = renderer.draw_to_array(canvas)
            elif hasattr(renderer, "draw"):
                # fallback: PIL → NumPy
                img = renderer.draw()
                layer_frame = np.array(img)

            if layer_frame is None:
                continue

            # 确保 layer_frame 是 NumPy 数组（类型安全）
            if not isinstance(layer_frame, np.ndarray):
                layer_frame = np.array(layer_frame, dtype=np.uint8)
            # 尺寸对齐（draw_renderer 可能输出非目标尺寸）
            if layer_frame.shape[:2] != frame.shape[:2]:
                from PIL import Image as PILImg
                im = PILImg.fromarray(layer_frame)
                im = im.resize((frame.shape[1], frame.shape[0]), PILImg.BILINEAR)
                layer_frame = np.array(im)

            # 图层淡入淡出（smoothstep 缓动转场）
            fade_dur = getattr(layer_cfg, "fade", 0.5)
            fade_in = getattr(layer_cfg, "fade_in", fade_dur)
            fade_out = getattr(layer_cfg, "fade_out", fade_dur)
            fopacity = layer_cfg.opacity
            if t < layer_cfg.start_time + fade_in:
                p = (t - layer_cfg.start_time) / max(fade_in, 1e-6)
                fopacity *= _smoothstep(min(max(p, 0.0), 1.0))
            if layer_cfg.duration is not None:
                end_t = layer_cfg.start_time + layer_cfg.duration
                if t > end_t - fade_out:
                    p = (end_t - t) / max(fade_out, 1e-6)
                    fopacity *= _smoothstep(min(max(p, 0.0), 1.0))

            # NumPy 合成
            frame = composite_frames_numpy(
                frame, layer_frame,
                fopacity, layer_cfg.blend_mode
            )

        return frame

    def _apply_global_postprocess_numpy(self, arr: np.ndarray,
                                        frame_idx: int = 0) -> np.ndarray:
        """后处理（NumPy 版本，避免 PIL 转换）+ 音频驱动调制"""
        pp = self.comp.global_postprocess

        if pp:
            if "vignette" in pp:
                if not hasattr(self, "_vignette_mask"):
                    y, x = np.ogrid[:self.comp.height, :self.comp.width]
                    cx, cy = self.comp.width / 2, self.comp.height / 2
                    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    max_dist = np.sqrt(cx ** 2 + cy ** 2)
                    vignette = (1.0 - 0.6 * (dist / max_dist) ** 2)
                    self._vignette_mask = np.clip(vignette, 0, 1)
                arr = (arr * self._vignette_mask[:, :, np.newaxis]).astype(np.uint8)

            if "glow" in pp:
                # 音频能量驱动辉光强度（beat 时更亮）
                strength = 0.5
                if self.audio_reactive is not None:
                    strength = 0.3 + 0.5 * self.audio_reactive.envelope_at(frame_idx)
                img = Image.fromarray(arr)
                blur = img.filter(ImageFilter.GaussianBlur(radius=4))
                arr = np.clip(
                    arr.astype(np.float32) * 0.7 + np.array(blur) * strength, 0, 255
                ).astype(np.uint8)

            if "edge" in pp:
                img = Image.fromarray(arr)
                edges = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
                arr = np.clip(
                    np.array(img) * 0.7 + np.array(edges) * 0.3, 0, 255
                ).astype(np.uint8)

        # 音频频谱质心 → 色温平衡（冷/暖随明亮度漂移）
        if (self.audio_reactive is not None and self.audio_balance
                and getattr(self.audio_reactive, "centroid", None) is not None
                and self.audio_reactive.centroid.size > 1):
            warmth = self.audio_reactive.centroid_at(frame_idx) * 2.0 - 1.0
            if abs(warmth) > 0.02:
                arr = _color_temperature(arr, warmth)

        # 音频 beat 触发数据冲击转场（轻量 glitch，独立于后处理开关）
        # 注：开启 Demucs 分轨后，鼓点冲击由 drums 分轨接管（避免双冲），
        #     此时 stems_drive_beat=True，跳过整曲混合的 beat glitch。
        if (self.audio_reactive is not None
                and not getattr(self, "stems_drive_beat", False)
                and self.audio_reactive.beat_at(frame_idx)):
            h, w = arr.shape[:2]
            band = np.random.randint(0, max(1, h - 8))
            shift = np.random.randint(-20, 21)
            arr = arr.copy()
            arr[band:band + 8] = np.roll(arr[band:band + 8], shift, axis=1)

        # 音频色相随节拍偏移（beat 驱动 + 包络漂移，见 audio_reactive）
        if (self.audio_reactive is not None and self.audio_hue_shift
                and getattr(self.audio_reactive, "hue", None) is not None
                and self.audio_reactive.hue.size > 1):
            hue_deg = self.audio_reactive.hue_shift_at(frame_idx) * 360.0
            if hue_deg > 0.01:
                arr = _hsv_hue_rotate(arr, hue_deg)

        # P0-3：Demucs 分轨音频驱动（鼓→冲击/色相迸发，贝斯→对比脉动，
        #        人声→色温，环境→辉光）；各通道独立，缺失分轨自动跳过。
        if getattr(self, "audio_stems", None) is not None:
            arr = audio_stems.apply_stem_modulation(arr, frame_idx, self.audio_stems)

        return arr

    def _apply_global_postprocess(self, img: Image.Image) -> Image.Image:
        """应用全局后处理（vignette 用 PIL composite，避免每帧 NumPy 转换）"""
        pp = self.comp.global_postprocess

        if not pp:
            return img

        if "vignette" in pp:
            # 暗角：缓存 PIL 渐变图，用 Image.composite 混合（无 NumPy 转换）
            if not hasattr(self, '_vignette_grad_pil'):
                # 用 NumPy 算一次，转 PIL 缓存
                y, x = np.ogrid[:self.comp.height, :self.comp.width]
                cx, cy = self.comp.width / 2, self.comp.height / 2
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                max_dist = np.sqrt(cx ** 2 + cy ** 2)
                vignette = 1.0 - 0.6 * (dist / max_dist) ** 2
                vignette = np.clip(vignette * 255, 0, 255).astype(np.uint8)
                self._vignette_grad_pil = Image.fromarray(vignette, mode="L")
            # composite: img * (grad/255) + black * (1 - grad/255)
            img = Image.composite(
                img,
                Image.new("RGB", img.size, (0, 0, 0)),
                self._vignette_grad_pil
            )

        if "glow" in pp:
            # 辉光
            blur = img.filter(ImageFilter.GaussianBlur(radius=4))
            arr = np.clip(np.array(img) * 0.7 + np.array(blur) * 0.5, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        if "edge" in pp:
            # 边缘增强
            edges = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
            img = Image.blend(img, edges, 0.3)

        return img


# ══════════════════════════════════════════════════════════════════
#  命令行接口
# ══════════════════════════════════════════════════════════════════

def _write_album_manifest(album_name: str, tracks: List[dict],
                          out_dir: str, generated_paths: List[str],
                          keep_top: Optional[int]):
    """P1-3：写专辑 manifest（合并 curate 的美学评分 + keep 标记）。

    输出：<out_dir>/<album>/album_manifest.json，含专辑标题、曲目数、
    每首元数据、美学评分、keep 标记，以及 curate 的连镜表/筛选 manifest 路径。
    """
    safe_album = "".join(c for c in album_name if c.isalnum() or c in "_- ") \
        or "album"
    album_dir = os.path.join(out_dir, safe_album)
    os.makedirs(album_dir, exist_ok=True)

    # 复用 curate 已写的 curate_manifest.json（若存在）合并评分/keep
    score_by_path = {}
    curate_manifest = os.path.join(out_dir, "curate_manifest.json")
    if os.path.isfile(curate_manifest):
        try:
            with open(curate_manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
            for rec in data:
                score_by_path[os.path.abspath(rec["path"])] = rec
        except Exception:
            pass

    ranked = []
    for t in tracks:
        absp = os.path.abspath(t["path"])
        rec = score_by_path.get(absp, {})
        entry = dict(t)
        entry["score"] = rec.get("score", None)
        entry["kept"] = rec.get("kept", True)
        entry["metrics"] = rec.get("metrics", None)
        ranked.append(entry)
    # 按评分排序（无评分落末尾），赋予 rank
    ranked.sort(key=lambda e: (e["score"] is not None, e["score"] or 0.0),
                reverse=True)
    for i, e in enumerate(ranked):
        e["rank"] = i + 1

    manifest = {
        "album": album_name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "track_count": len(ranked),
        "keep_top": keep_top,
        "tracks": ranked,
        "contact_sheet": os.path.join(out_dir, "contact_sheet.png")
        if os.path.isfile(os.path.join(out_dir, "contact_sheet.png")) else None,
        "curate_manifest": curate_manifest
        if os.path.isfile(curate_manifest) else None,
    }
    man_path = os.path.join(album_dir, "album_manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[Album] manifest → {man_path}（{len(ranked)} 首，"
          f"kept={sum(1 for e in ranked if e['kept'])}）")


def main():
    parser = argparse.ArgumentParser(
        description="随机视频生成器 — 结合 draw/effects 和 pywallpaper/effects"
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（可复现）")
    parser.add_argument("--keywords", type=str, nargs="+", default=None,
                        help="关键词（按关键词选择效果）")
    parser.add_argument("--duration", type=int, default=None,
                        help="视频时长（秒）")
    parser.add_argument("--fps", type=int, default=None,
                        help="帧率")
    parser.add_argument("--width", type=int, default=None,
                        help="视频宽度")
    parser.add_argument("--height", type=int, default=None,
                        help="视频高度")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--preview", action="store_true",
                        help="预览模式（仅渲染前30帧）")
    parser.add_argument("--list-effects", action="store_true",
                        help="列出所有可用效果")
    parser.add_argument("--version", action="store_true",
                        help="显示版本信息")
    parser.add_argument("--audio", type=str, default=None,
                        help="音频文件或 'random'（从 sfx/sounds/ 随机选一个）")
    parser.add_argument("--audio-volume", type=float, default=1.0,
                        help="音频音量倍率（默认 1.0）")
    parser.add_argument("--cloud", action="store_true",
                        help="使用云端 API 生成（glitch-video-generator 模式）")
    parser.add_argument("--hw-enc", type=str, default="auto",
                        choices=["auto", "nvenc", "qsv", "cpu"],
                        help="硬件编码: auto(默认,检测)=nvenc>qsv>cpu")
    parser.add_argument("--codec", type=str, default="h264",
                        choices=["h264", "hevc", "av1"],
                        help="编码格式: h264(默认)/hevc/av1")
    parser.add_argument("--resolution", type=str, default=None,
                        choices=["4k", "1080p", "720p"],
                        help="分辨率预设(覆盖 --width/--height): 4k/1080p/720p")
    # ── P2-3 导出与分发增强 ──
    parser.add_argument("--container", type=str, default="mp4",
                        choices=["mp4", "webm", "gif"],
                        help="封装格式: mp4(默认)/webm(VP9,社媒)/gif(动态图)")
    parser.add_argument("--bit-depth", type=int, default=8,
                        choices=[8, 10],
                        help="位深: 8(默认)/10(hevc/vp9 走 yuv420p10le + bt2020 "
                             "HDR 元数据；h264 忽略)")
    parser.add_argument("--aspect", type=str, default=None,
                        help="社媒比例预设(覆盖 --width/--height): 9:16 / 1:1 / "
                             "16:9 / 4:3，按当前长边重算分辨率(竖屏=9:16)")
    # ── P2-1 扩散基础层 ──
    parser.add_argument("--diffusion", type=str, default=None,
                        choices=["auto", "video", "cloud", "local", "procedural"],
                        help="扩散基础层模式: auto(视频→云端→程序化兜底)/video/"
                             "cloud/local/procedural；启用后程序化原语在底片上叠加")
    parser.add_argument("--base-video", type=str, default=None,
                        help="--diffusion video/auto 时用作基础层的本地视频路径")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="--diffusion local 时本地扩散模型目录")
    parser.add_argument("--audio-hue-shift", action="store_true",
                        help="音频驱动色相随节拍偏移（音乐可视化色彩脉动）")
    parser.add_argument("--audio-balance", action="store_true",
                        help="音频驱动色调随频谱质心漂移（明亮度→暖/冷色温）")
    parser.add_argument("--audio-stems", action="store_true",
                        help="Demucs 分轨音频驱动（鼓→冲击/色相迸发，贝斯→对比脉动，"
                             "人声→色温，环境→辉光）；需 `pip install demucs`，"
                             "缺失则自动降级整曲混合驱动")
    # ── P1-1：GPU(GLSL) 渲染后端 ──
    parser.add_argument("--gl", action="store_true",
                        help="启用 GPU(GLSL) 渲染后端（glfw+PyOpenGL，需可用 OpenGL "
                             "上下文）；着色器产出基底帧，音频后处理/分轨调制全复用。"
                             "上下文不可用则自动降级 CPU 渲染")
    # ── P0-1：语义组合层 ──
    parser.add_argument("--prompt", type=str, default=None,
                        help="自然语言描述（如'深海发光 慢节奏 冷色调'），"
                             "映射到效果组合；需 --llm 且 DEEPSEEK_API_KEY 才走模型，"
                             "否则用本地语义引擎")
    parser.add_argument("--llm", action="store_true",
                        help="启用 LLM 后端（需 DEEPSEEK_API_KEY 环境变量；"
                             "缺失则自动回退本地引擎）")
    # ── P0-2：批量生成 + 自动筛选 ──
    parser.add_argument("--batch", type=int, default=1,
                        help="批量生成数量（默认 1）；与 --prompt 配合时按 seed 变化")
    parser.add_argument("--keep-top", type=int, default=None,
                        help="自动美学筛选后保留前 K 名（写 manifest + 标记 kept）")
    parser.add_argument("--contact-sheet", action="store_true",
                        help="生成连镜表 contact_sheet.png（批量时默认开启）")
    # ── P1-3：专辑批量模式 + manifest ──
    parser.add_argument("--album", type=str, default=None,
                        help="专辑模式：以统一主题/语义生成一批曲目（默认 8 首，"
                             "可用 --batch 指定数量），写 album_manifest.json "
                             "（含每首的元数据/美学评分/keep 标记），并默认生成连镜表")
    args = parser.parse_args()

    # ── 云端生成模式 ──
    if args.cloud:
        print("[Main] 启用云端生成模式...")
        # 构建 cloud_generator 的命令行参数
        import sys as _sys
        cloud_args = []
        if args.seed is not None:
            cloud_args.append(str(args.seed))
        if args.output:
            cloud_args.extend(["--output", args.output])
        if args.audio:
            cloud_args.extend(["--audio", args.audio])

        # 保存原 argv，调用 cloud_generator
        _orig_argv = _sys.argv
        _sys.argv = [_orig_argv[0].replace("video_generator.py", "cloud_generator.py")] + cloud_args
        try:
            cloud_generator.main()
        finally:
            _sys.argv = _orig_argv
        return

    if args.version:
        print(f"随机视频生成器 v{VERSION}")
        return

    # 加载效果库
    print("加载效果库...")
    registry = get_default_registry()

    if args.list_effects:
        print("\n所有效果：")
        for effect in registry.all()[:50]:  # 只显示前50个
            print(f"  {effect.summary()}")
        print(f"\n... 共 {len(registry.all())} 个效果")
        return

    # 创建组合器
    composer = RandomComposer(registry)
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    batch = max(1, args.batch)
    # P1-3：专辑模式默认 8 首（可被 --batch 覆盖），且默认生成连镜表
    if args.album is not None:
        if args.batch == 1:  # 用户未显式指定数量时才用专辑默认
            batch = 8
        album_mode = True
    else:
        album_mode = False
    base_seed = args.seed
    seeds = ([base_seed + i for i in range(batch)]
             if base_seed is not None else [None] * batch)

    generated_paths = []
    album_tracks = []  # P1-3：专辑每首元数据
    for i, sd in enumerate(seeds):
        # 1) 组合（语义 / 关键词 / 随机）
        if args.prompt:
            print(f"\n[Prompt] 语义组合（第 {i+1}/{batch} 次）: {args.prompt}")
            comp, meta = llm_composer.compose_from_prompt(
                args.prompt, registry, seed=sd, use_llm=args.llm)
            print(f"  引擎={meta['engine']}  回退={meta.get('fallback')}")
            if meta.get("issues"):
                print(f"  Critic: {meta['issues']}")
        elif args.keywords:
            print(f"\n根据关键词生成：{args.keywords}")
            comp = composer.compose_from_keywords(args.keywords, seed=sd)
        else:
            print(f"\n随机生成视频组合 (seed={sd})...")
            comp = composer.compose(seed=sd)

        # 2) 覆盖命令行参数
        if args.resolution:
            _RES = {"4k": (3840, 2160), "1080p": (1920, 1080), "720p": (1280, 720)}
            comp.width, comp.height = _RES[args.resolution]
        if args.duration:
            comp.duration = args.duration
        if args.fps:
            comp.fps = args.fps
        if args.width:
            comp.width = args.width
        if args.height:
            comp.height = args.height

        # 3) 输出路径（批量时加序号，避免覆盖）
        # P2-3：扩展名跟随 --container（mp4/webm/gif）
        _ext = {"mp4": ".mp4", "webm": ".webm", "gif": ".gif"}.get(
            (args.container or "mp4").lower(), ".mp4")
        if args.output and batch == 1:
            if os.path.dirname(args.output) == "":
                _base = args.output
                if not _base.lower().endswith(_ext):
                    _base = os.path.splitext(_base)[0] + _ext
                output_path = os.path.join(OUT_DIR, _base)
            else:
                output_path = args.output
                if not output_path.lower().endswith(_ext):
                    output_path = os.path.splitext(output_path)[0] + _ext
        else:
            desc = (comp.description or "video")[:40]
            safe = "".join(c for c in desc if c.isalnum() or c in "_-") or "video"
            stamp = int(time.time()) if sd is None else sd
            output_path = os.path.join(OUT_DIR, f"{safe}_{i+1}_{stamp}{_ext}")

        # 4) 生成视频（ffmpeg NVENC / CPU fallback；音频在内部混入）
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # P2-1：构造扩散基础层（可选；失败优雅降级为 None=程序化主题）
        _diff_base = None
        if args.diffusion:
            try:
                import diffusion_base as _db
                _diff_base = _db.make_diffusion_base(
                    args.diffusion, comp.width, comp.height, comp.fps,
                    prompt=args.prompt, video_path=args.base_video,
                    model_path=args.model_dir, seed=(sd or 1),
                    duration=comp.duration)
                print(f"[Diffusion] 基础层已启用: {_diff_base.source}")
            except Exception as e:  # noqa: BLE001
                print(f"[Diffusion] 不可用，降级程序化主题: {e}")
                _diff_base = None

        generator = VideoGenerator(
            comp, output_path, prefer_hw=args.hw_enc,
            codec=args.codec, audio_path=args.audio,
            audio_hue_shift=args.audio_hue_shift,
            audio_balance=args.audio_balance,
            audio_volume=args.audio_volume,
            use_audio_stems=args.audio_stems,
            use_gl=args.gl,
            container=args.container,
            bit_depth=args.bit_depth,
            aspect=args.aspect,
            diffusion_base=_diff_base,
        )
        generator.generate(preview=args.preview)
        generated_paths.append(output_path)
        # P1-3：收集专辑元数据
        if album_mode:
            album_tracks.append({
                "index": i + 1,
                "title": f"Track {i + 1:02d} — {desc}" if desc else f"Track {i + 1:02d}",
                "path": output_path,
                "theme": comp.theme,
                "description": comp.description,
                "width": comp.width,
                "height": comp.height,
                "fps": comp.fps,
                "duration": comp.duration,
                "seed": sd,
                "used_gl": args.gl,
            })

    # 5) 自动美学筛选（P0-2）
    if args.keep_top is not None or args.contact_sheet or album_mode:
        os.makedirs(OUT_DIR, exist_ok=True)
        curator.curate(
            generated_paths, out_dir=OUT_DIR,
            keep_top=args.keep_top,
            contact_sheet=(args.contact_sheet or batch > 1 or album_mode),
        )

    # 6) P1-3：专辑 manifest（含美学评分 + keep 标记，合并 curate 结果）
    if album_mode:
        _write_album_manifest(args.album, album_tracks, OUT_DIR,
                              generated_paths, args.keep_top)

    if batch == 1:
        print(f"\n[完成] 完成！最终视频：{generated_paths[0]}")
    else:
        print(f"\n[完成] 共生成 {len(generated_paths)} 个视频：")
        for p in generated_paths:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
