"""
gen_sample.py — 端到端样例视频生成器（全面增强验证用）

演示本次增强的全部新能力：
  - draw 层走 scipy/Numba 卷积加速演化（BZReaction）
  - pw 层使用新增原语 wireframe_3d（旋转 3D 线框）
  - 音频驱动：色相随节拍偏移（--audio-hue-shift 等价逻辑）
  - 音频合成：节拍 wav 经 VideoGenerator 直接混入 mp4（generate() 内部 mux）
  - 编码器：h264（libx264 / NVENC 自动），glow 后处理

用法：
  # 指定 ffmpeg 二进制（不指定时自动用本项目 bundled 的 essentials build；
  # 也可用本机任意 ffmpeg，如雷电云 LD-Yun 自带的 F:/ldyun/ffmpeg.exe）
  set FFMPEG_BIN=F:/ldyun/ffmpeg.exe
  python gen_sample.py
"""
import os
import sys
import numpy as np

# 确保能 import 本项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy.io import wavfile

from effect_loader import get_default_registry
from effect_model import EffectSource
from random_composer import VideoComposition, LayerConfig
from video_generator import VideoGenerator


def build_beat_wav(path: str, duration: float = 6.0, sr: int = 22050):
    """生成一段带稳定节拍脉冲的 wav（用于音频驱动色相偏移演示）"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = 0.25 * np.sin(2 * np.pi * 220.0 * t)
    for beat in np.arange(0.5, duration, 0.5):
        idx = int(beat * sr)
        L = int(0.06 * sr)
        if idx + L <= len(sig):
            sig[idx:idx + L] += 0.9 * np.hanning(L)
    sig = sig / np.max(np.abs(sig))
    wavfile.write(path, sr, (sig * 32767).astype(np.int16))
    return path


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_mp4 = os.path.join(out_dir, "sample_bz_wireframe.mp4")
    wav_path = os.path.join(out_dir, "_sample_beat.wav")

    # 1) 构造组合：BZ draw 层 + wireframe_3d pw 层
    registry = get_default_registry()
    draw_eff = None
    pw_eff = None
    for e in registry.all():
        if e.source == EffectSource.DRAW and "BZReaction" in e.name and draw_eff is None:
            draw_eff = e
        if (e.source == EffectSource.PYWALLPAPER
                and any(l.get("type") == "wireframe_3d" for l in e.pw_layers)
                and pw_eff is None):
            pw_eff = e
    if draw_eff is None or pw_eff is None:
        raise SystemExit(f"未找到所需效果: draw={draw_eff} pw={pw_eff}")

    comp = VideoComposition()
    comp.width = 512
    comp.height = 512
    comp.fps = 30
    comp.duration = 6
    comp.theme = "gradient"
    comp.global_postprocess = "glow"

    draw_layer = LayerConfig(draw_eff, 0.0, None)
    draw_layer.opacity = 0.9
    draw_layer.blend_mode = "overlay"
    draw_layer.params = {"palette": "Neon", "grid_w": 64, "grid_h": 64, "steps_per_frame": 3}
    comp.add_layer(draw_layer)

    pw_layer = LayerConfig(pw_eff, 0.0, None)
    pw_layer.opacity = 0.85
    pw_layer.blend_mode = "add"
    comp.add_layer(pw_layer)

    print(f"[Sample] draw 层: {draw_eff.name}")
    print(f"[Sample] pw 层: {pw_eff.name} (wireframe_3d)")
    print(comp.summary())

    # 2) 音频：节拍 wav（既驱动色相偏移，又会被 VideoGenerator 混入 mp4 音轨）
    build_beat_wav(wav_path, duration=comp.duration)
    print(f"[Sample] 音频: {wav_path}（将混入 mp4 音轨）")

    # 3) ffmpeg 二进制：优先环境变量 FFMPEG_BIN
    ffmpeg_bin = os.environ.get("FFMPEG_BIN", "")
    if ffmpeg_bin:
        print(f"[Sample] 使用 FFMPEG_BIN={ffmpeg_bin}")
    else:
        print("[Sample] 未设置 FFMPEG_BIN，将尝试系统 ffmpeg")

    # 4) 生成（CPU 编码保证可移植；codec=h264）
    gen = VideoGenerator(
        comp, out_mp4,
        prefer_hw="cpu",
        codec="h264",
        audio_path=wav_path,
        audio_hue_shift=True,
    )
    gen.generate(preview=False)

    # 5) 报告
    if os.path.isfile(out_mp4) and os.path.getsize(out_mp4) > 1024:
        size_mb = os.path.getsize(out_mp4) / 1024 / 1024
        print(f"\n[Sample] ✓ 生成成功: {out_mp4} ({size_mb:.2f} MB)")
    else:
        print(f"\n[Sample] ✗ 生成失败，检查 ffmpeg 是否可用")

    # 清理临时 wav
    try:
        os.remove(wav_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
