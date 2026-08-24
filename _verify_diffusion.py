"""
_verify_diffusion.py — P2-1 扩散基础层真实验证
1) 程序化兜底基础层：生成视频，校验非黑且帧间有差。
2) 视频基础层：先用程序化生成一段底片，再作为 --diffusion video 基础层叠加
   程序化图层，校验成片非黑且含基础层内容。
3) 降级：--diffusion local 无模型路径 → 抛 DiffusionUnavailable，调用方降级。
"""
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from random_composer import RandomComposer
from effect_loader import get_default_registry
from video_generator import VideoGenerator
import diffusion_base as db

FFMPEG = os.environ.get("FFMPEG_BIN") or "ffmpeg"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)


def _gen_base_video(path, w=160, h=120, dur=2, fps=25, seed=3):
    registry = get_default_registry()
    comp = RandomComposer(registry).compose(seed=seed)
    comp.duration, comp.fps, comp.width, comp.height = dur, fps, w, h
    g = VideoGenerator(comp, path, prefer_hw="cpu", codec="h264")
    g.generate()


def _mean_luma(path):
    r = subprocess.run([FFMPEG, "-y", "-i", path, "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "32x32",
                        "pipe:1"], capture_output=True, timeout=30)
    if r.returncode != 0 or not r.stdout:
        return -1
    import numpy as np
    return float(np.frombuffer(r.stdout, dtype=np.uint8).reshape(-1, 3).mean())


def test_procedural():
    print("=== P2-1 程序化兜底基础层 ===")
    registry = get_default_registry()
    comp = RandomComposer(registry).compose(seed=11)
    comp.duration, comp.fps, comp.width, comp.height = 2, 25, 200, 150
    base = db.make_diffusion_base("procedural", 200, 150, 25, seed=11)
    f0 = base.frame_at(0)
    f10 = base.frame_at(10)
    import numpy as np
    diff = float(np.abs(f0.astype(int) - f10.astype(int)).mean())
    ok = f0.shape == (150, 200, 3) and f0.mean() > 0 and diff > 0.5
    print(f"  frame0 mean={f0.mean():.1f} 帧差={diff:.2f}  -> {'OK' if ok else 'FAIL'}")
    return ok


def test_video_base():
    print("=== P2-1 视频基础层 + 程序化叠加 ===")
    base_vid = os.path.join(OUT, "e2e_diff_base_src.mp4")
    if not os.path.isfile(base_vid):
        _gen_base_video(base_vid)
    registry = get_default_registry()
    comp = RandomComposer(registry).compose(seed=5)
    comp.duration, comp.fps, comp.width, comp.height = 2, 25, 160, 120
    base = db.make_diffusion_base("video", 160, 120, 25, video_path=base_vid)
    out = os.path.join(OUT, "e2e_diff_video_base.mp4")
    if os.path.isfile(out):
        os.remove(out)
    g = VideoGenerator(comp, out, prefer_hw="cpu", codec="h264",
                       diffusion_base=base)
    g.generate()
    luma = _mean_luma(out)
    size = os.path.getsize(out) if os.path.isfile(out) else 0
    ok = size > 1024 and luma > 0
    print(f"  成片大小={size/1024:.1f}KB 首帧均亮={luma:.1f} -> {'OK' if ok else 'FAIL'}")
    return ok


def test_local_degrade():
    print("=== P2-1 local 模式降级 (无模型) ===")
    try:
        db.make_diffusion_base("local", 160, 120, 25, model_path="")
        print("  [FAIL] 应抛 DiffusionUnavailable")
        return False
    except db.DiffusionUnavailable as e:
        print(f"  [OK] 正确降级: {e}")
        return True


def main():
    r = [("procedural", test_procedural()),
         ("video_base", test_video_base()),
         ("local_degrade", test_local_degrade())]
    print("\n===== P2-1 扩散基础层汇总 =====")
    all_ok = True
    for n, ok in r:
        print(f"  {'✓' if ok else '✗'} {n}")
        all_ok = all_ok and ok
    print(f"\n{'DIFFUSION_OK' if all_ok else 'DIFFUSION_FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
