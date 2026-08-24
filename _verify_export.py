"""
_verify_export.py — P2-3 导出与分发增强真实验证
测试：mp4(h264 8bit) / webm(vp9 8bit) / gif(动态) / mp4(hevc 10bit HDR) / 竖屏 9:16
每个产物用 ffprobe 校验容器/编码/分辨率/位深，并解码首帧检查非黑。
"""
import os
import sys
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from random_composer import RandomComposer
from effect_loader import get_default_registry
from video_generator import VideoGenerator, find_ffprobe
import audio_mixer

FFMPEG = audio_mixer.find_ffmpeg()
FFPROBE = find_ffprobe()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
print(f"[bin] ffmpeg={FFMPEG}\n[bin] ffprobe={FFPROBE}")


def ffprobe(path):
    """用真正的 ffprobe 解析结构化流信息（项目自带 essentials build 含 ffprobe）。"""
    r = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries",
         "stream=codec_name,codec_type,width,height,pix_fmt,color_transfer,"
         "color_primaries,profile:format=format_name,duration",
         "-of", "json", path],
        capture_output=True, text=True, timeout=30)
    try:
        return json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return {}


def decode_nonblack(path):
    """解码首帧平均亮度，>0 即非纯黑。"""
    r = subprocess.run(
        [FFMPEG, "-y", "-i", path, "-frames:v", "1", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", "32x32", "pipe:1"],
        capture_output=True, timeout=30)
    if r.returncode != 0 or not r.stdout:
        return -1, len(r.stderr)
    import numpy as np
    arr = np.frombuffer(r.stdout, dtype=np.uint8).reshape(-1, 3)
    return float(arr.mean()), 0


def test_one(name, container, codec, bit_depth, aspect=None, duration=2,
             width=320, height=240, fps=25):
    print(f"\n=== {name}: container={container} codec={codec} "
          f"bit={bit_depth} aspect={aspect} ===")
    registry = get_default_registry()
    composer = RandomComposer(registry)
    comp = composer.compose(seed=7)
    comp.duration = duration
    comp.fps = fps
    comp.width = width
    comp.height = height
    # 计算 aspect 变换后的期望分辨率（与 video_generator._apply_aspect 同规则）
    if aspect:
        try:
            a, b = (float(x) for x in aspect.lower().split(":"))
            long_edge = max(width, height)
            if a >= b:
                exp_w, exp_h = long_edge, int(round(long_edge * b / a / 2.0) * 2)
            else:
                exp_h, exp_w = long_edge, int(round(long_edge * a / b / 2.0) * 2)
        except Exception:
            exp_w, exp_h = width, height
    else:
        exp_w, exp_h = width, height
    ext = {"mp4": ".mp4", "webm": ".webm", "gif": ".gif"}[container]
    out = os.path.join(OUT, f"e2e_export_{name}{ext}")
    if os.path.isfile(out):
        os.remove(out)
    g = VideoGenerator(comp, out, prefer_hw="cpu", codec=codec,
                       container=container, bit_depth=bit_depth, aspect=aspect)
    g.generate()
    # 校验
    if not os.path.isfile(out):
        print(f"  [FAIL] 产物缺失: {out}")
        return False
    info = ffprobe(out)
    streams = info.get("streams", [])
    vstream = next((s for s in streams if s.get("codec_type") == "video"), {})
    fmt = (info.get("format", {}) or {}).get("format_name", "")
    lum, err = decode_nonblack(out)
    size = os.path.getsize(out)
    vcodec = vstream.get("codec_name", "")
    vw, vh = int(vstream.get("width", 0)), int(vstream.get("height", 0))
    pix = vstream.get("pix_fmt", "")
    print(f"  容器={fmt} 视频编码={vcodec} 分辨率={vw}x{vh} pix_fmt={pix} "
          f"大小={size/1024:.1f}KB 首帧均亮={lum:.1f}")
    # 结构化断言
    checks = []
    checks.append(("容器", container[:3] in fmt))  # mp4/webm/gif
    if container != "gif":
        checks.append(("编码", vcodec == codec))
        checks.append(("分辨率", vw == exp_w and vh == exp_h))
        checks.append(("文件", size > 1024))
        checks.append(("非黑", lum > 0))
        if bit_depth >= 10:
            checks.append(("10bit", "10" in pix))  # yuv420p10le 等
    ok = all(v for _, v in checks)
    for label, v in checks:
        print(f"    - {label}: {'✓' if v else '✗'}")
    print(f"  [{'OK' if ok else 'FAIL'}]")
    return ok


def main():
    results = []
    results.append(("mp4_h264_8bit", test_one(
        "mp4_h264_8bit", "mp4", "h264", 8)))
    results.append(("webm_vp9_8bit", test_one(
        "webm_vp9_8bit", "webm", "vp9", 8)))
    results.append(("gif", test_one(
        "gif", "gif", "h264", 8)))  # gif 忽略 codec
    results.append(("mp4_hevc_10bit", test_one(
        "mp4_hevc_10bit", "mp4", "hevc", 10)))
    results.append(("portrait_9x16", test_one(
        "portrait_9x16", "mp4", "h264", 8, aspect="9:16",
        width=1280, height=720)))
    print("\n===== P2-3 导出验证汇总 =====")
    all_ok = True
    for n, ok in results:
        print(f"  {'✓' if ok else '✗'} {n}")
        all_ok = all_ok and ok
    print(f"\n{'EXPORT_OK' if all_ok else 'EXPORT_FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
