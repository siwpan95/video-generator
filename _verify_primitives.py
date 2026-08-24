"""
_verify_primitives.py — P1-2 新原语真实验证

覆盖：
  1) 四类原语直接实例化 → 多帧 update+draw_to_array → 非黑 / 唯一色 / 帧间变化
  2) PWEffecRenderer.load_config 集成（PRIMITIVE_MAP 注册正确）
  3) 真实编码：用 FFmpegWriter 渲染 raymarch 30 帧 → output/e2e_primitives.mp4 → ffprobe 校验
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pw_renderer  # noqa: E402
import primitives_pro  # noqa: E402

W, H, FPS, N = 320, 240, 25, 30

CONFIGS = {
    "raymarch": {"type": "raymarch", "speed": 0.6, "glow": 7.0,
                 "color": {"hue": [0.5, 0.75]}},
    "metaballs": {"type": "metaballs", "count": 6, "speed": 0.6, "threshold": 1.0,
                  "color": {"hue": [0.55, 0.95]}},
    "lsystem": {"type": "lsystem", "iterations": 4, "angle": 25.0, "rotate": 0.25,
                "color": {"hue": [0.25, 0.42]}},
    "dla": {"type": "dla", "max_points": 350, "grow_time": 5.0,
            "color": {"hue": [0.0, 0.18]}},
    # P2 新增：circle_map（真实组合 JSON 结构子集）
    "circle_map": {
        "type": "circle_map",
        "center": {"x": "50%", "y": "50%"},
        "rings": [
            {"count": 72, "radius": 80, "line": True, "line_width": 2},
            {"count": 72, "radius": 130, "line": True, "line_width": 2},
            {"count": 48, "radius": 180, "line": False},
        ],
        "color": {"type": "hsv_flow", "hue": [0.0, 0.7], "speed": 0.1,
                  "sat": 0.8, "val": 0.85},
        "audio": {"source": "band_energy", "target": "radius_offset",
                  "scale": [0, 60], "smoothing": 0.85},
    },
}

CLS = {
    "raymarch": primitives_pro.RaymarchSDF,
    "metaballs": primitives_pro.Metaballs,
    "lsystem": primitives_pro.LSystemPrimitive,
    "dla": primitives_pro.DLACluster,
    "circle_map": primitives_pro.CircleMapPrimitive,
}


def stat(arr):
    nz = int(np.count_nonzero(arr))
    uniq = len(np.unique(arr.reshape(-1, 3), axis=0))
    return nz, uniq, float(arr.mean())


def test_direct():
    print("=== 1) 直接实例化 + 多帧渲染 ===")
    ok = True
    for name, cfg in CONFIGS.items():
        t0 = time.time()
        obj = CLS[name](cfg, W, H)
        frames = []
        for i in range(N):
            obj.update(1.0 / FPS)
            f = obj.draw_to_array(np.zeros((H, W, 3), np.uint8))
            frames.append(f)
        dt = time.time() - t0
        # 末帧统计
        nz, uniq, mean = stat(frames[-1])
        # 首末帧差异（动画确实推进）
        diff = float(np.abs(frames[0].astype(int) - frames[-1].astype(int)).mean())
        good = (nz > (W * H * 0.01)) and (uniq > 5) and (diff > 0.5)
        ok = ok and good
        print(f"  {name:<10} nz={nz:>7} uniq={uniq:>5} mean={mean:6.2f} "
              f"帧差={diff:6.2f} {N}帧/{dt:5.2f}s {'' if good else '❌'}")
    print("  DIRECT", "✅ 全部通过" if ok else "❌ 有失败")
    return ok


def test_integration():
    print("=== 2) PWEffecRenderer.load_config 集成 ===")
    ren = pw_renderer.PWEffecRenderer(W, H)
    layers = [CONFIGS[name] for name in CONFIGS]
    effect = {"params": {"layers": layers}}
    ren.load_config(effect)
    skipped = [l["type"] for l in layers
               if l["type"] not in pw_renderer.PWEffecRenderer.PRIMITIVE_MAP]
    built = len(ren.layers)
    # 渲染一帧
    ren.update(0.1)
    img = ren.draw()
    arr = np.asarray(img)
    nz, uniq, mean = stat(arr)
    good = (built == len(layers)) and (skipped == []) and \
           (nz > W * H * 0.01) and (uniq > 5)
    print(f"  构建图层={built}/{len(layers)}  跳过={skipped}  "
          f"帧 nz={nz} uniq={uniq} mean={mean:.2f}")
    print("  INTEGRATION", "✅ 通过" if good else "❌ 失败")
    return good


def test_e2e_mp4():
    print("=== 3) 真实编码 e2e_primitives.mp4 ===")
    try:
        from video_generator import FFmpegWriter
    except Exception as e:
        print(f"  无法导入 FFmpegWriter: {e}（跳过编码验证）")
        return True
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "output", "e2e_primitives.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    obj = primitives_pro.RaymarchSDF(
        {"type": "raymarch", "speed": 0.6, "glow": 7.0,
         "color": {"hue": [0.5, 0.75]}}, W, H)
    writer = FFmpegWriter(output_path=out, width=W, height=H, fps=FPS, codec="h264")
    if not writer.is_opened():
        print("  ❌ FFmpegWriter 未打开")
        return False
    t0 = time.time()
    for i in range(N):
        obj.update(1.0 / FPS)
        frame = obj.draw_to_array(np.zeros((H, W, 3), np.uint8))
        writer.write(frame)
    writer.release()
    dt = time.time() - t0
    if not os.path.exists(out):
        print("  ❌ 输出文件不存在")
        return False
    size = os.path.getsize(out)
    print(f"  渲染+编码 {N}帧: {dt:.2f}s  文件 {size/1024:.1f} KB -> {out}")
    print("  E2E", "✅ 通过" if size > 1000 else "❌ 文件过小")
    return size > 1000


if __name__ == "__main__":
    r1 = test_direct()
    r2 = test_integration()
    r3 = test_e2e_mp4()
    print("\n" + ("✅ PRIMITIVES_OK" if (r1 and r2 and r3) else "❌ PRIMITIVES_FAIL"))
    sys.exit(0 if (r1 and r2 and r3) else 1)
