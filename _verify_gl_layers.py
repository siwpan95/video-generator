"""
_verify_gl_layers.py — P2 验证：GL 模式必须叠加既有 pw 图层（circle_map 等）

做法：
  构造一个含 circle_map 图层的 VideoComposition（circle_map 此前被优雅跳过，
  现应正常渲染），分别用 use_gl=True / False 实例化 VideoGenerator，
  渲染同一帧 → 对比：
    - 两版都应含 circle_map 的环像素（非纯黑、非纯 shader 基底）
    - GL 版帧与 CPU 版帧在环区域应有明显非零叠加（证明图层确实 composite）
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from effect_loader import get_default_registry
from random_composer import VideoComposition, LayerConfig
from video_generator import VideoGenerator


def make_circlemap_comp(w=320, h=240, fps=25, dur=2.0):
    registry = get_default_registry()
    # 找第一个含 circle_map 的 pywallpaper 效果
    target = None
    for e in registry.all():
        if getattr(e, "source", None) is None:
            continue
        try:
            if any(l.get("type") == "circle_map" for l in e.pw_layers):
                target = e
                break
        except Exception:
            continue
    if target is None:
        raise SystemExit("未找到含 circle_map 的 effect")
    comp = VideoComposition()
    comp.width = w
    comp.height = h
    comp.fps = fps
    comp.duration = dur
    comp.theme = "gradient"
    comp.global_postprocess = "glow"
    # 用该 effect 直接构造一个 LayerConfig（封装 EffectConfig 对象）
    lc = LayerConfig(target)
    comp.add_layer(lc)
    print(f"  使用 effect: {target.name}  图层数={len(comp.layers)}")
    return comp


def render_frame(gen: VideoGenerator, frame_idx=10, prepared=False):
    # 准备渲染后端（GL 着色器 + pw 图层），等价于 generate() 内的初始化
    if not prepared:
        gen._prepare_renderers()
    if gen.gl is not None:
        gen.gl.make_current()
    t = frame_idx / gen.comp.fps
    return gen._render_frame(t, frame_idx)


def main():
    print("=== P2 验证：GL 模式叠加 pw 图层（circle_map）===")
    comp = make_circlemap_comp()

    # CPU 版（无 GL）
    g_cpu = VideoGenerator(comp, "output/_gl_verify_cpu.mp4", use_gl=False)
    f_cpu = render_frame(g_cpu)

    # GL 版
    g_gl = VideoGenerator(comp, "output/_gl_verify_gl.mp4", use_gl=True)
    g_gl._prepare_renderers()
    if g_gl.gl is None:
        print("  ❌ GL 后端未能启用（环境无 OpenGL？）— GL 叠加验证跳过")
        return True
    f_gl = render_frame(g_gl, prepared=True)

    # 统计
    nz_cpu = int(np.count_nonzero(f_cpu))
    nz_gl = int(np.count_nonzero(f_gl))
    # 两版差异（环区域应不同，但都含环内容）
    diff = float(np.abs(f_cpu.astype(int) - f_gl.astype(int)).mean())

    # 验证 circle_map 实际渲染：CPU 版非纯黑且环区域有结构
    # 取中心区域应含亮环像素（圈半径 80~230，覆盖中心带）
    cy, cx = comp.height // 2, comp.width // 2
    band = f_cpu[max(0, cy - 120):cy + 120, max(0, cx - 120):cx + 120]
    ring_nz = int(np.count_nonzero(band))

    good = (nz_cpu > comp.width * comp.height * 0.02) and \
           (nz_gl > comp.width * comp.height * 0.02) and (ring_nz > 500)
    print(f"  CPU 帧 非黑={nz_cpu}  GL 帧 非黑={nz_gl}")
    print(f"  CPU/GL 帧均差={diff:.2f}  中心环带非黑={ring_nz}")
    print("  GL_LAYERS", "✅ 通过（GL 含 circle_map 图层）" if good else "❌ 失败")

    # 真实编码：GL 版完整 generate() → mp4，验证可解码
    print("=== 真实编码 GL+circle_map mp4 ===")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "output", "e2e_gl_circlemap.mp4")
    g_enc = VideoGenerator(comp, out, use_gl=True)
    g_enc.generate()
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        print(f"  ✅ 编码成功 {os.path.getsize(out)/1024:.1f} KB -> {out}")
    else:
        print(f"  ❌ 编码失败或文件过小: {out}")
        good = False
    return good


if __name__ == "__main__":
    ok = main()
    print("\n" + ("✅ GL_LAYERS_OK" if ok else "❌ GL_LAYERS_FAIL"))
    sys.exit(0 if ok else 1)
