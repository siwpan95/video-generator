import time
import numpy as np

import gl_renderer as GL

print("HAS_GL", GL._HAS_GL)
assert GL._HAS_GL, "glfw/PyOpenGL 未安装"

W = H = 128
gl = GL.GLRenderer(W, H, 25.0, family="noise", palette="neon", seed=2.0)

ok = True
for fam in GL.GLRenderer.FAMILIES:
    gl.set_family(fam)
    fr = gl.render_frame(1.0, 0, 0.6)
    assert fr.shape == (H, W, 3) and fr.dtype == np.uint8, fr.shape
    uniq = len(np.unique(fr.reshape(-1, 3), axis=0))
    mean = float(fr.mean())
    print(f"  {fam:10s} shape={fr.shape} 唯一色={uniq:4d} mean={mean:.1f}")
    if uniq <= 5:
        print(f"    ✗ {fam} 图案过于单一"); ok = False
    if mean <= 1.0:
        print(f"    ✗ {fam} 几乎全黑"); ok = False
gl.close()

# 主题 → (家族, 调色板) 映射
print("theme map:",
      {t: (GL.family_for_theme(t), GL.palette_for_theme(t))
       for t in ["empty", "starfield", "gradient", "aurora"]})

# 速度：单上下文 60 帧计时（320x240）
gl = GL.GLRenderer(320, 240, 25.0, family="flow", palette="neon", seed=1.0)
t0 = time.time()
N = 60
for i in range(N):
    gl.render_frame(i / 25.0, i, 0.5)
dt = time.time() - t0
print(f"GL 渲染 {N}帧@320x240: {dt:.3f}s -> {N/dt:.1f} fps (仅渲染，不含编码)")
gl.close()

print("✅ GL_OK" if ok else "❌ GL 验证未通过")
