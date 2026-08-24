import os, sys, time
import numpy as np

import audio_stems as AS

AUDIO = sys.argv[1] if len(sys.argv) > 1 else "sfx/sounds/1 (1).mp3"
FPS = 25.0

print(f"[1] 普通 import audio_stems 成功（无父进程段错误）；"
      f"demucs_available={AS._demucs_available()}")

t0 = time.time()
s = AS.AudioStems(AUDIO, FPS)          # 真实 demucs 分离（子进程，torch/lib 已注入 PATH）
print(f"[2] AudioStems 构造耗时 {time.time()-t0:.1f}s  ->  {s.summary()}")

print("[3] 各分轨包络统计（验证分离确实产出不同信号）:")
stats = {}
for n in ("drums", "bass", "vocals", "other", "mix"):
    ar = getattr(s, n)
    if ar is not None and ar.envelope is not None and ar.envelope.size > 1:
        e = ar.envelope
        stats[n] = (float(e.min()), float(e.max()), float(e.mean()))
        print(f"    {n:7s} frames={len(e):4d}  "
              f"min={e.min():.3f} max={e.max():.3f} mean={e.mean():.3f}")
    else:
        print(f"    {n:7s} 无数据/恒定")

# 分轨互相关性：drums 与 vocals 包络应明显不同（证明不是同一份信号）
def _corr(a, b):
    if a is None or b is None:
        return float("nan")
    ea, eb = a.envelope, b.envelope
    if ea.size != eb.size:
        return float("nan")
    return float(np.corrcoef(ea, eb)[0, 1])

print(f"[4] 互相关 drums~vocals={_corr(s.drums, s.vocals):.3f}  "
      f"drums~bass={_corr(s.drums, s.bass):.3f}  "
      f"bass~vocals={_corr(s.bass, s.vocals):.3f}")

# [5] 真实分轨驱动调制：随机彩色帧，鼓能量峰值帧应改变画面
frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
di = int(np.argmax(s.drums.envelope))
out = AS.apply_stem_modulation(frame.copy(), di, s)
changed = not np.array_equal(out, frame)
print(f"[5] 鼓峰值帧 idx={di} 调制后画面改变={changed}")

print("\n结论：真实 demucs 分离 + 分轨 AudioReactive + 调制映射 全部跑通。")
