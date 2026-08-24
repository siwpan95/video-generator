"""回合3 真实验证脚本（F:/Python314，numpy/scipy/librosa 可用）。"""
import sys, os, time, wave, struct, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=== 1) palettes_extra.turbo ===")
import palettes_extra as pe
assert "turbo" in pe.PERCEPTUAL_NAMES, pe.PERCEPTUAL_NAMES
assert pe.is_perceptual("Turbo")
lut = pe.build_lut("turbo")
assert lut.shape == (256, 3) and lut.dtype == np.uint8
assert lut.min() >= 0 and lut.max() <= 255
img = pe.apply_perceptual(np.linspace(0, 1, 256).reshape(1, -1), "turbo")
assert img.shape == (1, 256, 3)
print(f"  turbo LUT ok  range=({lut.min()},{lut.max()})  "
      f"endpoints R0={lut[0,0]} R255={lut[255,0]}")

print("=== 2) noise_field: simplex / worley / warp ===")
import noise_field as nf
sn = nf.SimplexNoise(7)
sf = sn.render_field(64, 64, t=1.0)
assert sf.shape == (64, 64) and sf.min() >= 0 and sf.max() <= 1.0, (sf.min(), sf.max())
# 时间连续
sf2 = sn.render_field(64, 64, t=1.05)
assert np.sqrt(((sf - sf2) ** 2).mean()) < 1.0
wn = nf.WorleyNoise(7, cell=32.0)
wf = wn.render_field(64, 64, scale=8.0)
assert wf.shape == (64, 64) and wf.min() >= 0 and wf.max() <= 1.0, (wf.min(), wf.max())
wf2 = wn.render_field(64, 64, scale=8.0, t=0.05)
assert np.sqrt(((wf - wf2) ** 2).mean()) < 1.0
pn = nf.PerlinNoise(7)
t0 = time.perf_counter()
warp = nf.warp_field(pn, 128, 128, t=1.0, scale=8.0, warp=4.0)
dt = time.perf_counter() - t0
assert warp.shape == (128, 128) and warp.min() >= 0 and warp.max() <= 1.0
# warp 与纯 fbm 应不同（层次更丰富）
plain = pn.render_field(128, 128, t=1.0, scale=8.0)
assert not np.allclose(warp, plain), "warp 应与 fbm 不同"
print(f"  simplex ok[{sf.min():.2f},{sf.max():.2f}]  worley ok  "
      f"warp 128^2 {dt*1000:.1f}ms  warp!=plain:{not np.allclose(warp,plain)}")

print("=== 3) pw_renderer: SimpleFlowField (curl + angle) ===")
import pw_renderer as pw
for mode in ("curl", "angle"):
    cfg = {"params": {"layers": [
        {"type": "flow_field", "count": 400, "mode": mode,
         "scale": 80.0, "speed": 2.0, "trail": 0.93,
         "color": {"type": "rainbow", "speed": 0.03}}]}}
    r = pw.PWEffecRenderer(256, 256)
    r.load_config(cfg)
    assert len(r.layers) == 1, len(r.layers)
    assert isinstance(r.layers[0], pw.SimpleFlowField)
    canvas = np.zeros((256, 256, 3), dtype=np.uint8)
    r.update(0.033)
    f1 = np.array(r.draw())
    r.update(0.033)
    f2 = np.array(r.draw())
    assert f1.shape == (256, 256, 3)
    nz = int((f2.sum(axis=2) > 0).sum())
    assert nz > 0, "flow field 应画出内容"
    assert not np.array_equal(f1, f2), "帧间应变化"
    print(f"  flow[{mode}] ok  非黑像素={nz}  帧变化={not np.array_equal(f1,f2)}")

# noise 原语 noise_type 分支
for nt in ("simplex", "worley", "warp"):
    cfg = {"params": {"layers": [
        {"type": "noise", "noise_type": nt, "scale": 10.0,
         "octaves": 4, "color": {"type": "hsv_flow"}}]}}
    r = pw.PWEffecRenderer(64, 64)
    r.load_config(cfg)
    r.update(0.033)
    f = np.array(r.draw())
    assert f.shape == (64, 64, 3) and f.sum() > 0
    print(f"  noise[{nt}] ok  sum={int(f.sum())}")

print("=== 4) audio_reactive: centroid / flux / bpm ===")
import audio_reactive as ar
# 造一段 3s 正弦+节拍的 wav
sr = 22050
dur = 3.0
t = np.linspace(0, dur, int(sr * dur), endpoint=False)
y = (np.sin(2 * np.pi * 440 * t) * 0.5
     + np.sin(2 * np.pi * 3 * np.ones_like(t) * 2)).astype(np.float32)
# 简单：叠加周期性幅度脉冲模拟节拍
beat = np.zeros_like(t)
for b in range(0, int(sr * dur), int(sr * 0.5)):
    beat[b:b + int(0.05 * sr)] = 0.5
sig = (y + beat).astype(np.float32)
tmp = "_verify_tone.wav"
with wave.open(tmp, "w") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
    wf.writeframes((sig * 32767).clip(-1, 1).astype("<i2").tobytes())
try:
    a = ar.AudioReactive(tmp, 30.0, dur)
    assert a.envelope.size > 1
    assert a.centroid.size == a.envelope.size, (a.centroid.shape, a.envelope.shape)
    assert 0.0 <= a.centroid.min() and a.centroid.max() <= 1.0
    assert a.flux.size == a.envelope.size
    assert a.flux.max() <= 1.0 + 1e-6
    # 频谱通量应在节拍处出现峰值
    assert a.flux.max() > 0.1, "通量应检出节拍脉冲"
    print(f"  librosa 路径 ok  env={a.envelope.shape} centroid=[{a.centroid.min():.2f},{a.centroid.max():.2f}] "
          f"flux_peak={a.flux.max():.2f} bpm={a.bpm:.1f}")
finally:
    try: os.remove(tmp)
    except Exception: pass

# numpy 回退特征函数（不依赖 librosa）
c, f = ar._spectral_features_numpy(sig.astype(np.float64), sr, 30.0)
assert c.size == 90 and f.size == 90
assert f.max() > 0.1
print(f"  numpy 回退 ok  centroid[{c.min():.2f},{c.max():.2f}] flux_peak={f.max():.2f}")

print("\nALL GREEN ✅")
