"""
_verify_stems.py — P0-3 Demucs 分轨音频驱动 验证

不依赖 demucs（用 from_paths + 合成分轨 wav），验证：
  A) 四个视觉通道辅助函数确实变换画面（单元级证明）
  B) from_paths 正常构造、drums 能检出节拍、apply_stem_modulation 在落拍帧改画
  C) DemucsUnavailable 降级路径（缺文件 / demucs 不可用均抛出，调用方可 catch）
"""
import os
import sys
import tempfile
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_stems as AS  # noqa: E402

SR = 22050
DUR = 2.0
N = int(SR * DUR)
FPS = 25
T = np.arange(N) / SR


def make_wav(path, signal):
    y = np.clip(np.asarray(signal, dtype=np.float64), -1, 1)
    data = (y * 32767.0).astype(np.int16).tobytes()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(data)


def silent():
    return np.zeros(N)


def drums_sig():
    sig = np.zeros(N)
    for bt in [0.4, 0.8, 1.2, 1.6]:
        i0 = int(bt * SR)
        i1 = int((bt + 0.15) * SR)
        seg = np.sin(2 * np.pi * 60 * T[i0:i1])
        env = np.linspace(0, 1, max(1, i1 - i0))
        sig[i0:i1] = 0.9 * seg * env
    return sig


def bass_sig():
    return 0.3 * np.sin(2 * np.pi * 80 * T) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * T))


def vocals_sig():
    return 0.4 * np.sin(2 * np.pi * 400 * T) * (0.5 + 0.5 * np.sin(2 * np.pi * 5 * T))


def other_sig():
    rng = np.random.RandomState(0)
    return 0.2 * rng.randn(N)


def gray_frame(h=64, w=64):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def striped_frame(h=64, w=64):
    # 水平条纹（非均匀），用于验证数据撕裂会真正改动像素
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[: h // 2] = (200, 50, 50)
    f[h // 2:] = (50, 50, 200)
    return f


def colored_frame(h=64, w=64):
    # 左红右蓝（有饱和度），用于验证色相旋转会改变颜色
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[:, : w // 2] = (220, 30, 30)
    f[:, w // 2:] = (30, 30, 220)
    return f


def main():
    tmp = tempfile.mkdtemp(prefix="stems_verify_")
    paths = {
        "drums": os.path.join(tmp, "drums.wav"),
        "bass": os.path.join(tmp, "bass.wav"),
        "vocals": os.path.join(tmp, "vocals.wav"),
        "other": os.path.join(tmp, "other.wav"),
        "mix": os.path.join(tmp, "mix.wav"),
    }
    make_wav(paths["drums"], drums_sig())
    make_wav(paths["bass"], bass_sig())
    make_wav(paths["vocals"], vocals_sig())
    make_wav(paths["other"], other_sig())
    make_wav(paths["mix"], drums_sig() + bass_sig() + vocals_sig() + other_sig())

    ok = True

    # ── A) 单元：四个视觉通道辅助函数 ──────────────────────────────
    # 注意：drum_glitch / hsv_hue_rotate 在「均匀灰」画面上不可见
    #      （滚动相同像素 / 灰饱和度为 0），须用非均匀/彩色画面验证。
    for name, fn, frame in [
        ("drum_glitch", lambda a: AS._drum_glitch(a, 0.9), colored_frame()),
        ("contrast_pulse", lambda a: AS._contrast_pulse(a, 0.8), gray_frame()),
        ("color_temperature", lambda a: AS._color_temperature(a, 0.5), gray_frame()),
        ("bloom", lambda a: AS._bloom(a, 0.5), gray_frame()),
        ("hsv_hue_rotate", lambda a: AS._hsv_hue_rotate(a, 90.0), colored_frame()),
    ]:
        out = fn(frame.copy())
        changed = not np.array_equal(out, frame)
        valid = out.shape == frame.shape and out.min() >= 0 and out.max() <= 255
        print(f"[A] {name:16s} changed={changed} valid={valid}")
        ok = ok and changed and valid

    # ── B) from_paths + 节拍检测 + 集成调制 ───────────────────────
    # 仅 drums 激活（其余分轨不传入 → None，被安全跳过）
    stems = AS.AudioStems.from_paths(
        {"drums": paths["drums"], "mix": paths["drums"]}, FPS)
    print(f"[B] {stems.summary()}")
    env = stems.drums.envelope
    env_nonconst = float(env.max() - env.min()) > 0.05
    beats = [i for i in range(len(env)) if stems.drums.beat_at(i)]
    print(f"[B] drums env range={float(env.max()-env.min()):.3f} "
          f"beats={beats}")
    ok = ok and env_nonconst and len(beats) > 0

    # 落拍帧：apply_stem_modulation 应改变画面；非落拍帧：理论不变（其余 None）
    # 用有色彩/水平变化的画面（鼓冲击/色相旋转对均匀灰不可见，真实帧不会如此）
    f_beat = beats[0]
    f_off = 0
    arr_beat = AS.apply_stem_modulation(colored_frame(), f_beat, stems)
    arr_off = AS.apply_stem_modulation(colored_frame(), f_off, stems)
    beat_changed = not np.array_equal(arr_beat, colored_frame())
    off_unchanged = np.array_equal(arr_off, colored_frame())
    print(f"[B] beat-frame changed={beat_changed}  off-frame unchanged={off_unchanged}")
    ok = ok and beat_changed and off_unchanged

    # 仅 vocals 激活 → 色温应改变画面
    voc_stems = AS.AudioStems.from_paths(
        {"vocals": paths["vocals"], "mix": paths["vocals"]}, FPS)
    vframe = gray_frame()
    vout = AS.apply_stem_modulation(vframe, 20, voc_stems)
    voc_changed = not np.array_equal(vout, vframe)
    print(f"[B] vocals-only stem changed={voc_changed}")
    ok = ok and voc_changed

    # ── C) DemucsUnavailable 降级路径 ─────────────────────────────
    c1 = c2 = False
    try:
        AS.AudioStems(os.path.join(tmp, "nope.mp3"), FPS)
    except AS.DemucsUnavailable:
        c1 = True
    print(f"[C] missing file -> DemucsUnavailable: {c1}")

    # 强制 demucs 不可用（即使本机装了），真实音频也应优雅抛出
    orig = AS._DEMUCS_OK
    AS._DEMUCS_OK = False
    AS._HAS_DEMUCS = False
    try:
        AS.AudioStems(paths["mix"], FPS)
    except AS.DemucsUnavailable:
        c2 = True
    finally:
        AS._DEMUCS_OK = orig
    print(f"[C] demucs disabled -> DemucsUnavailable: {c2}")
    ok = ok and c1 and c2

    print("\n" + ("✅ P0-3 验证全部通过" if ok else "❌ 验证失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
