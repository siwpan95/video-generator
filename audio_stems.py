"""
audio_stems.py — Demucs 分轨音频驱动（P0-3）

把一段音频用 Demucs 拆成 drums / bass / vocals / other 四条分轨，对每条
分轨各自跑一遍 AudioReactive，得到逐帧的「鼓点能量、贝斯能量、人声明亮度、
环境能量」等专属调制信号。视频生成器据此把不同乐器映射到不同的视觉通道
（鼓→冲击/色相迸发，贝斯→对比度脉动，人声→色温，环境→辉光），由「整曲
平均能量」升级为「听感对齐」的分层音乐可视化。

联网调研要点：
  - Demucs (facebookresearch) 是 SOTA 开源人声/乐器分离模型，htdemucs 四路
    输出 drums/bass/other/vocals；CLI：`python -m demucs -n htdemucs -o OUT audio.mp3`。
  - 分轨后每条轨各自算 RMS 包络 + onset，比整曲混合包络更能体现「哪个乐器在
    主导」，可驱动更有层次的视觉（副歌人声拉暖色温、鼓点砸出数据撕裂等）。

降级策略（与项目一致：无重依赖也能跑，绝不崩溃）：
  - 优先用 `python -m demucs`（需要 `pip install demucs` + torch，较重）。
  - 若 demucs 不可用 / 分离失败 → 抛出 DemucsUnavailable，调用方降级为整曲
    混合驱动（AudioReactive 原路径）。

对外 API：
  try:
      stems = AudioStems(audio_path, fps, duration)
      stems.drums.envelope_at(i) / beat_at(i)        # 鼓点能量 / 落拍
      stems.vocals.centroid_at(i)                     # 人声明亮度
      arr = audio_stems.apply_stem_modulation(arr, i, stems)
  except DemucsUnavailable:
      # 降级为整曲混合驱动
  # 测试 / 无 demucs 时直接用预分离好的 wav：
  stems = AudioStems.from_paths({"drums":..., "bass":..., "vocals":...,
                                 "other":..., "mix":...}, fps)
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

try:
    from audio_reactive import AudioReactive
except Exception:  # noqa: BLE001
    # 允许作为独立脚本导入时 audio_reactive 不在路径（已通过 video_generator 的 sys.path 注入）
    AudioReactive = None  # type: ignore

try:
    _HAS_DEMUCS = importlib.util.find_spec("demucs") is not None
except Exception:  # noqa: BLE001
    _HAS_DEMUCS = False

_DEMUCS_OK = None  # 缓存可用性探测结果


def _torch_lib_dir() -> "str | None":
    """定位 torch 的原生 DLL 目录（torch/lib），不真正 import torch，
    避免本机 torch 在父进程中段错误（其 C 扩展需 torch/lib 前置到 PATH）。"""
    try:
        spec = importlib.util.find_spec("torch")
    except Exception:  # noqa: BLE001
        return None
    if not spec or not spec.origin:
        return None
    lib = os.path.join(os.path.dirname(spec.origin), "lib")
    return lib if os.path.isdir(lib) else None


def _build_env() -> dict:
    """为 demucs 子进程构造环境：把 torch/lib 前置到 PATH，并放开 OpenMP
    重复库，规避本机 libiomp5md.dll 冲突导致的 access violation（段错误）。"""
    env = os.environ.copy()
    tlib = _torch_lib_dir()
    if tlib:
        env["PATH"] = tlib + os.pathsep + env.get("PATH", "")
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return env


def _demucs_cmd():
    """返回 (cmd_prefix, label)。DEMUCS_BIN 优先；否则用当前解释器 -m demucs；
    二者皆无则返回 (None, 'unavailable')。"""
    bin_ = os.environ.get("DEMUCS_BIN")
    if bin_ and os.path.isfile(bin_):
        return [bin_], f"DEMUCS_BIN={bin_}"
    try:
        has_demucs = importlib.util.find_spec("demucs") is not None
    except Exception:  # noqa: BLE001
        has_demucs = False
    if has_demucs:
        return [sys.executable, "-m", "demucs"], f"{sys.executable} -m demucs"
    return None, "unavailable"


class DemucsUnavailable(Exception):
    """demucs 不可用或分离失败时抛出，调用方据此降级。"""


def _demucs_available() -> bool:
    """探测 demucs 是否真正可用（实际拉起 CLI --help 一次）。结果带缓存。"""
    global _DEMUCS_OK
    if _DEMUCS_OK is not None:
        return _DEMUCS_OK
    cmd, _label = _demucs_cmd()
    if cmd is None:
        _DEMUCS_OK = False
        return False
    try:
        proc = subprocess.run(
            cmd + ["--help"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=60, env=_build_env(),
        )
        _DEMUCS_OK = (proc.returncode == 0)
    except Exception:  # noqa: BLE001
        _DEMUCS_OK = False
    return _DEMUCS_OK


class AudioStems:
    """四条分轨（drums/bass/vocals/other）+ 整曲 mix 的 AudioReactive 集合。"""

    STEMS = ("drums", "bass", "vocals", "other")

    def __init__(self, audio_path: str, fps: float,
                 duration: float = None, model: str = "htdemucs"):
        self.audio_path = audio_path
        self.fps = fps
        self.duration = duration
        self.model = model
        self.drums = self.bass = self.vocals = self.other = None
        self.mix = None
        self._stem_paths: dict = {}

        # 整曲兜底基础调制（始终可用，是降级安全网）
        if AudioReactive is not None:
            try:
                self.mix = AudioReactive(audio_path, fps, duration)
            except Exception:  # noqa: BLE001
                self.mix = None

        self._stem_paths, _tmp = self._separate(audio_path, model)
        for name in self.STEMS:
            p = self._stem_paths.get(name)
            if p and os.path.isfile(p) and AudioReactive is not None:
                try:
                    setattr(self, name, AudioReactive(p, fps, duration))
                except Exception:  # noqa: BLE001
                    setattr(self, name, None)
        # 分轨 wav 已读入内存，清理 demucs 临时目录（避免每次生成泄漏临时文件）
        if _tmp and os.path.isdir(_tmp):
            shutil.rmtree(_tmp, ignore_errors=True)

    @classmethod
    def from_paths(cls, paths: dict, fps: float, duration: float = None):
        """直接用已分离好的分轨 wav 构造（无需 demucs，便于测试 / 外部预分离）。

        paths 键：'drums'/'bass'/'vocals'/'other'（必需其一），'mix'（可选，整曲）。
        """
        obj = cls.__new__(cls)
        obj.audio_path = paths.get("mix") or paths.get("drums") or ""
        obj.fps = fps
        obj.duration = duration
        obj.model = "from_paths"
        obj.drums = obj.bass = obj.vocals = obj.other = None
        obj.mix = None
        obj._stem_paths = {k: v for k, v in paths.items() if k in cls.STEMS}
        if AudioReactive is not None:
            if paths.get("mix"):
                try:
                    obj.mix = AudioReactive(paths["mix"], fps, duration)
                except Exception:  # noqa: BLE001
                    obj.mix = None
            for name in cls.STEMS:
                p = paths.get(name)
                if p and os.path.isfile(p):
                    try:
                        setattr(obj, name, AudioReactive(p, fps, duration))
                    except Exception:  # noqa: BLE001
                        setattr(obj, name, None)
        return obj

    # ── Demucs 分离 ─────────────────────────────────────────────────
    def _separate(self, audio_path: str, model: str) -> dict:
        if not audio_path or not os.path.isfile(audio_path):
            raise DemucsUnavailable("音频文件不存在，无法分轨")
        if not _demucs_available():
            raise DemucsUnavailable(
                "未安装 demucs（请执行 `pip install demucs` 后重试）；"
                "已降级为整曲混合驱动")

        out_dir = tempfile.mkdtemp(prefix="demucs_")
        cmd, _label = _demucs_cmd()
        if cmd is None:
            raise DemucsUnavailable(
                "未找到 demucs（请 `pip install demucs` 或设置 DEMUCS_BIN）；"
                "已降级为整曲混合驱动")
        full_cmd = cmd + ["-n", model, "-o", out_dir, audio_path]
        try:
            proc = subprocess.run(
                full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=1800, env=_build_env(),
            )
        except FileNotFoundError as e:  # 极端：解释器无法拉起
            raise DemucsUnavailable(f"无法调用 demucs: {e}")
        if proc.returncode != 0:
            raise DemucsUnavailable(f"demucs 分离失败（退出码 {proc.returncode}）")

        track = os.path.splitext(os.path.basename(audio_path))[0]
        stem_dir = os.path.join(out_dir, model, track)
        if not os.path.isdir(stem_dir):
            # 兜底：在 model 目录下找第一个含 .wav 的目录（处理文件名奇奇怪怪的情况）
            for root, _, files in os.walk(os.path.join(out_dir, model)):
                if any(f.lower().endswith(".wav") for f in files):
                    stem_dir = root
                    break
        result = {}
        for name in self.STEMS:
            p = os.path.join(stem_dir, f"{name}.wav")
            if os.path.isfile(p):
                result[name] = p
        if not result:
            raise DemucsUnavailable("demucs 未产出任何分轨文件")
        return result, out_dir

    # ── 便捷查询（带 None 保护）────────────────────────────────────
    def drums_energy_at(self, i: int) -> float:
        return self.drums.envelope_at(i) if self.drums else 0.0

    def bass_energy_at(self, i: int) -> float:
        return self.bass.envelope_at(i) if self.bass else 0.0

    def vocals_energy_at(self, i: int) -> float:
        return self.vocals.envelope_at(i) if self.vocals else 0.0

    def other_energy_at(self, i: int) -> float:
        return self.other.envelope_at(i) if self.other else 0.0

    def drums_beat_at(self, i: int) -> bool:
        return self.drums.beat_at(i) if self.drums else False

    def summary(self) -> str:
        avail = [n for n in self.STEMS if getattr(self, n) is not None]
        return (f"AudioStems(model={self.model}, "
                f"available={avail or 'none'}, "
                f"mix={'✓' if self.mix else '✗'})")


# ═════════════════════════════════════════════════════════════════
#  分轨 → 视觉通道映射（核心调制逻辑）
# ═════════════════════════════════════════════════════════════════

def apply_stem_modulation(arr: np.ndarray, frame_idx: int,
                          stems: "AudioStems") -> np.ndarray:
    """把四条分轨的专属信号映射到独立视觉通道。

    映射（听感对齐）：
      - drums  → 数据撕裂冲击 + 色相迸发（落拍时砸一下，能量越大越狠）
      - bass   → 对比度/亮度脉动（低频冲击感，punch）
      - vocals → 色温（人声明亮度 → 暖/冷漂移）
      - other  → 环境辉光/泛光（铺底氛围）
    返回调制后的 uint8 数组。各通道独立，缺失分轨自动跳过。
    """
    # 1) drums → 冲击 + 色相迸发
    if stems.drums is not None:
        d_e = stems.drums.envelope_at(frame_idx)
        if stems.drums.beat_at(frame_idx):
            arr = _drum_glitch(arr, d_e)
            arr = _hsv_hue_rotate(arr, 6.0 + 28.0 * d_e)

    # 2) bass → 对比度/亮度脉动
    #    注意：AudioReactive 包络按各轨自身 max 归一化，恒定（静音/持续）分轨
    #    会被误读为满能量；仅当包络确有动态（max-min 明显）时才驱动，避免误触发。
    if (stems.bass is not None
            and getattr(stems.bass, "envelope", None) is not None
            and stems.bass.envelope.size > 1
            and (stems.bass.envelope.max() - stems.bass.envelope.min()) > 0.05):
        b_e = stems.bass.envelope_at(frame_idx)
        if b_e > 0.02:
            arr = _contrast_pulse(arr, b_e)

    # 3) vocals → 色温
    if (stems.vocals is not None
            and getattr(stems.vocals, "centroid", None) is not None
            and stems.vocals.centroid.size > 1):
        warmth = stems.vocals.centroid_at(frame_idx) * 2.0 - 1.0
        if abs(warmth) > 0.02:
            arr = _color_temperature(arr, warmth)

    # 4) other → 环境辉光（同样加动态门限，避免恒定铺底被误读为满能量）
    if (stems.other is not None
            and getattr(stems.other, "envelope", None) is not None
            and stems.other.envelope.size > 1
            and (stems.other.envelope.max() - stems.other.envelope.min()) > 0.05):
        o_e = stems.other.envelope_at(frame_idx)
        if o_e > 0.05:
            arr = _bloom(arr, 0.12 + 0.5 * o_e)

    return arr


# ── 视觉通道辅助（纯 numpy / PIL，无额外重依赖）──────────────────
def _drum_glitch(arr: np.ndarray, energy: float) -> np.ndarray:
    """落拍时的数据撕裂：随机若干水平条带做横向错位，强度随鼓能量。"""
    h, _ = arr.shape[:2]
    if h < 8:
        return arr
    n = max(1, int(2 + energy * 6))
    out = arr.copy()
    for _ in range(n):
        band = np.random.randint(0, max(1, h - 6))
        shift = np.random.randint(-30, 31)
        out[band:band + 6] = np.roll(out[band:band + 6], shift, axis=1)
    return out


def _contrast_pulse(arr: np.ndarray, energy: float) -> np.ndarray:
    """低频冲击：能量越高 → 对比度越高、轻微提亮（punch 感）。"""
    factor = 1.0 + 0.6 * float(energy)
    mean = arr.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    out = mean + (arr.astype(np.float32) - mean) * factor
    out = out * (1.0 + 0.1 * float(energy))
    return np.clip(out, 0, 255).astype(np.uint8)


def _color_temperature(arr: np.ndarray, warmth: float) -> np.ndarray:
    """色温漂移：warmth>0 偏暖（增 R 减 B），warmth<0 偏冷（减 R 增 B）。"""
    out = arr.astype(np.float32)
    if warmth >= 0:
        out[..., 0] += 35.0 * warmth
        out[..., 2] -= 22.0 * warmth
    else:
        w = -warmth
        out[..., 0] -= 22.0 * w
        out[..., 2] += 35.0 * w
    return np.clip(out, 0, 255).astype(np.uint8)


def _bloom(arr: np.ndarray, strength: float) -> np.ndarray:
    """环境辉光：高斯模糊叠加，制造铺底泛光。"""
    from PIL import Image, ImageFilter
    img = Image.fromarray(arr)
    blur = img.filter(ImageFilter.GaussianBlur(radius=6))
    out = np.clip(
        arr.astype(np.float32) * 0.8 + np.array(blur) * float(strength),
        0, 255,
    ).astype(np.uint8)
    return out


def _hsv_hue_rotate(arr: np.ndarray, deg: float) -> np.ndarray:
    """HSV 色相旋转（uint8 环回，避免溢出）。"""
    from PIL import Image
    img = Image.fromarray(arr, "RGB").convert("HSV")
    h = np.asarray(img, dtype=np.uint16)
    shift = int((float(deg) % 360.0) / 360.0 * 255.0)
    h[..., 0] = (h[..., 0] + shift) % 256
    out = Image.fromarray(h.astype(np.uint8), mode="HSV").convert("RGB")
    return np.asarray(out)


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 2:
        print("用法: python audio_stems.py <audio_file> [fps]")
        print(f"  demucs 可用: {_demucs_available()}")
        _sys.exit(1)
    fps = float(_sys.argv[2]) if len(_sys.argv) > 2 else 25.0
    try:
        s = AudioStems(_sys.argv[1], fps)
        print(f"[audio_stems] {s.summary()}")
    except DemucsUnavailable as e:
        print(f"[audio_stems] 分轨降级: {e}")
