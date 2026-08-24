"""
audio_reactive.py — 音频驱动效果调制（可选依赖 librosa）

联网调研（librosa onset / energy）：
  - RMS 能量包络是最直接的"音量 -> 亮度 / 辉光 / 不透明度"驱动信号。
  - onset_strength / beat 检测可触发转场冲击或节奏脉冲。
  - 降级策略（保证无重依赖也能跑）：
      librosa  ->  ffmpeg 抽取 PCM wav 后 numpy 分帧 RMS
               ->  标准库 wave 直接读 .wav
               ->  全失败则返回常数包络（不崩溃）。

对外 API：
  ar = AudioReactive(audio_path, fps, duration=None)
  ar.envelope        -> np.ndarray (n_frames,) 归一化到 [0,1]
  ar.onsets          -> np.ndarray (n_frames,) 0/1 节拍触发
  ar.envelope_at(i)  -> float  第 i 帧的能量（0-1）
  ar.beat_at(i)      -> bool   第 i 帧是否落拍
  ar.hue_shift_at(i) -> float  第 i 帧色相偏移（圈，1.0=360°，beat 驱动+包络漂移）
"""

import os
import subprocess
import tempfile
from typing import Optional

import numpy as np

try:
    import librosa
    _HAS_LIBROSA = True
except Exception:  # noqa: BLE001
    librosa = None
    _HAS_LIBROSA = False

_SR = 22050          # 重采样率（与 librosa 默认一致）
_FRAME_LEN = 2048    # 分析帧长
_HOP = 512           # 帧移


def _rms_envelope(y: np.ndarray, sr: int, fps: float) -> np.ndarray:
    """从 mono 信号分帧计算 RMS 能量，并线性重采样到视频帧率 fps。

    返回长度 = round(时长 * fps) 的包络，归一化到 [0,1]。
    """
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < _FRAME_LEN:
        # 太短：整体一个能量值
        e = np.sqrt(np.mean(y ** 2) + 1e-12)
        return np.clip(np.array([e]), 0.0, 1.0)

    # 按 hop 分帧求 RMS
    env_hop = []
    for i in range(0, n - _FRAME_LEN, _HOP):
        seg = y[i:i + _FRAME_LEN]
        env_hop.append(np.sqrt(np.mean(seg ** 2) + 1e-12))
    env_hop = np.array(env_hop, dtype=np.float64)
    if env_hop.size == 0:
        env_hop = np.array([0.0])

    dur = n / sr
    n_video = max(1, int(round(dur * fps)))
    hop_rate = sr / _HOP
    x_hop = np.arange(env_hop.size) / hop_rate
    x_video = np.arange(n_video) / fps
    env_video = np.interp(x_video, x_hop, env_hop)

    mx = env_video.max()
    if mx > 0:
        env_video = env_video / mx
    return env_video.astype(np.float64)


def _spectral_features_numpy(y: np.ndarray, sr: int, fps: float,
                             frame_len: int = _FRAME_LEN,
                             hop: int = _HOP):
    """无 librosa 时，用 numpy STFT 估算频谱质心与频谱通量。

    质心（明亮度）= Σ f·|X| / Σ |X|；通量 = 相邻帧幅度谱正差分之和。
    返回已重采样到逐视频帧的 (centroid_norm, flux_norm)，各为 [0,1]。
    """
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < frame_len:
        return np.array([0.5]), np.array([0.0])

    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
    half = frame_len // 2 + 1
    win = np.hanning(frame_len)
    n_frames = 1 + (n - frame_len) // hop
    cents, fluxes = [], []
    prev_mag = None
    for i in range(n_frames):
        seg = y[i * hop:i * hop + frame_len] * win
        mag = np.abs(np.fft.rfft(seg))
        total = mag.sum() + 1e-12
        cents.append((freqs * mag).sum() / total)
        if prev_mag is None:
            fluxes.append(0.0)
        else:
            fluxes.append(np.maximum(mag - prev_mag, 0.0).sum())
        prev_mag = mag

    cents = np.clip(np.array(cents) / (sr / 2.0), 0.0, 1.0)
    fluxes = np.array(fluxes, dtype=np.float64)
    fmax = fluxes.max()
    if fmax > 0:
        fluxes = fluxes / fmax

    dur = n / sr
    n_video = max(1, int(round(dur * fps)))
    src_rate = sr / hop
    x_src = np.arange(cents.size) / src_rate
    x_video = np.arange(n_video) / fps
    return (np.interp(x_video, x_src, cents),
            np.interp(x_video, x_src, fluxes))


class AudioReactive:
    """把一段音频变成逐帧调制信号，供视频生成器驱动效果参数。"""

    def __init__(self, audio_path: str, fps: float,
                 duration: Optional[float] = None):
        self.audio_path = audio_path
        self.fps = fps
        self.duration = duration
        self.sr = _SR
        self.envelope = np.array([0.5])
        self.onsets = np.array([0])
        self.hue = np.array([0.0])  # 每帧色相偏移（cyclic，范围 [0,1) 圈）
        # 联网调研新增：频谱质心（明亮度）/ 频谱通量（变化度）/ BPM
        self.centroid = np.array([0.5])  # 归一化 [0,1]，高=明亮
        self.flux = np.array([0.0])       # 归一化 [0,1]，高=变化剧烈
        self.bpm = 0.0
        self._load()

    # ── 加载 ───────────────────────────────────────────────────────
    def _load(self):
        # 1) librosa（最准，带 onset/beat）
        if _HAS_LIBROSA and librosa is not None:
            try:
                y, sr = librosa.load(self.audio_path, sr=_SR, mono=True)
                self._from_signal(y, sr, use_librosa_onset=True)
                return
            except Exception:  # noqa: BLE001
                pass

        # 2) .wav 直接用标准库 wave 读（无 ffmpeg 也能跑）
        if isinstance(self.audio_path, str) and self.audio_path.lower().endswith(".wav"):
            try:
                y, sr = self._read_wav_numpy(self.audio_path)
                self._from_signal(y, sr, use_librosa_onset=False)
                return
            except Exception:  # noqa: BLE001
                pass

        # 3) ffmpeg 抽 PCM wav 后 numpy 读
        wav = self._extract_wav(self.audio_path)
        if wav is not None:
            try:
                y, sr = self._read_wav_numpy(wav)
                self._from_signal(y, sr, use_librosa_onset=False)
                return
            except Exception:  # noqa: BLE001
                pass
            finally:
                try:
                    os.remove(wav)
                except Exception:  # noqa: BLE001
                    pass

        # 4) 全失败：常数包络（调用方用其做中性调制，不崩溃）
        self.envelope = np.array([0.5])
        self.onsets = np.array([0])

    def _from_signal(self, y, sr, use_librosa_onset: bool):
        self.sr = sr
        self.envelope = _rms_envelope(y, sr, self.fps)

        if use_librosa_onset and _HAS_LIBROSA and librosa is not None:
            try:
                self._from_librosa(y, sr)
                self._compute_hue()   # 修复：librosa 路径也必须算 hue
                return
            except Exception:  # noqa: BLE001
                pass

        # numpy 路径：能量上升沿 + 频谱通量近似
        self._compute_numpy_features(y, sr)
        self._compute_hue()

    def _from_librosa(self, y, sr):
        """用 librosa 计算 onset / 频谱质心 / 频谱通量 / BPM"""
        hop_rate = sr / _HOP
        frames = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=_HOP, units="frames")
        n_video = len(self.envelope)
        onsets = np.zeros(n_video, dtype=int)
        for f in frames:
            tv = float(f) / hop_rate
            idx = int(round(tv * self.fps))
            if 0 <= idx < n_video:
                onsets[idx] = 1
        self.onsets = onsets

        # 频谱质心（明亮度）→ 归一化 [0,1]
        c = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=_HOP)[0]
        c = np.clip(c / (sr / 2.0), 0.0, 1.0)
        # 频谱通量（变化度）→ 归一化 [0,1]
        fl = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP)
        if fl.max() > 0:
            fl = fl / fl.max()
        # 重采样到逐视频帧（与 _rms_envelope 同口径）
        src_rate = sr / _HOP
        x_src = np.arange(c.size) / src_rate
        x_video = np.arange(n_video) / self.fps
        self.centroid = np.interp(x_video, x_src, c)
        self.flux = np.interp(x_video, x_src, fl)

        # BPM（节拍速度估算）
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            self.bpm = float(np.atleast_1d(tempo)[0])
        except Exception:  # noqa: BLE001
            self.bpm = 0.0

    def _compute_numpy_features(self, y, sr):
        """无 librosa 时：numpy STFT 计算质心/通量，从通量峰值估 onset"""
        c, f = _spectral_features_numpy(y, sr, self.fps)
        self.centroid = c
        self.flux = f
        self.bpm = 0.0
        n = f.size
        if n > 1:
            thr = f.mean() + 0.3 * (f.max() - f.min())
            self.onsets = (f > max(thr, 1e-6)).astype(int)
        else:
            self.onsets = np.array([0])

    # ── 外部工具 ───────────────────────────────────────────────────
    @staticmethod
    def _extract_wav(path: str) -> Optional[str]:
        if not os.path.isfile(path):
            return None
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, tmp.name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=60,
            )
            if os.path.getsize(tmp.name) > 44:
                return tmp.name
        except Exception:  # noqa: BLE001
            pass
        try:
            os.remove(tmp.name)
        except Exception:  # noqa: BLE001
            pass
        return None

    @staticmethod
    def _read_wav_numpy(wav_path: str):
        import wave
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
        y = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        return y, sr

    # ── 查询接口 ───────────────────────────────────────────────────
    def envelope_at(self, frame_idx: int) -> float:
        if self.envelope.size == 1:
            return float(self.envelope[0])
        i = min(max(int(frame_idx), 0), self.envelope.size - 1)
        return float(self.envelope[i])

    def beat_at(self, frame_idx: int) -> bool:
        if self.onsets.size == 1:
            return False
        i = min(max(int(frame_idx), 0), self.onsets.size - 1)
        return bool(self.onsets[i])

    # ── 色相随节拍偏移 ─────────────────────────────────────────────
    def _compute_hue(self, attack: float = 0.18, drift: float = 0.015):
        """计算每帧色相偏移（cyclic 范围 [0,1) 圈）。

        驱动逻辑（联网调研：节拍同步视觉是最直接的"音乐可视化"手法）：
          - 每个落拍帧把目标色相按黄金比共轭 (0.61803...) 推进 → 相邻节拍
            色相隔得最开，避免相邻拍撞色，视觉更跳跃、更有律动感。
          - 帧间用 attack 斜坡缓动逼近目标（而非硬跳），形成"打拍子时
            色相被轻轻推一下"的攻击感。
          - 叠加随 RMS 包络的轻微连续漂移，无拍段落也有呼吸感。
        返回写入 self.hue（长度 = 帧数）。无音频/失败时保持 [0.0]。
        """
        n = len(self.envelope)
        hue = np.zeros(n, dtype=np.float64)
        golden = 0.6180339887498949
        target = 0.0
        cur = 0.0
        beat_frames = (self.onsets == 1) if self.onsets.size > 1 else np.zeros(n, dtype=bool)
        for i in range(n):
            if beat_frames[i]:
                target = (target + golden) % 1.0
            cur += (target - cur) * attack
            cur += drift * float(self.envelope[i])
            cur %= 1.0
            hue[i] = cur
        self.hue = hue

    def hue_shift_at(self, frame_idx: int, scale: float = 1.0) -> float:
        """第 frame_idx 帧的色相偏移，单位：圈（1.0 = 360°）。

        调用方通常：hue_deg = hue_shift_at(i) * 360，再对 RGB 做 HSV 旋转。
        """
        if getattr(self, "hue", None) is None or self.hue.size <= 1:
            return 0.0
        i = min(max(int(frame_idx), 0), self.hue.size - 1)
        return float(self.hue[i]) * scale

    def centroid_at(self, frame_idx: int) -> float:
        """第 frame_idx 帧的频谱质心（明亮度），归一化 [0,1]，高=明亮。

        调研：明亮度可映射为色温（低=冷/暗，高=暖/亮），用于音频驱动色调。
        """
        if getattr(self, "centroid", None) is None or self.centroid.size <= 1:
            return 0.5
        i = min(max(int(frame_idx), 0), self.centroid.size - 1)
        return float(self.centroid[i])

    def flux_at(self, frame_idx: int) -> float:
        """第 frame_idx 帧的频谱通量（变化度），归一化 [0,1]，高=变化剧烈。

        可用于比能量更敏锐的转场/冲击触发（副歌通常比主歌通量更大）。
        """
        if getattr(self, "flux", None) is None or self.flux.size <= 1:
            return 0.0
        i = min(max(int(frame_idx), 0), self.flux.size - 1)
        return float(self.flux[i])


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python audio_reactive.py <audio_file> [fps]")
        sys.exit(1)
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
    ar = AudioReactive(sys.argv[1], fps)
    print(f"[audio_reactive] librosa={_HAS_LIBROSA} frames={len(ar.envelope)}")
    print(f"  envelope[0:5] = {np.round(ar.envelope[:5], 3)}")
    beats = int(ar.onsets.sum()) if ar.onsets.size > 1 else 0
    print(f"  detected beats = {beats}")
