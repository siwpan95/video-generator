"""
diffusion_base.py — P2-1 扩散模型基础层（可选）

把"扩散模型生成的底片"作为视频的底层，再用既有程序化原语（色图/噪声/
音频调制）做风格化叠加。对应 ENHANCEMENT_PROPOSAL_2026.md 的 P2-1。

设计遵循项目"优雅降级"铁律：
  * 不强制任何重依赖（torch / 扩散模型权重）。
  * 可用来源（按优先级）：
      1) 本地视频 / 帧序列（from_video）—— 始终可用，ffmpeg 解码。
      2) 云端生成（from_cloud）—— 复用 cloud_generator 凭证，缺失则降级。
      3) 本地扩散模型（from_local_model）—— 仅当设置了模型路径且环境可用，
         否则抛 DiffusionUnavailable，绝不静默假装成功。
      4) 程序化兜底（procedural_fallback）—— 零依赖，供测试与无模型时使用。
  * VideoGenerator 接入后：base 帧替代主题背景，程序化图层在其上叠加。
"""
import os
import subprocess
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


class DiffusionUnavailable(Exception):
    """扩散基础层不可用（无模型/无凭证/环境缺失），调用方应降级。"""


def _find_ffmpeg() -> str:
    env = os.environ.get("FFMPEG_BIN", "").strip()
    if env:
        return env
    for c in ("ffmpeg",):   # 2026-09-23 去本地化：删除已失效的 C:/ffmpeg/bin/ffmpeg.exe 候选
        try:
            subprocess.run([c, "-version"], capture_output=True, timeout=5)
            return c
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    return "ffmpeg"


class DiffusionBase:
    """扩散基础层：产出 (H,W,3) uint8 帧序列，按帧索引取帧。"""

    def __init__(self, width: int, height: int, fps: int,
                 source: str = "procedural"):
        self.w = width
        self.h = height
        self.fps = fps
        self.source = source
        self._frames = None      # 预解码的视频帧列表（from_video）
        self._proc = None        # 流式解码子进程（from_video 流式）
        self._frame_idx = 0
        self._seed = 1

    # ──────────────────────────────────────────────────────────
    # 工厂
    # ──────────────────────────────────────────────────────────
    @classmethod
    def from_video(cls, path: str, width: int, height: int, fps: int):
        """用本地视频/帧序列作基础层（始终可用，ffmpeg 解码）。"""
        if not os.path.isfile(path):
            raise DiffusionUnavailable(f"基础层视频不存在: {path}")
        obj = cls(width, height, fps, source=f"video:{os.path.basename(path)}")
        obj._video_path = path
        # 预解码前 N 帧循环使用（视频短于成片时循环）
        obj._frames = obj._decode_all(path)
        if not obj._frames:
            raise DiffusionUnavailable("基础层视频解码为空")
        return obj

    @classmethod
    def from_cloud(cls, prompt: str, width: int, height: int, fps: int,
                   duration: float = 4.0):
        """用云端 API 生成基础层（复用 cloud_generator 凭证）。"""
        try:
            import cloud_generator
        except Exception as e:  # noqa: BLE001
            raise DiffusionUnavailable(f"cloud_generator 不可用: {e}")
        out = os.path.join(_SCRIPT_DIR, "output", "_diffusion_cloud_base.mp4")
        try:
            ok = cloud_generator.generate(
                prompt=prompt, output=out, width=width, height=height,
                fps=fps, duration=duration)
        except Exception as e:  # noqa: BLE001
            raise DiffusionUnavailable(f"云端生成失败: {e}")
        if not ok or not os.path.isfile(out):
            raise DiffusionUnavailable("云端未产出基础层文件")
        return cls.from_video(out, width, height, fps)

    @classmethod
    def from_local_model(cls, model_path: str, prompt: str,
                         width: int, height: int, fps: int,
                         duration: float = 4.0):
        """本地扩散模型（Wan2.2 / LTX-2.3 等）。

        仅当显式设置模型路径且 torch 可用时才尝试，否则抛
        DiffusionUnavailable。本机未预置权重，默认不启用，绝不静默降级成假片。
        """
        if not model_path or not os.path.isdir(model_path):
            raise DiffusionUnavailable("未设置本地扩散模型路径")
        # 探测 torch（不真正加载，避免父进程段错误；沿用 audio_stems 哲学）
        if __import__("importlib").util.find_spec("torch") is None:
            raise DiffusionUnavailable("本环境无 torch，无法跑本地扩散")
        # 真实推理入口：留作 hooks（需模型权重 + 显存），此处仅占位并明确失败。
        raise DiffusionUnavailable(
            "本地扩散推理未实现（需权重+显存）；请用 from_video/from_cloud 或 "
            "procedural_fallback")

    @classmethod
    def procedural_fallback(cls, width: int, height: int, fps: int,
                            seed: int = 1):
        """零依赖程序化兜底基础层（确定性动画渐变），供测试/无模型时使用。"""
        obj = cls(width, height, fps, source="procedural")
        obj._seed = seed
        return obj

    # ──────────────────────────────────────────────────────────
    # 帧获取
    # ──────────────────────────────────────────────────────────
    def frame_at(self, idx: int) -> "np.ndarray":
        import numpy as np
        if self.source.startswith("video:"):
            if self._frames:
                return self._frames[idx % len(self._frames)].copy()
            raise DiffusionUnavailable("视频帧为空")
        # 程序化兜底：缓慢流动的对角渐变 + 微噪点（确定性，按 idx 推进）
        t = idx / max(self.fps, 1)
        rg = np.linspace(0, 1, self.w, dtype=np.float64)
        phase = (t * 0.3 + self._seed * 0.13) % 1.0
        grad = np.tile(((rg + phase) % 1.0), (self.h, 1))  # (H,W)
        # HSV→RGB 简化（色相沿对角线流动）
        h = grad
        s = 0.6 * np.ones_like(h)
        v = 0.35 + 0.25 * np.sin(np.linspace(0, 6.28, self.h).reshape(-1, 1)
                                 + t * 2.0)
        v = np.clip(v, 0, 1)
        r = np.clip(np.abs(((h * 6.0) % 6.0) - 3.0) - 1.0, 0, 1)
        g = np.clip(2.0 - np.abs(((h * 6.0) % 6.0) - 2.0), 0, 1)
        b = np.clip(2.0 - np.abs(((h * 6.0) % 6.0) - 4.0), 0, 1)
        out = np.stack([r, g, b], axis=-1) * s.reshape(s.shape[0], s.shape[1], 1)
        out = out * v.reshape(v.shape[0], v.shape[1], 1)
        return (out * 255).astype(np.uint8)

    # ──────────────────────────────────────────────────────────
    # 内部：ffmpeg 解码视频为帧列表
    # ──────────────────────────────────────────────────────────
    def _decode_all(self, path: str):
        import numpy as np
        ff = _find_ffmpeg()
        try:
            proc = subprocess.Popen(
                [ff, "-v", "error", "-i", path, "-f", "rawvideo",
                 "-pix_fmt", "rgb24", "-s", f"{self.w}x{self.h}", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            frames = []
            frame_bytes = self.w * self.h * 3
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                frames.append(np.frombuffer(buf, dtype=np.uint8)
                              .reshape(self.h, self.w, 3).copy())
            proc.stdout.close()
            proc.wait()
            return frames
        except Exception as e:  # noqa: BLE001
            raise DiffusionUnavailable(f"解码失败: {e}")


def make_diffusion_base(mode: str, width: int, height: int, fps: int,
                        prompt: str = None, video_path: str = None,
                        model_path: str = None, seed: int = 1,
                        duration: float = 4.0):
    """统一入口：按 mode 构造基础层，失败抛 DiffusionUnavailable 由调用方降级。

    mode: "video" | "cloud" | "local" | "procedural" | "auto"
      auto：video → cloud → procedural（不尝试 local，避免无谓失败）
    """
    if mode == "auto" or mode == "video":
        if video_path and os.path.isfile(video_path):
            return DiffusionBase.from_video(video_path, width, height, fps)
    if mode in ("auto", "cloud") and prompt:
        try:
            return DiffusionBase.from_cloud(prompt, width, height, fps, duration)
        except DiffusionUnavailable:
            if mode == "cloud":
                raise
    if mode == "local":
        return DiffusionBase.from_local_model(
            model_path or os.environ.get("DIFFUSION_MODEL_DIR", ""),
            prompt or "", width, height, fps, duration)
    # 兜底
    return DiffusionBase.procedural_fallback(width, height, fps, seed)
