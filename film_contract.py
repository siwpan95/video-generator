#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""帧确定性契约 + 节拍时间轴（反哺 OpenFilm one-prompt 的两条思路）。

1. ``assert_deterministic(frame, times)``：OpenFilm 的 ``frame(t)`` 约定要求
   同一时刻永远画同一帧。本仓库 9685 个效果原本靠 ``--seed`` 复现随机组合，
   但"随机种子固定"≠"任意 t 的帧可复现"。这里用逐帧 SHA-256 双向渲染把
   这件事变成一条可跑的门禁，不依赖 ffmpeg，也不渲染整段视频。

2. ``BeatMap``：直接吃 ``openfilm/examples/one-prompt/assets/data/audio.json``
   的数据结构（beats / downbeats / sections / fps / rms / low / mid / high /
   onsets），提供按秒取包络、取最近拍、取分轨 onset 强度的纯函数接口。
   现有 ``audio_reactive`` 是"从音频现算特征"，这里是"读已经算好的时间轴"——
   两者并存，不互相覆盖。

用法::

    python film_contract.py <audio.json>          # 检查节拍时间轴并打印摘要
    python film_contract.py --selfcheck           # 自带合成样例自检
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

FrameFn = Callable[[float], object]


class DeterminismError(AssertionError):
    """同一 t 两次渲染得到不同字节。"""


def frame_digest(frame: FrameFn, t: float) -> str:
    """取 ``frame(t)`` 的字节摘要；支持 bytes / bytearray / 带 tobytes() 的图像。"""
    img = frame(t)
    raw = img if isinstance(img, (bytes, bytearray)) else img.tobytes()
    return hashlib.sha256(raw).hexdigest()


def assert_deterministic(frame: FrameFn, times: Iterable[float], *, label: str = 'frame') -> list[str]:
    """每个 t 渲染两次并逐字节比对；返回按输入顺序的摘要列表。"""
    digests = []
    for t in times:
        a = frame_digest(frame, t)
        b = frame_digest(frame, t)
        if a != b:
            raise DeterminismError(f'{label}({t}) 不确定：{a[:12]} != {b[:12]}')
        digests.append(a)
    return digests


@dataclass(frozen=True)
class BeatMap:
    """只读节拍时间轴；所有数组按 ``fps`` 等间隔采样，越界钳到端点。"""

    duration: float
    bpm: float
    beat_period: float
    beats: tuple[float, ...]
    downbeats: tuple[float, ...]
    sections: tuple[dict, ...]
    fps: int
    bands: dict[str, tuple[float, ...]]
    onsets: dict[str, tuple[tuple[float, float], ...]]

    @classmethod
    def from_dict(cls, data: dict) -> 'BeatMap':
        required = ('duration', 'bpm', 'beat_period', 'beats', 'downbeats',
                    'sections', 'fps', 'rms', 'low', 'mid', 'high', 'onsets')
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f'audio.json 缺字段: {", ".join(missing)}')
        fps = int(data['fps'])
        if fps <= 0:
            raise ValueError('fps 必须为正')
        bands = {}
        for key in ('rms', 'low', 'mid', 'high'):
            seq = data[key]
            if not isinstance(seq, list) or not seq:
                raise ValueError(f'{key} 必须是非空数组')
            bands[key] = tuple(float(v) for v in seq)
        onsets = {}
        for key, pairs in data['onsets'].items():
            onsets[str(key)] = tuple((float(p[0]), float(p[1])) for p in pairs)
        return cls(
            duration=float(data['duration']), bpm=float(data['bpm']),
            beat_period=float(data['beat_period']),
            beats=tuple(float(v) for v in data['beats']),
            downbeats=tuple(float(v) for v in data['downbeats']),
            sections=tuple(dict(s) for s in data['sections']),
            fps=fps, bands=bands, onsets=onsets,
        )

    @classmethod
    def load(cls, path: str | Path) -> 'BeatMap':
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    def band_at(self, name: str, t: float) -> float:
        """按 fps 线性插值取包络值；越界钳到端点。"""
        seq = self.bands[name]
        pos = min(max(float(t) * self.fps, 0.0), len(seq) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(seq) - 1)
        frac = pos - lo
        return seq[lo] * (1.0 - frac) + seq[hi] * frac

    def nearest_beat(self, t: float) -> float:
        """返回不晚于 t 的最近一拍（无拍返回 0.0）。"""
        idx = bisect_left(self.beats, float(t))
        return self.beats[idx - 1] if idx > 0 else 0.0

    def onset_strength(self, kind: str, t: float, window: float = 0.08) -> float:
        """取 ``kind`` 分轨在 t 附近 window 秒内的最大 onset 强度。"""
        best = 0.0
        for at, strength in self.onsets.get(kind, ()):
            if at > t + window:
                break
            if at >= t - window and strength > best:
                best = strength
        return best

    def section_at(self, t: float) -> str | None:
        for s in self.sections:
            if float(s['start']) <= t < float(s['end']):
                return str(s['name'])
        return None


def selfcheck() -> int:
    frame = lambda t: bytes([int(t * 17) % 256]) * 8
    ds = assert_deterministic(frame, [0, 0.5, 1.25], label='synthetic')
    assert len(set(ds)) == 3, '不同时刻摘要不应相同'

    bad_calls = {'n': 0}

    def flaky(t):
        bad_calls['n'] += 1
        return bytes([bad_calls['n'] % 256])

    try:
        assert_deterministic(flaky, [1.0], label='flaky')
    except DeterminismError:
        pass
    else:
        raise AssertionError('不确定帧没有被拦住')

    bm = BeatMap.from_dict({
        'duration': 4.0, 'bpm': 60.0, 'beat_period': 1.0,
        'beats': [0.0, 1.0, 2.0, 3.0], 'downbeats': [0.0, 2.0],
        'sections': [{'name': 'intro', 'start': 0, 'end': 2},
                     {'name': 'verse', 'start': 2, 'end': 4}],
        'fps': 2, 'rms': [0.0, 0.5, 1.0, 0.5],
        'low': [0.0, 0.0, 0.0, 0.0], 'mid': [0.0, 0.0, 0.0, 0.0],
        'high': [0.0, 0.0, 0.0, 0.0],
        'onsets': {'kick': [[1.0, 0.9], [3.0, 0.7]]},
    })
    assert abs(bm.band_at('rms', 0.25) - 0.25) < 1e-9
    assert bm.nearest_beat(1.9) == 1.0
    assert bm.onset_strength('kick', 1.02) == 0.9
    assert bm.section_at(2.5) == 'verse'
    print('FILM_CONTRACT_SELFCHECK_PASSED')
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='帧确定性 / 节拍时间轴检查')
    ap.add_argument('audio_json', nargs='?', help='one-prompt 风格 audio.json')
    ap.add_argument('--selfcheck', action='store_true')
    args = ap.parse_args(argv)
    if args.selfcheck:
        return selfcheck()
    if not args.audio_json:
        ap.error('需要 <audio.json> 或 --selfcheck')
    bm = BeatMap.load(args.audio_json)
    print(f'时长 {bm.duration:.3f}s / BPM {bm.bpm:g} / 拍 {len(bm.beats)} / '
          f'段 {len(bm.sections)} / onset 分轨 {list(bm.onsets)}')
    print('BEATMAP_OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
