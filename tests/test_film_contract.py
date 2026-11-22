# -*- coding: utf-8 -*-
"""帧确定性契约 + one-prompt 风格节拍时间轴测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from film_contract import (  # noqa: E402
    BeatMap, DeterminismError, assert_deterministic, frame_digest, main,
)


def _sample() -> dict:
    return {
        'duration': 4.0, 'bpm': 60.0, 'beat_period': 1.0,
        'beats': [0.0, 1.0, 2.0, 3.0], 'downbeats': [0.0, 2.0],
        'sections': [{'name': 'intro', 'start': 0, 'end': 2},
                     {'name': 'verse', 'start': 2, 'end': 4}],
        'fps': 2, 'rms': [0.0, 0.5, 1.0, 0.5],
        'low': [0, 0, 0, 0], 'mid': [0, 0, 0, 0], 'high': [0, 0, 0, 0],
        'onsets': {'kick': [[1.0, 0.9], [3.0, 0.7]]},
    }


def test_frame_digest_and_determinism():
    frame = lambda t: bytes([int(t * 10) % 256]) * 4
    ds = assert_deterministic(frame, [0, 0.5, 1.0])
    assert ds == [frame_digest(frame, t) for t in (0, 0.5, 1.0)]
    assert len(set(ds)) == 3


def test_flaky_frame_fails_loudly():
    n = {'v': 0}

    def flaky(t):
        n['v'] += 1
        return bytes([n['v']])

    with pytest.raises(DeterminismError):
        assert_deterministic(flaky, [1.0])


def test_beatmap_requires_fields():
    with pytest.raises(ValueError, match='缺字段'):
        BeatMap.from_dict({'duration': 1})


def test_band_interpolation_and_clamp():
    bm = BeatMap.from_dict(_sample())
    assert bm.band_at('rms', 0.25) == pytest.approx(0.25)
    assert bm.band_at('rms', -5) == 0.0
    assert bm.band_at('rms', 99) == 0.5


def test_beat_section_and_onset_queries():
    bm = BeatMap.from_dict(_sample())
    assert bm.nearest_beat(1.9) == 1.0
    assert bm.nearest_beat(-1) == 0.0
    assert bm.section_at(2.5) == 'verse'
    assert bm.section_at(9.0) is None
    assert bm.onset_strength('kick', 1.02) == 0.9
    assert bm.onset_strength('kick', 2.0) == 0.0


def test_selfcheck_entry():
    assert main(['--selfcheck']) == 0
