# -*- coding: utf-8 -*-
"""
test_genlog.py —— 视频生成任务日志测试（P4，与 audio_pipeline pipelog 同构）
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from genlog import PipelineLog  # noqa: E402


def test_genlog_append_and_read(tmp_path: Path):
    log = PipelineLog(tmp_path / "demo.pipeline.log")
    log.append("run_start", run_id="r1", output="demo.mp4", width=1920, height=1080)
    log.append("stage_done", stage="encode", run_id="r1", frames=300)
    entries = log.read()
    assert len(entries) == 2
    assert entries[0]["type"] == "run_start" and entries[0]["seq"] == 1
    assert entries[1]["type"] == "stage_done" and entries[1]["frames"] == 300


def test_genlog_seq_continues(tmp_path: Path):
    path = tmp_path / "x.pipeline.log"
    PipelineLog(path).append("run_start", run_id="r1")
    e = PipelineLog(path).append("run_end", run_id="r1", ok=True)
    assert e["seq"] == 2


def test_genlog_empty_when_no_file(tmp_path: Path):
    assert PipelineLog(tmp_path / "missing.log").read() == []
