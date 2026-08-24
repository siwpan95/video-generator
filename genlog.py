# -*- coding: utf-8 -*-
"""
genlog.py —— 视频生成任务事件日志（P4：日志即事实源，与 WorldLog 同构）

追加式 JSONL：{seq, ts, type, stage?, run_id?, ...}。
事件类型：run_start / stage_start / stage_done / stage_skip / stage_error / run_end。

默认落盘 <输出文件>.pipeline.log —— 与产物同区，审计/续跑有据可查。
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

__all__ = ["PipelineLog"]


class PipelineLog:
    """追加式生成任务日志（线程安全，写失败静默不阻断生成）。"""

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._seq = 0

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event_type: str, stage: Optional[str] = None,
               run_id: Optional[str] = None, **payload: Any) -> dict:
        with self._lock:
            if self._seq == 0:
                self._seq = self._read_max_seq()
            self._seq += 1
            entry: dict = {"seq": self._seq,
                           "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "type": event_type}
            if run_id:
                entry["run_id"] = run_id
            if stage:
                entry["stage"] = stage
            entry.update(payload)
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError:
                pass
            return entry

    def _read_max_seq(self) -> int:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                seq = 0
                for line in f:
                    try:
                        seq = max(seq, int(json.loads(line).get("seq", 0)))
                    except (ValueError, json.JSONDecodeError):
                        continue
                return seq
        except OSError:
            return 0

    def read(self, limit: int = 100) -> list[dict]:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return []
        entries: list[dict] = []
        for line in lines[-limit:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    @staticmethod
    def new_run_id() -> str:
        return uuid.uuid4().hex[:8]
