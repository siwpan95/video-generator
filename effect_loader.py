"""
effect_loader.py — 从两个项目的 effects/ 目录加载效果配置

功能：
  1. 扫描 draw/effects/*.json  →  解析为 EffectConfig（draw 格式）
  2. 扫描 pywallpaper/effects/*.json  →  解析为 EffectConfig（pw 格式）
  3. 提供统一查询接口（按类别/标签/名称过滤）
  4. 支持随机选择
"""

import json
import os
import re
import random
from typing import Optional

from effect_model import (
    EffectConfig, EffectSource,
    PALETTE_DRAW, PALETTE_PW,
)


# ─────────────────────────────────────────────────────────
# 加载器
# ─────────────────────────────────────────────────────────

def load_draw_effects(effects_dir: str) -> list[EffectConfig]:
    """加载 draw/effects/*.json"""
    results = []
    if not os.path.isdir(effects_dir):
        print(f"[Loader] draw effects 目录不存在: {effects_dir}")
        return results

    json_files = sorted([
        f for f in os.listdir(effects_dir)
        if f.endswith(".json")
    ])

    for fname in json_files:
        fpath = os.path.join(effects_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                raw = json.load(f)
            cfg = EffectConfig.from_draw_json(fpath, raw)
            results.append(cfg)
        except Exception as e:
            print(f"[Loader] 跳过 {fname}: {e}")

    print(f"[Loader] draw effects 加载了 {len(results)} 个效果")
    return results


def load_pw_effects(effects_dir: str) -> list[EffectConfig]:
    """加载 pywallpaper/effects/*.json（按数字ID前缀排序）"""
    results = []
    if not os.path.isdir(effects_dir):
        print(f"[Loader] pywallpaper effects 目录不存在: {effects_dir}")
        return results

    json_files = sorted(
        [f for f in os.listdir(effects_dir) if f.endswith(".json")],
        key=_pw_sort_key,
    )

    for fname in json_files:
        if fname == "themes.json":
            continue  # themes.json 是 list 格式，不是效果配置
        fpath = os.path.join(effects_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # 确保 raw 是 dict（效果配置），而不是 list（如 themes.json）
            if not isinstance(raw, dict):
                print(f"[Loader] 跳过 {fname}: 不是有效的效果配置（类型为 {type(raw).__name__}）")
                continue
            cfg = EffectConfig.from_pw_json(fpath, raw)
            results.append(cfg)
        except Exception as e:
            print(f"[Loader] 跳过 {fname}: {e}")

    print(f"[Loader] pywallpaper effects 加载了 {len(results)} 个效果")
    return results


def _pw_sort_key(fname: str) -> int:
    """按文件名开头的数字排序，无数字则排最后"""
    m = re.match(r"(\d+)", fname)
    return int(m.group(1)) if m else 999999


# ─────────────────────────────────────────────────────────
# 统一注册表
# ─────────────────────────────────────────────────────────

class EffectRegistry:
    """统一效果注册表，合并两个来源的效果"""

    def __init__(self):
        self.draw_effects: list[EffectConfig] = []
        self.pw_effects: list[EffectConfig] = []
        self._all: list[EffectConfig] = []

    def load_all(self, draw_dir: str, pw_dir: str):
        """加载两个目录，构建注册表"""
        self.draw_effects = load_draw_effects(draw_dir)
        self.pw_effects = load_pw_effects(pw_dir)
        self._all = self.draw_effects + self.pw_effects
        print(
            f"[Registry] 共加载 {len(self._all)} 个效果 "
            f"（draw:{len(self.draw_effects)} + "
            f"pw:{len(self.pw_effects)}）"
        )

    def all(self) -> list[EffectConfig]:
        return self._all

    def by_source(self, source: str) -> list[EffectConfig]:
        if source == EffectSource.DRAW:
            return self.draw_effects
        elif source == EffectSource.PYWALLPAPER:
            return self.pw_effects
        return []

    def by_category(self, category: str) -> list[EffectConfig]:
        return [e for e in self._all if e.category == category]

    def by_categories(self, categories: list[str]) -> list[EffectConfig]:
        return [e for e in self._all if e.category in categories]

    def random_one(self, source: Optional[str] = None,
                    categories: Optional[list[str]] = None) -> Optional[EffectConfig]:
        """随机选一个效果"""
        pool = self._all
        if source:
            pool = [e for e in pool if e.source == source]
        if categories:
            pool = [e for e in pool if e.category in categories]
        if not pool:
            return None
        return random.choice(pool)

    def random_many(self, n: int = 3,
                     source: Optional[str] = None,
                     categories: Optional[list[str]] = None) -> list[EffectConfig]:
        """随机选 n 个效果（不重复）"""
        pool = self._all
        if source:
            pool = [e for e in pool if e.source == source]
        if categories:
            pool = [e for e in pool if e.category in categories]
        n = min(n, len(pool))
        return random.sample(pool, n) if n > 0 else []

    def summary(self) -> str:
        """返回注册表摘要"""
        cats_draw = set(e.category for e in self.draw_effects)
        cats_pw = set(e.category for e in self.pw_effects)
        lines = [
            f"效果注册表摘要：",
            f"  draw 效果: {len(self.draw_effects)} 个，"
            f"  类别: {sorted(cats_draw)}",
            f"  pywallpaper 效果: {len(self.pw_effects)} 个，"
            f"  类别: {sorted(cats_pw)}",
            f"  总计: {len(self._all)} 个",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# 快捷函数
# ─────────────────────────────────────────────────────────

# effects 目录与 video_generator.py 同级，避免跨项目依赖
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DRAW_DIR = os.path.join(_SCRIPT_DIR, "effects", "draw")
DEFAULT_PW_DIR = os.path.join(_SCRIPT_DIR, "effects", "pywallpaper")


def get_default_registry() -> EffectRegistry:
    """获取默认注册表（自动加载两个项目）"""
    registry = EffectRegistry()
    registry.load_all(DEFAULT_DRAW_DIR, DEFAULT_PW_DIR)
    return registry


if __name__ == "__main__":
    # 测试加载
    reg = get_default_registry()
    print(reg.summary())

    print("\n随机选 3 个效果：")
    for cfg in reg.random_many(3):
        print(" -", cfg.summary())
