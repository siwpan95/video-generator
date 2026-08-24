"""
effect_model.py — 统一 Effect 数据模型

将两个项目的 effect JSON 解析为统一的 EffectConfig 对象。
支持两种来源：
  - draw/effects/*.json  （网格CA类效果）
  - pywallpaper/effects/*.json （EngineEffect 原语图层类效果）
"""

from dataclasses import dataclass, field
from typing import Optional
import random


# ─────────────────────────────────────────────────────────
# 调色板 / 颜色预设
# ─────────────────────────────────────────────────────────

PALETTE_DRAW = [
    "Cool", "Hot", "Grayscale", "Winter", "Rainbow",
    "JetSimple", "DivergingBWR", "DivergingRdBu",
    "SequentialYlGn", "Categorical8", "Binary",
    "IQ_Palette", "IQ_Palette2", "SinePalette",
    "LorenzPalette", "RosslerPalette",
    "WarmCool", "PosterSolar", "Thermal", "Aurora",
    "Cyberpunk", "Vaporwave", "Neon", "Ocean", "Sunset",
]

PALETTE_PW = [
    "hsv_flow", "rainbow", "cool_warm", "ocean_deep",
    "sunset_fire", "neon_pulse", "mono_perceptual",
    "gradient_2", "gradient_4", "party",
]


# ─────────────────────────────────────────────────────────
# Effect 来源类型
# ─────────────────────────────────────────────────────────

class EffectSource:
    DRAW = "draw"           # draw/effects/ 网格CA效果
    PYWALLPAPER = "pw"      # pywallpaper/effects/ 原语图层效果


# ─────────────────────────────────────────────────────────
# 统一 EffectConfig
# ─────────────────────────────────────────────────────────

@dataclass
class EffectConfig:
    """统一效果配置"""
    source: str                # EffectSource.DRAW or .PYWALLPAPER
    effect_id: str             # 文件ID或名称
    name: str                  # 显示名称
    category: str              # 类别（CA/粒子/流体/几何/...）
    file_path: str = ""        # 完整文件路径（新增）

    # ── draw 效果专用字段 ──
    draw_grid_w: int = 64
    draw_grid_h: int = 64
    draw_rule: str = ""
    draw_color_red: str = "v"
    draw_color_green: str = "v"
    draw_color_blue: str = "v"
    draw_postprocess: str = ""  # vignette / edge / glow / ...

    # ── pywallpaper 效果专用字段 ──
    pw_layers: list = field(default_factory=list)   # 原语图层配置列表
    pw_has_audio: bool = False

    # ── 通用字段 ──
    tags: list = field(default_factory=list)
    theme_compatible: list = field(default_factory=list)  # 兼容的主题

    @classmethod
    def from_draw_json(cls, filepath: str, raw: dict) -> "EffectConfig":
        """从 draw/effects/*.json 解析"""
        name = raw.get("name", filepath)
        color = raw.get("color", {})
        grid = raw.get("grid", {})
        post = raw.get("postprocess", {})

        w = grid.get("width", 64)
        h = grid.get("height", 64)

        return cls(
            source=EffectSource.DRAW,
            effect_id=filepath,
            name=name,
            category=_guess_draw_category(name),
            file_path=filepath,
            draw_grid_w=w,
            draw_grid_h=h,
            draw_rule=raw.get("evolution", {}).get("rule", ""),
            draw_color_red=color.get("red", "v"),
            draw_color_green=color.get("green", "v"),
            draw_color_blue=color.get("blue", "v"),
            draw_postprocess=post.get("effect", ""),
            tags=_extract_draw_tags(name),
        )

    @classmethod
    def from_pw_json(cls, filepath: str, raw: dict) -> "EffectConfig":
        """从 pywallpaper/effects/*.json 解析"""
        return cls(
            source=EffectSource.PYWALLPAPER,
            effect_id=raw.get("id", filepath),
            name=raw.get("name", filepath),
            category=raw.get("category", "未分类"),
            file_path=filepath,
            pw_layers=raw.get("params", {}).get("layers", []),
            pw_has_audio=bool(raw.get("params", {}).get("audio_modulation")),
            tags=raw.get("tags", []),
        )

    def summary(self) -> str:
        """返回效果摘要"""
        if self.source == EffectSource.DRAW:
            return (
                f"[DRAW] {self.name} "
                f"({self.draw_grid_w}x{self.draw_grid_h}) "
                f"palette={self._guess_palette()}"
            )
        else:
            layer_types = [l.get("type", "?") for l in self.pw_layers]
            return (
                f"[PW] {self.name} "
                f"({self.category}) "
                f"layers={layer_types}"
            )

    def _guess_palette(self) -> str:
        """从颜色表达式猜测调色板名称"""
        r, g, b = self.draw_color_red, self.draw_color_green, self.draw_color_blue
        if "sin" in r or "sin" in g or "sin" in b:
            return "Rainbow"
        if "1-v" in r or "invert" in r:
            return "Invert"
        return "Custom"


# ─────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────

def _guess_draw_category(name: str) -> str:
    """从 draw effect 名称猜测类别"""
    name_lower = name.lower()
    if "bzreaction" in name_lower or "oregonator" in name_lower:
        return "反应扩散"
    if "diffusion" in name_lower:
        return "扩散"
    if "turing" in name_lower or "meinhardt" in name_lower or "gierer" in name_lower:
        return "图灵图案"
    if "ca_" in name_lower or "super_cml" in name_lower or "coupledmap" in name_lower:
        return "细胞自动机"
    if "schrodinger" in name_lower:
        return "量子"
    if "kuramoto" in name_lower or "networkkuramoto" in name_lower:
        return "同步"
    if "particle" in name_lower or "fountain" in name_lower:
        return "粒子"
    if "fluid" in name_lower or "navier" in name_lower:
        return "流体"
    if "wave" in name_lower:
        return "波动"
    if "spiral" in name_lower or "annulus" in name_lower:
        return "螺旋"
    return "数学"


def _extract_draw_tags(name: str) -> list:
    """从 draw effect 名称提取标签"""
    tags = []
    nl = name.lower()
    if "super" in nl:
        tags.append("super")
    if "anisotropic" in nl:
        tags.append("anisotropic")
    if "threshold" in nl:
        tags.append("threshold")
    return tags


# ─────────────────────────────────────────────────────────
# 随机化工具
# ─────────────────────────────────────────────────────────

def random_palette() -> str:
    return random.choice(PALETTE_DRAW + PALETTE_PW)


def random_category_filter() -> list:
    """返回随机的类别过滤列表（可能为空，表示不过滤）"""
    all_cats = [
        "反应扩散", "扩散", "图灵图案", "细胞自动机",
        "量子", "同步", "粒子", "流体", "波动", "螺旋",
        "数学", "L系统", "分形",
        "细胞自动机", "粒子系统", "流体模拟",
        "几何图形", "波浪", "文本雨", "3D线框",
    ]
    if random.random() < 0.5:
        return []  # 不过滤
    n = random.randint(1, 3)
    return random.sample(all_cats, min(n, len(all_cats)))
