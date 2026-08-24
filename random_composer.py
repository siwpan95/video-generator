"""
random_composer.py — 随机视频组合器

功能：
  1. 随机选择效果（从两个来源）
  2. 随机参数（调色板、网格尺寸、粒子数量...）
  3. 随机图层组合（多个效果叠加）
  4. 随机转场和节奏
  5. 生成视频描述（用于文件名和日志）
"""

import math
import random
import os
from typing import Optional, List, Tuple

from effect_model import EffectConfig, EffectSource
from effect_loader import EffectRegistry


# ── 视频参数 ─────────────────────────────────────────────────────────

DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_FPS = 30
DEFAULT_DURATION = 10  # 秒


# ── 随机组合配置 ────────────────────────────────────────────────────────

class VideoComposition:
    """视频组合配置（描述一个视频的生成参数）"""

    def __init__(self):
        self.width: int = DEFAULT_WIDTH
        self.height: int = DEFAULT_HEIGHT
        self.fps: int = DEFAULT_FPS
        self.duration: int = DEFAULT_DURATION

        # 图层列表（按从底到顶的顺序）
        self.layers: List["LayerConfig"] = []

        # 主题（背景）
        self.theme: str = "empty"  # empty / starfield / gradient / aurora

        # 全局效果
        self.global_postprocess: str = ""  # vignette / glow / edge

        # 视频描述
        self.description: str = ""

    def add_layer(self, layer: "LayerConfig"):
        self.layers.append(layer)

    def summary(self) -> str:
        lines = [
            f"视频组合：{self.width}x{self.height} @{self.fps}fps，"
            f"{self.duration}s",
            f"  主题: {self.theme}",
            f"  图层数: {len(self.layers)}",
        ]
        for i, layer in enumerate(self.layers):
            lines.append(f"  [{i}] {layer.summary()}")
        if self.global_postprocess:
            lines.append(f"  后处理: {self.global_postprocess}")
        return "\n".join(lines)


class LayerConfig:
    """单个图层配置"""

    def __init__(self, effect: EffectConfig, start_time: float = 0.0,
                 duration: float = None):
        self.effect = effect
        self.start_time = start_time  # 图层开始时间（秒）
        self.duration = duration  # None 表示到视频结束
        self.opacity: float = 1.0  # 不透明度 [0,1]
        self.blend_mode: str = "overlay"  # overlay / add / multiply
        self.params: dict = {}  # 随机化的参数

    def summary(self) -> str:
        src = "DRAW" if self.effect.source == EffectSource.DRAW else "PW"
        dur_str = f"{self.duration}s" if self.duration else "全程"
        return (
            f"{src} {self.effect.name} "
            f"({dur_str}，opacity={self.opacity:.1f}，"
            f"blend={self.blend_mode})"
        )


# ── 随机生成器 ─────────────────────────────────────────────────────────

class RandomComposer:
    """随机视频组合生成器"""

    THEMES = ["empty", "starfield", "gradient", "aurora"]
    BLEND_MODES = ["overlay", "add", "multiply", "screen"]
    POSTPROCESSES = ["", "vignette", "glow", "edge", "vignette+glow"]

    def __init__(self, registry: EffectRegistry):
        self.registry = registry

    def compose(self, seed: Optional[int] = None) -> VideoComposition:
        """生成一个随机视频组合"""
        if seed is not None:
            random.seed(seed)

        comp = VideoComposition()

        # 1. 随机视频参数
        comp.width = random.choice([512, 768, 1024])
        comp.height = random.choice([512, 768, 1024])
        comp.fps = random.choice([24, 30, 60])
        comp.duration = random.choice([5, 10, 15, 30])

        # 2. 随机主题
        comp.theme = random.choice(self.THEMES)

        # 3. 随机图层（1~4 层）
        n_layers = random.choices([1, 2, 3, 4], weights=[20, 40, 30, 10])[0]

        # 决定图层来源分布
        source_choices = self._choose_sources(n_layers)

        for i, source in enumerate(source_choices):
            layer = self._random_layer(source, comp.duration, i, n_layers)
            if layer:
                comp.add_layer(layer)

        # 4. 随机全局后处理
        comp.global_postprocess = random.choice(self.POSTPROCESSES)

        # 5. 生成描述
        comp.description = self._generate_description(comp)

        return comp

    def _choose_sources(self, n: int) -> List[str]:
        """为 n 个图层选择来源（draw/pw/混合）"""
        if n == 1:
            return [random.choice([EffectSource.DRAW, EffectSource.PYWALLPAPER])]
        # 多层：可能全来自一个来源，或混合
        if random.random() < 0.3:
            # 30% 概率：全 draw
            return [EffectSource.DRAW] * n
        elif random.random() < 0.6:
            # 30% 概率：全 pw
            return [EffectSource.PYWALLPAPER] * n
        else:
            # 40% 概率：混合
            return [random.choice([EffectSource.DRAW, EffectSource.PYWALLPAPER])
                    for _ in range(n)]

    def _random_layer(self, source: str, total_duration: float,
                      index: int, total_layers: int) -> Optional[LayerConfig]:
        """生成一个随机图层"""
        # 随机选择效果
        effect = self.registry.random_one(source=source)
        if not effect:
            return None

        # 随机时间窗口
        if total_layers > 1 and random.random() < 0.5:
            # 50% 概率：图层有开始时间和持续时间（非全程）
            start = random.uniform(0, total_duration * 0.5)
            dur = random.uniform(total_duration * 0.3, total_duration - start)
            start = min(start, total_duration - 1)
            dur = min(dur, total_duration - start)
        else:
            start = 0.0
            dur = None  # 全程

        layer = LayerConfig(effect, start, dur)

        # 随机不透明度
        layer.opacity = random.choice([0.5, 0.7, 0.85, 1.0])

        # 随机混合模式
        layer.blend_mode = random.choice(self.BLEND_MODES)

        # 随机参数
        layer.params = self._random_params(effect)

        return layer

    def _random_params(self, effect: EffectConfig) -> dict:
        """生成随机参数"""
        params = {}
        if effect.source == EffectSource.DRAW:
            # draw 效果：随机调色板、网格尺寸
            params["palette"] = random.choice(
                ["Rainbow", "Cool", "Hot", "Ocean", "Sunset",
                 "Neon", "Thermal", "Winter", "Cyberpunk", "Turbo"]
            )
            params["grid_w"] = random.choice([32, 48, 64, 80, 128])
            params["grid_h"] = random.choice([32, 48, 64, 80, 128])
            params["steps_per_frame"] = random.choice([1, 3, 5, 10])
        else:
            # pywallpaper 效果：根据类型随机参数
            params["speed"] = random.uniform(0.01, 0.1)
            if effect.category == "粒子系统":
                params["n_particles"] = random.randint(100, 2000)
            elif effect.category in ["波浪", "频谱"]:
                params["amplitude"] = random.randint(20, 200)
        return params

    def _generate_description(self, comp: VideoComposition) -> str:
        """生成视频描述（用于文件名）"""
        parts = []
        for layer in comp.layers:
            src = "D" if layer.effect.source == EffectSource.DRAW else "P"
            name = layer.effect.name[:20].replace(" ", "_")
            parts.append(f"{src}_{name}")
        theme_short = comp.theme[:3]
        return f"{theme_short}_" + "_".join(parts)[:100]

    def compose_from_keywords(self, keywords: List[str],
                             seed: Optional[int] = None) -> VideoComposition:
        """根据关键词生成视频组合（非随机）"""
        if seed is not None:
            random.seed(seed)

        comp = VideoComposition()
        comp.width = 768
        comp.height = 768
        comp.fps = 30
        comp.duration = 15

        # 根据关键词选择效果
        for kw in keywords[:4]:  # 最多4个关键词
            effect = self._find_effect_by_keyword(kw)
            if effect:
                layer = LayerConfig(effect, 0.0, None)
                layer.opacity = 0.8
                layer.blend_mode = "add"
                comp.add_layer(layer)

        if not comp.layers:
            # 没找到匹配的效果，随机选一个
            comp = self.compose(seed)

        comp.description = "_".join(keywords[:3])
        return comp

    def _find_effect_by_keyword(self, kw: str) -> Optional[EffectConfig]:
        """根据关键词查找效果"""
        kw_lower = kw.lower()
        for effect in self.registry.all():
            if (kw_lower in effect.name.lower() or
                kw_lower in effect.category.lower() or
                any(kw_lower in tag.lower() for tag in effect.tags)):
                return effect
        return None

    def build_from_spec(self, spec: dict,
                       seed: Optional[int] = None) -> "VideoComposition":
        """从结构化 spec 构建视频组合（LLM 语义组合层 / 本地语义引擎共用落地入口）。

        spec 结构（所有字段可选）：
          {
            "width": int, "height": int, "fps": int, "duration": int,
            "theme": str,                       # empty/starfield/gradient/aurora
            "global_postprocess": str,          # ""/vignette/glow/edge/vignette+glow
            "palette_bias": [str, ...],         # draw 调色板候选池
            "layers": [
              {"source": "draw"|"pw",
               "keywords": [str, ...],          # 命中原语/类别名；空则按 source 随机
               "blend_mode": str, "opacity": float,
               "params": { ... }}              # 覆盖/补充参数
            ]
          }
        约束由 llm_composer.Critic 在 spec 阶段校验；此处再做一次安全兜底。
        """
        if seed is not None:
            random.seed(seed)

        comp = VideoComposition()
        comp.width = int(spec.get("width", 768))
        comp.height = int(spec.get("height", 768))
        comp.fps = int(spec.get("fps", 30))
        comp.duration = int(spec.get("duration", 12))

        theme = spec.get("theme", "empty")
        comp.theme = theme if theme in self.THEMES else "empty"

        pp = spec.get("global_postprocess", "")
        comp.global_postprocess = pp if pp in self.POSTPROCESSES else ""

        palette_bias = spec.get("palette_bias") or []

        for ls in spec.get("layers", [])[:4]:
            src = ls.get("source", EffectSource.PYWALLPAPER)
            if src not in (EffectSource.DRAW, EffectSource.PYWALLPAPER):
                src = EffectSource.PYWALLPAPER

            effect = None
            for kw in ls.get("keywords", []):
                effect = self._find_effect_by_keyword(kw)
                if effect:
                    break
            if effect is None:
                effect = self.registry.random_one(source=src)
            if not effect:
                continue

            layer = LayerConfig(effect, 0.0, None)
            layer.opacity = float(ls.get("opacity", 0.85))
            layer.blend_mode = ls.get("blend_mode", "add")
            if layer.blend_mode not in self.BLEND_MODES:
                layer.blend_mode = "add"

            params = self._random_params(effect)
            params.update(dict(ls.get("params", {})))
            # 语义调色板优先（冷/暖/主题色），未显式指定时才用随机/调色板池
            if palette_bias and "palette" not in ls.get("params", {}):
                params["palette"] = random.choice(palette_bias)
            layer.params = params
            comp.add_layer(layer)

        if not comp.layers:
            # spec 完全没解析出图层 → 退回纯随机，保证总有产出
            comp = self.compose(seed=seed)

        comp.description = spec.get("description") or self._generate_description(comp)
        return comp


# ── 测试 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from effect_loader import get_default_registry

    print("加载效果库...")
    registry = get_default_registry()

    print("\n生成随机视频组合...")
    composer = RandomComposer(registry)

    for i in range(3):
        print(f"\n=== 组合 {i+1} ===")
        comp = composer.compose(seed=i * 42)
        print(comp.summary())
