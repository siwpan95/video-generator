"""
llm_composer.py — 语义组合层（P0-1）

把自然语言描述（如"深海发光、慢节奏、冷色调"）映射为结构化 spec，
再由 RandomComposer.build_from_spec 落地为 VideoComposition。

设计遵循项目既有"优雅降级"哲学：
  * LocalSemanticComposer  —— 纯本地词典引擎，零依赖、零联网，始终可用。
  * LLMComposer（可选）     —— 仅当设置 DEEPSEEK_API_KEY 环境变量且 --llm 时激活；
                              调用 DeepSeek 结构化输出，失败/缺失即回退本地引擎。
  * Critic                  —— Actor-Critic 双智能体中的"校验者"，约束图层数/
                              调色板/来源/参数范围，硬失败则回退纯随机。

入口：compose_from_prompt(prompt, registry, seed, use_llm) -> (VideoComposition, meta)
"""

import os
from typing import Dict, List, Optional, Tuple

from random_composer import RandomComposer, EffectSource


# ─────────────────────────────────────────────────────────────────────────
# 语义词典（本地引擎）
# ─────────────────────────────────────────────────────────────────────────

# 情绪/色调 → 调色板候选池（draw 调色板名，沿用 effect_model.PALETTE_DRAW）
MOOD_PALETTE = {
    "冷":   ["Cool", "Ocean", "Cyberpunk", "Turbo"],
    "寒冷": ["Cool", "Ocean"],
    "深海": ["Ocean", "Cool", "Turbo"],
    "海洋": ["Ocean", "Cool"],
    "水":   ["Ocean", "Cool"],
    "冰":   ["Cool", "Winter"],
    "科技": ["Cyberpunk", "Neon", "Turbo"],
    "赛博": ["Cyberpunk", "Neon"],
    "电路": ["Cyberpunk", "IQ_Palette"],
    "霓虹": ["Neon", "Vaporwave"],
    "暖":   ["Sunset", "Hot", "Neon"],
    "温暖": ["Sunset", "Hot"],
    "日落": ["Sunset", "Hot"],
    "火焰": ["Hot", "Sunset"],
    "熔岩": ["Hot", "Thermal"],
    "火":   ["Hot", "Sunset"],
    "梦幻": ["Vaporwave", "Neon", "Rainbow"],
    "彩虹": ["Rainbow"],
    "自然": ["gradient_2", "Aurora", "Ocean"],
    "森林": ["Aurora", "gradient_4"],
    "黄金": ["Sunset", "PosterSolar"],
    "暗":   ["Grayscale", "Cyberpunk"],
    "黑":   ["Grayscale", "IQ_Palette"],
}

# 情绪/主题 → 背景主题
MOOD_THEME = {
    "星空": "starfield", "宇宙": "starfield", "星": "starfield",
    "极光": "aurora", "天空": "gradient", "渐变": "gradient",
    "夜": "starfield", "空": "empty",
}

# 运动/节奏 → 渲染参数
MOTION = {
    "慢":   {"fps": 24, "speed": 0.02, "duration": 15, "steps": 3},
    "舒缓": {"fps": 24, "speed": 0.02, "duration": 15, "steps": 3},
    "温柔": {"fps": 24, "speed": 0.02, "duration": 15, "steps": 3},
    "快":   {"fps": 60, "speed": 0.09, "duration": 8,  "steps": 5},
    "动感": {"fps": 60, "speed": 0.08, "duration": 8,  "steps": 5},
    "激烈": {"fps": 60, "speed": 0.1,  "duration": 6,  "steps": 10},
    "冲击": {"fps": 60, "speed": 0.1,  "duration": 6,  "steps": 10},
    "节奏": {"fps": 30, "speed": 0.06, "duration": 10, "steps": 5},
    "舞":   {"fps": 30, "speed": 0.06, "duration": 10, "steps": 5},
}

# 运动词优先级（强语义词先于中性词，避免"快节奏"被"节奏"覆盖）
MOTION_PRIORITY = ["激烈", "冲击", "快", "舒缓", "慢", "动感", "节奏", "舞", "温柔"]

# 风格/原语类型 → 来源倾向（draw=网格CA / pw=原语图层）
STYLE_SOURCE = {
    "反应": EffectSource.DRAW, "图灵": EffectSource.DRAW,
    "数学": EffectSource.DRAW, "细胞": EffectSource.DRAW,
    "扩散": EffectSource.DRAW, "波": EffectSource.DRAW,
    "粒子": EffectSource.PYWALLPAPER, "流体": EffectSource.PYWALLPAPER,
    "波浪": EffectSource.PYWALLPAPER, "线框": EffectSource.PYWALLPAPER,
    "3d": EffectSource.PYWALLPAPER, "三维": EffectSource.PYWALLPAPER,
    "流场": EffectSource.PYWALLPAPER, "噪声": EffectSource.PYWALLPAPER,
    "纹理": EffectSource.PYWALLPAPER, "几何": EffectSource.PYWALLPAPER,
    "文本": EffectSource.PYWALLPAPER, "文字": EffectSource.PYWALLPAPER,
}

# 领域关键词 → 效果类别提示（命中原语/类别名，喂给 _find_effect_by_keyword）
DOMAIN_CATEGORY = {
    "深海": ["波浪", "流体", "粒子"], "海洋": ["波浪", "流体", "粒子"],
    "海浪": ["波浪", "流体"],
    "水": ["流体", "波浪"], "宇宙": ["几何", "粒子"],
    "星": ["几何", "粒子"], "电路": ["细胞自动机", "数学"],
    "网格": ["细胞自动机", "数学"], "火焰": ["扩散", "流体"],
    "熔岩": ["扩散", "流体"], "植物": ["L系统", "分形"],
    "森林": ["L系统", "分形"], "花": ["L系统", "分形"],
    "城市": ["几何", "线框"], "建筑": ["几何", "线框"],
}

# 后处理倾向
POSTPROCESS = {
    "发光": "vignette+glow", "光": "glow", "辉光": "glow",
    "电影": "vignette", "氛围": "vignette", "暗角": "vignette",
    "锐利": "edge", "科技感": "edge", "线": "edge",
}

# 画幅倾向（关键词 → 尺寸）
ASPECT = {
    "竖屏": (720, 1280), "手机": (720, 1280), "故事": (720, 1280),
    "方形": (1024, 1024), "正方形": (1024, 1024),
    "横屏": (1280, 720), "宽屏": (1280, 720),
    "大": (1024, 1024), "高清": (1024, 1024),
}


# ─────────────────────────────────────────────────────────────────────────
# 本地语义引擎
# ─────────────────────────────────────────────────────────────────────────

class LocalSemanticComposer:
    """纯本地、零依赖的自然语言 → spec 映射器。"""

    def compose(self, prompt: str, seed: Optional[int] = None) -> Dict:
        p = prompt.lower()
        palette_bias: List[str] = []
        theme = None
        motion = None
        sources: List[str] = []
        categories: List[str] = []
        postprocess = None
        aspect = None

        # 逐词典扫描（子串匹配，兼容中英文混写）
        for key, pal in MOOD_PALETTE.items():
            if key in p:
                palette_bias.extend(pal)
        for key, th in MOOD_THEME.items():
            if key in p and theme is None:
                theme = th
        for key in MOTION_PRIORITY:
            if key in p:
                motion = MOTION[key]
                break
        for key, src in STYLE_SOURCE.items():
            if key in p:
                sources.append(src)
        for key, cats in DOMAIN_CATEGORY.items():
            if key in p:
                categories.extend(cats)
        for key, pp in POSTPROCESS.items():
            if key in p and postprocess is None:
                postprocess = pp
        for key, dim in ASPECT.items():
            if key in p and aspect is None:
                aspect = dim

        # 去重并保持顺序
        palette_bias = list(dict.fromkeys(palette_bias)) or ["Cool", "Ocean", "Turbo"]
        sources = list(dict.fromkeys(sources))
        categories = list(dict.fromkeys(categories))

        # 全局参数
        width, height = aspect or (768, 768)
        fps = (motion or {}).get("fps", 30)
        duration = (motion or {}).get("duration", 12)
        steps = (motion or {}).get("steps", 5)
        speed = (motion or {}).get("speed", 0.05)

        # 图层规划：领域类别驱动图层数，否则默认 2~3 层
        if categories:
            n_layers = min(max(len(categories), 1), 4)
        else:
            n_layers = 2 if not sources else min(len(sources) + 1, 4)

        if not sources:
            # 无风格倾向 → 按 draw/pw 混合
            sources = [EffectSource.PYWALLPAPER, EffectSource.DRAW,
                       EffectSource.PYWALLPAPER, EffectSource.DRAW]

        layers = []
        for i in range(n_layers):
            src = sources[i % len(sources)]
            kw = categories[i % len(categories)] if categories else ""
            layer: Dict = {
                "source": src,
                "keywords": [kw] if kw else [],
                "blend_mode": "add",
                "opacity": 0.85,
            }
            if src == EffectSource.DRAW:
                layer["params"] = {
                    "palette": None,  # 由 palette_bias 填充
                    "steps_per_frame": steps,
                }
            else:
                layer["params"] = {"speed": speed}
            layers.append(layer)

        spec = {
            "width": width,
            "height": height,
            "fps": fps,
            "duration": duration,
            "theme": theme or "empty",
            "global_postprocess": postprocess or "",
            "palette_bias": palette_bias,
            "layers": layers,
            "description": "prompt_" + _safe(prompt),
        }
        return spec


def _safe(text: str, n: int = 40) -> str:
    return "".join(c for c in text if c.isalnum() or c in "_- ")[:n].strip()


# ─────────────────────────────────────────────────────────────────────────
# Critic（Actor-Critic 双智能体中的"校验者"）
# ─────────────────────────────────────────────────────────────────────────

KNOWN_PALETTES = {
    "Cool", "Hot", "Grayscale", "Winter", "Rainbow", "JetSimple",
    "DivergingBWR", "DivergingRdBu", "SequentialYlGn", "Categorical8",
    "Binary", "IQ_Palette", "IQ_Palette2", "SinePalette", "LorenzPalette",
    "RosslerPalette", "WarmCool", "PosterSolar", "Thermal", "Aurora",
    "Cyberpunk", "Vaporwave", "Neon", "Ocean", "Sunset",
    "hsv_flow", "rainbow", "cool_warm", "ocean_deep", "sunset_fire",
    "neon_pulse", "mono_perceptual", "gradient_2", "gradient_4", "party",
    "Turbo",
}

THEMES = {"empty", "starfield", "gradient", "aurora"}
POSTPROCESSES = {"", "vignette", "glow", "edge", "vignette+glow"}


def critic(spec: Dict) -> Tuple[Dict, List[str]]:
    """校验并修正 spec，返回 (corrected_spec, issues)。

    硬失败（无法修正）时 issues 含 'HARD' 标记，调用方应回退纯随机。
    """
    issues: List[str] = []
    s = dict(spec)

    # 图层数
    layers = s.get("layers", [])
    if not isinstance(layers, list) or len(layers) == 0:
        issues.append("HARD: no layers")
        return s, issues
    layers = layers[:4]
    for L in layers:
        if L.get("source") not in (EffectSource.DRAW, EffectSource.PYWALLPAPER):
            L["source"] = EffectSource.PYWALLPAPER
        L.setdefault("keywords", [])
        L.setdefault("blend_mode", "add")
        L.setdefault("opacity", 0.85)
        L.setdefault("params", {})
    s["layers"] = layers

    # 数值范围夹紧
    s["width"] = _clamp_int(s.get("width", 768), 64, 4096)
    s["height"] = _clamp_int(s.get("height", 768), 64, 4096)
    s["fps"] = _clamp_int(s.get("fps", 30), 8, 120)
    s["duration"] = _clamp_int(s.get("duration", 12), 1, 300)

    # 枚举校验
    if s.get("theme") not in THEMES:
        s["theme"] = "empty"
    if s.get("global_postprocess") not in POSTPROCESSES:
        s["global_postprocess"] = ""

    # 调色板白名单
    pb = [p for p in s.get("palette_bias", []) if p in KNOWN_PALETTES]
    if not pb:
        pb = ["Cool", "Ocean", "Turbo"]
    s["palette_bias"] = pb

    return s, issues


def _clamp_int(v, lo, hi):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return lo


# ─────────────────────────────────────────────────────────────────────────
# 可选 LLM 后端（DeepSeek，仅 env 激活）
# ─────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个程序化视频生成器的参数规划器。
根据用户的中文/英文自然语言描述，输出一个严格的 JSON（不要任何解释、不要 markdown 代码块）：

{
  "width": 768, "height": 768, "fps": 30, "duration": 12,
  "theme": "empty|starfield|gradient|aurora",
  "global_postprocess": ""|"vignette"|"glow"|"edge"|"vignette+glow",
  "palette_bias": ["Cool", "Ocean"],
  "layers": [
    {"source": "draw"|"pw",
     "keywords": ["BZReaction"],
     "blend_mode": "add", "opacity": 0.85,
     "params": {"palette": "Cool", "steps_per_frame": 5}}
  ]
}

约束：layers 数组长度 1~4；source 只能是 "draw"(网格CA/数学) 或 "pw"(粒子/流体/几何/3D/流场/噪声)；
palette 必须来自已知集合（Cool/Hot/Ocean/Sunset/Neon/Cyberpunk/Vaporwave/Rainbow/Thermal/Aurora/Turbo/Winter/Grayscale 等）；
根据描述选择合适的来源、情绪调色板、主题与运动速度。只输出 JSON。"""


def _call_deepseek(prompt: str, api_key: str) -> Optional[Dict]:
    """调用 DeepSeek chat 接口，返回解析后的 spec dict；任何失败返回 None。"""
    import json
    import urllib.request

    url = "https://api.deepseek.com/v1/chat/completions"
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.7,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:  # noqa: BLE001
        print(f"[LLM] DeepSeek 调用失败，回退本地引擎: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────────────────────────────────

def compose_from_prompt(prompt: str, registry,
                        seed: Optional[int] = None,
                        use_llm: bool = False
                        ) -> Tuple["VideoComposition", Dict]:
    """自然语言 → VideoComposition。

    返回 (composition, meta)，meta 含 engine / issues / spec，便于 CLI 打印诊断。
    """
    from random_composer import VideoComposition  # noqa: F401 (type only)

    composer = RandomComposer(registry)
    local_engine = LocalSemanticComposer()

    spec = None
    engine = "local"

    if use_llm:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if api_key:
            print(f"[LLM] 使用 DeepSeek 语义规划（prompt={prompt!r}）")
            llm_spec = _call_deepseek(prompt, api_key)
            if llm_spec:
                spec = llm_spec
                engine = "llm-deepseek"
            else:
                print("[LLM] 模型返回无效，回退本地引擎")
        else:
            print("[LLM] 未设置 DEEPSEEK_API_KEY，回退本地语义引擎")

    if spec is None:
        spec = local_engine.compose(prompt, seed=seed)

    # Critic 校验
    spec, issues = critic(spec)
    if any("HARD" in i for i in issues):
        print(f"[Critic] 硬失败，回退纯随机组合: {issues}")
        comp = composer.compose(seed=seed)
        return comp, {"engine": engine, "issues": issues, "spec": spec,
                      "fallback": "random"}

    comp = composer.build_from_spec(spec, seed=seed)
    return comp, {"engine": engine, "issues": issues, "spec": spec,
                  "fallback": None}
