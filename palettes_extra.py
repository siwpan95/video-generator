"""
palettes_extra.py — 感知均匀 (perceptually-uniform) 色图

自包含、无外部依赖：用公开的 viridis / plasma / inferno /
magma / cividis 锚点（来自 Matplotlib 官方采样）线性插值生成
256 级 RGB 查找表 (LUT)。

为什么用这些而不是 Rainbow/Jet？
  - 联网调研 (matplotlib.org 官方 Colormaps 教程、IBM / Crameri 等)
    一致结论：Jet/Rainbow 不是感知均匀的，等数值步长看起来不等视觉
    步长，且亮度非单调，会产生虚假"边界"与条带 (banding)；
    约 1/12 男性有色觉缺陷 (CVD) 无法区分其色相。
  - viridis 家族亮度单调递增，灰阶可读，过 CVD 模拟，是连续数据的
    默认首选。

提供：
  - PERCEPTUAL_NAMES: 支持的色图名列表
  - build_lut(name, n=256) -> (n,3) uint8 查找表
  - apply_perceptual(v, name) -> (H,W,3) uint8  （v ∈ [0,1]）
"""

from typing import Dict, List, Tuple

import numpy as np


# ── 官方采样锚点（位置 0..1 → (R,G,B) 0..255）─────────────
_ANCHORS: Dict[str, List[Tuple[float, int, int, int]]] = {
    # viridis
    "viridis": [
        (0.0, 68, 1, 84), (0.1, 72, 40, 120), (0.2, 62, 74, 137),
        (0.3, 49, 104, 142), (0.4, 38, 130, 142), (0.5, 31, 158, 137),
        (0.6, 53, 183, 121), (0.7, 110, 206, 88), (0.8, 181, 222, 43),
        (0.9, 245, 229, 27), (1.0, 253, 231, 37),
    ],
    # plasma
    "plasma": [
        (0.0, 13, 8, 135), (0.1, 75, 3, 161), (0.2, 125, 3, 168),
        (0.3, 168, 34, 150), (0.4, 203, 70, 121), (0.5, 229, 107, 93),
        (0.6, 248, 148, 65), (0.7, 253, 195, 40), (0.8, 245, 232, 45),
        (0.9, 240, 249, 33), (1.0, 240, 249, 33),
    ],
    # inferno
    "inferno": [
        (0.0, 0, 0, 4), (0.1, 22, 11, 57), (0.2, 66, 10, 104),
        (0.3, 106, 23, 110), (0.4, 147, 38, 103), (0.5, 188, 55, 84),
        (0.6, 221, 81, 58), (0.7, 243, 121, 45), (0.8, 240, 165, 38),
        (0.9, 249, 214, 76), (1.0, 252, 255, 164),
    ],
    # magma
    "magma": [
        (0.0, 0, 0, 4), (0.1, 28, 16, 68), (0.2, 79, 18, 123),
        (0.3, 129, 37, 129), (0.4, 181, 54, 122), (0.5, 229, 80, 100),
        (0.6, 233, 92, 75), (0.7, 245, 157, 101), (0.8, 250, 204, 144),
        (0.9, 252, 241, 191), (1.0, 252, 253, 191),
    ],
    # cividis（显式设计为色盲友好）
    "cividis": [
        (0.0, 0, 32, 76), (0.2, 43, 62, 119), (0.4, 121, 118, 163),
        (0.6, 196, 186, 207), (0.8, 236, 221, 168), (1.0, 255, 233, 69),
    ],
}

# 别名（兼容常见写法）
_ALIASES = {
    "vir": "viridis", "plas": "plasma", "inf": "inferno",
    "mag": "magma", "covid": "cividis", "civid": "cividis",
    "turbo": "turbo",   # 感知增强彩虹（Google 2019）
}

# ── Turbo（Google 2019, Anton Mikhailov；多项式近似 by Ruofei Du, Apache-2.0）
# 调研结论：Jet/Rainbow 不是感知均匀的，而 Turbo 在设计上保留了彩虹的
# 高区分度，同时亮度更平滑、条带更少，是可视化"连续标量 + 想要彩色"的
# 现代首选（取代 jet）。下面用官方 GLSL 多项式系数做向量化求值，
# 与 matplotlib/官网 LUT 数值一致。
_TURBO_R4 = (0.13572138, 4.61539260, -42.66032258, 132.13108234)
_TURBO_G4 = (0.09140261, 2.19418839, 4.84296658, -14.18503333)
_TURBO_B4 = (0.10667330, 12.64194608, -60.58204836, 110.36276771)
_TURBO_R2 = (-152.94239396, 59.28637943)
_TURBO_G2 = (4.27729857, 2.82956604)
_TURBO_B2 = (-89.90310912, 27.34824973)


def _turbo_lut(n: int = 256) -> np.ndarray:
    """构建 n 级 Turbo 查找表 (n,3) uint8（多项式近似）

    vec4 = (1, x, x^2, x^3); vec2 = (x^4, x^5)
    chan = dot(vec4, kX4) + dot(vec2, kX2)，再 *255。
    """
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    v4 = np.stack([np.ones_like(x), x, x * x, x * x * x], axis=-1)   # (n,4)
    v2 = np.stack([x * x * x * x, x * x * x * x * x], axis=-1)        # (n,2)
    r = v4 @ np.array(_TURBO_R4) + v2 @ np.array(_TURBO_R2)
    g = v4 @ np.array(_TURBO_G4) + v2 @ np.array(_TURBO_G2)
    b = v4 @ np.array(_TURBO_B4) + v2 @ np.array(_TURBO_B2)
    lut = np.stack([r, g, b], axis=-1)
    return np.clip(lut * 255.0, 0, 255).astype(np.uint8)


PERCEPTUAL_NAMES: List[str] = list(_ANCHORS.keys()) + ["turbo"]


def _normalize_name(name: str) -> str:
    n = name.lower().strip()
    if n in _ALIASES:
        return _ALIASES[n]
    return n


def is_perceptual(name: str) -> bool:
    """名称是否属于感知均匀色图（含 turbo）"""
    key = _normalize_name(name)
    return key in _ANCHORS or key == "turbo"


def build_lut(name: str, n: int = 256) -> np.ndarray:
    """构建 n 级 RGB 查找表 (n,3) uint8

    在锚点之间做线性插值；锚点已覆盖 0..1 全覆盖。
    turbo 走独立的多项式近似路径。
    """
    key = _normalize_name(name)
    if key == "turbo":
        return _turbo_lut(n)
    if key not in _ANCHORS:
        raise KeyError(f"未知感知色图: {name}")
    anchors = _ANCHORS[key]
    positions = np.array([a[0] for a in anchors], dtype=np.float64)
    r = np.array([a[1] for a in anchors], dtype=np.float64)
    g = np.array([a[2] for a in anchors], dtype=np.float64)
    b = np.array([a[3] for a in anchors], dtype=np.float64)

    x = np.linspace(0.0, 1.0, n)
    # 边界外钳制到首/尾锚点
    r_i = np.interp(x, positions, r)
    g_i = np.interp(x, positions, g)
    b_i = np.interp(x, positions, b)
    lut = np.stack([r_i, g_i, b_i], axis=-1)
    return np.clip(lut, 0, 255).astype(np.uint8)


# 进程级 LUT 缓存（构建一次即可复用）
_LUT_CACHE: Dict[str, np.ndarray] = {}


def _get_lut(name: str, n: int = 256) -> np.ndarray:
    key = f"{_normalize_name(name)}:{n}"
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = build_lut(name, n)
    return _LUT_CACHE[key]


def apply_perceptual(v: np.ndarray, name: str) -> np.ndarray:
    """把归一化数据 v ∈ [0,1] 映射为 RGB uint8 (H,W,3)

    v: (H,W) float 数组（会被钳制到 [0,1]）
    返回: (H,W,3) uint8

    实现用 LUT 索引 → 单条对 Python 循环无关，速度极快。
    """
    v = np.asarray(v, dtype=np.float32)
    v = np.clip(v, 0.0, 1.0)
    lut = _get_lut(name, 256)
    idx = (v * 255.0).astype(np.int32)
    idx = np.clip(idx, 0, 255)
    return lut[idx]


if __name__ == "__main__":
    # 简单自测
    for nm in PERCEPTUAL_NAMES:
        lut = build_lut(nm)
        assert lut.shape == (256, 3), nm
        assert lut.dtype == np.uint8, nm
    # 渲染一张渐变条验证
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    grad = np.repeat(grad, 32, axis=0)
    img = apply_perceptual(grad, "viridis")
    from PIL import Image
    Image.fromarray(img).save("test_perceptual_viridis.png")
    print("[palettes_extra] 自测通过，色图:", PERCEPTUAL_NAMES)
