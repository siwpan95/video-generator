"""
draw_renderer.py — draw/effects 网格CA效果渲染器

使用 numpy 实现网格演化 + PIL 渲染帧。
支持 draw/effects/*.json 中定义的大部分效果类型。

依赖：numpy, PIL
可选：scipy（卷积加速）, asteval（公式求值）
"""

import math
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# ── Numba JIT（可选）────────────────────────────────────────────
try:
    import numba as nb
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:  # 损坏/缺省均优雅降级到纯 Python 路径
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper
    prange = range

# ── scipy 卷积加速（可选，调研：比 np.roll 快 3-5×）────────
try:
    from scipy.ndimage import convolve as _scipy_convolve
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── 感知均匀色图（可选）────────────────────────────────────────
try:
    import palettes_extra
except ImportError:
    palettes_extra = None

# ── asteval 安全公式求值（可选）────────────────────────────────
try:
    import formula_eval
except ImportError:
    formula_eval = None

# 卷积内核（与原始 np.roll 邻居定义一致：
#   _NEIGH4 / _LAP 对应原始 4 邻；_NEIGH8 对应生命游戏 8 邻）
_NEIGH4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
_NEIGH8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float64)
_LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

# ── 调色板 ID（Numba 兼容，避免字符串匹配）────────────────
_PALETTE_COOL = 0
_PALETTE_HOT = 1
_PALETTE_GRAYSCALE = 2
_PALETTE_WINTER = 3
_PALETTE_RAINBOW = 4
_PALETTE_OCEAN = 5
_PALETTE_SUNSET = 6
_PALETTE_THERMAL = 7
_PALETTE_ICE = 8
_PALETTE_PURPLE = 9
_PALETTE_WARM = 10
_PALETTE_NEON = 11
_PALETTE_OTHER = 99

# ── 调色板函数 ──────────────────────────────────────────────────────

def apply_palette(data: np.ndarray, palette_name: str) -> np.ndarray:
    """将归一化网格数据 [0,1] 映射为 RGB uint8 [w,h,3]"""
    v = np.clip(data, 0.0, 1.0)
    h, w = v.shape

    # 感知均匀色图（可选，联网调研：viridis/plasma/inferno/magma/cividis
    # 感知均匀，避免 jet 伪彩的条带与色觉缺陷问题）
    if palettes_extra is not None and palettes_extra.is_perceptual(palette_name):
        return palettes_extra.apply_perceptual(v, palette_name)

    pal = palette_name.lower()

    if "cool" in pal:
        r = 1.0 - v
        g = v
        b = np.ones_like(v) * 0.5
    elif "hot" in pal:
        r = np.clip(v * 3.0, 0, 1)
        g = np.clip((v - 0.333) * 3.0, 0, 1)
        b = np.clip((v - 0.666) * 3.0, 0, 1)
    elif "grayscale" in pal or "gray" in pal:
        r = g = b = v
    elif "winter" in pal:
        r = v * 0.2
        g = v * 0.6
        b = 0.8 + v * 0.2
    elif "rainbow" in pal or "neon" in pal:
        r = 0.5 + 0.5 * np.sin(v * 12.0)
        g = 0.5 + 0.5 * np.sin(v * 12.0 + 2.094)
        b = 0.5 + 0.5 * np.sin(v * 12.0 + 4.189)
    elif "ocean" in pal:
        r = v * 0.1
        g = v * 0.6 + 0.2 * np.sin(v * 15.0)
        b = v * 0.8 + 0.2
    elif "sunset" in pal or "fire" in pal:
        r = v * 0.9 + (1 - v) * 0.05
        g = v * 0.3 + (1 - v) * 0.05
        b = v * 0.1 + (1 - v) * 0.3
    elif "thermal" in pal:
        r = v
        g = v * 0.5
        b = 0.5 + 0.5 * np.sin(v * 10.0)
    elif "ice" in pal:
        r = v * 0.3
        g = v * 0.8
        b = 1.0 - v * 0.4
    elif "purple" in pal:
        r = v * 0.5
        g = v * 0.2
        b = 0.5 + v * 0.5
    elif "warm" in pal:
        r = 0.8 + v * 0.2
        g = 0.4 + v * 0.4
        b = v * 0.3
    elif "neon" in pal:
        r = v * 0.5 + 0.5 * np.sin(v * 25.0)
        g = v * 0.5 + 0.5 * np.sin(v * 25.0 + 2.094)
        b = v * 0.8 + 0.2 * np.sin(v * 20.0)
    else:
        # 默认：灰度
        r = g = b = v

    rgb = np.stack([
        np.clip(r * 255, 0, 255).astype(np.uint8),
        np.clip(g * 255, 0, 255).astype(np.uint8),
        np.clip(b * 255, 0, 255).astype(np.uint8),
    ], axis=-1)  # (h,w,3)
    return rgb


# ── 网格演化规则 ──────────────────────────────────────────────────────

class GridEvolver:
    """网格演化器（简化版，支持常见 CA 规则）"""

    @staticmethod
    def bz_reaction(grid: np.ndarray, steps: int = 1) -> np.ndarray:
        """BZ 反应（简化版 Belousov-Zhabotinsky）"""
        h, w = grid.shape
        result = grid.copy()
        for _ in range(steps):
            # 简化 BZ: 每个细胞根据邻居状态更新
            neighbors = np.roll(result, 1, 0) + np.roll(result, -1, 0) + \
                       np.roll(result, 1, 1) + np.roll(result, -1, 1)
            neighbors /= 4.0
            # BZ 规则（简化）
            new = np.where(result > 0.5, neighbors * 0.5, 1.0 - neighbors * 0.5)
            # 添加随机扰动
            new += np.random.rand(h, w) * 0.02
            result = np.clip(new, 0.0, 1.0)
        return result

    @staticmethod
    def diffusion(grid: np.ndarray, steps: int = 1, d: float = 0.1) -> np.ndarray:
        """扩散（拉普拉斯平滑）"""
        result = grid.copy()
        for _ in range(steps):
            # 四邻拉普拉斯
            laplacian = (
                np.roll(result, 1, 0) + np.roll(result, -1, 0) +
                np.roll(result, 1, 1) + np.roll(result, -1, 1) - 4 * result
            )
            result = np.clip(result + d * laplacian, 0.0, 1.0)
        return result

    @staticmethod
    def life(grid: np.ndarray, steps: int = 1) -> np.ndarray:
        """生命游戏（Game of Life）"""
        result = (grid > 0.5).astype(np.float64)
        h, w = result.shape
        for _ in range(steps):
            # 统计活邻居
            neighbors = (
                np.roll(result, 1, 0) + np.roll(result, -1, 0) +
                np.roll(result, 1, 1) + np.roll(result, -1, 1) +
                np.roll(np.roll(result, 1, 0), 1, 1) +
                np.roll(np.roll(result, 1, 0), -1, 1) +
                np.roll(np.roll(result, -1, 0), 1, 1) +
                np.roll(np.roll(result, -1, 0), -1, 1)
            )
            # 生命游戏规则
            new = np.where(
                result == 1.0,
                ((neighbors == 2) | (neighbors == 3)).astype(np.float64),
                (neighbors == 3).astype(np.float64)
            )
            result = new
        return result

    @staticmethod
    def wave(grid: np.ndarray, steps: int = 1, c: float = 0.5) -> np.ndarray:
        """波传播（波动方程简化）"""
        result = grid.copy()
        prev = result.copy()
        for _ in range(steps):
            laplacian = (
                np.roll(result, 1, 0) + np.roll(result, -1, 0) +
                np.roll(result, 1, 1) + np.roll(result, -1, 1) - 4 * result
            )
            # 波动方程：u(t+1) = 2*u(t) - u(t-1) + c^2 * laplacian
            new = 2 * result - prev + (c ** 2) * laplacian
            prev = result.copy()
            result = np.clip(new, 0.0, 1.0)
        return result

    @staticmethod
    def spiral(grid: np.ndarray, steps: int = 1, speed: float = 0.1) -> np.ndarray:
        """螺旋波（简化版）"""
        result = grid.copy()
        h, w = result.shape
        cx, cy = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        angle = np.arctan2(y - cy, x - cx)
        radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        for _ in range(steps):
            # 螺旋波：角度 + 半径调制
            wave_front = np.sin(angle * 3 + radius * 0.3 - speed * _ * 10)
            result = (result + 0.1 * wave_front) % 1.0
        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def random_walk(grid: np.ndarray, steps: int = 1, p: float = 0.01) -> np.ndarray:
        """随机游走 / 噪声"""
        result = grid.copy()
        h, w = result.shape
        for _ in range(steps):
            noise = (np.random.rand(h, w) < p).astype(np.float64)
            result = np.clip(result + noise * 0.5, 0.0, 1.0)
            # 轻微扩散
            result = GridEvolver.diffusion(result, 1, 0.05)
        return result


# ── scipy 卷积加速演化（可选，联网调研：比 np.roll 快 3-5×）────────
if _HAS_SCIPY:
    def bz_reaction_scipy(grid, steps=1):
        """BZ 反应 — scipy 卷积加速版（与原 bz_reaction 等价，mode='wrap'）"""
        result = grid.copy()
        h, w = grid.shape
        for _ in range(steps):
            neigh = _scipy_convolve(grid, _NEIGH4, mode='wrap') / 4.0
            new = np.where(result > 0.5, neigh * 0.5, 1.0 - neigh * 0.5)
            new = new + np.random.rand(h, w) * 0.02
            result = np.clip(new, 0.0, 1.0)
        return result

    def diffusion_scipy(grid, steps=1, d=0.1):
        """扩散 — scipy 卷积加速版（与原 diffusion 等价）"""
        result = grid.copy()
        for _ in range(steps):
            lap = _scipy_convolve(result, _LAP, mode='wrap')
            result = np.clip(result + d * lap, 0.0, 1.0)
        return result

    def life_scipy(grid, steps=1):
        """生命游戏 — scipy 卷积加速版（与原 life 等价）"""
        result = (grid > 0.5).astype(np.float64)
        for _ in range(steps):
            neigh = _scipy_convolve(result, _NEIGH8, mode='wrap')
            new = np.where(
                result == 1.0,
                ((neigh == 2) | (neigh == 3)).astype(np.float64),
                (neigh == 3).astype(np.float64),
            )
            result = new
        return result

    def wave_scipy(grid, steps=1, c=0.5):
        """波传播 — scipy 卷积加速版（与原 wave 等价）"""
        result = grid.copy()
        prev = result.copy()
        for _ in range(steps):
            lap = _scipy_convolve(result, _LAP, mode='wrap')
            new = 2 * result - prev + (c ** 2) * lap
            prev = result.copy()
            result = np.clip(new, 0.0, 1.0)
        return result


# ── 后处理 ───────────────────────────────────────────────────────────

def apply_postprocess(rgb: np.ndarray, effect_name: str) -> np.ndarray:
    """应用后处理效果（vignette / edge / glow）"""
    h, w = rgb.shape[:2]
    result = rgb.copy()

    ename = effect_name.lower()

    if "vignette" in ename:
        # 暗角
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        vignette = 1.0 - 0.7 * (dist / max_dist) ** 2
        vignette = vignette[:, :, np.newaxis]
        result = (result * vignette).astype(np.uint8)

    if "edge" in ename or "glow" in ename:
        # 边缘发光（简化：用 PIL 的 edge enhance）
        img = Image.fromarray(result)
        if "edge" in ename:
            img = img.filter(ImageFilter.EDGE_ENHANCE)
        if "glow" in ename:
            # 简单辉光：模糊后叠加
            glow = img.filter(ImageFilter.GaussianBlur(radius=3))
            img = Image.blend(img, glow, 0.3)
        result = np.array(img)

    return result


# ── Numba JIT 加速函数 ─────────────────────────────────────────────
#    以下函数在 HAS_NUMBA=True 时被 JIT 编译，否则直接执行

def _palette_name_to_id(name: str) -> int:
    """把调色板名称转为整数 ID（Numba 兼容）"""
    nl = name.lower()
    if "cool" in nl:
        return _PALETTE_COOL
    if "hot" in nl:
        return _PALETTE_HOT
    if "gray" in nl or "grey" in nl:
        return _PALETTE_GRAYSCALE
    if "winter" in nl:
        return _PALETTE_WINTER
    if "rainbow" in nl or "neon" in nl:
        return _PALETTE_RAINBOW
    if "ocean" in nl:
        return _PALETTE_OCEAN
    if "sunset" in nl or "fire" in nl:
        return _PALETTE_SUNSET
    if "thermal" in nl:
        return _PALETTE_THERMAL
    if "ice" in nl:
        return _PALETTE_ICE
    if "purple" in nl:
        return _PALETTE_PURPLE
    if "warm" in nl:
        return _PALETTE_WARM
    if "neon" in nl:
        return _PALETTE_NEON
    return _PALETTE_OTHER


if _HAS_NUMBA:
    @nb.njit(cache=True)
    def _apply_palette_nb(data, palette_id):
        """JIT 调色板映射（避免字符串操作，纯数值计算）"""
        h, w = data.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                v = data[i, j]
                if palette_id == _PALETTE_COOL:
                    r = (1.0 - v) * 255
                    g = v * 255
                    b = 128.0
                elif palette_id == _PALETTE_HOT:
                    r = min(max(v * 3.0, 0.0), 1.0) * 255
                    g = min(max((v - 0.333) * 3.0, 0.0), 1.0) * 255
                    b = min(max((v - 0.666) * 3.0, 0.0), 1.0) * 255
                elif palette_id == _PALETTE_GRAYSCALE:
                    r = g = b = v * 255
                elif palette_id == _PALETTE_WINTER:
                    r = v * 0.2 * 255
                    g = v * 0.6 * 255
                    b = min(1.0, 0.8 + v * 0.2) * 255
                elif palette_id == _PALETTE_RAINBOW:
                    r = (0.5 + 0.5 * np.sin(v * 12.0)) * 255
                    g = (0.5 + 0.5 * np.sin(v * 12.0 + 2.094)) * 255
                    b = (0.5 + 0.5 * np.sin(v * 12.0 + 4.189)) * 255
                elif palette_id == _PALETTE_OCEAN:
                    r = v * 0.1 * 255
                    g = min(1.0, v * 0.6 + 0.2 * np.sin(v * 15.0)) * 255
                    b = min(1.0, v * 0.8 + 0.2) * 255
                elif palette_id == _PALETTE_SUNSET:
                    r = (v * 0.9 + (1.0 - v) * 0.05) * 255
                    g = (v * 0.3 + (1.0 - v) * 0.05) * 255
                    b = ((1.0 - v) * 0.3) * 255
                elif palette_id == _PALETTE_THERMAL:
                    r = v * 255
                    g = v * 0.5 * 255
                    b = min(1.0, 0.5 + 0.5 * np.sin(v * 10.0)) * 255
                elif palette_id == _PALETTE_ICE:
                    r = v * 0.3 * 255
                    g = v * 0.8 * 255
                    b = min(1.0, 1.0 - v * 0.4) * 255
                elif palette_id == _PALETTE_PURPLE:
                    r = v * 0.5 * 255
                    g = v * 0.2 * 255
                    b = min(1.0, 0.5 + v * 0.5) * 255
                elif palette_id == _PALETTE_WARM:
                    r = min(1.0, 0.8 + v * 0.2) * 255
                    g = (0.4 + v * 0.4) * 255
                    b = v * 0.3 * 255
                elif palette_id == _PALETTE_NEON:
                    r = min(1.0, v * 0.5 + 0.5 * np.sin(v * 25.0)) * 255
                    g = min(1.0, v * 0.5 + 0.5 * np.sin(v * 25.0 + 2.094)) * 255
                    b = min(1.0, v * 0.8 + 0.2 * np.sin(v * 20.0)) * 255
                else:
                    r = g = b = v * 255
                r8 = int(min(max(r, 0.0), 255.0))
                g8 = int(min(max(g, 0.0), 255.0))
                b8 = int(min(max(b, 0.0), 255.0))
                rgb[i, j, 0] = r8
                rgb[i, j, 1] = g8
                rgb[i, j, 2] = b8
        return rgb

    @nb.njit(parallel=True, cache=True)
    def _convolve3_nb(grid, kernel):
        """3x3 卷积辅助（mode='wrap'，环形边界）。

        与 scipy.ndimage.convolve(..., mode='wrap') 数值等价：
        scipy 的 convolve 在核对称时即为不翻转的相关运算，本函数同样不翻转。
        供 Numba 路径下的网格演化复用，使 scipy 风格卷积也进入 JIT 路径，
        与 *_scipy 系列共用同一套 3x3 内核（_NEIGH4/_LAP/_NEIGH8）。"""
        h, w = grid.shape
        kh, kw = kernel.shape
        half_h = kh // 2
        half_w = kw // 2
        out = np.empty_like(grid)
        for i in nb.prange(h):
            for j in range(w):
                s = 0.0
                for ki in range(kh):
                    for kj in range(kw):
                        gi = (i + ki - half_h) % h
                        gj = (j + kj - half_w) % w
                        s += grid[gi, gj] * kernel[ki, kj]
                out[i, j] = s
        return out

    @nb.njit(parallel=True, cache=True)
    def _bz_step_nb(grid, steps, noise):
        """BZ 反应 — JIT 编译（scipy 风格卷积版，与 bz_reaction_scipy 等价）"""
        result = grid.copy()
        for _ in range(steps):
            neigh = _convolve3_nb(result, _NEIGH4) / 4.0
            new = np.where(result > 0.5, neigh * 0.5, 1.0 - neigh * 0.5)
            new = np.clip(new + noise * 0.02, 0.0, 1.0)
            result = new
        return result

    @nb.njit(parallel=True, cache=True)
    def _diffusion_step_nb(grid, steps, d):
        """扩散 — JIT 编译（scipy 风格卷积版，与 diffusion_scipy 等价）"""
        result = grid.copy()
        for _ in range(steps):
            lap = _convolve3_nb(result, _LAP)
            result = np.clip(result + d * lap, 0.0, 1.0)
        return result

    @nb.njit(parallel=True, cache=True)
    def _life_step_nb(grid, steps):
        """生命游戏 — JIT 编译（scipy 风格卷积版，与 life_scipy 等价）"""
        result = (grid > 0.5).astype(np.float64)
        for _ in range(steps):
            neigh = _convolve3_nb(result, _NEIGH8)
            new = np.where(
                result == 1.0,
                ((neigh == 2.0) | (neigh == 3.0)).astype(np.float64),
                (neigh == 3.0).astype(np.float64),
            )
            result = new
        return result

    @nb.njit(parallel=True, cache=True)
    def _wave_step_nb(grid, steps, c):
        """波传播 — JIT 编译（scipy 风格卷积版，与 wave_scipy 等价）"""
        result = grid.copy()
        prev = result.copy()
        c2 = c * c
        for _ in range(steps):
            lap = _convolve3_nb(result, _LAP)
            new = np.clip(2.0 * result - prev + c2 * lap, 0.0, 1.0)
            prev = result.copy()
            result = new
        return result

    @nb.njit(parallel=True, cache=True)
    def _spiral_step_nb(grid, steps, speed, cx_f, cy_f):
        """螺旋波 — JIT 编译（cx_f, cy_f 是归一化中心）"""
        h, w = grid.shape
        result = grid.copy()
        for _ in range(steps):
            new = np.empty_like(result)
            for i in nb.prange(h):
                for j in range(w):
                    angle = np.arctan2(float(i) - cy_f * float(h),
                                        float(j) - cx_f * float(w))
                    radius = np.sqrt((float(j) - cx_f * float(w)) ** 2 +
                                   (float(i) - cy_f * float(h)) ** 2)
                    wave_front = np.sin(angle * 3.0 + radius * 0.3 - speed * float(_) * 10.0)
                    new[i, j] = min(max(result[i, j] + 0.1 * wave_front, 0.0), 1.0)
            result = new
        return result

else:
    # 无 Numba：用普通 NumPy 实现（接口保持一致）
    def _apply_palette_nb(data, palette_id):
        return apply_palette(data, "Cool")  # fallback 忽略 ID

    def _bz_step_nb(grid, steps, noise):
        return GridEvolver.bz_reaction(grid, steps)

    def _diffusion_step_nb(grid, steps, d):
        return GridEvolver.diffusion(grid, steps, d)

    def _life_step_nb(grid, steps):
        return GridEvolver.life(grid, steps)

    def _wave_step_nb(grid, steps, c):
        return GridEvolver.wave(grid, steps, c)

    def _spiral_step_nb(grid, steps, speed, cx_f, cy_f):
        return GridEvolver.spiral(grid, steps, speed)
class DrawEffectRenderer:
    """draw/effects JSON 渲染器"""

    def __init__(self, width: int = 512, height: int = 512,
                 grid_w: int = 64, grid_h: int = 64):
        self.width = width
        self.height = height
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.grid = None
        self.t = 0
        self.effect_name = ""
        self.palette = "Rainbow"
        self.evolution_type = "bz"  # bz / diffusion / life / wave / spiral
        self.postprocess = ""
        # Numba JIT 加速
        self._palette_id = _PALETTE_RAINBOW
        self._use_jit = _HAS_NUMBA

    def load_config(self, config: dict):
        """从 draw effect JSON 加载配置"""
        grid = config.get("grid", {})
        self.grid_w = grid.get("width", self.grid_w)
        self.grid_h = grid.get("height", self.grid_h)

        # 初始化网格
        init_cfg = grid.get("initial", {})
        self._init_grid(init_cfg)

        # 颜色
        color = config.get("color", {})
        self.palette = self._guess_palette_from_color(color)
        self._palette_id = _palette_name_to_id(self.palette)

        # 后处理
        post = config.get("postprocess", {})
        self.postprocess = post.get("effect", "")

        # 演化规则（从 evolution.rule 猜测类型）
        evo = config.get("evolution", {})
        rule = evo.get("rule", "")
        self.evolution_type = self._guess_evolution_type(rule)

        self.effect_name = config.get("name", "unknown")

    def _init_grid(self, init_cfg: dict):
        """初始化网格"""
        init_type = init_cfg.get("type", "random")
        h, w = self.grid_h, self.grid_w

        if init_type == "random":
            self.grid = np.random.rand(h, w)
        elif init_type == "formula":
            expr = init_cfg.get("expression", "0.5")
            self.grid = self._eval_formula_grid(expr, h, w)
        elif init_type == "center_dot":
            self.grid = np.zeros((h, w))
            cy, cx = h // 2, w // 2
            radius = init_cfg.get("radius", 3)
            y, x = np.ogrid[:h, :w]
            mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2
            self.grid = np.zeros((h, w))
            self.grid[mask] = 1.0
        else:
            self.grid = np.random.rand(h, w)

    def _eval_formula_grid(self, expr: str, h: int, w: int) -> np.ndarray:
        """公式初始化网格（优先 asteval 安全求值，失败回退原简化逻辑）

        联网调研：asteval 基于 Python ast 的受限解释器，自动导入 numpy
        多数数学函数，支持 if 表达式/条件，比裸 eval() 安全得多。
        """
        y, x = np.ogrid[:h, :w]
        x_norm = x / max(w - 1, 1)
        y_norm = y / max(h - 1, 1)

        # 优先走 asteval 安全求值（支持嵌套条件、三角函数、cos/sin 等）
        if formula_eval is not None and formula_eval.is_available():
            try:
                r = np.clip(
                    formula_eval.eval_grid_expr(expr, x_norm, y_norm),
                    0.0, 1.0,
                )
                if r.shape == (h, w):
                    return r
                # 形状异常（如条件表达式未向量化）→ 落入下方回退路径
            except Exception:
                # 求值失败（如表达式含未知符号），回退到下方简化路径
                pass

        # 回退：极简支持（与原始行为一致）
        try:
            e = expr.replace("x", "x_norm").replace("y", "y_norm")
            e = e.replace("0.5", "0.5*np.ones((h,w))")
            if "if" in e or "else" in e:
                import re
                m = re.match(r"([0-9.]+)\s*if\s*(.+)\s*else\s*(.+)", e)
                if m:
                    val_true = float(m.group(1))
                    return np.where(
                        (x_norm - 0.5) ** 2 + (y_norm - 0.5) ** 2 < 0.25,
                        val_true, 0.0
                    )
            return np.random.rand(h, w)
        except Exception:
            return np.random.rand(h, w)

    def _guess_palette_from_color(self, color: dict) -> str:
        """从 color 配置猜测调色板"""
        r = color.get("red", "v").lower()
        g = color.get("green", "v").lower()
        b = color.get("blue", "v").lower()
        if "sin" in r or "sin" in g or "sin" in b:
            return "Rainbow"
        if "1-v" in r:
            return "Invert"
        return "Rainbow"  # 默认

    def _guess_evolution_type(self, rule: str) -> str:
        """从规则字符串猜测演化类型"""
        rl = rule.lower()
        if "bz" in rl or "reaction" in rl:
            return "bz"
        if "diffusion" in rl or "laplacian" in rl:
            return "diffusion"
        if "life" in rl or "neighbor" in rl:
            return "life"
        if "wave" in rl or "propagat" in rl:
            return "wave"
        if "spiral" in rl or "annulus" in rl:
            return "spiral"
        return "bz"  # 默认

    def step(self, n: int = 1):
        """演化 n 步（优先 Numba JIT）"""
        if self.grid is None:
            return
        if self._use_jit and _HAS_NUMBA:
            # JIT 路径：一次调用处理 n 步
            if self.evolution_type == "bz":
                noise = np.random.rand(*self.grid.shape)
                self.grid = _bz_step_nb(self.grid, n, noise)
            elif self.evolution_type == "diffusion":
                self.grid = _diffusion_step_nb(self.grid, n, 0.1)
            elif self.evolution_type == "life":
                self.grid = _life_step_nb(self.grid, n)
            elif self.evolution_type == "wave":
                self.grid = _wave_step_nb(self.grid, n, 0.5)
            elif self.evolution_type == "spiral":
                self.grid = _spiral_step_nb(
                    self.grid, n, 0.1, 0.5, 0.5)
            else:
                for _ in range(n):
                    self.grid = GridEvolver.random_walk(
                        self.grid, 1, 0.01)
        else:
            for _ in range(n):
                et = self.evolution_type
                if _HAS_SCIPY and et in ("bz", "diffusion", "life", "wave"):
                    # scipy 卷积加速路径（与原 np.roll 逻辑等价，速度更快）
                    if et == "bz":
                        self.grid = bz_reaction_scipy(self.grid, 1)
                    elif et == "diffusion":
                        self.grid = diffusion_scipy(self.grid, 1, 0.1)
                    elif et == "life":
                        self.grid = life_scipy(self.grid, 1)
                    elif et == "wave":
                        self.grid = wave_scipy(self.grid, 1, 0.5)
                else:
                    # 原 np.roll 路径（无 scipy 时，或 spiral/random_walk）
                    if et == "bz":
                        self.grid = GridEvolver.bz_reaction(self.grid, 1)
                    elif et == "diffusion":
                        self.grid = GridEvolver.diffusion(self.grid, 1, 0.1)
                    elif et == "life":
                        self.grid = GridEvolver.life(self.grid, 1)
                    elif et == "wave":
                        self.grid = GridEvolver.wave(self.grid, 1, 0.5)
                    elif et == "spiral":
                        self.grid = GridEvolver.spiral(self.grid, 1, 0.1)
                    else:
                        self.grid = GridEvolver.random_walk(self.grid, 1, 0.01)
        self.t += n

    def render_frame(self) -> Image.Image:
        """渲染当前帧为 PIL Image"""
        if self.grid is None:
            img = Image.new("RGB", (self.width, self.height), (0, 0, 0))
            return img

        # 调色板映射（优先 Numba JIT）
        if self._use_jit and _HAS_NUMBA:
            rgb = _apply_palette_nb(self.grid, self._palette_id)
        else:
            rgb = apply_palette(self.grid, self.palette)

        # 缩放到目标尺寸
        from PIL import Image as PILImage
        img = PILImage.fromarray(rgb)
        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height), PILImage.NEAREST)

        # 后处理
        if self.postprocess:
            rgb = apply_postprocess(np.array(img), self.postprocess)
            img = PILImage.fromarray(rgb)

        return img


# ── 测试 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import os

    # 测试：加载一个 draw effect 并渲染几帧
    draw_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "draw", "effects"
    )

    # 找一个 BZReaction 效果
    test_file = None
    for f in os.listdir(draw_dir):
        if "BZReaction" in f and f.endswith(".json"):
            test_file = os.path.join(draw_dir, f)
            break

    if test_file:
        print(f"测试渲染: {test_file}")
        with open(test_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        renderer = DrawEffectRenderer(width=512, height=512, grid_w=64, grid_h=64)
        renderer.load_config(config)

        frames = []
        for i in range(60):
            renderer.step(5)
            frame = renderer.render_frame()
            frames.append(frame)
            if i % 10 == 0:
                print(f"  帧 {i}")

        # 保存为 GIF（前20帧）
        out_path = os.path.join(os.path.dirname(__file__), "test_draw.gif")
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:20],
            duration=50,
            loop=0
        )
        print(f"测试 GIF 已保存: {out_path}")
    else:
        print("未找到测试文件")
