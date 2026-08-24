"""
pw_renderer.py — pywallpaper/effects 效果渲染器（简化版）

用 PIL.ImageDraw 实现常用原语的渲染。
支持 pywallpaper/effects/*.json 中定义的图层效果。

依赖：PIL, numpy, math
"""

import math
import random
import inspect
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import List, Dict, Tuple, Optional

# ── 安全导入新增模块（自包含，无外部依赖）──────────────
try:
    import noise_field
except ImportError:
    noise_field = None
try:
    import wireframe3d
except ImportError:
    wireframe3d = None

# ── P1-2 新增程序化原语（自包含，零外部依赖）──────────────
try:
    import primitives_pro
    _HAS_PRIMITIVES_PRO = True
except ImportError:
    primitives_pro = None
    _HAS_PRIMITIVES_PRO = False

# ── Numba JIT（可选）────────────────────────────────────────────
try:
    import numba as nb
    from numba import njit, prange
    _HAS_NUMBA_PW = True
except Exception:  # 损坏/缺省均优雅降级到纯 Python 路径
    _HAS_NUMBA_PW = False
    def njit(*args, **kwargs):
        def wrapper(func): return func
        return wrapper
    prange = range

# ── 直接粒子渲染（Numba JIT，避免 PIL 逐粒子 Python 调用）────────
if _HAS_NUMBA_PW:
    @njit(parallel=True, cache=True)
    def _render_particles_nb(canvas, x_arr, y_arr,
                            size_arr, r_arr, g_arr, b_arr, count):
        """直接把粒子写进 canvas NumPy 数组（无 PIL）

        r/g/b_arr 已经乘过 alpha，直接叠加即可。
        """
        h, w = canvas.shape[0], canvas.shape[1]
        for idx in prange(count):
            xc = int(x_arr[idx] + 0.5)
            yc = int(y_arr[idx] + 0.5)
            s = int(size_arr[idx] + 0.5)
            if s < 1:
                s = 1
            r = r_arr[idx]
            g = g_arr[idx]
            b = b_arr[idx]
            s2 = s * s
            for dy in range(-s, s + 1):
                ddy = dy * dy
                for dx in range(-s, s + 1):
                    if dx*dx + ddy > s2:
                        continue
                    ny = yc + dy
                    nx = xc + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        vr = canvas[ny, nx, 0] + r
                        vg = canvas[ny, nx, 1] + g
                        vb = canvas[ny, nx, 2] + b
                        canvas[ny, nx, 0] = vr if vr < 255 else 255
                        canvas[ny, nx, 1] = vg if vg < 255 else 255
                        canvas[ny, nx, 2] = vb if vb < 255 else 255
        return canvas
else:
    def _render_particles_nb(*args, **kwargs):
        return None  # fallback 未使用


# ── 颜色生成器（简化版 pywallpaper/engine/color_system.py） ──────────

class SimpleColorGen:
    """简化颜色生成器"""

    def __init__(self, config: dict):
        self.config = config
        self.type = config.get("type", "solid")
        self.hue_range = config.get("hue", [0.0, 1.0])
        self.sat = config.get("sat", 0.8)
        self.val = config.get("val", 0.9)
        self.speed = config.get("speed", 0.02)
        self.time = 0.0

    def get_color(self, index: int = 0, total: int = 1,
                  t: float = None) -> Tuple[int, int, int]:
        """返回 (R,G,B) uint8"""
        t = self.time if t is None else t
        if self.type == "hsv_flow":
            hue = (self.hue_range[0] +
                    (self.hue_range[1] - self.hue_range[0]) *
                    ((t * self.speed + index / max(total, 1)) % 1.0))
            return hsv_to_rgb(hue, self.sat, self.val)
        elif self.type == "rainbow":
            hue = (index / max(total, 1) + t * 0.01) % 1.0
            return hsv_to_rgb(hue, 0.8, 0.9)
        elif self.type == "solid":
            r = self.config.get("r", 200)
            g = self.config.get("g", 200)
            b = self.config.get("b", 255)
            return (r, g, b)
        else:
            # 默认随机色
            hue = (index / max(total, 1) + t * 0.005) % 1.0
            return hsv_to_rgb(hue, 0.7, 0.85)


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """HSV → RGB（h,s,v ∈ [0,1]）"""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


# ── 粒子系统（NumPy 数组存储 + Numba JIT 渲染）─────────────────────

class SimpleParticleSystem:
    """简化粒子系统（NumPy 数组存储，Numba JIT 渲染）"""

    def __init__(self, config: dict, width: int, height: int):
        self.w = width
        self.h = height
        self.time = 0.0
        self._use_jit = _HAS_NUMBA_PW

        emitters_cfg = config.get("emitters", [])
        self.emitters: List[dict] = emitters_cfg if emitters_cfg else []
        self.max_particles = config.get("max_particles", 500)
        self.particle_type = config.get("particle_type", "circle")

        forces_cfg = config.get("forces", [])
        self.gravity_y = 0.0
        for f in forces_cfg:
            if f.get("type") == "gravity":
                self.gravity_y = f.get("y", 150)

        # 颜色
        color_cfg = config.get("color", {})
        self.color_gen = SimpleColorGen(color_cfg)

        # NumPy 数组存储粒子数据（预分配）
        self._max = self.max_particles
        self._count = 0  # 当前粒子数
        self._x = np.zeros(self._max, dtype=np.float64)
        self._y = np.zeros(self._max, dtype=np.float64)
        self._vx = np.zeros(self._max, dtype=np.float64)
        self._vy = np.zeros(self._max, dtype=np.float64)
        self._life = np.zeros(self._max, dtype=np.float64)
        self._age = np.zeros(self._max, dtype=np.float64)
        self._size = np.zeros(self._max, dtype=np.float64)
        self._color_idx = np.zeros(self._max, dtype=np.int64)

    def _spawn_particles(self):
        for em in self.emitters:
            rate = em.get("rate", 10)
            n = max(1, int(rate / 30))
            for _ in range(n):
                if self._count >= self._max:
                    break
                pos = em.get("position", [0.5, 0.5])
                x = pos[0] * self.w
                y = pos[1] * self.h
                speed_r = em.get("speed", [100, 200])
                dir_r = em.get("direction", [0, 360])
                lifetime_r = em.get("lifetime", [1.0, 3.0])
                size_r = em.get("size", [2, 6])

                speed = speed_r[0] + random.random() * (speed_r[1] - speed_r[0])
                angle = math.radians(dir_r[0] + random.random() * (dir_r[1] - dir_r[0]))
                idx = self._count
                self._x[idx] = x
                self._y[idx] = y
                self._vx[idx] = math.cos(angle) * speed
                self._vy[idx] = math.sin(angle) * speed
                self._life[idx] = lifetime_r[0] + random.random() * (lifetime_r[1] - lifetime_r[0])
                self._age[idx] = 0.0
                self._size[idx] = size_r[0] + random.random() * (size_r[1] - size_r[0])
                self._color_idx[idx] = idx % 8
                self._count += 1

    def update(self, dt: float):
        self.time += dt
        self._spawn_particles()

        if self._count == 0:
            return
        # 向量化更新
        age = self._age[:self._count]
        life = self._life[:self._count]
        vx = self._vx[:self._count]
        vy = self._vy[:self._count]
        alive_mask = age < life
        # 阻尼 + 重力
        vx *= 0.99
        vy *= 0.99
        vy += self.gravity_y * dt
        self._vx[:self._count] = vx
        self._vy[:self._count] = vy
        self._x[:self._count] += vx * dt
        self._y[:self._count] += vy * dt
        self._age[:self._count] += dt
        # 收集存活粒子到前面
        alive = np.where(alive_mask)[0]
        if len(alive) < self._count:
            self._x[:len(alive)] = self._x[alive]
            self._y[:len(alive)] = self._y[alive]
            self._vx[:len(alive)] = self._vx[alive]
            self._vy[:len(alive)] = self._vy[alive]
            self._life[:len(alive)] = self._life[alive]
            self._age[:len(alive)] = self._age[alive]
            self._size[:len(alive)] = self._size[alive]
            self._color_idx[:len(alive)] = self._color_idx[alive]
            self._count = len(alive)

    def draw(self, drawer: ImageDraw.ImageDraw):
        """渲染粒子到 PIL ImageDraw（fallback 路径，未被 PWEffecRenderer 使用）"""
        if self._count == 0:
            return
        self.color_gen.time = self.time
        for i in range(self._count):
            alpha = 1.0 - self._age[i] / max(self._life[i], 1e-6)
            r, g, b = self.color_gen.get_color(
                int(self._color_idx[i]), self._count, self.time)
            r = int(r * alpha)
            g = int(g * alpha)
            b = int(b * alpha)
            x, y = self._x[i], self._y[i]
            s = int(self._size[i])
            if self.particle_type == "circle":
                drawer.ellipse([x - s, y - s, x + s, y + s], fill=(r, g, b))
            elif self.particle_type == "square":
                drawer.rectangle([x - s, y - s, x + s, y + s], fill=(r, g, b))
            else:
                drawer.ellipse([x - s, y - s, x + s, y + s], fill=(r, g, b))

    def draw_to_array(self, canvas: np.ndarray) -> np.ndarray:
        """直接渲染粒子到 NumPy 数组（Numba JIT 加速）"""
        if self._count == 0:
            return canvas
        if not self._use_jit:
            # fallback：逐像素 Python 循环（慢）
            for i in range(self._count):
                alpha = 1.0 - self._age[i] / max(self._life[i], 1e-6)
                r, g, b = self.color_gen.get_color(
                    int(self._color_idx[i]), self._count, self.time)
                x = int(self._x[i] + 0.5)
                y = int(self._y[i] + 0.5)
                s = int(self._size[i] + 0.5)
                if s < 1:
                    s = 1
                for dy in range(-s, s + 1):
                    for dx in range(-s, s + 1):
                        if dx * dx + dy * dy > s * s:
                            continue
                        ny = y + dy
                        nx = x + dx
                        if 0 <= ny < self.h and 0 <= nx < self.w:
                            canvas[ny, nx, 0] = min(canvas[ny, nx, 0] + int(r * alpha), 255)
                            canvas[ny, nx, 1] = min(canvas[ny, nx, 1] + int(g * alpha), 255)
                            canvas[ny, nx, 2] = min(canvas[ny, nx, 2] + int(b * alpha), 255)
            return canvas
        # Numba JIT 路径：颜色已乘 alpha
        x_arr = self._x[:self._count].copy()
        y_arr = self._y[:self._count].copy()
        size_arr = self._size[:self._count].copy()
        r_arr = np.zeros(self._count, dtype=np.uint8)
        g_arr = np.zeros(self._count, dtype=np.uint8)
        b_arr = np.zeros(self._count, dtype=np.uint8)
        for i in range(self._count):
            alpha = 1.0 - self._age[i] / max(self._life[i], 1e-6)
            r, g, b = self.color_gen.get_color(
                int(self._color_idx[i]), self._count, self.time)
            r_arr[i] = int(r * alpha)
            g_arr[i] = int(g * alpha)
            b_arr[i] = int(b * alpha)
        canvas = _render_particles_nb(
            canvas, x_arr, y_arr, size_arr,
            r_arr, g_arr, b_arr, self._count)
        return canvas


# ── 波形绘制（简化版 pywallpaper bars/spikes） ───────────────────────

class SimpleWaveform:
    """简化波形绘制器"""

    def __init__(self, config: dict, width: int, height: int):
        self.w = width
        self.h = height
        self.time = 0.0
        self.n_bars = config.get("n_bars", 32)
        self.amplitude = config.get("amplitude", 80)
        self.color_gen = SimpleColorGen(config.get("color", {}))

    def update(self, dt: float):
        self.time += dt

    def draw(self, drawer: ImageDraw.ImageDraw):
        self.color_gen.time = self.time
        bar_w = self.w / max(self.n_bars, 1)
        for i in range(self.n_bars):
            phase = i / self.n_bars
            val = 0.5 + 0.5 * math.sin(self.time * 2.0 + phase * 6.28)
            bar_h = val * self.amplitude
            x0 = i * bar_w
            y0 = self.h / 2 - bar_h / 2
            r, g, b = self.color_gen.get_color(i, self.n_bars, self.time)
            drawer.rectangle(
                [x0, y0, x0 + bar_w - 1, y0 + bar_h],
                fill=(r, g, b)
            )


# ── 流体模拟（简化版） ─────────────────────────────────────────────────

class SimpleFluid:
    """简化流体效果 — 用 Perlin 噪声模拟"""

    def __init__(self, config: dict, width: int, height: int):
        self.w = width
        self.h = height
        self.time = 0.0
        self.color_gen = SimpleColorGen(config.get("color", {}))

    def update(self, dt: float):
        self.time += dt

    def draw(self, img: Image.Image) -> Image.Image:
        """直接修改 Image（像素级操作）"""
        self.color_gen.time = self.time
        arr = np.array(img)
        y, x = np.ogrid[:self.h, :self.w]
        t = self.time

        # 用正弦波模拟流体图案
        wave1 = np.sin(x * 0.02 + t * 0.5) * 0.5 + 0.5
        wave2 = np.cos(y * 0.03 + t * 0.3) * 0.5 + 0.5
        pattern = (wave1 + wave2) / 2.0

        r, g, b = self.color_gen.get_color(0, 1, t)
        arr[:, :, 0] = (pattern * r).astype(np.uint8)
        arr[:, :, 1] = (pattern * g).astype(np.uint8)
        arr[:, :, 2] = (pattern * b).astype(np.uint8)

        return Image.fromarray(arr)


# ── CA 网格（简化版，复用 draw_renderer） ───────────────────────────

class SimpleCAGrid:
    """简化 CA 网格渲染器"""

    def __init__(self, config: dict, width: int, height: int):
        self.w = width
        self.h = height
        self.time = 0.0

        grid_size = config.get("grid_size", [64, 64])
        self.grid_w = grid_size[0]
        self.grid_h = grid_size[1]
        self.grid = np.random.rand(self.grid_h, self.grid_w)
        self.color_gen = SimpleColorGen(config.get("color", {}))

    def update(self, dt: float):
        self.time += dt
        # 简化 BZ 演化
        neighbors = (
            np.roll(self.grid, 1, 0) + np.roll(self.grid, -1, 0) +
            np.roll(self.grid, 1, 1) + np.roll(self.grid, -1, 1)
        ) / 4.0
        self.grid = np.clip(
            np.where(self.grid > 0.5, neighbors * 0.5, 1.0 - neighbors * 0.5),
            0.0, 1.0
        )

    def draw(self, drawer: ImageDraw.ImageDraw):
        # 栅格化到画布
        import PIL.Image as PILImage
        self.color_gen.time = self.time

        r, g, b = self.color_gen.get_color(0, 1, self.time)
        # 简化：用矩形网格表示
        cell_w = self.w / self.grid_w
        cell_h = self.h / self.grid_h
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                v = self.grid[gy, gx]
                if v > 0.5:
                    x0 = gx * cell_w
                    y0 = gy * cell_h
                    alpha = int(v * 255)
                    drawer.rectangle(
                        [x0, y0, x0 + cell_w, y0 + cell_h],
                        fill=(r, g, b)
                    )


# ── 图层渲染器（主入口） ──────────────────────────────────────────────

# ── Perlin 噪声流体（新增原语：noise / perlin） ───────────
class SimpleNoise:
    """Perlin 噪声流体/纹理层（自包含，纯 NumPy）

    用 noise_field.PerlinNoise 生成连续噪声场，按强度叠加到画布，
    比纯随机的值噪声更自然（调研：Perlin 噪声是程序化生成的"盐"）。

    config 字段：
      scale: 细节频率（越大越细，默认 12）
      octaves: fBm 倍频数（默认 4）
      scroll: [vx, vy] 时间滚动速度（默认 [0.6, 0.35]）
      mode: glow（叠加发光）/ displace（扰动亮度）/ mix
      color: SimpleColorGen 配置
    """

    def __init__(self, config: dict, width: int, height: int):
        self.w = width
        self.h = height
        self.time = 0.0
        if noise_field is None:
            raise RuntimeError("noise_field 模块缺失，无法使用 noise 原语")
        self.noise_type = str(config.get("noise_type", "perlin")).lower()
        if self.noise_type == "simplex":
            self.noise = noise_field.SimplexNoise(config.get("seed"))
        elif self.noise_type == "worley":
            self.noise = noise_field.WorleyNoise(
                config.get("seed"), cell=float(config.get("cell", 64.0)))
        else:
            self.noise = noise_field.PerlinNoise(config.get("seed"))
        self.warp = float(config.get("warp", 4.0))
        self.scale = float(config.get("scale", 12.0))
        self.octaves = int(config.get("octaves", 4))
        self.scroll = tuple(config.get("scroll", [0.6, 0.35]))
        self.mode = config.get("mode", "glow")
        self.color_gen = SimpleColorGen(config.get("color", {}))

    def update(self, dt: float = 0.033):
        self.time += dt

    def draw_to_array(self, canvas: np.ndarray) -> np.ndarray:
        self.color_gen.time = self.time
        if self.noise_type == "warp":
            field = noise_field.warp_field(
                self.noise, self.h, self.w, t=self.time,
                scale=self.scale, octaves=self.octaves,
                warp=self.warp, scroll=self.scroll,
            )
        else:
            field = self.noise.render_field(
                self.h, self.w, t=self.time,
                scale=self.scale, octaves=self.octaves, scroll=self.scroll,
            )
        r, g, b = self.color_gen.get_color(0, 1, self.time)
        f = field[:, :, np.newaxis].astype(np.float32)  # (h,w,1)
        canvas = canvas.astype(np.float32)
        if self.mode == "displace":
            d = (field - 0.5)[:, :, np.newaxis]
            canvas = canvas * (1.0 + d * 0.8)
        else:  # glow / mix：加性叠加
            canvas[..., 0] += r * f[..., 0]
            canvas[..., 1] += g * f[..., 0]
            canvas[..., 2] += b * f[..., 0]
        return np.clip(canvas, 0, 255).astype(np.uint8)


# ── 文本雨（新增原语：text_rain / text） ───────────────────
class SimpleTextRain:
    """文本雨（Matrix 风格），用 PIL 默认字体渲染到画布

    config 字段：
      density: 列密度（0..1，默认 0.5）
      speed: 下落速度（像素/秒，默认 120）
      color: 字体颜色（默认 (0,255,120) 绿）
      charset: 字符集（默认片假名+数字）
    """

    _DEFAULT_CHARSET = "アイウエオカキクケコサシスセソタチツテトナニヌネノ0123456789#$%&*+=<>"

    def __init__(self, config: dict, width: int, height: int):
        self.w = width
        self.h = height
        self.time = 0.0

        def _num(key, default):
            # pywallpaper JSON 常用 {min,max} 范围对象；标量/缺省回退
            v = config.get(key, default)
            if isinstance(v, dict):
                try:
                    lo = float(v.get("min", default))
                    hi = float(v.get("max", default))
                    return (lo + hi) / 2.0
                except (TypeError, ValueError):
                    return float(default)
            try:
                return float(v)
            except (TypeError, ValueError):
                return float(default)

        self.density = _num("density", 0.5)
        self.speed = _num("speed", 120.0)
        # color 可能为 list(rgb) 或 hsv_flow 配置 dict → 容错回退绿
        col = config.get("color", [0, 255, 120])
        if isinstance(col, (list, tuple)) and len(col) >= 3:
            self.color = (int(col[0]), int(col[1]), int(col[2]))
        else:
            self.color = (0, 255, 120)
        self.charset = config.get("charset") or config.get("chars",
                                                           self._DEFAULT_CHARSET)
        self.n_cols = max(1, int(self.w / 14 * self.density))
        self._cols: List[dict] = []
        self._init_cols()
        try:
            self._font = ImageFont.load_default()
        except Exception:
            self._font = None

    def _init_cols(self):
        self._cols = []
        for i in range(self.n_cols):
            x = int((i + 0.5) / self.n_cols * self.w)
            self._cols.append({
                "x": x,
                "y": random.uniform(-self.h, 0.0),
                "ch": random.choice(self.charset),
            })

    def update(self, dt: float = 0.033):
        self.time += dt
        for c in self._cols:
            c["y"] += self.speed * dt
            if random.random() < 0.05:
                c["ch"] = random.choice(self.charset)
            if c["y"] > self.h + 20:
                c["y"] = random.uniform(-self.h * 0.5, 0.0)
                c["x"] = random.randint(0, self.w - 1)

    def draw(self, img: Image.Image) -> Image.Image:
        drawer = ImageDraw.Draw(img, "RGBA")
        head = (min(255, self.color[0] + 120),
                min(255, self.color[1] + 120),
                min(255, self.color[2] + 120))
        for c in self._cols:
            y = int(c["y"])
            if 0 <= y < self.h:
                if self._font:
                    drawer.text((c["x"], y), c["ch"], fill=head[:3],
                               font=self._font)
                else:
                    drawer.text((c["x"], y), c["ch"], fill=head[:3])
        return img


# ── Flow-field 粒子（联网调研新增原语：flow_field / curl / flow） ──
class SimpleFlowField:
    """Flow-field 粒子流——生成艺术经典（调研：curl 无散度场避免粒子堆积/消散）。

    - mode="angle"：经典值噪声角度场，angle = noise(x,y,t)*2π，粒子沿朝向前进。
    - mode="curl" ：对 fBm 标量势 ψ 取旋度 ∇×ψ 得到无散度速度场，
                    流动有机、无汇无源（研究证实 ∇·(∇×ψ)=0）。
    通过"每帧半透明淡出上一帧 + 在当前点叠加亮点"实现拖尾流线（trail fade）。

    config 字段：
      count: 粒子数（默认 1500）
      scale: 流场空间尺度（像素/特征，越大越平缓；默认 90）
      speed: 速度（默认 2.0）
      trail: 拖尾保留系数 [0.8,0.98]，越小拖尾越短（默认 0.93）
      mode:  angle / curl（默认 curl）
      gain:  叠加亮度增益（默认 1.0）
      color: SimpleColorGen 配置
      seed:  随机种子
    """

    def __init__(self, config: dict, width: int, height: int):
        self.w = width
        self.h = height
        self.time = 0.0
        if noise_field is None:
            raise RuntimeError("noise_field 模块缺失，无法使用 flow_field 原语")
        self.seed = config.get("seed")
        self.noise = noise_field.PerlinNoise(self.seed)
        self.count = int(config.get("count", 1500))
        self.scale = float(config.get("scale", 90.0))
        self.speed = float(config.get("speed", 2.0))
        self.trail = float(config.get("trail", 0.93))
        self.mode = str(config.get("mode", "curl")).lower()
        self.gain = float(config.get("gain", 1.0))
        self.color_gen = SimpleColorGen(config.get("color", {}))

        rng = np.random.RandomState(self.seed)
        self._x = rng.uniform(0, self.w, self.count).astype(np.float64)
        self._y = rng.uniform(0, self.h, self.count).astype(np.float64)
        self._px = self._x.copy()
        self._py = self._y.copy()
        self._age = rng.uniform(0, 1, self.count).astype(np.float64)
        self._life = rng.uniform(2.0, 8.0, self.count).astype(np.float64)
        # 持久拖尾缓冲（跨帧保留，形成流线）
        self._trail = np.zeros((self.h, self.w, 3), dtype=np.float64)
        self._eps = 1.0 / max(self.scale, 1e-3)   # 旋度有限差分步长

    def _velocity(self, x, y):
        if self.mode == "curl":
            e = self._eps
            t = self.time * 0.05
            psi = self.noise.fbm(x / self.scale, y / self.scale + t)
            psi_x = self.noise.fbm((x + e) / self.scale, y / self.scale + t) - psi
            psi_y = self.noise.fbm(x / self.scale, (y + e) / self.scale + t) - psi
            vx = psi_y
            vy = -psi_x
        else:
            ang = self.noise.noise2(x / self.scale, y / self.scale + self.time * 0.05)
            ang = ang * math.pi * 2.0
            vx = np.cos(ang)
            vy = np.sin(ang)
        return vx, vy

    def update(self, dt: float = 0.033):
        self.time += dt
        self._px[:] = self._x
        self._py[:] = self._y
        vx, vy = self._velocity(self._x, self._y)
        if self.mode == "curl":
            mag = np.sqrt(vx * vx + vy * vy) + 1e-6
            vx = vx / mag
            vy = vy / mag
        self._x += vx * self.speed * dt * 60.0
        self._y += vy * self.speed * dt * 60.0
        # 越界或寿命到 → 重生（重生瞬间不画拖尾）
        self._age += dt
        out = ((self._x < -5) | (self._x > self.w + 5) |
               (self._y < -5) | (self._y > self.h + 5) |
               (self._age > self._life))
        n = int(out.sum())
        if n:
            r = np.random.RandomState()
            self._x[out] = r.uniform(0, self.w, n)
            self._y[out] = r.uniform(0, self.h, n)
            self._age[out] = 0.0
            self._life[out] = r.uniform(2.0, 8.0, n)
            self._px[out] = self._x[out]
            self._py[out] = self._y[out]

    def draw_to_array(self, canvas: np.ndarray) -> np.ndarray:
        self.color_gen.time = self.time
        canvas = canvas.astype(np.float64)
        # 拖尾淡出
        self._trail *= self.trail
        r, g, b = self.color_gen.get_color(0, 1, self.time)
        xi = np.clip(self._x.astype(np.int64), 0, self.w - 1)
        yi = np.clip(self._y.astype(np.int64), 0, self.h - 1)
        alpha = (1.0 - self._age / np.maximum(self._life, 1e-6)) * self.gain
        alpha = np.clip(alpha, 0.0, 1.0)
        # 半径 1 的十字亮点，便宜且足以形成连续流线
        for dx, dy in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)):
            xx = np.clip(xi + dx, 0, self.w - 1)
            yy = np.clip(yi + dy, 0, self.h - 1)
            self._trail[yy, xx, 0] += r * alpha
            self._trail[yy, xx, 1] += g * alpha
            self._trail[yy, xx, 2] += b * alpha
        out = np.clip(canvas + self._trail, 0, 255)
        return out.astype(np.uint8)


class PWEffecRenderer:
    """pywallpaper/effects JSON 渲染器（简化版）

    每个 EngineEffect 由多个 Layer 组成。
    每个 Layer 使用一种原语类型。
    """

    PRIMITIVE_MAP = {
        "particle": SimpleParticleSystem,
        "particles": SimpleParticleSystem,
        "waveform": SimpleWaveform,
        "bars": SimpleWaveform,
        "spikes": SimpleWaveform,
        "fluid": SimpleFluid,
        "ca": SimpleCAGrid,
        "cellular": SimpleCAGrid,
        # ── 新增原语（全面增强）──
        "noise": SimpleNoise,
        "perlin": SimpleNoise,
        "wireframe_3d": (wireframe3d.Wireframe3D if wireframe3d else None),
        "wireframe": (wireframe3d.Wireframe3D if wireframe3d else None),
        "text_rain": SimpleTextRain,
        "text": SimpleTextRain,
        # ── 联网调研新增：flow-field 粒子（curl 无散度 / 经典角度场）──
        "flow_field": SimpleFlowField,
        "flowfield": SimpleFlowField,
        "flow": SimpleFlowField,
        "curl": SimpleFlowField,
        # ── P1-2 新增程序化原语（raymarch SDF / metaballs / lsystem / dla）──
        "raymarch": (primitives_pro.RaymarchSDF if _HAS_PRIMITIVES_PRO else None),
        "sdf": (primitives_pro.RaymarchSDF if _HAS_PRIMITIVES_PRO else None),
        "metaballs": (primitives_pro.Metaballs if _HAS_PRIMITIVES_PRO else None),
        "metaball": (primitives_pro.Metaballs if _HAS_PRIMITIVES_PRO else None),
        "lsystem": (primitives_pro.LSystemPrimitive if _HAS_PRIMITIVES_PRO else None),
        "fractal": (primitives_pro.LSystemPrimitive if _HAS_PRIMITIVES_PRO else None),
        "dla": (primitives_pro.DLACluster if _HAS_PRIMITIVES_PRO else None),
        "aggregation": (primitives_pro.DLACluster if _HAS_PRIMITIVES_PRO else None),
        # ── P2 新增：径向环 / 同心圆 / 频谱圆（circle_map，300+ 组合 JSON 使用）──
        "circle_map": (primitives_pro.CircleMapPrimitive if _HAS_PRIMITIVES_PRO else None),
        "circlemap": (primitives_pro.CircleMapPrimitive if _HAS_PRIMITIVES_PRO else None),
        "circlemaps": (primitives_pro.CircleMapPrimitive if _HAS_PRIMITIVES_PRO else None),
    }

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        self.time = 0.0
        self.layers: List[object] = []

    def load_config(self, effect_config: dict):
        """从 pywallpaper effect JSON 加载配置"""
        params = effect_config.get("params", {})
        layers_cfg = params.get("layers", [])

        self.layers = []
        for layer_cfg in layers_cfg:
            ltype = layer_cfg.get("type", "unknown")
            if ltype in self.PRIMITIVE_MAP:
                cls = self.PRIMITIVE_MAP[ltype]
                try:
                    layer = cls(layer_cfg, self.width, self.height)
                    self.layers.append(layer)
                except Exception as e:
                    print(f"[PW] 跳过图层 {ltype}: {e}")
            else:
                print(f"[PW] 未支持的原语类型: {ltype}")

    def update(self, dt: float = 0.033):
        self.time += dt
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(dt)

    def draw(self) -> Image.Image:
        """渲染当前帧为 PIL Image（粒子层使用 Numba JIT 加速）"""
        # 先用 NumPy 数组渲染粒子层（JIT 加速）
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for layer in self.layers:
            if hasattr(layer, "draw_to_array"):
                canvas = layer.draw_to_array(canvas)

        # 转 PIL，再渲染其余层
        img = Image.fromarray(canvas)
        drawer = ImageDraw.Draw(img)

        for layer in self.layers:
            if hasattr(layer, "draw_to_array"):
                continue  # 已处理
            if hasattr(layer, "draw"):
                sig = inspect.signature(layer.draw)
                params = list(sig.parameters.keys())
                if "img" in params or "image" in params:
                    img = layer.draw(img)
                    drawer = ImageDraw.Draw(img)
                else:
                    layer.draw(drawer)

        return img


# ── 测试 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import os

    pw_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "pywallpaper", "effects"
    )

    # 测试粒子效果
    test_file = None
    for f in os.listdir(pw_dir):
        if "particle" in f and f.endswith(".json"):
            test_file = os.path.join(pw_dir, f)
            break

    if test_file:
        print(f"测试渲染 PW 效果: {test_file}")
        with open(test_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        renderer = PWEffecRenderer(width=512, height=512)
        renderer.load_config(config)

        frames = []
        for i in range(60):
            renderer.update(0.033)
            frame = renderer.draw()
            frames.append(frame)
            if i % 10 == 0:
                print(f"  帧 {i}")

        out_path = os.path.join(os.path.dirname(__file__), "test_pw.gif")
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:20],
            duration=50,
            loop=0
        )
        print(f"测试 GIF 已保存: {out_path}")
    else:
        print("未找到粒子测试文件")
