"""
noise_field.py — 自包含向量化 2D Perlin 噪声场

纯 NumPy 实现（无外部依赖），包含：
  - 经典 Ken Perlin "improved" 噪声（2D）
  - 多倍频 fBm（fractal Brownian motion）
  - 时间滚动（让噪声场随时间流动，适合视频）

用途：
  - 作为 pywallpaper 新原语 `noise`（流体/纹理背景）
  - 作为主题背景的丰富纹理层
  - 调制其他效果的参数（如亮度扰动）

调研：Perlin 噪声被誉为程序化生成的"盐"，比纯随机的值噪声
在视觉上更连续、更自然（常用于地形、云雾、流体、火焰等）。
"""

from typing import Optional, Tuple

import math
import numpy as np


def _fade(t: np.ndarray) -> np.ndarray:
    """Perlin 经典 6t^5-15t^4+10t^3 缓动"""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def _grad(hash_v: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """2D 梯度函数（经典 Perlin）"""
    h = hash_v & 3
    u = np.where(h < 2, x, y)
    v = np.where(h < 2, y, x)
    return np.where(h & 1, -u, u) + np.where(h & 2, -v, v)


class PerlinNoise:
    """向量化 2D Perlin 噪声生成器

    用法：
        n = PerlinNoise(seed=42)
        field = n.render_field(64, 64, t=3.0, scale=8.0)
        # field: (64,64) float ∈ [0,1]
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed(seed)

    def seed(self, seed: Optional[int] = None):
        """用种子（重）初始化置换表"""
        base = np.arange(256, dtype=np.uint8)
        rng = np.random.RandomState(seed)
        rng.shuffle(base)
        self._p = np.concatenate([base, base]).astype(np.int64)  # 512

    def noise2(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """计算 2D Perlin 噪声，输入为任意形状的数组（同形）

        返回与输入同形的 float，值域约 [-1, 1]。
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xi = x.astype(np.int64) & 255
        yi = y.astype(np.int64) & 255
        xf = x - np.floor(x)
        yf = y - np.floor(y)
        u = _fade(xf)
        v = _fade(yf)

        p = self._p
        n00 = _grad(p[p[xi] + yi], xf, yf)
        n10 = _grad(p[p[xi + 1] + yi], xf - 1.0, yf)
        n01 = _grad(p[p[xi] + yi + 1], xf, yf - 1.0)
        n11 = _grad(p[p[xi + 1] + yi + 1], xf - 1.0, yf - 1.0)

        x1 = _lerp(n00, n10, u)
        x2 = _lerp(n01, n11, u)
        return _lerp(x1, x2, v)

    def fbm(self, x: np.ndarray, y: np.ndarray,
            octaves: int = 4, persistence: float = 0.5,
            lacunarity: float = 2.0) -> np.ndarray:
        """分形布朗运动：多个倍频叠加"""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        total = np.zeros_like(x, dtype=np.float64)
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        for _ in range(octaves):
            total += amplitude * self.noise2(x * frequency, y * frequency)
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        # 归一化到 [-1,1] 附近，再映射到 [0,1]
        if max_value > 0:
            total /= max_value
        return np.clip(total * 0.5 + 0.5, 0.0, 1.0)

    def render_field(self, h: int, w: int, t: float = 0.0,
                        scale: float = 8.0,
                        octaves: int = 4,
                        scroll: Tuple[float, float] = (0.6, 0.35),
                        persistence: float = 0.5) -> np.ndarray:
        """渲染一张 (h,w) 的噪声场，值域 [0,1]

        scale: 越大细节越细（频率越高）
        scroll: (vx, vy) 时间滚动速度（单位：网格/秒）
        """
        ys = np.arange(h, dtype=np.float64) / scale + t * scroll[1]
        xs = np.arange(w, dtype=np.float64) / scale + t * scroll[0]
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        return self.fbm(xx, yy, octaves=octaves,
                        persistence=persistence)


# ═════════════════════════════════════════════════════════════════
#  进阶噪声：Simplex / Worley(元胞) / 域扭曲 fBm（联网调研新增）
# ═════════════════════════════════════════════════════════════════

# 经典 2D Simplex 噪声（Ken Perlin 2002；公开领域参考实现改编）。
# 调研结论：Simplex 用单形（2D 三角形）替代超立方体网格，消除 Perlin 的
# 轴向对齐伪影、各向同性更好；2D 仅需 3 个角点（Perlin 需 4 个）。
_GRAD3 = np.array([
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
], dtype=np.float64)

_F2 = 0.5 * (math.sqrt(3.0) - 1.0)
_G2 = (3.0 - math.sqrt(3.0)) / 6.0


class SimplexNoise:
    """向量化 2D Simplex 噪声（各向同性、无轴向伪影；值域约 [-1,1]）。"""

    def __init__(self, seed: Optional[int] = None):
        self.seed(seed)

    def seed(self, seed: Optional[int] = None):
        base = np.arange(256, dtype=np.int64)
        rng = np.random.RandomState(seed)
        rng.shuffle(base)
        self._perm = np.concatenate([base, base]).astype(np.int64)
        self._perm_mod12 = (self._perm & 11).astype(np.int64)

    def noise2(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        s = (x + y) * _F2
        i = np.floor(x + s).astype(np.int64)
        j = np.floor(y + s).astype(np.int64)
        t = (i + j) * _G2
        x0 = x - (i - t)
        y0 = y - (j - t)
        i1 = (x0 > y0).astype(np.int64)
        j1 = 1 - i1
        x1 = x0 - i1 + _G2
        y1 = y0 - j1 + _G2
        x2 = x0 - 1.0 + 2.0 * _G2
        y2 = y0 - 1.0 + 2.0 * _G2
        ii = i & 255
        jj = j & 255
        gi0 = self._perm_mod12[ii + self._perm[jj]]
        gi1 = self._perm_mod12[ii + i1 + self._perm[jj + j1]]
        gi2 = self._perm_mod12[ii + 1 + self._perm[jj + 1]]

        def _corner(xc, yc, gi):
            t0 = 0.5 - xc * xc - yc * yc
            t0 = np.clip(t0, 0.0, None)
            t0 *= t0
            g = _GRAD3[gi]            # (...,3)
            return t0 * t0 * (g[..., 0] * xc + g[..., 1] * yc)

        n0 = _corner(x0, y0, gi0)
        n1 = _corner(x1, y1, gi1)
        n2 = _corner(x2, y2, gi2)
        return 70.0 * (n0 + n1 + n2)  # 标准缩放，约 [-1,1]

    def fbm(self, x, y, octaves: int = 4,
            persistence: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        total = np.zeros_like(x, dtype=np.float64)
        amp = 1.0
        freq = 1.0
        max_value = 0.0
        for _ in range(octaves):
            total += amp * self.noise2(x * freq, y * freq)
            max_value += amp
            amp *= persistence
            freq *= lacunarity
        if max_value > 0:
            total /= max_value
        return np.clip(total * 0.5 + 0.5, 0.0, 1.0)

    def render_field(self, h, w, t: float = 0.0, scale: float = 8.0,
                     octaves: int = 4, scroll=(0.6, 0.35),
                     persistence: float = 0.5) -> np.ndarray:
        ys = np.arange(h, dtype=np.float64) / scale + t * scroll[1]
        xs = np.arange(w, dtype=np.float64) / scale + t * scroll[0]
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        return self.fbm(xx, yy, octaves=octaves, persistence=persistence)


class WorleyNoise:
    """元胞/沃罗诺伊噪声（F1：到最近特征点距离，值域 [0,1]）。

    调研结论：Worley 属于与值/梯度噪声不同的家族——撒布特征点后取
    最近距离，得到蜂窝/细胞状纹理，是生成艺术中 Voronoi 外观的来源。
    """

    def __init__(self, seed: Optional[int] = None, cell: float = 64.0,
                 metric: str = "euclidean"):
        self.cell = float(cell)
        self.metric = metric
        self.seed_int = (int(seed) if isinstance(seed, int) else hash(str(seed))) & 0x7fffffff
        if self.seed_int == 0:
            self.seed_int = 123456789

    def _hash12(self, a, b):
        a = a.astype(np.int64)
        b = b.astype(np.int64)
        # 用 64-bit 整数哈希（uint64 取模，避免 numpy int64 溢出）
        ua = a.astype(np.uint64)
        ub = b.astype(np.uint64)
        h = (ua * np.uint64(374761393)) ^ (ub * np.uint64(668265263))
        h = h ^ np.uint64(self.seed_int & 0x7fffffff)
        h = (h ^ (h >> np.uint64(15))) * np.uint64(0x2C1B3C6D)
        h = h ^ (h >> np.uint64(12))
        r1 = (h & np.uint64(0xFFFF)).astype(np.float64) / 65536.0
        r2 = ((h >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.float64) / 65536.0
        return r1, r2

    def noise2(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        cx = np.floor(x / self.cell).astype(np.int64)
        cy = np.floor(y / self.cell).astype(np.int64)
        fmin = np.full(x.shape, np.inf, dtype=np.float64)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cxi = cx + dx
                cyi = cy + dy
                jx, jy = self._hash12(cxi, cyi)
                px = (cxi + jx) * self.cell
                py = (cyi + jy) * self.cell
                if self.metric == "manhattan":
                    d = np.abs(x - px) + np.abs(y - py)
                else:
                    d = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                fmin = np.minimum(fmin, d)
        return np.clip(fmin / self.cell, 0.0, 1.0)

    def render_field(self, h, w, t: float = 0.0, scale: float = 8.0,
                     octaves: int = 4, scroll=(0.0, 0.0)) -> np.ndarray:
        ys = np.arange(h, dtype=np.float64) / scale + t * scroll[1]
        xs = np.arange(w, dtype=np.float64) / scale + t * scroll[0]
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        return self.noise2(xx, yy)


def warp_field(noise, h, w, t: float = 0.0, scale: float = 8.0,
               octaves: int = 4, warp: float = 4.0,
               scroll=(0.6, 0.35), persistence: float = 0.5) -> np.ndarray:
    """Inigo Quilez 风格域扭曲 fBm：用噪声扭曲坐标，得到有机流体纹理。

        q = fbm(p);  r = fbm(p + warp*q);  f = fbm(p + warp*r)
    比单层 fBm 层次更丰富，常用于火焰/云雾/大理石背景。
    """
    ys = np.arange(h, dtype=np.float64) / scale + t * scroll[1]
    xs = np.arange(w, dtype=np.float64) / scale + t * scroll[0]
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    qx = noise.fbm(xx, yy, octaves=octaves, persistence=persistence)
    qy = noise.fbm(xx + 5.2, yy + 1.3, octaves=octaves, persistence=persistence)
    rx = noise.fbm(xx + warp * qx, yy + warp * qy,
                   octaves=octaves, persistence=persistence)
    ry = noise.fbm(xx + warp * qx + 1.7, yy + warp * qy + 9.2,
                   octaves=octaves, persistence=persistence)
    f = noise.fbm(xx + warp * rx, yy + warp * ry,
                  octaves=octaves, persistence=persistence)
    return np.clip(f, 0.0, 1.0)


if __name__ == "__main__":
    import time
    n = PerlinNoise(seed=7)
    t0 = time.perf_counter()
    field = n.render_field(256, 256, t=2.0, scale=12.0)
    dt = time.perf_counter() - t0
    print(f"[noise_field] 256x256 fBm 渲染耗时 {dt*1000:.1f} ms")
    print(f"  shape={field.shape} dtype={field.dtype} "
          f"min={field.min():.3f} max={field.max():.3f} mean={field.mean():.3f}")
    # 连续帧应平滑变化（相邻帧差应远小于 1）
    f1 = n.render_field(64, 64, t=1.0)
    f2 = n.render_field(64, 64, t=1.05)
    print(f"  相邻帧均方根差: {np.sqrt(((f1-f2)**2).mean()):.4f} "
          f"(应明显 < 1，表示时间连续)")
    print("OK")
