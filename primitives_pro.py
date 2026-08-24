"""
primitives_pro.py — P1-2 新增程序化原语（CPU，纯 NumPy/PIL，零硬依赖）

四种新原语，均实现 pywallpaper 图层接口：
  __init__(self, config: dict, width: int, height: int)
  update(self, dt: float = 0.033)        # 推进时间（动画）
  draw_to_array(self, canvas) -> canvas   # 加性叠加到传入画布（可分层）

原语家族：
  RaymarchSDF   (raymarch / sdf)    : 2D 有符号距离场场景，smin 融合 + 边缘辉光
  Metaballs     (metaballs)         : 经典元球场，阈值填充 + 辉光
  LSystemPrimitive (lsystem / fractal) : L-system 重写 + 海龟绘图，逐帧生长/旋转
  DLACluster    (dla / aggregation) : 扩散限制凝聚，预计算簇 + 逐帧渐进显现

降级铁律（与项目一致）：不引入任何外部依赖，纯 Python 即可跑，绝不崩溃。
"""

import math
import numpy as np
from PIL import Image, ImageDraw


def _hsv_to_rgb(h: float, s: float, v: float):
    """HSV → (R,G,B) uint8；h,s,v ∈ [0,1]"""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """多项式平滑最小值（IQ），k 越大融合越柔。"""
    h = np.clip(0.5 + 0.5 * (b - a) / max(k, 1e-6), 0.0, 1.0)
    return b * h + a * (1.0 - h) - k * h * (1.0 - h)


class RaymarchSDF:
    """2D 有符号距离场（SDF）场景渲染。

    用屏幕坐标映射为 p∈[-aspect,aspect]×[-1,1]，对若干基本图元
    （圆/盒）求距离场并 smin 融合，按距离做填充 + 边缘辉光。
    图元中心随时间旋转，呈现有机流动观感。

    config: scale, speed, glow, smooth, color{hue:[a,b],sat,val}
    """

    def __init__(self, config: dict, width: int, height: int):
        self.w = width
        self.h = height
        self.time = 0.0
        self.speed = float(config.get("speed", 0.4))
        self.glow = float(config.get("glow", 6.0))
        self.smooth = float(config.get("smooth", 0.3))
        c = config.get("color", {})
        self.hue = c.get("hue", [0.5, 0.72])
        self.sat = c.get("sat", 0.85)
        self.val = c.get("val", 0.95)

    def update(self, dt: float = 0.033):
        self.time += dt

    def draw_to_array(self, canvas: np.ndarray) -> np.ndarray:
        t = self.time * self.speed
        aspect = self.w / self.h
        ys, xs = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        px = (xs / (self.w - 1)) * 2.0 * aspect - aspect
        py = 1.0 - (ys / (self.h - 1)) * 2.0

        a1 = t
        c1x = 0.42 * math.cos(a1)
        c1y = 0.30 * math.sin(a1 * 1.3)
        c2x = 0.46 * math.cos(a1 * 0.7 + 2.0)
        c2y = 0.34 * math.sin(a1 * 0.9)
        d1 = np.sqrt((px - c1x) ** 2 + (py - c1y) ** 2) - 0.34
        d2 = np.sqrt((px - c2x) ** 2 + (py - c2y) ** 2) - 0.27
        # 中心旋转方块
        bx = np.abs(px) - 0.24
        by = np.abs(py) - 0.24
        d3 = (np.sqrt(np.maximum(bx, 0) ** 2 + np.maximum(by, 0) ** 2)
              + np.minimum(np.maximum(bx, by), 0.0) - 0.05)

        d = _smin(d1, d2, self.smooth)
        d = _smin(d, d3, self.smooth)

        fill = 1.0 - np.clip(d / 0.04 + 0.5, 0.0, 1.0)   # 内部=1
        halo = np.exp(-np.abs(d) * self.glow)
        inten = fill * 0.9 + halo * 0.7

        hue = self.hue[0] + (self.hue[1] - self.hue[0]) * ((t * 0.05) % 1.0)
        r, g, b = _hsv_to_rgb(hue, self.sat, self.val)
        f = inten[:, :, None].astype(np.float32)
        out = canvas.astype(np.float32)
        out[..., 0] += r * f[..., 0]
        out[..., 1] += g * f[..., 0]
        out[..., 2] += b * f[..., 0]
        return np.clip(out, 0, 255).astype(np.uint8)


class Metaballs:
    """经典元球（Metaballs）场渲染。

    field = Σ r_i² / ((x-x_i)² + (y-y_i)² + ε)，field 越阈值即内部，
    叠加辉光。球心随时间沿椭圆轨道运动。

    config: count, speed, threshold, glow, color{hue,sat,val}
    """

    def __init__(self, config: dict, width: int, height: int):
        self.w = width
        self.h = height
        self.time = 0.0
        self.n = int(config.get("count", 5))
        self.speed = float(config.get("speed", 0.5))
        self.threshold = float(config.get("threshold", 1.0))
        self.glow = float(config.get("glow", 1.5))
        c = config.get("color", {})
        self.hue = c.get("hue", [0.55, 0.95])
        self.sat = c.get("sat", 0.8)
        self.val = c.get("val", 0.95)

        rng = np.random.default_rng(0)
        self.radii = rng.uniform(0.16, 0.40, self.n)
        self.phases = rng.uniform(0.0, 6.283, self.n)
        self.axes = rng.uniform(0.30, 0.85, self.n)

    def update(self, dt: float = 0.033):
        self.time += dt

    def draw_to_array(self, canvas: np.ndarray) -> np.ndarray:
        t = self.time * self.speed
        aspect = self.w / self.h
        ys, xs = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        px = (xs / (self.w - 1)) * 2.0 * aspect - aspect
        py = 1.0 - (ys / (self.h - 1)) * 2.0

        field = np.zeros((self.h, self.w), np.float32)
        for i in range(self.n):
            cx = self.axes[i] * math.cos(t + self.phases[i])
            cy = self.axes[i] * math.sin(t * 1.1 + self.phases[i])
            r = self.radii[i]
            field += (r * r) / ((px - cx) ** 2 + (py - cy) ** 2 + 0.02)

        fill = np.clip((field - self.threshold) / 0.25, 0.0, 1.0)
        halo = np.exp(-np.abs(field - self.threshold) * self.glow)
        inten = fill * 0.95 + halo * 0.5

        hue = self.hue[0] + (self.hue[1] - self.hue[0]) * ((t * 0.04) % 1.0)
        r, g, b = _hsv_to_rgb(hue, self.sat, self.val)
        f = inten[:, :, None].astype(np.float32)
        out = canvas.astype(np.float32)
        out[..., 0] += r * f[..., 0]
        out[..., 1] += g * f[..., 0]
        out[..., 2] += b * f[..., 0]
        return np.clip(out, 0, 255).astype(np.uint8)


class LSystemPrimitive:
    """L-system 分形（海龟绘图），逐帧生长 + 整体旋转。

    config:
      axiom, rules(dict), iterations, angle(deg), step,
      rotate(rad/s), color{hue,sat,val}

    内置默认：经典分形植物（Barnsley-ish Lindenmayer 植物）。
    字符串长度上限保护，避免迭代爆炸。
    """

    _DEFAULT_RULES = {"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"}

    def __init__(self, config: dict, width: int, height: int):
        self.w = width
        self.h = height
        self.time = 0.0
        self.axiom = config.get("axiom", "X")
        self.rules = config.get("rules", dict(self._DEFAULT_RULES))
        self.iterations = int(min(config.get("iterations", 4), 6))
        self.angle = math.radians(float(config.get("angle", 25.0)))
        self.step = float(config.get("step", 10.0))
        self.rotate_speed = float(config.get("rotate", 0.2))
        c = config.get("color", {})
        self.hue = c.get("hue", [0.25, 0.42])
        self.sat = c.get("sat", 0.7)
        self.val = c.get("val", 0.92)
        self._build()

    def _build(self):
        s = self.axiom
        for _ in range(self.iterations):
            ns = []
            for ch in s:
                ns.append(self.rules.get(ch, ch))
            s = "".join(ns)
            if len(s) > 200000:
                break

        segs = []
        stack = []
        x = y = 0.0
        hd = math.pi / 2.0
        for ch in s:
            if ch in ("F", "G"):
                nx = x + math.cos(hd) * self.step
                ny = y + math.sin(hd) * self.step
                segs.append((x, y, nx, ny))
                x, y = nx, ny
            elif ch == "+":
                hd += self.angle
            elif ch == "-":
                hd -= self.angle
            elif ch == "[":
                stack.append((x, y, hd))
            elif ch == "]":
                if stack:
                    x, y, hd = stack.pop()

        if not segs:
            segs = [(0.0, 0.0, 1.0, 0.0)]
        segs = np.array(segs, dtype=np.float32)
        xs = segs[:, [0, 2]]
        ys_ = segs[:, [1, 3]]
        minx, maxx = xs.min(), xs.max()
        miny, maxy = ys_.min(), ys_.max()
        span = max(maxx - minx, maxy - miny, 1e-6)
        segs[:, 0] = (segs[:, 0] - minx) / span * 1.8 - 0.9
        segs[:, 2] = (segs[:, 2] - minx) / span * 1.8 - 0.9
        segs[:, 1] = (segs[:, 1] - miny) / span * 1.8 - 0.9
        segs[:, 3] = (segs[:, 3] - miny) / span * 1.8 - 0.9
        self.segs = segs
        self.n = len(segs)

    def update(self, dt: float = 0.033):
        self.time += dt

    def draw_to_array(self, canvas: np.ndarray) -> np.ndarray:
        if self.n == 0:
            return canvas
        # 逐帧生长：约 iterations*0.5+1 秒内长满
        reveal = int(min(1.0, self.time / (self.iterations * 0.5 + 1.0)) * self.n)
        reveal = max(1, reveal)
        rot = self.time * self.rotate_speed
        ca, sa = math.cos(rot), math.sin(rot)

        scale = min(self.w, self.h) * 0.46
        ox, oy = self.w / 2.0, self.h / 2.0
        hue = self.hue[0] + (self.hue[1] - self.hue[0]) * ((self.time * 0.03) % 1.0)
        r, g, b = _hsv_to_rgb(hue, self.sat, self.val)

        img = Image.new("RGB", (self.w, self.h), (0, 0, 0))
        d = ImageDraw.Draw(img)
        lw = 2 if max(self.w, self.h) >= 240 else 1
        for i in range(reveal):
            # 沿枝条推进做色相渐变（尖端偏暖），避免整株单色 → 唯一色过少
            frac = i / max(reveal, 1)
            hue = (self.hue[0] + (self.hue[1] - self.hue[0]) * frac
                   + (self.time * 0.03) % 1.0) % 1.0
            r, g, b = _hsv_to_rgb(hue, self.sat, self.val)
            x0, y0, x1, y1 = self.segs[i]
            rx0 = x0 * ca - y0 * sa
            ry0 = x0 * sa + y0 * ca
            rx1 = x1 * ca - y1 * sa
            ry1 = x1 * sa + y1 * ca
            sx0 = int(ox + rx0 * scale)
            sy0 = int(oy - ry0 * scale)
            sx1 = int(ox + rx1 * scale)
            sy1 = int(oy - ry1 * scale)
            d.line([sx0, sy0, sx1, sy1], fill=(r, g, b), width=lw)
        # 加性叠加（1.6x 提亮，不破坏颜色多样性）
        arr = np.clip(canvas.astype(np.float32) + np.asarray(img, np.float32) * 1.6,
                      0, 255).astype(np.uint8)
        return arr


class DLACluster:
    """扩散限制凝聚（DLA）：随机游走粒子黏附到已有簇。

    预计算（一次性）聚合成簇点集，归一化到 [-1,1]；逐帧渐进显现，
    呈现生长动画。用固定 max_points + 步数上限 + 总尝试预算保证不失控。

    config: max_points, grow_time, cell, color{hue,sat,val}
    """

    def __init__(self, config: dict, width: int, height: int):
        self.w = width
        self.h = height
        self.time = 0.0
        self.max_points = int(config.get("max_points", 350))
        self.grow_time = float(config.get("grow_time", 6.0))
        self.cell = int(config.get("cell", 120))
        c = config.get("color", {})
        self.hue = c.get("hue", [0.0, 0.18])
        self.sat = c.get("sat", 0.8)
        self.val = c.get("val", 0.95)
        self._grow()

    def _grow(self):
        S = self.cell
        grid = np.zeros((S, S), np.int8)
        cx = cy = S // 2
        grid[cy, cx] = 1
        pts = [(cx, cy)]
        rng = np.random.default_rng(7)
        # 跟踪簇当前最大半径，发射环随簇生长收缩，保证游走粒子能触及并黏附
        cluster_r = 1.0
        while len(pts) < self.max_points:
            ang = rng.uniform(0.0, 6.283)
            rad = min(S * 0.48, cluster_r + max(6.0, cluster_r * 0.5))
            x = int(cx + math.cos(ang) * rad)
            y = int(cy + math.sin(ang) * rad)
            stuck = False
            for _ in range(4000):
                # 八邻域随机游走（更连续，收敛更快）
                x += int(rng.integers(-1, 2))
                y += int(rng.integers(-1, 2))
                if x < 1 or x >= S - 1 or y < 1 or y >= S - 1:
                    break
                if (grid[y - 1, x] or grid[y + 1, x]
                        or grid[y, x - 1] or grid[y, x + 1]):
                    grid[y, x] = 1
                    dx = x - cx
                    dy = y - cy
                    d = math.sqrt(dx * dx + dy * dy)
                    if d > cluster_r:
                        cluster_r = d
                    pts.append((x, y))
                    stuck = True
                    break
            if not stuck:
                # 极少数游走逃逸：簇已接近满，提前结束
                if cluster_r >= S * 0.46:
                    break
        pts = np.array(pts, dtype=np.float32)
        self.pts = (pts - cx) / (S * 0.5)
        self.n = len(self.pts)

    def update(self, dt: float = 0.033):
        self.time += dt

    def draw_to_array(self, canvas: np.ndarray) -> np.ndarray:
        if self.n == 0:
            return canvas
        # 前 1.5s 生长（1→max），之后保持完整，避免末帧仍稀疏
        reveal = int(min(1.0, self.time / min(self.grow_time, 1.5)) * self.n)
        reveal = max(1, reveal)
        scale = min(self.w, self.h) * 0.92
        ox = self.w / 2.0
        oy = self.h / 2.0
        # 预计算整数像素坐标（reveal 范围内）
        nx = self.pts[:reveal, 0]
        ny = self.pts[:reveal, 1]
        sx = np.clip((ox + nx * scale).astype(np.int32), 0, self.w - 1)
        sy = np.clip((oy - ny * scale).astype(np.int32), 0, self.h - 1)
        # 半径渐变着色（核心→边缘色相漂移），制造颜色多样性
        rad = np.sqrt(nx * nx + ny * ny)
        rmax = float(rad.max()) if rad.size else 1.0
        frac = (rad / rmax) if rmax > 0 else rad
        hue = (self.hue[0] + (self.hue[1] - self.hue[0]) * frac
               + (self.time * 0.03) % 1.0) % 1.0
        cols = np.array([_hsv_to_rgb(h, self.sat, self.val) for h in hue],
                        dtype=np.float32)  # (reveal,3)
        out = canvas.astype(np.float32)
        # 每点画 3x3 光点，向量化批量叠加（核心亮、环略暗）
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy = np.clip(sy + dy, 0, self.h - 1)
                xx = np.clip(sx + dx, 0, self.w - 1)
                out[yy, xx, 0] += cols[:, 0] * 0.8
                out[yy, xx, 1] += cols[:, 1] * 0.8
                out[yy, xx, 2] += cols[:, 2] * 0.8
        return np.clip(out, 0, 255).astype(np.uint8)


class CircleMapPrimitive:
    """径向环 / 同心圆 / 频谱圆原语（circle_map）。

    对应 pywallpaper 组合 JSON 里大量出现的 `circle_map` 图层：
      center{ x:"50%", y:"50%" }
      rings[ {count, radius, line(bool), line_width} ]
      color {type:"hsv_flow", hue:[a,b], speed, sat, val}
      audio {source:"band_energy", target:"radius_offset", scale:[0,60], smoothing}

    渲染：对每个 ring，若 line=True 画整圈描边圆（line_width px），
    否则沿圆周等距布 count 个点（带辉光）。hue 随时间在 [a,b] 内流动；
    audio 半径偏移把音频能量映射到每圈半径扩张（频谱环观感）。

    降级铁律：纯 NumPy/PIL，缺字段一律安全默认，绝不崩溃。
    """

    def __init__(self, config: dict, width: int, height: int):
        self.w = width
        self.h = height
        self.time = 0.0
        # center：百分比 → 像素（默认正中）
        ctr = config.get("center", {"x": "50%", "y": "50%"})
        self.cx = self._pct(ctr.get("x", "50%"), width)
        self.cy = self._pct(ctr.get("y", "50%"), height)
        # rings：归一化为统一结构
        raw_rings = config.get("rings", [])
        self.rings = []
        for r in raw_rings:
            try:
                self.rings.append({
                    "count": int(r.get("count", 1)),
                    "radius": float(r.get("radius", 100)),
                    "line": bool(r.get("line", False)),
                    "line_width": max(1, int(r.get("line_width", 2))),
                })
            except Exception:
                continue
        if not self.rings:
            # 无 ring 时给一个默认值，避免空画布
            self.rings = [{"count": 1, "radius": min(width, height) * 0.3,
                           "line": True, "line_width": 2}]
        # color（hsv_flow）
        c = config.get("color", {})
        self.hue = c.get("hue", [0.55, 0.75])
        if not isinstance(self.hue, (list, tuple)) or len(self.hue) < 2:
            self.hue = [float(self.hue) if isinstance(self.hue, (int, float)) else 0.55,
                        float(self.hue) + 0.2 if isinstance(self.hue, (int, float)) else 0.75]
        self.color_speed = float(c.get("speed", 0.1))
        self.sat = float(c.get("sat", 0.8))
        self.val = float(c.get("val", 0.85))
        # audio：band_energy → radius_offset
        a = config.get("audio", {})
        self.audio_scale = a.get("scale", [0.0, 60.0])
        if not isinstance(self.audio_scale, (list, tuple)) or len(self.audio_scale) < 2:
            self.audio_scale = [0.0, 60.0]
        self.audio_smoothing = float(a.get("smoothing", 0.85))
        self._audio_level = 0.0  # 平滑后的能量 [0,1]

    @staticmethod
    def _pct(v, dim):
        """'50%' → 像素；数字视为像素；默认居中。"""
        try:
            if isinstance(v, str):
                v = v.strip()
                if v.endswith("%"):
                    return dim * float(v[:-1]) / 100.0
                return float(v)
            return float(v)
        except Exception:
            return dim / 2.0

    def update(self, dt: float = 0.033, audio_level: float = None):
        self.time += dt
        if audio_level is not None:
            # 指数平滑（attack/decay 通用）
            a = self.audio_smoothing
            self._audio_level = a * self._audio_level + (1.0 - a) * max(0.0, min(1.0, audio_level))

    def _radius_offset(self):
        """把平滑能量映射到半径扩张像素（scale[0]→scale[1]）。"""
        lo, hi = self.audio_scale[0], self.audio_scale[1]
        return lo + (hi - lo) * self._audio_level

    def draw_to_array(self, canvas: np.ndarray) -> np.ndarray:
        out = canvas.astype(np.float32)
        # 当前 hue（在 [a,b] 内随时间流动）
        span = self.hue[1] - self.hue[0]
        hue = (self.hue[0] + span * ((self.time * self.color_speed) % 1.0)) % 1.0
        off = self._radius_offset()
        img = Image.new("RGB", (self.w, self.h), (0, 0, 0))
        d = ImageDraw.Draw(img)
        SEG = 24  # 圆周分色段数（制造彩虹环，避免整圈单色）
        for ri, ring in enumerate(self.rings):
            r = ring["radius"] + off
            if r <= 0:
                continue
            bbox = [self.cx - r, self.cy - r, self.cx + r, self.cy + r]
            if ring["line"]:
                lw = ring["line_width"]
                # 沿圆周分段上色（彩虹环），每段约 15°
                for s in range(SEG):
                    a0 = 360.0 * s / SEG
                    a1 = 360.0 * (s + 1) / SEG
                    sh = (hue + 0.04 * ri + span * (s / SEG)) % 1.0
                    col = _hsv_to_rgb(sh, self.sat, self.val)
                    d.arc(bbox, a0, a1, fill=col, width=lw)
            else:
                cnt = max(1, ring["count"])
                for k in range(cnt):
                    ang = 2.0 * math.pi * k / cnt
                    px = self.cx + r * math.cos(ang)
                    py = self.cy + r * math.sin(ang)
                    # 点阵按角度做 hue 渐变
                    ah = (hue + 0.04 * ri + span * (k / cnt)) % 1.0
                    col = _hsv_to_rgb(ah, self.sat, self.val)
                    d.ellipse([px - 2, py - 2, px + 2, py + 2], fill=col)
        arr = np.asarray(img, np.float32)
        # 加性叠加（与既有图层同约定）
        out = np.clip(out + arr, 0, 255).astype(np.uint8)
        return out
