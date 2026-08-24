"""
wireframe3d.py — 旋转 3D 线框渲染器（自包含，纯 NumPy）

用 4×4 旋转矩阵 + 透视投影把 3D 顶点投到 2D，
再用 DDA 把每条棱边栅格化进 NumPy 画布（加性发光）。

支持形状：cube（立方体）、icosahedron（二十面体）、
sphere（经纬球网）、tetrahedron（四面体）。

调研依据：计算机图形学标准管线 = 模型矩阵(旋转/缩放)
→ 视图矩阵(相机) → 投影矩阵(透视/正交)。本模块用齐次
坐标 + 透视除法实现，与标准教程 (datafield.dev 等) 一致。
"""

from typing import Dict, List, Tuple

import numpy as np


# ── 形状定义（顶点 + 棱边索引，已居中、半径≈1）─────────────
def _cube() -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    v = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64)
    e = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
    return v, e


def _tetrahedron() -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    v = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
    ], dtype=np.float64) / np.sqrt(3)
    e = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return v, e


def _icosahedron() -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    v /= np.linalg.norm(v[0])
    e = [(0, 11), (0, 5), (0, 1), (0, 7), (0, 10),
         (1, 5), (1, 7), (1, 8), (1, 9),
         (2, 3), (2, 4), (2, 6), (2, 10), (2, 11),
         (3, 4), (3, 6), (3, 8), (3, 9),
         (4, 5), (4, 9), (4, 11),
         (5, 9), (5, 11),
         (6, 7), (6, 8), (6, 10),
         (7, 8), (7, 10),
         (8, 9), (9, 10), (10, 11), (11, 5)]
    return v, e


def _sphere(lat: int = 8, lon: int = 12) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    verts = []
    edges = []
    idx = {}
    for i in range(lat + 1):
        theta = np.pi * i / lat
        for j in range(lon):
            phi = 2 * np.pi * j / lon
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            idx[(i, j)] = len(verts)
            verts.append([x, y, z])
    for i in range(lat + 1):
        for j in range(lon):
            cur = idx[(i, j)]
            nxt_j = idx[(i, (j + 1) % lon)]
            edges.append((cur, nxt_j))
            if i < lat:
                nxt_i = idx[(i + 1, j)]
                edges.append((cur, nxt_i))
    return np.array(verts, dtype=np.float64), edges


_SHAPES: Dict[str, Tuple[np.ndarray, List[Tuple[int, int]]]] = {
    "cube": _cube(),
    "tetrahedron": _tetrahedron(),
    "icosahedron": _icosahedron(),
    "sphere": _sphere(),
}


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0],
                    [0, c, -s, 0],
                    [0, s, c, 0],
                    [0, 0, 0, 1]], dtype=np.float64)


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s, 0],
                    [0, 1, 0, 0],
                    [-s, 0, c, 0],
                    [0, 0, 0, 1]], dtype=np.float64)


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0],
                    [s, c, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=np.float64)


def _dda_line(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int,
             r: int, g: int, b: int, intensity: float):
    """DDA 栅格化一条直线，加性写进 canvas（越界则跳过）"""
    h, w = canvas.shape[0], canvas.shape[1]
    # 快速跳过：两端都在画布外同侧（线段不可能穿入画布）
    if (x0 < 0 and x1 < 0) or (x0 >= w and x1 >= w) or \
       (y0 < 0 and y1 < 0) or (y0 >= h and y1 >= h):
        return
    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy), 1))
    # 防御：限制步数，避免极端坐标导致的超长循环
    cap = max(w, h) * 2
    if steps > cap:
        steps = cap
    xs = np.linspace(x0, x1, steps + 1)
    ys = np.linspace(y0, y1, steps + 1)
    for px, py in zip(xs, ys):
        ix, iy = int(round(px)), int(round(py))
        if 0 <= ix < w and 0 <= iy < h:
            vr = canvas[iy, ix, 0] + r * intensity
            vg = canvas[iy, ix, 1] + g * intensity
            vb = canvas[iy, ix, 2] + b * intensity
            canvas[iy, ix, 0] = vr if vr < 255 else 255
            canvas[iy, ix, 1] = vg if vg < 255 else 255
            canvas[iy, ix, 2] = vb if vb < 255 else 255


class Wireframe3D:
    """旋转 3D 线框图层（自包含，纯 NumPy）

    config 字段：
      shape: cube / icosahedron / sphere / tetrahedron
      speed: 旋转速度（弧度/秒，可分别给 x/y/z）
      scale: 屏幕占比（0..1，占短边比例）
      fov: 视场角（度）
      color: 复用 pywallpaper 的 SimpleColorGen 配置
    """

    def __init__(self, config: dict, width: int, height: int):
        self.w = width
        self.h = height
        self.time = 0.0

        shape = config.get("shape", "icosahedron")
        self.vertices, self.edges = _SHAPES.get(shape, _SHAPES["icosahedron"])
        self.n_verts = self.vertices.shape[0]

        sp = config.get("speed", 0.6)
        if isinstance(sp, (list, tuple)):
            self.spd = (float(sp[0]), float(sp[1]), float(sp[2]) if len(sp) > 2 else 0.0)
        else:
            self.spd = (float(sp) * 0.7, float(sp), 0.0)

        raw_scale = float(config.get("scale", 0.32))
        # JSON 中 scale 可能用像素单位（如 150），>1 时按画布短边归一化，
        # 避免坐标爆炸导致 DDA 栅格化循环过长（性能陷阱）。
        if raw_scale > 1.0:
            raw_scale = raw_scale / max(self.w, self.h)
        self.scale = min(max(raw_scale, 0.02), 0.98)
        self.fov = float(config.get("fov", 55.0))
        self.intensity = float(config.get("intensity", 0.9))
        self.dist = 3.2  # 相机到原点距离

        # 颜色（延迟导入避免循环依赖）
        from pw_renderer import SimpleColorGen
        self.color_gen = SimpleColorGen(config.get("color", {}))

        # 齐次坐标缓存
        self._hom = np.ones((4, self.n_verts), dtype=np.float64)
        self._hom[:3, :] = self.vertices.T

    def update(self, dt: float = 0.033):
        self.time += dt

    def _project(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回屏幕坐标 (2,N) 与可见性 (N,)"""
        ax, ay, az = tuple(self.time * s for s in self.spd)
        R = _rz(az) @ _rx(ax) @ _ry(ay)
        cam = R @ self._hom  # (4,N)
        # 视图：相机在原点看 -z，物体整体前移 dist → z 为正
        zc = cam[2, :] + self.dist
        visible = zc > 0.05
        f = np.tan(np.radians(self.fov) / 2.0) * self.dist
        sc = np.where(zc > 0.05, f / zc, 0.0)
        sx = cam[0, :] * sc
        sy = cam[1, :] * sc
        px = self.w / 2 + sx * self.scale * min(self.w, self.h)
        py = self.h / 2 - sy * self.scale * min(self.w, self.h)  # y 轴向下
        return np.stack([px, py], axis=0), visible

    def draw_to_array(self, canvas: np.ndarray) -> np.ndarray:
        """把线框加性绘制进 canvas（H,W,3）uint8"""
        self.color_gen.time = self.time
        pts, vis = self._project()
        r, g, b = self.color_gen.get_color(0, 1, self.time)
        for i, j in self.edges:
            if not (vis[i] and vis[j]):
                continue
            _dda_line(canvas,
                       int(round(pts[0, i])), int(round(pts[1, i])),
                       int(round(pts[0, j])), int(round(pts[1, j])),
                       int(r * self.intensity), int(g * self.intensity),
                       int(b * self.intensity), 1.0)
        return canvas


if __name__ == "__main__":
    import time
    for shape in ["cube", "icosahedron", "sphere", "tetrahedron"]:
        wf = Wireframe3D({"shape": shape, "scale": 0.3}, 256, 256)
        t0 = time.perf_counter()
        for _ in range(10):
            wf.update(0.033)
            c = wf.draw_to_array(np.zeros((256, 256, 3), dtype=np.uint8))
        dt = (time.perf_counter() - t0) / 10 * 1000
        lit = int((c.sum(axis=2) > 0).sum())
        print(f"[wireframe3d] {shape:14s} 单帧 {dt:.2f} ms  点亮像素 {lit}")
    print("OK")
