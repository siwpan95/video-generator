"""
gl_renderer.py — GPU(GLSL) 渲染后端（P1-1）

用 OpenGL(GLSL 着色器) 在 GPU 上渲染程序化图案，作为视频的「基底帧」，
再交给 video_generator 既有的 _apply_global_postprocess_numpy 做音频调制
（辉光/暗角/色相/beat 撕裂/分轨调制）与 ffmpeg 编码——GPU 提速 + 全部
音频交互不变。

为什么不用 ModernGL：ModernGL 的上下文依赖 glcontext 需编译原生扩展，
而本机无 C 编译器（MSVC/gcc 均缺），pip 编译失败。改用**预编译 wheel**
的 glfw（建上下文）+ PyOpenGL（系统 opengl32.dll 的 ctypes 封装，纯 Python），
零编译即可在本机跑真·GLSL。

降级铁律（与项目一致）：glfw/PyOpenGL 缺失、或上下文创建失败 → 抛
GLUnavailable，调用方降级为现有 CPU(pw_renderer) 管线，绝不崩溃。

着色器设计：单一 fragment 程序 + `u_family` int 分支切换 6 种图案家族
（noise/flow/plasma/aurora/starfield/ca），统一用 IQ 余弦调色板
pal(t)=a+b*cos(2π(c·t+d))，由 u_pa..u_pd 四组 vec3 uniform 驱动，主题
映射到不同预设。音频能量 u_energy 注入着色器做亮度/脉动。

对外 API：
    try:
        gl = GLRenderer(w, h, fps, family="flow", palette="neon", seed=1)
        arr = gl.render_frame(t=1.0, frame_idx=25, energy=0.6)  # (H,W,3) uint8
        gl.close()
    except GLUnavailable:
        # 降级 CPU
"""

import numpy as np

try:
    import glfw
    from OpenGL import GL
    _HAS_GL = True
except Exception:  # noqa: BLE001
    glfw = None
    GL = None
    _HAS_GL = False


class GLUnavailable(Exception):
    """GL 后端不可用（缺依赖/无上下文）时抛出，调用方据此降级。"""


# ── IQ 余弦调色板预设：pal(t) = a + b*cos(2π(c·t + d)) ──────────────────
_PALETTES = {
    "neon": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0),
             (0.00, 0.33, 0.67)),
    "cool": ((0.5, 0.5, 0.6), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0),
             (0.30, 0.20, 0.60)),
    "warm": ((0.6, 0.4, 0.3), (0.4, 0.3, 0.2), (1.0, 1.0, 1.0),
             (0.00, 0.10, 0.20)),
    "green": ((0.4, 0.6, 0.5), (0.4, 0.4, 0.4), (1.0, 1.0, 1.0),
              (0.10, 0.20, 0.30)),
    "mono": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0),
             (0.00, 0.00, 0.00)),
    "fire": ((0.6, 0.3, 0.2), (0.5, 0.3, 0.1), (1.0, 1.0, 1.0),
             (0.00, 0.15, 0.25)),
}

# 主题 → (着色器家族, 调色板预设)
_THEME_MAP = {
    "empty":     ("noise",  "neon"),
    "starfield": ("starfield", "cool"),
    "gradient":  ("plasma", "neon"),
    "aurora":    ("aurora", "green"),
}


def family_for_theme(theme: str) -> str:
    return _THEME_MAP.get(theme, ("noise", "neon"))[0]


def palette_for_theme(theme: str) -> str:
    return _THEME_MAP.get(theme, ("noise", "neon"))[1]


_VERT = """#version 330 core
// 全屏三角形，无需 VBO（用 gl_VertexID）
void main() {
    vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0,
                  (gl_VertexID == 2) ? 3.0 : -1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

_FRAG = """#version 330 core
precision highp float;
out vec4 fragColor;

uniform vec2  u_res;
uniform float u_time;
uniform float u_energy;     // 0..1 音频能量
uniform float u_seed;
uniform float u_speed;
uniform float u_complexity; // 0..1 细节密度
uniform int   u_family;     // 0 noise 1 flow 2 plasma 3 aurora 4 starfield 5 ca
uniform vec3  u_pa; uniform vec3 u_pb; uniform vec3 u_pc; uniform vec3 u_pd;

vec3 pal(float t) { return u_pa + u_pb * cos(6.28318530718 * (u_pc * t + u_pd)); }

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21) + u_seed * 17.0);
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    float a = hash(i), b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0)), d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 6; i++) {
        s += a * vnoise(p);
        p *= 2.02; a *= 0.5;
    }
    return s;
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_res;
    vec2 p = (gl_FragCoord.xy - 0.5 * u_res) / u_res.y;
    float t = u_time * u_speed;
    float e = u_energy;
    vec3 col = vec3(0.0);

    if (u_family == 0) {            // noise：域扭曲 fbm
        vec2 q = vec2(fbm(p * 3.0 + t * 0.1), fbm(p * 3.0 - t * 0.12 + 5.2));
        float f = fbm(p * (3.0 + u_complexity * 4.0) + q * (2.0 + e * 2.0));
        col = pal(f + 0.1 * t + e * 0.3);
    } else if (u_family == 1) {     // flow：流场条带
        float a = fbm(p * 2.5 + t * 0.15);
        vec2 dir = vec2(cos(a * 6.28), sin(a * 6.28));
        float f = fbm(p * 4.0 + dir * (t * 0.4) + e);
        col = pal(0.5 + 0.5 * f + 0.15 * t);
    } else if (u_family == 2) {     // plasma：正弦等离子
        float v = sin(p.x * 6.0 + t) + sin(p.y * 6.0 - t * 1.3)
                + sin((p.x + p.y) * 5.0 + t * 0.7)
                + sin(length(p) * 8.0 - t);
        col = pal(0.5 + 0.25 * v + e * 0.2);
    } else if (u_family == 3) {     // aurora：层叠极光带
        float band = 0.0;
        for (int i = 0; i < 4; i++) {
            float fi = float(i);
            band += sin(p.x * (2.0 + fi) + t * (0.5 + 0.2 * fi)
                        + fbm(vec2(p.x * 2.0, fi)) * 3.0)
                    * exp(-abs(p.y - 0.2 * fi + 0.3 * sin(t + fi)) * (2.0 + u_complexity * 3.0));
        }
        col = pal(0.55 + 0.35 * band + e * 0.25);
    } else if (u_family == 4) {     // starfield：星空
        vec2 g = floor(gl_FragCoord.xy / (3.0 + u_complexity * 6.0));
        float star = step(0.985 - e * 0.01, hash(g));
        float tw = 0.5 + 0.5 * sin(t * 3.0 + hash(g + 3.7) * 30.0);
        col = vec3(star * tw) * pal(0.6 + 0.3 * hash(g + 1.1));
        col += vec3(0.02, 0.03, 0.06);  // 夜空底
    } else {                        // ca：网格元胞
        float cells = 4.0 + u_complexity * 20.0;
        vec2 c = floor(p * cells);
        float m = hash(c + floor(t * (0.5 + e)));
        float edge = smoothstep(0.0, 0.08, fract(p.x * cells))
                    * smoothstep(1.0, 0.92, fract(p.x * cells))
                    * smoothstep(0.0, 0.08, fract(p.y * cells))
                    * smoothstep(1.0, 0.92, fract(p.y * cells));
        col = pal(m + 0.1 * t) * (0.3 + 0.7 * edge) * (0.5 + 0.8 * e);
    }

    // 轻微音频提亮（后处理还会再叠辉光/暗角/beat）
    col *= 0.8 + 0.5 * e;
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


# 模块级单例窗口/上下文：避免 glfw 在批量/多视频场景反复
# create/destroy 导致重建上下文失败（窗口拿不到 GL context）。
_gl_window = None
# 模块级单例 GL 资源：着色器程序与 VAO 整个上下文只建一次、永不删除，
# 跨所有 GLRenderer 实例/批量复用（进程退出时由 OS 回收）。
_gl_prog = None
_gl_u = None
_gl_vao = None


class GLRenderer:
    """GPU(GLSL) 程序化图案渲染器。"""

    FAMILIES = ["noise", "flow", "plasma", "aurora", "starfield", "ca"]

    @classmethod
    def family_for_theme(cls, theme: str) -> str:
        """主题 → 着色器家族（与模块级同名函数行为一致）。"""
        return family_for_theme(theme)

    @classmethod
    def palette_for_theme(cls, theme: str) -> str:
        """主题 → IQ 余弦调色板预设（与模块级同名函数行为一致）。"""
        return palette_for_theme(theme)

    def __init__(self, width: int, height: int, fps: float = 25.0,
                 family: str = "noise", palette: str = "neon",
                 seed: float = 1.0, speed: float = 1.0,
                 complexity: float = 0.5):
        if not _HAS_GL:
            raise GLUnavailable(
                "未安装 glfw / PyOpenGL（pip install glfw PyOpenGL）；"
                "已降级为 CPU 渲染")
        # 模块级单例窗口/上下文：仅首次创建；之后按需 set_window_size
        # 适配分辨率。彻底规避 glfw 反复 create/destroy 重建上下文失败的坑。
        global _gl_window
        if _gl_window is None:
            if not glfw.init():
                raise GLUnavailable("glfw.init() 失败（无可用 OpenGL 上下文）")
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            _gl_window = glfw.create_window(
                max(1, width), max(1, height), "vg_gl", None, None)
            if not _gl_window:
                raise GLUnavailable("glfw 创建窗口失败（无 GPU/上下文）")
        self.window = _gl_window
        glfw.set_window_size(self.window, max(1, width), max(1, height))
        glfw.make_context_current(self.window)

        self.width = width
        self.height = height
        self.fps = fps
        self.family_idx = self.FAMILIES.index(family) \
            if family in self.FAMILIES else 0
        pa, pb, pc, pd = _PALETTES.get(palette, _PALETTES["neon"])
        self.palette = (np.array(pa, np.float32), np.array(pb, np.float32),
                        np.array(pc, np.float32), np.array(pd, np.float32))
        self.seed = float(seed)
        self.speed = float(speed)
        self.complexity = float(complexity)

        self._ensure_gl_resources()
        GL.glViewport(0, 0, width, height)
        # 上下文保持 current：单线程直接可用；worker 线程会再 make_current 认领

    def _ensure_gl_resources(self):
        """模块级单例：整个 GL 上下文只编译一次着色器程序与 VAO，
        跨所有 GLRenderer 实例/批量复用；永不删除（进程退出时 OS 回收）。"""
        global _gl_prog, _gl_u, _gl_vao
        if _gl_prog is not None:
            return
        vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vs, _VERT)
        GL.glCompileShader(vs)
        self._check_shader(vs, "vertex")
        fsh = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fsh, _FRAG)
        GL.glCompileShader(fsh)
        self._check_shader(fsh, "fragment")
        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vs)
        GL.glAttachShader(prog, fsh)
        GL.glLinkProgram(prog)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            raise GLUnavailable("着色器链接失败: "
                                + GL.glGetProgramInfoLog(prog).decode())
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fsh)
        _gl_prog = prog
        _gl_u = {n: GL.glGetUniformLocation(prog, n) for n in (
            "u_res", "u_time", "u_energy", "u_seed", "u_speed",
            "u_complexity", "u_family", "u_pa", "u_pb", "u_pc", "u_pd")}
        _gl_vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(_gl_vao)
        GL.glViewport(0, 0, 1, 1)  # 占位，render_frame 按帧重设


    @staticmethod
    def _check_shader(shader, kind):
        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            raise GLUnavailable(f"{kind} 着色器编译失败: "
                                + GL.glGetShaderInfoLog(shader).decode())

    def make_current(self):
        """渲染线程调用：把 GL 上下文绑到当前线程。"""
        if _HAS_GL and getattr(self, "window", None) is not None:
            glfw.make_context_current(self.window)

    def release_context(self):
        """主线程建完上下文/编译完着色器后调用：释放当前线程的 GL 上下文，
        交由渲染 worker 线程认领。OpenGL 上下文同一时刻只能绑定一个线程，
        不释放则 worker 的 make_context_current 无法认领 → glUseProgram 等
        解析为 NULL → NullFunctionError。"""
        if _HAS_GL and getattr(self, "window", None) is not None:
            glfw.make_context_current(None)

    def set_family(self, family: str):
        """热切换着色器家族（u_family 是 uniform，单上下文即可）。"""
        if family in self.FAMILIES:
            self.family_idx = self.FAMILIES.index(family)

    def render_frame(self, t: float, frame_idx: int,
                     energy: float = 0.5) -> np.ndarray:
        self.make_current()  # 确保 GL 上下文绑定到本线程（主线程/worker 均安全）
        GL.glUseProgram(_gl_prog)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glUniform2f(_gl_u["u_res"], self.width, self.height)
        GL.glUniform1f(_gl_u["u_time"], float(t))
        GL.glUniform1f(_gl_u["u_energy"], float(energy))
        GL.glUniform1f(_gl_u["u_seed"], self.seed)
        GL.glUniform1f(_gl_u["u_speed"], self.speed)
        GL.glUniform1f(_gl_u["u_complexity"], self.complexity)
        GL.glUniform1i(_gl_u["u_family"], self.family_idx)
        GL.glUniform3fv(_gl_u["u_pa"], 1, self.palette[0])
        GL.glUniform3fv(_gl_u["u_pb"], 1, self.palette[1])
        GL.glUniform3fv(_gl_u["u_pc"], 1, self.palette[2])
        GL.glUniform3fv(_gl_u["u_pd"], 1, self.palette[3])

        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glBindVertexArray(_gl_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

        buf = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        GL.glReadPixels(0, 0, self.width, self.height,
                        GL.GL_RGB, GL.GL_UNSIGNED_BYTE, buf)
        # GL 底朝上 → 翻转为上朝下
        return buf[::-1].copy()

    def close(self):
        # 仅释放本实例的 GL 对象（着色器程序/VAO）；不销毁共享窗口、不 terminate
        # glfw（窗口为模块级单例，跨实例/批量复用，进程退出时由 OS 回收）。
        try:
            if _HAS_GL and getattr(self, "window", None) is not None:
                glfw.make_context_current(self.window)
                if getattr(self, "prog", None) is not None:
                    GL.glDeleteProgram(self.prog)
                if getattr(self, "_vao", None) is not None:
                    GL.glDeleteVertexArrays(1, [self._vao])
        except Exception:  # noqa: BLE001
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    import sys
    if not _HAS_GL:
        print("[gl_renderer] glfw/PyOpenGL 未安装，无法自检")
        sys.exit(1)
    w = h = 64
    for fam in GLRenderer.FAMILIES:
        gl = GLRenderer(w, h, 25.0, family=fam, palette="neon", seed=2.0)
        fr = gl.render_frame(1.0, 0, 0.6)
        uniq = len(np.unique(fr.reshape(-1, 3), axis=0))
        print(f"[gl_renderer] {fam:10s} shape={fr.shape} 唯一色={uniq}")
        gl.close()
    print("OK")
