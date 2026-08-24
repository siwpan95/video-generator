# G:\ai\video_generator 增强调研与方案

> 联网调研（2025–2026 公开资料）+ 源码复核，输出"还能怎么增强"的落地蓝图。
> 目标：在不破坏现有 9685 效果库 / 音频混入 / 多编码等核心能力的前提下，补齐瓶颈、接入新范式。
> 复查的对象：`random_composer.py`（`RandomComposer.compose/compose_from_keywords`）、`effect_model.py`（`EffectConfig.pw_layers`、`PRIMITIVE_MAP` 注册）、`video_generator.py`（`VideoGenerator.generate`、后处理）、`audio_reactive.py`、`cloud_generator.py`。

---

## 一、现状速写与瓶颈（基于源码）

**已具备的强项**
- 双源 9685 效果（draw 网格 CA + pywallpaper 原语），`EffectConfig` 统一模型 + `EffectRegistry`。
- 渲染：`draw_renderer`（BZ/扩散/CA/波，scipy + Numba JIT 卷积）、`pw_renderer`（粒子/流体/波浪/几何/3D 线框/流场/噪声/文本雨）。
- 调色：感知均匀色图（viridis/plasma/…/Turbo）+ `palettes_extra`。
- 噪声：`noise_field`（Perlin/Simplex/Worley/域扭曲）。
- 音频：`audio_reactive`（能量/beat/质心/通量/BPM 四重降级）+ `audio_mixer` 真混入 mp4（librosa→ffmpeg→numpy）。
- 编码：`h264/hevc/av1`、NVENC AQ/B 帧、4K、yuv420p+faststart。
- 云端：`cloud_generator`（glitch prompt → API → 下载）。

**真实瓶颈（决定增强优先级）**
1. **CPU 渲染上限**：域扭曲纯 Python 逐像素多趟 fbm ≈ 1.2 fps（ENHANCEMENT_OVERVIEW 自述）；BZ/扩散逐帧演化靠 Numba CPU JIT。分辨率一上 4K 即崩。
2. **组合无语义控制**：`compose_from_keywords` 是字符串模糊匹配（`kw in name/category/tags`），没有"语义 → 参数区间"的映射。
3. **无自动质量筛选**：随机种子产出大量"废片"，靠人眼挑，无法直接用于专辑/批量生产。
4. **无实时预览/交互**：每次都是离线条带式渲染 → 编码，调参闭环慢。
5. **原语库偏"老派程序化"**：缺 raymarching SDF、metaballs、strange attractors、L-system、DLA 等 2025 仍活跃的新原语。
6. **音频仅整体调制 + SFX**：`--audio random` 是 340+ 音效库，没有分轨（stem）驱动，也没有 AI 配乐。

---

## 二、联网调研要点（2025–2026）

| 方向 | 关键结论 | 对项目的启示 |
|------|----------|--------------|
| **LLM-Conditioned PCG** | 2024–2025 热点范式：用 LLM 作自然语言控制器，零样本/双智能体（Actor 规划 + Critic 校验约束）把抽象指令映射到结构化参数（EmergentMind 综述，arXiv 2512.10501 双智能体地图生成） | 把 `--keywords` 升级为"自然语言 → 效果选择 + 参数区间"，Critic 校验图层数/调色板兼容性 |
| **AI Co-Artist（LLM + GLSL）** | GPT-4 演化 GLSL 着色器做实时交互艺术（arXiv 2512.08951），证明"LLM 语义理解 + 程序合成"可泛化到创意编码 | LLM 不止选参数，还能**生成 GLSL 新原语**喂给 GPU 后端 |
| **实时 GPU 渲染** | ModernGL / VisPy / GLSL fragment shader 可在边缘设备 60fps 渲染（moderngl 5.12、VisPy 2025）；AI Co-Artist 同样依赖实时着色器 | 把噪声场/fBm/域扭曲/BZ 演化搬进 fragment shader（FBO → ffmpeg），10–50x 提速 + 实时预览 |
| **开源视频扩散 2026** | Wan2.2 5B（GGUF 8GB 可跑）、LTX-2.3（**原生音视频同步**、4K@50fps、20s）、HunyuanVideo 1.5（14GB FP8）、CogVideoX 5B（16GB）已可本地推理 | 可加"扩散基础层"：用 Wan2.2/LTX-2.3 生成底片/关键帧，再用现有程序化原语做风格化叠加；LTX-2.3 还能直接给**对齐音轨** |
| **新生成原语** | raymarching SDF（Mandelbulb/Julia）、metaballs、strange attractors（Lorenz/Clifford/De Jong）、L-system 植物、DLA 分形聚合（MDPI Fractal Fract 2025）仍是生成艺术活跃方向 | 作为新 `PRIMITIVE_MAP` 条目补齐，GLSL 友好、视觉冲击强 |
| **自动美学评分/筛选** | CLIP + MLP 回归头评视频美学（VideoAestheticEvaluator）、VBench 多维诊断、FVD/CLIP-SIM 文本对齐（Springer 2025 综述） | 生成后加"自动评分 + 连镜表（contact sheet）"，top-N 留存，直接服务批量/专辑生产 |

---

## 三、增强蓝图（落到具体模块，按优先级）

### P0 — 高价值、低风险、可增量（不动渲染核心）

**P0-1 · LLM 语义组合层（接 `RandomComposer`）**
- 新增 `llm_composer.py`：输入自然语言（如"深海发光、慢节奏、冷色调"）→ 调用 LLM（复用 `cloud_generator` 的云端凭证，或 WorkBuddy AgentKey / DeepSeek）输出结构化 JSON：`{layers:[{source, keywords, params_range}], theme, postprocess, palette_bias}`。
- 采用 **Actor–Critic 双智能体**：Actor 出方案，Critic 校验约束（图层数 1–4、调色板不过载、draw/pw 混排合理），对齐 arXiv 2512.10501。
- 复用现有 `compose_from_keywords` 的关键词检索做落地，参数区间用 `random.uniform` 采样（已有现成 `_random_params`）。
- CLI：`--prompt "..."`（替代/补充 `--keywords`）。

**P0-2 · 自动美学筛选 + 连镜表（接 `VideoGenerator.generate` 之后）**
- 新增 `curator.py`：轻量评分，**优先启发式**（零额外依赖）：亮度分布熵、运动幅度（帧差 RMS）、beat 同步度（若有音频）；可选 CLIP+MLP（CLIP ViT-B/32 本地跑）做语义美学分。
- 批量模式 `--batch N --keep-top K`：生成 N 个，按分留存 top-K，输出 `contact_sheet.png` + `manifest.json`（参考 musickit 的 `manifest_index.json` 模式）。
- 不通过评分绝不直接删片，仅"推荐留存"。

**P0-3 · Demucs 分轨音频驱动（升级 `audio_reactive`）**
- 你本机已有 demucs 实战经验（记忆：音频拆轨/人声分离）。新增 `audio_stems.py`：用 demucs 把 `--audio` 拆成 vocals/bass/drums/other。
- 多流调制：`drums → beat 触发转场/数据冲击`、`bass → 几何密度/缩放`、`vocals → 色温`、`other → 全局辉光`。比现在"整体能量"信息量大一个量级。
- 降级：无 demucs 时退回现有单流路径（沿用四重降级哲学）。

### P1 — 性能与表现力跃迁

**P1-1 · ModernGL GLSL 渲染后端（替换 Pillow 逐像素）**
- 新增 `gl_renderer.py`：`moderngl` 创建离屏 context，把 `noise_field`（fBm/域扭曲）、`draw_renderer` 的 BZ/扩散 3×3 卷积、wireframe 投影搬进 fragment/compute shader，渲到 FBO → `np.frombuffer` → ffmpeg。
- 预期：域扭曲从 ~1.2 fps → 实时级，4K 可行。顺带实现 `--preview-live`（moderngl-window 实时拖参）。
- 依赖：`pip install moderngl`（headless `glcontext` 本机 Win+GPU 可用），缺 GPU 时回退 CPU 路径。

**P1-2 · 新生成原语（注册进 `pw_renderer.PRIMITIVE_MAP`）**
- `raymarch_sdf`：球体/环面/Mandelbulb/Julia 集 SDF + 软阴影/环境光，3D 纵深是 `wireframe_3d` 缺的。
- `attractors`：Lorenz / Clifford / De Jong 轨道迹（便宜、极美）。
- `lsystem`：L-system 植物/分形树（可变角度/长度随机化）。
- `metaballs` / `dla`：有机团块与分形聚合。
- 每个新原语写 1 个合成 JSON（沿用已落地的 7 个 `__builtin_*` 注册范式）进随机合成与关键词检索。

**P1-3 · 专辑/批量模式 + manifest（musickit 模式）**
- 语义连贯的 N 首（同主题/同源色板/连续 seed），输出目录 + `manifest.json`（source/original/path/score），可直接交给电台或 BloodMusic 专辑流程。

### P2 — 范式扩展（按需、风险/资源更高）

**P2-1 · 扩散模型图层**
- `--diffusion-base wan2.2`：本地 Wan2.2-TI2V-5B（GGUF 8GB）或 LTX-2.3 生成底片，再用现有程序化原语（色图/噪声/音频调制）做"风格化叠加"；LTX-2.3 可顺带产出**对齐音轨**替代 `--audio random`。
- 显存不足时优雅降级为纯程序化。

**P2-2 · 交互式 Web UI（Gradio/Streamlit）**
- 含 LLM 聊天框（`--prompt`）、实时参数滑杆、音频上传、社媒预设（9:16 / 1:1）、一键批量 + 连镜表。

**P2-3 · 导出与分发增强**
- HDR / 10-bit（hevc hdr10 / av1）、竖屏社媒预设、WebM/动态 GIF；产物可直推 `G:\radio` 的 HLS 直播流。

---

## 四、MVP 落地路线图

| 阶段 | 内容 | 风险 | 依赖 |
|------|------|------|------|
| 1 | P0-1 LLM 语义组合 + P0-2 自动筛选 | 低（纯 Python，不碰渲染） | LLM 凭证（复用 cloud_generator / AgentKey） |
| 2 | P0-3 Demucs 分轨 | 低（沿用你现有 demucs 环境） | demucs + torch |
| 3 | P1-1 ModernGL 后端 | 中（需 OpenGL 环境） | moderngl + glcontext |
| 4 | P1-2 新原语 + P1-3 专辑模式 | 中 | 无强依赖 |
| 5 | P2-1 扩散图层 + P2-2 UI + P2-3 导出 | 高（显存/复杂度） | 8–24GB 显存可选 |

> 建议先做阶段 1–2：纯增量、零渲染改动，立刻让"随机生成"变成"可控 + 可挑"，与你做专辑/电台的实际工作流最贴合。

---

## 五、风险与降级原则（沿用项目既有哲学）

- **优雅降级优先于崩溃**：所有新依赖（demucs / moderngl / CLIP / 扩散）缺省时回退现有路径（项目已是四重降级 + `except Exception` 风格）。
- **不破坏既有接口**：`VideoGenerator.generate` 签名、`PRIMITIVE_MAP`、9685 效果库一律向后兼容；新功能走新参数/新模块。
- **可复现性保留**：`--seed` 贯穿 LLM 组合（固定采样）与渲染，确保结果可重跑。
- **GLSL 需本机有 OpenGL**：本机 Windows + 独立 GPU 可用 `moderngl[headless]`；纯 CPU 环境走 numpy 回退。
- **扩散模型显存门槛**：8GB 起，设 `--diffusion-base` 显式开关，默认关闭。

---

## 六、调研来源（节选）

- EmergentMind, *LLM-Conditioned Procedural Generation* / *Procedural Content Generation with LLMs*（2024–2025 综述）
- arXiv 2512.08951 *AI Co-Artist: LLM-Powered GLSL Shader Animation Evolution*（2025-11）
- arXiv 2512.10501 *Zero-shot 3D Map Generation with LLM Agents*（双智能体 Actor–Critic，2025-12）
- thundercompute / localaimaster / apatero 2026 开源视频模型横评（Wan2.2 / LTX-2.3 / HunyuanVideo 1.5）
- ModernGL 5.12、VisPy 2025 GPU 渲染文档
- MDPI *Fractal Fract* 2025，DLA / L-system / strange attractors 生成艺术
- Springer *Text-to-video generators: a comprehensive survey*（2025，CLIP-SIM/FVD/VBench）
- OpenDCAI *VideoAestheticEvaluator*（CLIP + MLP 视频美学评分）
