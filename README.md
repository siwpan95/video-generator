# 随机视频生成器

> 仓库：<https://gitee.com/siwpan/video-generator> ｜ 许可：MIT（见 [LICENSE](LICENSE)）
> 注：文中 <PATH> 为路径占位符，请替换为你本机实际的目录 / 可执行文件路径。

结合 `draw/effects/` 和 `pywallpaper/effects/` 两个项目的共计 **9685 个效果预设**，构建全新的随机视频生成器。

## 功能特性

- **双源效果库**：自动加载两个项目的 effect JSON 文件
- **随机组合**：随机选效果、随机参数、随机图层叠加、随机主题背景
- **多种效果类型**：
  - draw 效果：网格 CA（反应扩散、图灵图案、细胞自动机、量子…）
  - pywallpaper 效果：粒子系统、流体模拟、波浪、几何、3D 线框…
- **主题背景**：星空 / 渐变 / 极光 / 微光
- **图层混合**：overlay / add / multiply / screen
- **后处理**：暗角（vignette）/ 辉光（glow）/ 边缘增强
- **可复现**：通过 `--seed` 参数固定随机种子
- **关键词模式**：通过 `--keywords` 按效果名称/类别筛选
- **音频合成（generate 内统一 mux）**：`--audio random` 或从文件指定音轨，由 `VideoGenerator.generate()` 在编码后直接混入 mp4——CLI 与 `gen_sample.py` 走同一条 mux 路径；`audio_mixer.find_ffmpeg` 同样优先读 `FFMPEG_BIN`
- **云端生成模式**：通过 `--cloud` 调用云端 API 生成 glitch 艺术视频（需凭证）
- **sfx 集成**：复用 `sfx/` 项目的音效库（340+ 音效）
- **音频驱动效果**：`--audio` 同时驱动视觉——能量包络调制主题亮度/辉光，节拍触发数据冲击转场
- **图层转场**：smoothstep 缓动淡入淡出（fade-in/out），图层切换更顺滑自然
- **感知均匀色图**：viridis / plasma / inferno / magma / cividis（避免 jet 伪彩的条带与色觉缺陷）
- **新增渲染原语**：Perlin 噪声流体（noise）、3D 旋转线框（wireframe_3d）、文本雨（text_rain）
- **Numba JIT 卷积加速**：draw 演化（bz/diffusion/life/wave）的 3×3 卷积同时接入 Numba JIT 路径，`@njit parallel` 编译，与 scipy 数值逐位等价
- **音频色相偏移**：`--audio-hue-shift` 让画面色相随节拍按黄金比脉动（HSV 旋转后处理，beat 同步）
- **draw JSON 公式**：asteval 安全求值，支持 `if/else` 与**链式比较**（`0.2 < r < 0.25` 类环形初始化），无静默降级
- **编码增强**：`--codec h264/hevc/av1`，NVENC 质量参数（spatial/temporal AQ + rc-lookahead + B 帧）+ yuv420p + faststart
- **分辨率预设**：`--resolution 4k/1080p/720p`（覆盖 --width/--height）
- **FFMPEG_BIN 环境变量**：可指向任意 ffmpeg 二进制（最高优先级，绕过系统 PATH）
- **Google Turbo 色图**：新增 Turbo 多项式色图（官方 GLSL 系数，端点 R0=34/R255=144），随 viridis/plasma 等一同随机选用，低光可读性强于 jet
- **新增噪声原语**：Simplex 噪声（各向同性、无轴向条带）、Worley/Cellular 噪声（Voronoi 观感）、域扭曲 Domain Warping（`fbm(p+warp·fbm)` 有机流体纹理）
- **流场粒子原语**：`flow_field`（curl 无散度噪声 / angle 角度场双模式，带拖尾流线）
- **音频频谱特征**：`--audio-balance` 按频谱质心调色温（明亮→暖、低→冷）；音频特征额外支持频谱通量（段落/onset）与 BPM
- **编码调优**：hevc 自动加 `-tag:v hvc1`（Apple 生态兼容）；av1 加 `-cpu-used 4 -row-mt 1`（CPU）/ `-tag:v av01`（NVENC）

## 安装依赖

```bash
pip install numpy pillow opencv-python-headless
# 可选增强依赖（按需安装，缺省时优雅降级）：
pip install scipy     # 网格演化卷积加速（比 np.roll 快 3-5x）
pip install asteval   # draw JSON 完整公式 / 颜色表达式安全求值
pip install librosa   # 音频驱动（精确 RMS 能量 + onset/beat 检测）
```
> 系统依赖：ffmpeg（视频编码与音频混入，需单独安装并加入 PATH）

## 使用方法

```bash
# 完全随机生成（默认 10 秒、30fps、768x768）
python video_generator.py

# 可复现随机（指定种子）
python video_generator.py --seed 42

# 按关键词选择效果
python video_generator.py --keywords BZReaction particle

# 自定义参数
python video_generator.py --duration 15 --fps 60 --width 1024 --height 1024

# 预览模式（仅渲染前 30 帧，快速验证）
python video_generator.py --preview

# 混入随机音效
python video_generator.py --audio random

# 指定音频文件
python video_generator.py --audio path/to/sound.mp3

# 调整音频音量
python video_generator.py --audio random --audio-volume 0.5

# 选择编码格式（默认 h264；支持 hevc / av1）
python video_generator.py --codec hevc

# 音频驱动视觉 + 色相随节拍偏移（HSV 旋转后处理）
python video_generator.py --audio beat.wav --audio-hue-shift

# 端到端样例（不指定 FFMPEG_BIN 时自动用 bundled essentials build；
# 也可手动指向任意 ffmpeg，例如雷电云 LD-Yun 自带的 <PATH>）
set FFMPEG_BIN=<PATH>
python gen_sample.py

# 分辨率预设（覆盖 --width/--height）：4k / 1080p / 720p
python video_generator.py --resolution 4k

# 音频驱动色温随频谱质心漂移（明亮→暖、低→冷）
python video_generator.py --audio music.mp3 --audio-balance

# 关键词命中新增原语（已注册进随机合成与检索）
python video_generator.py --keywords flow_field
python video_generator.py --keywords noise_simplex
python video_generator.py --keywords warp
python video_generator.py --keywords text_rain
python video_generator.py --keywords wireframe_3d

# 云端生成模式（glitch-video-generator 原始功能）
python video_generator.py --cloud

# 云端生成 + 指定种子 + 音频
python video_generator.py --cloud --seed 42 --audio random

# 指定输出路径
python video_generator.py --output my_video.mp4

# 列出所有可用效果
python video_generator.py --list-effects
```

## 云端生成模式（--cloud）

调用 `glitch-video-generator` 原始功能：
1. 用 `random_glitch_prompt.py` 生成随机 prompt
2. 通过 `buddy-cloud.py` 调用云端 API
3. 下载结果视频
4. 可选：混入音频

需要云端凭证（在 WorkBuddy 环境中自动获取）。

## 音频合成（mux 进 mp4）

音频混合并非 CLI 专属——`VideoGenerator.generate()` 在视频编码完成后，若 `audio_path` 非空且 `audio_mux=True`（默认），会直接调用 `audio_mixer.mux_audio` 把音轨写入 mp4。因此 **CLI 与 `gen_sample.py` 走的是同一条 mux 路径**，不会出现"只驱动视觉、音轨丢失"的情况。

支持两种方式：
- `--audio random`：从 `sfx/sounds/` 目录随机选一个音效（340+ 个）
- `--audio <path>`：指定音频文件路径（支持 .mp3/.wav/.ogg/.m4a）

mux 使用 `ffmpeg`（`-c:v copy -c:a aac -shortest`，音量可调 `--audio-volume`），其二进制解析优先读 `FFMPEG_BIN` 环境变量，与视频编码器保持一致。

## 项目结构

```
video_generator/
├── effect_model.py        # 统一 Effect 数据模型
├── effect_loader.py      # 从本地 effects/ 目录加载效果
├── draw_renderer.py     # draw 效果渲染器（网格 CA）
├── pw_renderer.py       # pywallpaper 效果渲染器（原语图层）
├── random_composer.py   # 随机组合生成器
├── video_generator.py    # 主脚本（本地视频生成入口）
├── audio_mixer.py       # 音频合成工具（ffmpeg 混入音效）
├── cloud_generator.py    # 云端生成器（glitch-video-generator 模式）
├── palettes_extra.py     # 感知均匀色图（viridis/plasma/inferno/magma/cividis + Turbo）
├── noise_field.py        # Perlin / Simplex / Worley 噪声场 + 域扭曲（流体 / 纹理背景）
├── wireframe3d.py        # 3D 旋转线框投影（立方体 / 二十面体 / 球 / 四面体）
├── formula_eval.py       # asteval 安全公式求值（draw JSON 公式 / 颜色表达式）
├── audio_reactive.py     # 音频驱动调制（可选 librosa / ffmpeg / 纯 numpy 降级）
├── random_glitch_prompt.py  # 随机 glitch prompt 生成器
├── gen_sample.py         # 端到端样例生成（BZ + 3D 线框 + 音频色相偏移 → sample_bz_wireframe.mp4）
├── sample_bz_wireframe.mp4  # 端到端样例产物（512x512 @30fps，6s，h264，**含 aac 节拍音轨**）
├── effects/               # 本地 effects 副本（避免跨项目依赖）
│   ├── draw/             # draw/effects/ 副本（4972 个）
│   └── pywallpaper/     # pywallpaper/effects/ 副本（4713 个）
├── sfx/                   # 音效下载器（从原项目复制）
│   └── sounds/            # 下载的音效库（340+ 个）
├── output/                # 输出视频目录
├── test_draw.gif          # draw 渲染器测试输出
├── test_pw.gif           # pw 渲染器测试输出
├── README.md             # 本文档
└── requirements.txt      # 依赖清单
```

## 效果库统计

| 来源 | 效果数量 | 主要类别 |
|------|----------|---------|
| `draw/effects/` | 4972 | 反应扩散、图灵图案、细胞自动机、量子、螺旋… |
| `pywallpaper/effects/` | 4713 | 粒子系统、流体模拟、波浪、几何、3D、频谱…（含 7 个新增合成特效 JSON） |
| **总计** | **9685** | 覆盖数学艺术、生成艺术主要方向 |

## 视频组合参数

每次随机生成包含：
- **1~4 个图层**，每个图层来自 draw 或 pywallpaper
- **主题背景**（星空 / 渐变 / 极光 / 微光）
- **全局后处理**（暗角 / 辉光 / 边缘增强）
- **随机参数**（调色板、网格尺寸、粒子数量、波幅…）

## 扩展方向（已落地 ✅）

- [x] 支持更多 pywallpaper 原语类型（wireframe_3d、text_rain、noise 已实现）
- [x] 使用 `scipy.ndimage.convolve` 加速 draw 网格演化（比 np.roll 快 3-5x，与原逻辑数值等价）
- [x] 使用 `asteval` 支持完整颜色公式求值（draw JSON 初始化/颜色表达式安全求值）
- [x] draw JSON 公式支持链式比较（`0.2 < r < 0.25` 类），asteval 安全求值无静默降级
- [x] 把 scipy 风格 3×3 卷积加速**也接进 Numba JIT 路径**（与 scipy 数值逐位等价）
- [x] 添加音频色相偏移（`--audio-hue-shift`：色相随节拍按黄金比脉动）
- [x] 添加音频驱动（读取音频文件，能量调制亮度/辉光、节拍触发数据冲击转场）
- [x] 音频真正混入输出 mp4（`VideoGenerator.generate()` 内部统一 mux，取代 CLI 单独 mux 块；`audio_mixer` 同样读 `FFMPEG_BIN`）
- [x] 添加转场效果（smoothstep 缓动淡入淡出 + beat 数据冲击）
- [x] 支持 4K 输出、更高帧率（`--resolution 4k/1080p/720p`）
- [x] 编码增强与 GPU 加速（`--codec h264/hevc/av1`；NVENC AQ/rc-lookahead/B 帧；yuv420p+faststart）
- [x] Google Turbo 多项式色图（官方 GLSL 系数，端点 R0=34/R255=144）
- [x] 新增 Simplex / Worley / 域扭曲噪声原语（各向同性、Voronoi 观感、有机流体）
- [x] 新增流场粒子原语 `flow_field`（curl 无散度 / angle 角度场双模式，带拖尾）
- [x] 音频频谱质心 / 通量 / BPM 特征（`--audio-balance` 按质心调色温）
- [x] hevc `-tag:v hvc1`、av1 `-cpu-used 4 -row-mt 1` / `-tag:v av01` 编码调优

## 技术栈

- **Python 3.13+**
- **NumPy**：网格演算
- **Pillow (PIL)**：帧渲染
- **ffmpeg**：硬件/软件视频编码与音频混入（系统依赖，替代 OpenCV VideoWriter）
- **可选增强**：scipy（卷积加速）、asteval（公式求值）、librosa（音频分析）

## 许可证

与父项目 `draw/` 和 `pywallpaper/` 保持一致。

- [x] P0：LLM 语义组合层（`--prompt`，本地词典引擎 + 可选 DeepSeek，Actor–Critic 校验）
- [x] P0：自动美学筛选 + 连镜表（`--batch N --keep-top K --contact-sheet`，curator 启发式评分）
- [x] P0：Demucs 分轨音频驱动（`--audio-stems`，鼓→冲击/贝斯→对比/人声→色温/环境→辉光，缺失降级）
- [x] P1：GPU GLSL 渲染后端（`--gl`，glfw+PyOpenGL，6 家族着色器，~2.5× 提速）
- [x] P1：新程序化原语（raymarch SDF / metaballs / l-system / dla / circle_map，注册进 PRIMITIVE_MAP）
- [x] P1：专辑批量模式 + manifest（`--album NAME`，album_manifest.json 含评分/keep）
- [x] P2：扩散基础层（`--diffusion {auto,video,cloud,local,procedural}`，底片 + 程序化叠加，四源优雅降级）
- [x] P2：交互式 Web UI（`web_ui.py`，Gradio 本地服务，prompt/滑杆/社媒比例/批量连镜表）
- [x] P2：导出增强（`--container mp4|webm|gif`、`--bit-depth 8|10` 10bit HDR、`--aspect 9:16|1:1|16:9|4:3` 竖屏）
