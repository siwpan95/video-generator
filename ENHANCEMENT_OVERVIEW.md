# G:\video_generator 增强总结

> 在联网调研基础上对随机视频生成器进行的全面增强。所有改动均通过隔离 venv 真实验证（非仅语法检查），本文件为累计记录。
>
> 验证环境：
> - `default`（numpy 2.5.1 / scipy 1.18 / asteval 1.0.9 / Pillow 12.3）— asteval + scipy 路径
> - `numba-verify`（numpy 2.4.6 / numba 0.66 / scipy 1.18 / Pillow 12.3 / asteval 1.0.9）— Numba JIT 路径
> - ffmpeg：ffmpeg 缺省用项目 bundled essentials build（G:/ai/video_generator/ffmpeg-...-essentials_build）；验证时曾用雷电云 LD-Yun 自带的 `D:/projects/repos/video-generator/ffmpeg-2026-06-08-git-6028720d70-essentials_build/bin/ffmpeg.exe` 作为外部 ffmpeg

---

## 回合 1（联网调研驱动全面增强）

### 新增模块（自包含、零硬依赖）
| 模块 | 能力 | 验证 |
|------|------|------|
| `palettes_extra.py` | 感知均匀色图 viridis/plasma/inferno/magma/cividis | 输出 (8,8,3) uint8 ✓ |
| `noise_field.py` | Ken Perlin 噪声场 + 多倍频 fBm + 时间滚动 | 256² fbm 数 ms ✓ |
| `wireframe3d.py` | 4×4 旋转/透视矩阵投影，立方体/二十面体/球/四面体线框 | 四形状 <1ms/帧 ✓ |
| `formula_eval.py` | asteval 受限解释器安全求值 draw JSON 公式/颜色表达式 | 全 fallback 降级 ✓ |
| `audio_reactive.py` | 音频→逐帧能量包络 + beat；librosa→ffmpeg→wave→常数 四重降级 | numpy 路径 50 帧包络 [0.46,1.0]，检出 6 拍 ✓ |

### 渲染器增强
- **`draw_renderer.py`**：`bz_reaction_scipy / diffusion_scipy / life_scipy / wave_scipy`（非 JIT 分支优先调用）；asteval 公式求值；感知色图接入 `apply_palette`。
- **`pw_renderer.py`**：新增原语 `noise` / `wireframe_3d` / `text_rain`，注册进 `PRIMITIVE_MAP`。

### 主生成器升级（`video_generator.py`）
- 编码增强（`--codec h264/hevc/av1`、NVENC AQ/rc-lookahead/B 帧、yuv420p+faststart）
- 图层转场（smoothstep 缓动）、分辨率预设（`--resolution`）、音频驱动（能量调制亮度/辉光 + beat 触发转场）

### 回合 1 修复的 Bug
| 文件 | 问题 | 处理 |
|------|------|------|
| `cloud_generator.py` | `generate_video` 在 `return False` 后 ~38 行不可达死代码 | 已删除 |
| `formula_eval.py` | `ae = Interpreter()` 拼写错误（应为 `Interpreter`），装 asteval 时必 NameError | 已修正 |

---

## 回合 2（本轮回应用户三项增强 + 真实验证）

### 增强 A — Numba JIT 路径接入 scipy 风格卷积（Task #10）
`draw_renderer.py` 新增 `@nb.njit` 辅助 `_convolve3_nb(grid, kernel)`（mode='wrap' 环形边界，与 `scipy.ndimage.convolve(..., mode='wrap')` 数值等价），并将 `_bz_step_nb / _diffusion_step_nb / _life_step_nb / _wave_step_nb` 从手写邻居索引改为走卷积。

**现场复验（numba-verify venv，本回合重跑）：**
```
conv _NEIGH4  max diff = 0.00e+00
conv _LAP     max diff = 0.00e+00
conv _NEIGH8  max diff = 0.00e+00
bz   njit(noise=0) vs scipy(random noise) max diff = 2.00e-02  (噪声上限，结构全等)
dif  njit vs scipy max diff = 0.00e+00
life njit vs scipy max diff = 0.00e+00
wave njit vs scipy max diff = 0.00e+00
```

### 增强 B — audio_reactive 色相随节拍偏移（Task #11）
- `audio_reactive.py`：`__init__` 增 `self.hue`；`_from_signal` 末尾调用 `_compute_hue()`；新增 `_compute_hue(attack, drift)`（beat 时按黄金比 0.61803 推进色相目标、帧间 attack 缓动、叠加包络漂移，cyclic [0,1)）与 `hue_shift_at(frame_idx, scale)` 查询接口。
- `video_generator.py`：新增 `_hsv_hue_rotate(arr, hue_deg)`（纯 NumPy RGB↔HSV 旋转）、构造参数 `audio_hue_shift`、CLI `--audio-hue-shift`、后处理 `_apply_global_postprocess_numpy` 接入音频色相旋转。

### 增强 C — 端到端样例视频（Task #12）
- 新增 `gen_sample.py`：BZ draw 层（Annulus_BZReaction_Cool）+ wireframe_3d pw 层（3D 立方体），生成带节拍 wav 驱动色相偏移，输出 `sample_bz_wireframe.mp4`。
- 本回合在修复后**重新生成**样例（numba-verify venv 全栈：Numba JIT + asteval + 色相旋转 + 音频混入）：
  ```
  h264 | 512x512 @30fps | 6.0s | 216 KB | EXIT=0 | 速度 12.8 fps
  音轨: aac / 22050Hz / mono / 6.04s | RMS=0.21 | peak=0.95 | 非静音 96.7%
  t=3s 帧 mean RGB = [102.6, 205.1, 20.7] | 非黑像素 100%
  ```

### 增强 D — 音频真正混入 mp4（本轮新增）
- **问题**：`VideoGenerator.generate()` 只用 `audio_path` 驱动色相/亮度，**从不把音频写进文件**；mux 逻辑只在 CLI `__main__` 块里，`gen_sample.py` 直接调 `generate()` 产出的是**静音 mp4**。
- **修复**：
  - `audio_mixer.find_ffmpeg` 新增 `FFMPEG_BIN` 环境变量最高优先级（与编码器用同一 ffmpeg）。
  - `VideoGenerator.__init__` 新增 `audio_volume`（默认 1.0）/ `audio_mux`（默认 True）参数。
  - `generate()` 末尾：`audio_path` 非空且 `audio_mux=True` 时调 `audio_mixer.mux_audio` 把音轨混进 mp4，失败降级为"视频已生成、仅无声"并打印告警。
  - CLI `__main__` 删除冗余的 mux 块（现由 `generate()` 统一处理），`audio_volume` 透传。
- **验证**：`sample_bz_wireframe.mp4` 现含两条流（video h264 + audio aac/22050/mono），RMS 0.21、peak 0.95、96.7% 非静音样本，节拍脉冲清晰可辨。

### 回合 2 修复的 Bug（现场发现并修复）
| 文件 | 问题 | 处理 | 验证 |
|------|------|------|------|
| `formula_eval.py` | **链式比较** `a < X < b` 被 asteval 求值为**标量 NaN**（非逐元素），导致 BZ 环形公式 `1 if (0.2 < (x-0.5)**2+(y-0.5)**2 < 0.25) else 0` 实际走旧逻辑降级，`_eval_formula_grid` 的形状守卫把它静默吞掉 | 新增 `_expand_chained_comparisons`（ast 把 `a OP1 b OP2 c` 展开为 `(a OP1 b) & (b OP2 c) & ...`），接进 `_preprocess`，两处 eval 入口改用 `_preprocess` | 现场复验：该公式现返回 (64,64) 网格（600 px），`0.2 < x < 0.5 < y` 三重链也正确 (608 px)，颜色表达式仍正常 |
| `wireframe3d.py` | **scale 语义爆炸**：JSON `scale:150` 被当短边比例再乘 min(w,h)=512 → 坐标炸到 ±84000，`_dda_line` 每帧约百万次，整体 0.4 fps | `scale>1` 时按 max(w,h) 归一化并 clamp [0.02,0.98]；`_dda_line` 加同侧越界快速返回 + 步数上限 | `Wireframe3D.draw_to_array` 2793ms → 23ms/帧（回合 1 验证） |
| `formula_eval.py` | **向量化 if/else 回归**（回合 1 引入）：asteval 不自动向量化 `A if 数组条件 else B`，返回标量 NaN | 新增 `_vectorize_if` 把顶层 `A if C else B` 改写 `where(C,A,B)`，符号表加 `where`/`np` | （回合 1 验证） |

> ⚠️ 关于"回合 1 的 ALL GREEN"：当时公式初始化检查在 `numba-verify` 环境跑，而该环境**没有 asteval**，`eval_grid_expr` 抛错后调用方的 try/except 落到旧逻辑（返回正确 (64,64)）——也就是说那次量到的是**降级路径**，并非 asteval 真正生效。本回合发现并在 default 环境复验，确认 asteval 对链式比较确实返回 NaN，现已根治。

---

## 回合 3（联网调研驱动：色图 / 噪声 / 流场 / 音频特征 / 编码调优）

### 调研来源（联网检索）
- **Google Turbo 多项式近似**（Anton Mikhailov 2019；Ruofei Du 公开 GLSL 系数，Apache-2.0）—— 比 jet 更感知均匀、低光可读性强，适合热力/数据可视化。
- **Simplex 噪声**（Perlin 2002）—— 单形格点、2D 仅 3 角、各向同性、无 Perlin 经典版的轴向条带伪影。
- **Worley / Cellular 噪声** —— F1（最近特征点距离）生成 Voronoi/细胞观感。
- **域扭曲 Domain Warping**（Inigo Quilez）—— `fbm(p + warp·fbm(p + warp·fbm(p)))`，多层 fbm 互相位移产生有机流体纹理（云/大理石/熔岩）。
- **流场粒子**：① curl noise `v = ∇×ψ`（无散度，无汇无源，流线自然不聚不散）；② 角度场 `angle = noise(x,y,t)·2π` + 拖尾淡出（streamline）。
- **音频特征**：频谱质心（=明亮度，映射色温）、频谱通量（帧间谱变化，onset/段落检测）、BPM（librosa `beat_track`）；numpy `rfft` STFT 回退（无 librosa 也能算 centroid/flux）。
- **ffmpeg 编码**：hevc 需 `-tag:v hvc1` 才能在 Apple 生态播放；libaom-av1 用 `-cpu-used 4 -row-mt 1` 提速。

### 新增 / 改动
| 模块 | 改动 | 验证 |
|------|------|------|
| `palettes_extra.py` | 新增 **Turbo** 多项式色图（`_turbo_lut`，官方 GLSL 系数，端点 R0=34 / R255=144），接入 `is_perceptual()` / `build_lut()`；`random_composer` 调色板池加入 `"Turbo"` | turbo LUT (256,3) uint8 ✓ |
| `noise_field.py` | 新增 `SimplexNoise`（2D simplex + fbm）、`WorleyNoise`（F1 距离场）、`warp_field`（域扭曲） | simplex/worley/warp 场 ✓ |
| `pw_renderer.py` | `SimpleNoise` 按 `noise_type∈{perlin,simplex,worley,warp}` 分派；新增 `SimpleFlowField`（curl/angle 双模式，持久拖尾 `_trail` + 速度场有限差分，粒子起点散布） | flow[curl]+[angle]+noise 三分支 ✓ |
| `audio_reactive.py` | 新增 `centroid`（频谱质心/明亮度）、`flux`（频谱通量/段落）、`bpm`（librosa beat_track）；librosa + numpy STFT 双路径 | centroid/flux/bpm 双路径 ✓ |
| `video_generator.py` | 新增 `--audio-balance`（按频谱质心调色温：明亮→暖、低→冷，`_color_temperature`）；hevc 加 `-tag:v hvc1`、av1_nvenc 加 `-tag:v av01`、CPU av1 加 `-cpu-used 4 -row-mt 1` | 端到端 ✓ |
| `draw_renderer.py` | numba 导入 `except ImportError` → `except Exception`（本机 numba 损坏仍会 `import` 成功但抛 AttributeError，原代码漏捕导致导入即崩） | 导入不再崩 ✓ |
| **7 个合成特效 JSON** | 注册进随机合成与关键词检索：`__builtin_flow_field_curl` / `__builtin_flow_field_angle`（flow_field）、`__builtin_noise_simplex` / `__builtin_noise_worley` / `__builtin_noise_warp`（noise）、`__builtin_wireframe3d`（wireframe_3d 二十面体）、`__builtin_text_rain`（text_rain）。标签含原语类型名 | `--keywords flow_field/noise_simplex/warp/text_rain/wireframe_3d` 命中 ✓ |

### 端到端真实验证（本回合）
**环境**：`/Python314`（numpy 2.4.6 / scipy 1.18.0 / librosa 0.11.0）；ffmpeg 曾用雷电云 LD-Yun 自带的 `D:/projects/repos/video-generator/ffmpeg-2026-06-08-git-6028720d70-essentials_build/bin/ffmpeg.exe`（版本未实测，不杜撰）。

单元测试 `_verify_r3.py` **ALL GREEN**（turbo LUT、simplex/worley/warp 场、flow curl+haromny、audio centroid/flux/bpm 双路径）。

两段端到端视频（均确认 video h264 + audio aac 双流、时长精确 2.00s）：
```
e2e_flow.mp4   --keywords flow_field --audio random --audio-balance
  480x270 @30fps 2s | 426 KB | EXIT 0
  前 3 帧拖尾缓冲预热(近黑)→ frame5 luma 8.6 → frame15 28.4，非黑帧确认

e2e_warp.mp4   --keywords noise_warp --audio random --audio-hue-shift
  480x270 @30fps 2s | 150 KB | EXIT 0（域扭曲纯 Python 逐像素多趟 fbm，~1.2 fps，慢但正确）
  噪声场铺满整帧 luma 42-44，非黑确认
```
注册表加载确认：`flow_field→FlowField_Angle/Curl`、`noise_simplex→Noise_Simplex`、`warp→Noise_Warp`、`text_rain→TextRain_Matrix`、`wireframe_3d→Wireframe3D_Icosa`。当前共 **9685** 个效果（draw 4972 + pw 4713，含本轮 7 个新 JSON）。

### 回合 3 修复的 Bug
| 文件 | 问题 | 处理 |
|------|------|------|
| `pw_renderer.py` / `draw_renderer.py` | 本机 numba 已装但损坏（`import numba` 抛 `AttributeError: module 'coverage' has no attribute 'types'`），原 `except ImportError` 漏捕 → 导入即崩 | 改为 `except Exception`，遵循优雅降级 |
| `audio_reactive.py` | librosa 分支 `_from_librosa` 末尾漏调 `_compute_hue()`，导致 `--audio-hue-shift` 在 librosa 路径下不生效 | 两条路径统一在末尾调用 `_compute_hue()` |
| `noise_field.py` | `WorleyNoise._hash12` 用 int64 哈希混合溢出 `OverflowError` | 改用 `np.uint64` XOR 混合 |
| `pw_renderer.py` | `SimpleNoise.draw_to_array` 调 `render_field(..., octaves=, scroll=)`，但 Worley 缺这两个 kwarg → `TypeError` | `WorleyNoise.render_field` 增加 `octaves`/`scroll` 形参 |
| `noise_field.py` | Simplex `_corner` 用 `g[0]`/`g[1]` 取错轴，形状广播失败 `ValueError` | 改为 `g[...,0]`/`g[...,1]` |

### 复现命令（回合 3）
```bash
# 单元测试（/Python314，需 numpy；scipy/librosa 可选）
python _verify_r3.py

# 端到端：流场 + 音频色温
set FFMPEG_BIN=D:/projects/repos/video-generator/ffmpeg-2026-06-08-git-6028720d70-essentials_build/bin/ffmpeg.exe（雷电云 LD-Yun 自带，仅为验证时使用的外部 ffmpeg；不设置则自动用项目 bundled essentials build）
python video_generator.py --keywords flow_field --audio random --audio-balance --duration 2 --fps 30 --width 480 --height 270 --output e2e_flow.mp4

# 端到端：域扭曲 + 音频色相
python video_generator.py --keywords noise_warp --audio random --audio-hue-shift --duration 2 --fps 30 --width 480 --height 270 --output e2e_warp.mp4
```

---

## 验证方式（复现命令）
```bash
# Numba 路径数值等价（numba-verify 环境）
python -c "import numpy as np, scipy.ndimage as ndi, draw_renderer as dr
g=np.random.default_rng(0).random((64,64))
for k in (dr._NEIGH4, dr._LAP, dr._NEIGH8):
    print(np.max(np.abs(dr._convolve3_nb(g,k)-ndi.convolve(g,k,mode='wrap'))))"

# asteval 链式比较（default 环境，需 asteval）
python -c "import numpy as np
from formula_eval import eval_grid_expr
ys,xs=np.mgrid[0:64,0:64]/63.0
print(eval_grid_expr('1 if (0.2<(x-0.5)**2+(y-0.5)**2<0.25) else 0',xs,ys).shape)"

# 端到端样例
set FFMPEG_BIN=D:/projects/repos/video-generator/ffmpeg-2026-06-08-git-6028720d70-essentials_build/bin/ffmpeg.exe（雷电云 LD-Yun 自带，仅为验证时使用的外部 ffmpeg；不设置则自动用项目 bundled essentials build）
python gen_sample.py
```

## 文档
- `README.md`：补充 Numba 卷积加速、音频色相偏移、链式比较公式、样例脚本等增强项；本轮新增「音频合成（mux 进 mp4）」小节，说明 `generate()` 内部统一 mux、`audio_mixer.find_ffmpeg` 同样读 `FFMPEG_BIN`、CLI 与 `gen_sample.py` 同一条 mux 路径。
- `requirements.txt`：scipy / asteval / librosa 标注为可选增强依赖。

---

## 回合 4（阶段 1：P0-1 语义组合层 + P0-2 自动美学筛选 — 联网调研驱动）

> 落实 ENHANCEMENT_PROPOSAL_2026.md 的阶段 1。全部改动经 `.venv` 真实验证（生成 + 解码评分）。

### P0-1 — LLM 语义组合层 `llm_composer.py`（新增）
- 自然语言 → 结构化 spec → `RandomComposer.build_from_spec`（新增公开方法）落地为 `VideoComposition`。
- **双引擎 + 优雅降级**：`LocalSemanticComposer`（纯本地词典，零依赖、零联网、始终可用）+ `LLMComposer`（可选，仅当 `--llm` 且环境变量 `DEEPSEEK_API_KEY` 存在时调用 DeepSeek 结构化输出，缺失/失败自动回退本地）。
- **Actor–Critic 双智能体**：`critic(spec)` 校验/夹紧图层数(1–4)、调色板白名单、fps/duration 范围、theme/postprocess 枚举；硬失败回退纯随机。
- 语义词典覆盖：情绪→调色板/主题、运动（慢/快/激烈/节奏，强语义词优先级避免"快节奏"被"节奏"覆盖）、风格→来源(draw/pw)、领域→类别（深海/海洋/火焰/植物…）、后处理（发光/电影/锐利）、画幅（竖屏/方形/横屏）。
- CLI：`--prompt "..."`（接 llm_composer）、`--llm`（激活 LLM 后端）。

### P0-2 — 自动美学筛选 `curator.py`（新增）
- 零额外依赖，复用 `audio_mixer.find_ffmpeg`（优先 `FFMPEG_BIN`）解码帧。
- 启发式指标：亮度均值/对比、亮度直方图熵、色彩多样性、相邻帧差分 RMS（动态能量）；全黑帧重罚。
- `curate()`：跨批次 min-max 归一化加权评分（motion 0.4 / entropy 0.35 / color_var 0.25）→ 排序 → 标记 kept/rejected → 写 `curate_manifest.json` + `contact_sheet.png`。
- CLI：`--batch N`（批量生成，按 seed 变化）、`--keep-top K`（保留前 K）、`--contact-sheet`（连镜表；批量时默认开启）。

### 本回合修复的 Bug（真实端到端验证发现）
| 文件 | 问题 | 处理 |
|------|------|------|
| `pw_renderer.py` | `SimpleTextRain` 对 pywallpaper JSON 的 `{min,max}` 范围式 `speed`/`color` 直接 `float()`/`int()` 崩溃，导致 text_rain 图层被静默跳过（"表面工程"式缺陷） | 新增 `_num()` 容错 range→均值；`color` 为 dict 时回退默认绿。text_rain 现可正常渲染 |

### 端到端真实验证（`.venv`，FFMPEG_BIN=D:/projects/repos/video-generator/ffmpeg-2026-06-08-git-6028720d70-essentials_build/bin/ffmpeg.exe（雷电云 LD-Yun 自带，仅为验证时使用的外部 ffmpeg；不设置则自动用项目 bundled essentials build））
- **语义映射单测**：`深海发光 慢节奏 冷色调`→fps24/dur15/pp=vignette+glow/pal=Turbo,Cool；`快节奏 霓虹 赛博 粒子`→fps60/pal=Neon,Cyberpunk；`火焰 熔岩 激烈`→fps60/pal=Hot；`森林 植物 舒缓 自然`→fps24/pal=gradient_4。Critic 对非法 spec 夹紧 fps→120、调色板回退白名单。
- **单段生成**：`--prompt "深海发光 慢节奏 冷色调" --duration 2` → `output/e2e_prompt.mp4`（480x270@30fps，3 图层，vignette+glow），8.5 fps。
- **定向修复验证**：`--keywords text_rain` → `output/e2e_textrain.mp4`（480x270@30fps 2s，h264，不再跳过）。
- **批量+筛选**：`--prompt "霓虹 赛博 快节奏" --batch 4 --keep-top 2` → 4 段生成 → 评分排序 0.895/0.374/0.245/0.172 → top-2 kept、bottom-2 rejected → `curate_manifest.json` + `contact_sheet.png`。top 视频 motion=50.52、nonblack 86.9%，最差 motion=5.37，排序判别有效。

### 已知待办（非本阶段，留待 P1）
- `[PW] 未支持的原语类型: circle_map`：组合 JSON（combo3_*_circle_map_*）含 `circle_map` 图层，但 `PRIMITIVE_MAP` 未注册，被优雅跳过（非崩溃）。属 P1 新原语范畴，未在本阶段处理。
- LLM 实时调用路径因本机无 `DEEPSEEK_API_KEY` 未对真实 API 联调，仅验证降级到本地引擎。

---

## 回合 5（阶段 1：P0-3 Demucs 分轨音频驱动）

### 新增模块 `audio_stems.py`
- `AudioStems`：对一段音频分别跑 Demucs 分离（drums/bass/vocals/other）与整曲 `mix`，**复用 `AudioReactive` 逐轨计算** 包络/onset/质心，零重复逻辑。
- 分轨 → 视觉通道映射 `apply_stem_modulation(arr, frame_idx, stems)`：
  - **drums** → 落拍数据撕裂冲击（`_drum_glitch`）+ 色相迸发（`_hsv_hue_rotate`）
  - **bass** → 对比度/亮度脉动（`_contrast_pulse`，低频 punch）
  - **vocals** → 色温漂移（`_color_temperature`，人声明亮度 → 暖/冷）
  - **other** → 环境辉光泛光（`_bloom`，铺底氛围）
  - 各通道独立，缺失分轨自动跳过。
- **降级策略（沿用项目铁律：无重依赖也能跑，绝不崩溃）**：
  - 优先 `python -m demucs -n htdemucs -o OUT audio`；demucs 缺失/分离失败 → 抛 `DemucsUnavailable`，调用方降级为整曲混合驱动（原 `AudioReactive` 路径）。
  - `from_paths()` 类方法：直接用预分离 wav 构造，**无需 demucs**，便于测试/外部预分离。
- **关键修正（真实验证发现）**：`AudioReactive` 包络按各轨自身 max 归一化，恒定（静音/持续）分轨会被误读为满能量。在 `apply_stem_modulation` 的 bass/other 分支加 `max-min > 0.05` 动态门限，仅在有真实动态时驱动，避免误触发。

### 主生成器集成（`video_generator.py`）
- 构造器新增 `use_audio_stems` 参数；`generate()` 在 `audio_reactive`（= mix）构造后尝试 `AudioStems(...)`，成功则 `self.audio_stems` 接管、`stems_drive_beat=True` 并复用 `stems.mix` 作基础调制（避免二次计算）。
- `_apply_global_postprocess_numpy`：`stems_drive_beat=True` 时跳过整曲 beat 冲击（改由 drums 分轨接管，避免双冲）；末尾调用 `apply_stem_modulation` 叠加四条分轨的专属调制。
- CLI：`--audio-stems`（`pip install demucs` 缺失则自动降级整曲混合驱动）。

### 真实验证（`.venv`，FFMPEG_BIN=D:/projects/repos/video-generator/ffmpeg-2026-06-08-git-6028720d70-essentials_build/bin/ffmpeg.exe（雷电云 LD-Yun 自带，仅为验证时使用的外部 ffmpeg；不设置则自动用项目 bundled essentials build））
- **单元 + 集成单测**（`_verify_stems.py`）：四个视觉通道辅助函数均确实变换画面；`from_paths` 正常构造、drums 合成鼓点检出节拍（beats=[10,15,21,24,31,34,41,44]）、落拍帧 `apply_stem_modulation` 改变画面而非落拍帧保持不变；`DemucsUnavailable` 对「缺文件」与「demucs 禁用」均正确抛出。✅ 全部通过。
- **端到端降级**：`--audio-stems --audio sfx/sounds/1\ (1\).mp3 --duration 3` → 打印 `[Audio] Demucs 不可用，降级为整曲混合驱动` → 正常产出 `output/e2e_stems_fallback.mp4`（含音频混入），无崩溃。✅

### 真实 Demucs 分离路径验证（补跑，回合 5 后续）
- **本机 torch 段错误坑（已解决）**：`.venv` 装了 `torch 2.13.0 + demucs 4.1.0`，但 `import torch` 因本机 PATH 上另一个 `libiomp5md.dll` 抢先加载，在加载 C 扩展时 access violation 段错误。`audio_stems.py` 改用 `importlib.util.find_spec` 探测（不再真正 `import demucs`，避免父进程崩），并在拉起 demucs **子进程**前把 `torch/lib` 前置到其 PATH + 设 `KMP_DUPLICATE_LIB_OK=TRUE`，规避冲突。另支持 `DEMUCS_BIN` 环境变量指定 demucs 可执行文件。
- **真实分离单测**（`_verify_stems_real.py`，普通 import 不触发段错误）：`AudioStems("sfx/sounds/1 (1).mp3", 25)` 真实 htdemucs 分离出 `['drums','bass','vocals','other']` + `mix`；各轨包络互相关 `drums~vocals=0.046`（鼓/人声确为不同信号，证明分离有效）、`drums~bass=0.839`、`bass~vocals=0.285`；鼓峰值帧经 `apply_stem_modulation` 改变画面。✅
- **真实端到端**：`--audio-stems --audio sfx/sounds/1\ (1\).mp3 --duration 4` → 日志 `[Audio] ✓ AudioStems(model=htdemucs, available=[...], mix=✓)`（确为真实分离、非降级）→ 产出 `output/e2e_stems_real.mp4`（320×240 25fps 4s，h264 + AAC 混入），ffprobe 校验通过。✅
- **临时文件清理**：`_separate` 用 `tempfile.mkdtemp`，分轨 wav 被 `AudioReactive` 读入内存后，构造结束即 `shutil.rmtree` 删临时目录，避免每次生成泄漏。
- **结论**：P0-3 Demucs 分轨音频驱动 全部完成并真实验证（映射逻辑 + 降级 + 真实分离三态均覆盖）。

---

## 回合 6（阶段 2 增强 P1：P1-1 GPU GLSL 渲染后端）

> 落实 ENHANCEMENT_PROPOSAL_2026.md 阶段 2 的 P1-1。经 `.venv` 真实验证（单测 `_verify_gl.py` 全 6 家族 + 真实端到端 `--gl` 解码校验）。

### 技术选型：为什么不用 ModernGL
- ModernGL 的上下文依赖 `glcontext`，需**编译原生 C 扩展**；本机**无任何 C 编译器**（MSVC / gcc 均缺），`pip install moderngl` 在 `Failed to build glcontext ... Microsoft Visual C++ 14.0 or greater required` 处失败。
- 改用**预编译 wheel**：`glfw`（建 OpenGL 上下文，纯 C 库打包进 wheel，无需本机编译）+ `PyOpenGL`（系统 `opengl32.dll` 的 ctypes 封装，纯 Python）。零编译即在本机跑真·GLSL。

### 新增模块 `gl_renderer.py`
- `GLRenderer`：用 GLSL fragment 着色器在 **GPU 上渲染程序化图案**作视频「基底帧」，再交给 `video_generator` 既有 `_apply_global_postprocess_numpy`（辉光/暗角/色相/beat 撕裂/分轨调制）与 ffmpeg 编码——GPU 提速 + 全部音频交互不变。
- **着色器设计**：单一 fragment 程序 + `u_family` int 分支切换 **6 种图案家族**（`noise`/`flow`/`plasma`/`aurora`/`starfield`/`ca`）；统一 **IQ 余弦调色板** `pal(t)=a+b*cos(2π(c·t+d))`，由 `u_pa..u_pd` 四组 vec3 uniform 驱动，6 套预设（neon/cool/warm/green/mono/fire）。音频能量 `u_energy` 注入做亮度/脉动。
- **主题映射** `_THEME_MAP`：`empty→(noise,neon)`、`starfield→(starfield,cool)`、`gradient→(plasma,neon)`、`aurora→(aurora,green)`；`family_for_theme` / `palette_for_theme`（模块级函数 + `GLRenderer` 同名 classmethod 双重 API）。
- **降级铁律**：`glfw`/`PyOpenGL` 缺失或上下文创建失败 → 抛 `GLUnavailable`，调用方降级现有 CPU(`pw_renderer`) 管线，**绝不崩溃**。

### 上下文生命周期（规避 glfw 在本机的若干坑）
- **模块级单例**：`_gl_window` 仅首次创建；`_gl_prog`/`_gl_u`/`_gl_vao` 编译一次、**永不删除**（进程退出 OS 回收）。彻底规避 glfw 反复 create/destroy 重建上下文失败、以及多实例/批量下 `glUseProgram`/`glBindVertexArray` 因复用已删对象 id 报 `GLError 1282` 的坑。
- **多线程上下文转移**：上下文在**主线程** `GLRenderer.__init__` 建好并编译着色器后，必须 `release_context()`（`glfw.make_context_current(None)`）释放；渲染 **worker 线程** 通过 `make_current()` 认领。否则 worker 调 `glUseProgram` 时上下文仍被主线程占用、`wglGetProcAddress` 返回 NULL → `NullFunctionError: glUseProgram`。

### 主生成器集成（`video_generator.py`）
- 构造器新增 `use_gl` 参数；`generate()` 先建 `GLRenderer`（成功则跳过 CPU 图层 `_init_layer_renderers`，并 `release_context()`）；`_render_worker` 开头 `make_current()`；`_render_frame` 顶部 GL 分支 `frame = gl.render_frame(t, frame_idx, env)` → `_apply_global_postprocess_numpy` 复用不变。
- CLI：`--gl`（开启 GPU 后端；失败/不可用自动降级 CPU，不报错退出）。

### 本回合修复的 Bug（真实验证发现）
| 文件 | 问题 | 处理 |
|------|------|------|
| `gl_renderer.py` | `_check_shader` 由 `def` 改为 `@staticmethod` 后，`raise` 体掉到列 0 → `IndentationError`，阻断整个模块 import | `raise GLUnavailable(...)` 缩进 4 空格归入 `if` 块 |
| `gl_renderer.py` + `video_generator.py` | `video_generator` 以 `GLRenderer.family_for_theme(...)`（classmethod）调用，但该 API 仅存在于模块级函数 → `AttributeError` → GL 全程降级 CPU | 在 `GLRenderer` 补 `family_for_theme` / `palette_for_theme` classmethod 包装，委托模块级同名函数 |
| `gl_renderer.py` + `video_generator.py` | GL 上下文在主线程建好后被主线程一直占用，worker 线程 `make_context_current` 认领失败 → `NullFunctionError: glUseProgram` | 新增 `release_context()`，主线程建完释放；worker `make_current()` 认领 |

### 端到端真实验证（`.venv`，FFMPEG_BIN=D:/projects/repos/video-generator/ffmpeg-2026-06-08-git-6028720d70-essentials_build/bin/ffmpeg.exe（雷电云 LD-Yun 自带，仅为验证时使用的外部 ffmpeg；不设置则自动用项目 bundled essentials build））
- **单测 `_verify_gl.py`**：6 家族均产出非平凡帧（noise 932 / flow 538 / plasma 1455 / aurora 1082 / ca 890 唯一色，mean>1；starfield 稀疏星点 11 色 mean 11.9 属正常）；主题映射正确；60 帧@320×240 仅渲染 **0.173s ≈ 347 fps**（GPU，不含编码）。✅ `GL_OK`
- **真实端到端**：`--gl --duration 4 --width 320 --height 240 --fps 25 --output output/e2e_gl.mp4` → 日志 `[GL] ✓ GPU 渲染后端已启用` → 渲染 **~50 fps（CPU 管线同参数仅 ~20 fps，提速约 2.5×）** → 产出 `output/e2e_gl.mp4`，ffprobe 校验：**h264 High / yuv420p / 320×240 / 25fps / 4.00s**，解码无误。✅
- **降级验证**：`glfw`/`PyOpenGL` 缺失或上下文失败路径抛 `GLUnavailable`，主流程打印降级提示并走 CPU 管线，无崩溃（代码路径，未强制卸载验证）。

### 已知限制（P1-1 范围内的合理简化，留待后续）
- GL 模式**替代**整条 CPU(`pw_renderer`) 图层管线：着色器产出基底帧后复用 `_apply_global_postprocess_numpy`（音频调制一致），但**不**逐层 composite 既有 pw_renderer 图层（波形/粒子/Rainbow 等）。这是后端替换的刻意简化，视觉走 shader 图案 + 统一后处理；后续若需 GL 基底 + CPU 图层叠加，可在 `_render_frame` GL 分支末尾接入 `pw_renderer` 合成。
- `starfield` 家族为稀疏星点（黑底+亮星），mean 偏低、唯一色少，属预期；如需繁星/星云背景可在 shader 内追加。

### 复现命令（P1-1）
```bash
# GL 单测（6 家族 + 性能）
.venv/Scripts/python.exe _verify_gl.py

# 真实端到端 GPU 渲染
set FFMPEG_BIN=D:/projects/repos/video-generator/ffmpeg-2026-06-08-git-6028720d70-essentials_build/bin/ffmpeg.exe（雷电云 LD-Yun 自带，仅为验证时使用的外部 ffmpeg；不设置则自动用项目 bundled essentials build）
.venv/Scripts/python.exe video_generator.py --gl --duration 4 --width 320 --height 240 --fps 25 --output output/e2e_gl.mp4
```

---

## 回合 7（阶段 2 增强 P1：P1-2 新程序化原语）

> 落实 ENHANCEMENT_PROPOSAL_2026.md 阶段 2 的 P1-2。经 `.venv` 真实验证（`_verify_primitives.py` 直测 + PWEffecRenderer 集成 + FFmpegWriter 真实编码）。

### 新增模块 `primitives_pro.py`（自包含，零外部依赖，纯 NumPy/PIL）
四种新原语，均实现 pywallpaper 图层接口（`__init__(config,w,h)` / `update(dt)` / `draw_to_array(canvas)` 加性叠加，可分层）：

| 原语类 | 别名 | 内容 |
|--------|------|------|
| `RaymarchSDF` | `raymarch` / `sdf` | 2D 有符号距离场场景：圆/盒 SDF 用 IQ 多项式 `smin` 融合 + 边缘辉光；图元中心随时间旋转 |
| `Metaballs` | `metaballs` / `metaball` | 经典元球场 `Σ r²/d²`，阈值填充 + 辉光；球心沿椭圆轨道运动 |
| `LSystemPrimitive` | `lsystem` / `fractal` | L-system 重写 + 海龟绘图（默认分形植物），逐帧生长 + 整体旋转；沿枝条色相渐变 |
| `DLACluster` | `dla` / `aggregation` | 扩散限制凝聚：预计算聚簇（收缩发射环保证黏附）+ 逐帧渐进显现；半径色相渐变 |

- **性能（30 帧@320×240，`.venv`）**：raymarch 0.22s / metaballs 0.34s / lsystem 0.18s / dla 4.5s（dla 含一次性预计算 ~2.5s，渲染本身快）。均向量化，无 Numba 依赖（规避本机损坏 numba）。
- **着色**：统一用本地 `_hsv_to_rgb`（与 `SimpleColorGen` 等价，避免与 `pw_renderer` 循环 import）；`color{hue,sat,val}` 配置驱动色相动画。

### 主渲染器集成（`pw_renderer.py`）
- 顶部安全导入 `primitives_pro`（缺失则 `_HAS_PRIMITIVES_PRO=False`，注册 `None` 优雅降级）。
- `PRIMITIVE_MAP` 新增 8 个别名：`raymarch/sdf/metaballs/metaball/lsystem/fractal/dla/aggregation`。`load_config` 对 `None` 条目打印跳过（与现有 wireframe3d 模式一致），不崩溃。

### 本回合修复的缺陷（真实验证发现）
| 原语 | 问题 | 处理 |
|------|------|------|
| `LSystemPrimitive` | 默认分形植物在 320×240 下 `width=1` 细线 + 单色乘 2.2 → 仅 2 唯一色、mean<1，被非平凡断言判失败 | 线宽随分辨率提至 2；沿枝条推进做**色相渐变**（尖端偏暖）让唯一色>5；加性 1.6x 提亮 |
| `DLACluster` | 原发射环固定 `0.48·S`，簇生长后游走粒子难以触及 → 仅黏附 ~13 点，末帧 nz=27 | 改用**跟踪簇当前最大半径的收缩发射环**（标准 DLA），预计算稳定达到 `max_points`（350）；同时 `draw_to_array` 改向量化整数坐标批量叠加，并加半径色相渐变（唯一色 5→91）；网格 `cell` 160→120 使一次性预计算 5.3s→~2.5s |

### 端到端真实验证（`.venv`，FFMPEG_BIN=D:/projects/repos/video-generator/ffmpeg-2026-06-08-git-6028720d70-essentials_build/bin/ffmpeg.exe（雷电云 LD-Yun 自带，仅为验证时使用的外部 ffmpeg；不设置则自动用项目 bundled essentials build））
- **直测 `_verify_primitives.py`**：四原语均非黑、唯一色>5、帧间有差异（动画推进）；DLA 半径 0~0.34、350 点；L-system 360 段。✅ `PRIMITIVES_OK`
- **集成**：`PWEffecRenderer.load_config` 用手写 4 图层 effect 构建 4/4、跳过 0；合成帧 nz=230400、uniq=4666、mean 71.2。✅
- **真实编码**：raymarch 30 帧经 `FFmpegWriter`（libx264）→ `output/e2e_primitives.mp4`，ffprobe 校验：**h264 High / yuv420p / 320×240 / 25fps / 1.20s**，解码无误。✅

### 复现命令（P1-2）
```bash
# 四原语直测 + 集成 + 真实编码
set FFMPEG_BIN=D:/projects/repos/video-generator/ffmpeg-2026-06-08-git-6028720d70-essentials_build/bin/ffmpeg.exe（雷电云 LD-Yun 自带，仅为验证时使用的外部 ffmpeg；不设置则自动用项目 bundled essentials build）
.venv/Scripts/python.exe _verify_primitives.py
```

---

## 回合 8（阶段 2 增强 P1：P1-3 专辑批量模式 + manifest）

> 落实 ENHANCEMENT_PROPOSAL_2026.md 阶段 2 的 P1-3。经 `.venv` 真实验证（3 首专辑端到端 + manifest 校验 + mp4 解码）。

### 主生成器集成（`video_generator.py`）
- 新增 CLI `--album NAME`：以统一主题/语义生成一批曲目（默认 **8 首**，可用 `--batch` 覆盖数量），复用既有批量生成循环（语义/关键词/随机组合 + 参数覆盖 + ffmpeg 编码）。
- 专辑模式默认开启**连镜表**（与批量行为一致），并默认复用 P0-2 的 `curator.curate` 美学筛选。
- 新增 `_write_album_manifest(album_name, tracks, out_dir, paths, keep_top)`：把每首元数据（index/title/path/theme/description/分辨率/fps/duration/seed/used_gl）与 curate 产出合并，写 `<out_dir>/<album>/album_manifest.json`：
  - 每首含 **美学评分 + kept 标记 + 指标**（亮度均值/对比、熵、色彩多样性、motion、帧数）；
  - 按评分排序赋予 `rank`；记录 `generated_at` / `track_count` / `keep_top` / 连镜表与 curate manifest 路径。
- 修复：本文件此前漏 `import json` → 首次写 album manifest 抛 `NameError`；补 `import json`。

### 端到端真实验证（`.venv`，FFMPEG_BIN=D:/projects/repos/video-generator/ffmpeg-2026-06-08-git-6028720d70-essentials_build/bin/ffmpeg.exe（雷电云 LD-Yun 自带，仅为验证时使用的外部 ffmpeg；不设置则自动用项目 bundled essentials build））
- `--album "Neon Dreams" --prompt "霓虹 赛博 快节奏" --batch 3 --duration 2 --fps 25 --width 320 --height 240`：
  - 3 首均正常生成（CPU 管线 ~46fps）；curator 评分 1.000 / 0.191 / 0.091，排序有效。
  - 写 `output/Neon Dreams/album_manifest.json`（3 首，kept=3，字段完整含 metrics）。
  - 连镜表 `output/contact_sheet.png`、筛选 `output/curate_manifest.json` 一并产出。
  - 3 个 mp4 经 ffprobe 校验：h264 High / yuv420p / 320×240 / 25fps / 2.00s，全部解码无误。✅

### 复现命令（P1-3）
```bash
set FFMPEG_BIN=D:/projects/repos/video-generator/ffmpeg-2026-06-08-git-6028720d70-essentials_build/bin/ffmpeg.exe（雷电云 LD-Yun 自带，仅为验证时使用的外部 ffmpeg；不设置则自动用项目 bundled essentials build）
# 专辑模式（默认 8 首，--batch 可改数量；--keep-top K 触发挑优标记）
.venv/Scripts/python.exe video_generator.py --album "Neon Dreams" --prompt "霓虹 赛博 快节奏" --batch 3 --duration 2 --width 320 --height 240
```

---

## 回合 9（阶段 2 收尾 P2：清理两个遗留待办）

> 清理 P1 阶段明确标注的两个遗留项：(1) `circle_map` 原语未注册；(2) GL 模式未叠加既有 pw 图层。经 `.venv` 真实验证。

### 待办 1：`circle_map` 原语注册（P2）
- **背景**：300+ 个组合 JSON（`effects/pywallpaper/*`）的 `layers[].type` 含 `"circle_map"`（径向环/同心圆/频谱圆），此前 `PRIMITIVE_MAP` 未注册 → 被 `load_config` 打印 `[PW] 未支持的原语类型: circle_map` 优雅跳过（不崩溃，但丢内容）。
- **实现**（`primitives_pro.py` 新增 `CircleMapPrimitive`）：
  - 解析真实组合 JSON 字段：`center{x,y 百分比}`、`rings[{count,radius,line,line_width}]`、`color{hue,sat,val,speed}`（hsv_flow）、`audio{scale,smoothing}`（band_energy→radius_offset 半径扩张）。
  - `line=True` 沿圆周 **24 段分色弧**（彩虹环，避免整圈单色）；`line=False` 沿圆周等距布点并按角度 hue 渐变。
  - 时间推进：`hue` 在 `[a,b]` 内流动；`_audio_level` 指数平滑后映射到半径偏移（频谱环观感）。
  - 降级铁律：缺字段一律安全默认，绝不崩溃；纯 NumPy/PIL。
- **注册**（`pw_renderer.py`）：`PRIMITIVE_MAP` 新增 `circle_map` / `circlemap` / `circlemaps` → `primitives_pro.CircleMapPrimitive`（受 `_HAS_PRIMITIVES_PRO` 守卫）。

### 待办 2：GL 模式叠加既有的 pw 图层（P2）
- **背景**：P1-1 的 GL 分支在 `_render_frame` 里 `render_frame → _apply_global_postprocess → return`，**绕过了既有 pw 图层合成**（CPU 路径 799–856 行那段），导致开启 `--gl` 时 circle_map/flow 等图层全部丢失，只剩 shader 基底。
- **实现**（`video_generator.py`）：
  - 把 CPU 路径的图层合成块抽成独立方法 **`_composite_layers_numpy(frame, t, dt)`**（处理：图层启停、update 推进、render_frame/draw_to_array/draw 三态 fallback、尺寸对齐、smoothstep 淡入淡出、最终 `composite_frames_numpy`）。
  - CPU 路径与 GL 分支**共用** `_composite_layers_numpy`；GL 分支改为 `gl.render_frame → _composite_layers_numpy → _apply_global_postprocess → return`。
  - 顺势把"GL 初始化 + 图层初始化"也抽成 **`_prepare_renderers()`**（原在 `generate()` 内联），`__init__` 末尾置 `self.gl = None`，`generate()` 与单帧测试共用，消除重复。
- **语义不变**：关闭 GL 时行为完全不变（仍走 CPU 全量管线）；开启 GL 时 = shader 基底 + pw 图层叠加 + 统一后处理，三类内容都在。

### 端到端真实验证（`.venv`，FFMPEG_BIN=D:/projects/repos/video-generator/ffmpeg-2026-06-08-git-6028720d70-essentials_build/bin/ffmpeg.exe（雷电云 LD-Yun 自带，仅为验证时使用的外部 ffmpeg；不设置则自动用项目 bundled essentials build））
- **circle_map 直测**（`_verify_primitives.py` 已扩展）：53 唯一色、非黑、帧间有差；集成 5/5 图层构建、跳过 0、合成帧 uniq=4129。✅ `PRIMITIVES_OK`
- **GL 叠加验证**（`_verify_gl_layers.py` 新增）：用真实含 circle_map 的 effect「频谱环」构造组合，
  - CPU 帧非黑=176972 / GL 帧非黑=228756 / CPU·GL 帧均差=143.70 / 中心环带非黑=132036 → 证明 GL 基底之上 circle_map 环已正确叠加（非纯 shader）。✅ `GL_LAYERS_OK`
  - 真实编码：`g_enc.generate()` 走完整 `--gl` 管线 → `output/e2e_gl_circlemap.mp4`，ffprobe 校验：**h264 High / yuv420p / 320×240 / 25fps / 2.00s**，解码无误。✅
- 两个遗留待办至此清零。

### 复现命令（P2）
```bash
set FFMPEG_BIN=D:/projects/repos/video-generator/ffmpeg-2026-06-08-git-6028720d70-essentials_build/bin/ffmpeg.exe（雷电云 LD-Yun 自带，仅为验证时使用的外部 ffmpeg；不设置则自动用项目 bundled essentials build）
# circle_map 注册 + 四原语回归
.venv/Scripts/python.exe _verify_primitives.py
# GL 模式叠加 pw 图层（含 circle_map）验证 + 真实编码
.venv/Scripts/python.exe _verify_gl_layers.py
```

---

## 阶段 2（P1）总览
- **P1-1 GPU GLSL 渲染后端**（回合 6）：glfw+PyOpenGL 替代 ModernGL（本机无编译器）；6 家族着色器 + IQ 余弦调色板；多线程上下文转移修复；~2.5× 提速。`gl_renderer.py` + `_verify_gl.py` + `output/e2e_gl.mp4`。
- **P1-2 新程序化原语**（回合 7）：`primitives_pro.py` 四原语（raymarch SDF / metaballs / lsystem / dla），注册进 `pw_renderer.PRIMITIVE_MAP`；`_verify_primitives.py` + `output/e2e_primitives.mp4`。
- **P1-3 专辑批量模式 + manifest**（回合 8）：`--album` 编排统一主题批次 + `album_manifest.json`（元数据/美学评分/keep）。
- 全部经 `.venv` + `FFMPEG_BIN=D:/projects/repos/video-generator/ffmpeg-2026-06-08-git-6028720d70-essentials_build/bin/ffmpeg.exe（雷电云 LD-Yun 自带，仅为验证时使用的外部 ffmpeg；不设置则自动用项目 bundled essentials build）` 真实验证，遵循项目优雅降级铁律（无硬依赖、不崩溃）。P1 阶段完成。





---

## 回合 10（阶段 3 P2：新增能力落地 — 扩散基础层 / Web UI / 导出增强）

> 落地 ENHANCEMENT_PROPOSAL_2026.md 的 P2 三项新增能力，全部经 `.venv` + `FFMPEG_BIN=D:/projects/repos/video-generator/ffmpeg-2026-06-08-git-6028720d70-essentials_build/bin/ffmpeg.exe（雷电云 LD-Yun 自带，仅为验证时使用的外部 ffmpeg；不设置则自动用项目 bundled essentials build）` 真实验证（RTX 3050 6GB 本机）。
> 遵循项目"优雅降级"铁律：重依赖（gradio/扩散模型）缺省或失败一律回退，绝不崩溃。

### P2-1 · 扩散模型基础层（`diffusion_base.py` + `video_generator.py` 集成）
- 新增 `diffusion_base.py`：把"扩散底片"作为视频底层，再叠程序化原语做风格化。来源按优先级：
  1. `from_video` —— 本地视频/帧序列（ffmpeg 解码，**始终可用**）；
  2. `from_cloud` —— 复用 `cloud_generator` 凭证生成（缺失降级）；
  3. `from_local_model` —— 仅当显式设模型路径且 torch 可用才尝试，否则抛 `DiffusionUnavailable`（**绝不静默假片**）；
  4. `procedural_fallback` —— 零依赖确定性动画渐变，供测试/无模型使用。
- `make_diffusion_base(mode, ...)` 统一入口：`auto`(video→cloud→procedural) / `video` / `cloud` / `local` / `procedural`。
- `VideoGenerator` 加 `diffusion_base` 参数；CPU 路径在 `_render_frame` 用其帧替代 `theme_renderer` 背景（取帧失败自动回退程序化主题）。GL 模式不与之混用（shader 基底独立）。
- CLI：`--diffusion {auto,video,cloud,local,procedural}` + `--base-video PATH` + `--model-dir DIR`。
- 验证（`_verify_diffusion.py`）：程序化兜底帧非黑+帧间有差；视频基础层成片非黑（底片+程序化叠加）；`local` 无模型路径正确抛 `DiffusionUnavailable` 降级。✅ `DIFFUSION_OK`

### P2-2 · 交互式 Web UI（`web_ui.py`）
- 新增 `web_ui.py`：Gradio 本地服务（`python web_ui.py`，默认 127.0.0.1:7860）。
- 模块级 `web_generate(...)` 引擎逻辑**无 gradio 依赖**（可独立批处理/测试）；`_build_ui()` 才懒加载 gradio，缺依赖启动时报清晰安装提示（exit 2）。
- 界面含：语义 prompt / 关键词 / 参数滑杆（时长·帧率·宽·高·种子·批量）/ 社媒比例下拉(9:16/1:1/16:9/4:3) / 封装格式(mp4/webm/gif) / GL 开关 / 挑优保留数 / 音频上传 / 一键生成 → Gallery + 连镜表 + 状态。
- 全部复用 `VideoGenerator` 引擎（与 CLI 同管线），零重复实现。
- 验证：gradio 6.25.0 本机装好；服务 HTTP 200 正常吐页；`web_generate` 头部测试 3 首（9:16+keep_top=2）→ curator 评分 0.4/0.4/0.0、kept 标记正确、`contact_sheet.png` 产出。✅

### P2-3 · 导出与分发增强（`video_generator.py` FFmpegWriter 改造）
- `FFmpegWriter` 扩展：容器 `mp4/webm/gif`、位深 `8/10`、竖屏比例。
  - **WebM(VP9)**：走 libvpx-vp9，容器强制 CPU 软编码（无稳定硬件 VP9）。✅
  - **动态 GIF**：split+palettegen/paletteuse(bayer 抖动) 两遍调色板，循环。✅
  - **10-bit HDR**：hevc/vp9 走 `yuv420p10le` + bt2020 primaries/transfer/matrix 元数据（h264 忽略 10bit）。✅
  - **竖屏/方形**：`--aspect 9:16` 按长边重算分辨率（如 1280×720→720×1280）。✅
- `VideoGenerator` 加 `container/bit_depth/aspect`；GIF 自动关音频混入；输出扩展名跟随容器。
- CLI：`--container mp4|webm|gif`、`--bit-depth 8|10`、`--aspect 9:16|1:1|16:9|4:3`。
- 验证（`_verify_export.py`）：mp4 h264 8bit / webm vp9 8bit / gif / mp4 hevc 10bit(Main10 yuv420p10le bt2020) / portrait 9:16(720×1280) 五项均 ffprobe 校验容器·编码·分辨率·位深 + 首帧非黑。✅ `EXPORT_OK`

### 复现命令（P2 新增能力）
```bash
set FFMPEG_BIN=D:/projects/repos/video-generator/ffmpeg-2026-06-08-git-6028720d70-essentials_build/bin/ffmpeg.exe（雷电云 LD-Yun 自带，仅为验证时使用的外部 ffmpeg；不设置则自动用项目 bundled essentials build）
# P2-1 扩散基础层（程序化兜底 + 视频底片）
.venv/Scripts/python.exe _verify_diffusion.py
# P2-2 Web UI（需 gradio；先装：.venv/Scripts/python.exe -m pip install gradio）
.venv/Scripts/python.exe web_ui.py --port 7860
# P2-3 导出增强（mp4/webm/gif/10bit/竖屏）
.venv/Scripts/python.exe _verify_export.py
# 串联示例：竖屏 HDR + 扩散底片
.venv/Scripts/python.exe video_generator.py --prompt "深海发光" --aspect 9:16 --container mp4 --bit-depth 10 --diffusion procedural --duration 4
```

## 阶段 3（P2 新增能力）总览
- **P2-1 扩散基础层**（回合 10）：`diffusion_base.py` 四源可选（video/cloud/local/procedural），`VideoGenerator.diffusion_base` 接入；优雅降级铁律。
- **P2-2 Web UI**（回合 10）：`web_ui.py` Gradio 本地服务，`web_generate` 引擎与 UI 解耦，复用生成器管线。
- **P2-3 导出增强**（回合 10）：`FFmpegWriter` 支持 webm/gif/10bit HDR/竖屏比例，CLI 三新开关。
- 三项均真实验证通过。至此前述提案 P0（语义+筛选+分轨）、P1（GLSL+新原语+专辑）、P2（扩散+UI+导出）全部落地。


### 回合 11（ffprobe 接入 + 类结构修复）
- 用户提供本机 ffmpeg essentials build（`G:\ai\video_generator\ffmpeg-2026-06-08-git-6028720d70-essentials_build\bin`，含 `ffprobe.exe`）。此前误判"无独立 ffprobe"，验证脚本改用 `ffmpeg -i` stderr 解析做伪 ffprobe。
- 修复：`video_generator.py` 新增模块级 `find_ffprobe()`（优先 FFMPEG_BIN 同目录 → bundled essentials build → 常见路径 → PATH 回退）；`FFmpegWriter._find_ffmpeg` 与 `audio_mixer.find_ffmpeg` 均把 bundled essentials build 的 `ffmpeg.exe` 加进候选（在 PATH 之前）。
- 顺带修复一处严重结构 bug：`find_ffprobe`/`shutil_which` 误插入在 `FFmpegWriter` 类体中间，导致 `_detect_hw_encoder`/`_open`/`write`/`is_opened` 被 Python 解释为 `find_ffprobe` 的局部函数、`FFmpegWriter` 丢失 `_open` 等方法（运行时 `AttributeError: 'FFmpegWriter' object has no attribute '_open'`）。已把两模块级函数正确移到类之外，`FFmpegWriter` 方法归位。
- `_verify_export.py` 改用真 `ffprobe`（JSON `-show_entries`）做结构化校验（容器/编码/分辨率/位深），不再依赖 stderr 解析；并修正竖屏断言（`portrait_9x16` 实际传 1280×720→9:16→720×1280，测试按 aspect 变换后期望值断言）。
- 验证：`_verify_export.py` 五项（mp4 h264 8bit / webm vp9 8bit / gif / mp4 hevc 10bit / portrait 9:16）全过 `EXPORT_OK`；`_verify_diffusion.py` 仍 `DIFFUSION_OK`。
