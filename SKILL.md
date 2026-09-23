---
name: video-generator
description: Maintain the video_generator project (本 SKILL.md 所在仓库根). Procedural audio-reactive video generator with GL/CPU renderers, diffusion base and task log.
agent_created: true
---

# video_generator 视频生成器维护

## 概述
ATT 集成音视频生成器：LLM/随机作曲 → 主题渲染（GL 着色器 + CPU 图层）→ 音频能量驱动逐帧调制 → ffmpeg 编码 → 音频混流。主模块 video_generator.py（66KB），另有 verify 脚本族（_verify_*.py）作门禁。

## 结构
- video_generator.py：VideoGenerator 主类（generate() 四阶段：renderers/audio/encode/mux）+ main()
- genlog.py：任务事件日志（<输出>.pipeline.log，JSONL）
- 渲染后端：gl_renderer.py（GLSL）/ pw_renderer.py（PIL）/ draw_renderer.py
- 音频：audio_reactive.py（整曲）/ audio_stems.py（Demucs 分轨）

## 常用命令
- 生成：python video_generator.py ...（参数见 main() argparse）
- 门禁：python _verify_gl.py 等 _verify_*.py 脚本族
- 日志测试：python -m pytest tests/test_genlog.py -q（3/3）

## 已知陷阱
- GL 上下文：主线程建完必须 release_context()，否则渲染 worker 线程 make_current 失败
- ffmpeg 编码并行：渲染线程→队列(maxsize=4)→主线程写 pipe
- 任务日志写失败静默（_log 助手），不阻断生成
