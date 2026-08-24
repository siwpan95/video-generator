"""
web_ui.py — P2-2 交互式 Web UI（Gradio 本地服务）

把既有 CLI 能力搬进浏览器：语义 prompt、实时参数滑杆、社媒比例预设、
音频上传、GL 开关、一键批量 + 连镜表。

设计原则（沿用项目降级哲学）：
  * Gradio 为可选依赖：web_generate 引擎逻辑在模块级、无 gradio 依赖，
    _build_ui 才懒加载 gradio，缺依赖时启动报清晰安装提示而非崩溃。
  * 所有渲染复用 VideoGenerator 引擎（与 CLI 同一条管线），不重复实现。

用法：
  python web_ui.py                 # 默认 http://127.0.0.1:7860
  python web_ui.py --port 8080 --share
"""
import argparse
import os
import sys
import time

# 确保脚本目录在 sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ── 引擎依赖（无 gradio 也可调用 web_generate 做测试/批处理）──
from effect_loader import get_default_registry
from random_composer import RandomComposer
import llm_composer
import curator
from video_generator import VideoGenerator

OUT_DIR = os.path.join(_SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)
_REGISTRY = get_default_registry()
_COMPOSER = RandomComposer(_REGISTRY)


def _safe_ext(container):
    return {"mp4": ".mp4", "webm": ".webm", "gif": ".gif"}.get(
        (container or "mp4").lower(), ".mp4")


def web_generate(prompt, keywords, use_prompt, duration, fps, width, height,
                 seed, batch, aspect, container, use_gl, audio_file,
                 keep_top, progress=None):
    """核心生成逻辑（无 gradio 依赖，可独立测试/批处理）。

    返回 (gallery[(path,label)...], sheet_path_or_None, status_text)。
    """
    paths = []
    t0 = time.time()
    n = max(1, int(batch))
    seeds = ([int(seed) + i for i in range(n)] if seed not in (None, "")
             else [None] * n)

    for i, sd in enumerate(seeds):
        if use_prompt and prompt:
            comp, meta = llm_composer.compose_from_prompt(
                prompt, _REGISTRY, seed=sd, use_llm=False)
        elif keywords:
            comp = _COMPOSER.compose_from_keywords(keywords.split(), seed=sd)
        else:
            comp = _COMPOSER.compose(seed=sd)
        comp.duration = int(duration)
        comp.fps = int(fps)
        comp.width = int(width)
        comp.height = int(height)
        ext = _safe_ext(container)
        desc = (comp.description or "video")[:40]
        safe = "".join(c for c in desc if c.isalnum() or c in "_-") or "video"
        stamp = int(time.time()) if sd is None else sd
        out = os.path.join(OUT_DIR, f"web_{safe}_{i+1}_{stamp}{ext}")
        g = VideoGenerator(
            comp, out, prefer_hw="auto", codec="h264",
            audio_path=(audio_file if audio_file else None),
            use_gl=bool(use_gl), container=container,
            aspect=(aspect if aspect and aspect != "none" else None),
        )
        g.generate()
        paths.append(out)
        if progress is not None:
            try:
                progress((i + 1) / n, desc=f"渲染 {i+1}/{n}")
            except Exception:
                pass

    # 连镜表（批量时）
    sheet = None
    if len(paths) > 1 or keep_top:
        try:
            curator.curate(
                paths, out_dir=OUT_DIR, keep_top=keep_top or None,
                contact_sheet=True)
            sheet = os.path.join(OUT_DIR, "contact_sheet.png")
            if not os.path.isfile(sheet):
                sheet = None
        except Exception as e:  # noqa: BLE001
            sheet = None
            print(f"[WebUI] 连镜表生成失败: {e}")

    elapsed = time.time() - t0
    gallery = [(p, os.path.basename(p)) for p in paths]
    status = (f"✓ 生成 {len(paths)} 个视频，耗时 {elapsed:.1f}s\n"
              f"输出目录: {OUT_DIR}")
    return gallery, sheet, status


def _build_ui():
    """构造 Gradio 界面；返回 demo。失败时抛 ImportError（缺 gradio）。"""
    import gradio as gr  # 懒加载：缺依赖时在此抛出

    with gr.Blocks(title="随机视频生成器 · Web UI") as demo:
        gr.Markdown("# 🎬 随机视频生成器 — 交互式 Web UI\n"
                    "语义描述 / 关键词 → 程序化视频；支持竖屏社媒、GL 加速、批量连镜表。")
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="语义描述（如：深海发光 慢节奏 冷色调）",
                    placeholder="留空则用关键词/随机")
                keywords = gr.Textbox(
                    label="关键词（空格分隔，如：BZReaction particle）",
                    placeholder="可选，覆盖语义")
                use_prompt = gr.Checkbox(value=True, label="优先使用语义描述")
                with gr.Row():
                    duration = gr.Slider(1, 30, value=5, step=1, label="时长(s)")
                    fps = gr.Slider(10, 60, value=25, step=1, label="帧率")
                with gr.Row():
                    width = gr.Slider(128, 1920, value=640, step=2, label="宽")
                    height = gr.Slider(128, 1920, value=640, step=2, label="高")
                with gr.Row():
                    seed = gr.Number(value=42, label="种子(空=随机)", precision=0)
                    batch = gr.Slider(1, 8, value=1, step=1, label="批量数量")
                with gr.Row():
                    aspect = gr.Dropdown(
                        ["none", "9:16", "1:1", "16:9", "4:3"], value="none",
                        label="社媒比例(覆盖宽高)")
                    container = gr.Dropdown(
                        ["mp4", "webm", "gif"], value="mp4", label="封装格式")
                with gr.Row():
                    use_gl = gr.Checkbox(value=False, label="GPU(GLSL) 加速")
                    keep_top = gr.Number(value=0, label="挑优保留数(0=全留)",
                                         precision=0)
                audio_file = gr.File(
                    label="音频文件(可选，混入 mp4)",
                    file_types=[".mp3", ".wav", ".ogg", ".m4a"])
                btn = gr.Button("🎬 生成", variant="primary")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="生成结果", columns=2, height=420)
                sheet = gr.Image(label="连镜表", visible=True)
                status = gr.Textbox(label="状态", lines=4)

        btn.click(
            web_generate,
            inputs=[prompt, keywords, use_prompt, duration, fps, width, height,
                    seed, batch, aspect, container, use_gl, audio_file,
                    keep_top],
            outputs=[gallery, sheet, status],
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description="视频生成器 Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="生成公网 share 链接（需联网）")
    args = parser.parse_args()

    try:
        demo = _build_ui()
    except ImportError:
        sys.stderr.write(
            "\n[WebUI] 未安装 gradio，无法启动 Web UI。\n"
            "  安装：.venv/Scripts/python.exe -m pip install gradio\n"
            "  或回退 CLI：python video_generator.py --prompt \"...\"\n")
        sys.exit(2)

    print(f"[WebUI] 启动 http://{args.host}:{args.port} (Ctrl+C 退出)")
    demo.launch(server_name=args.host, server_port=args.port,
                share=args.share)


if __name__ == "__main__":
    main()
