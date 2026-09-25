#!/usr/bin/env python3
"""
cloud_generator.py — 云端视频生成（glitch-video-generator 原始功能）

功能：
1. 调用 random_glitch_prompt.py 生成随机 prompt
2. 通过 buddy-cloud.py 调用云端 API 生成视频
3. 下载结果到本地

用法：
  python cloud_generator.py                     # 完全随机
  python cloud_generator.py 42                  # 可复现（seed=42）
  python cloud_generator.py --prompt "自定义提示词"  # 自定义提示词
  python cloud_generator.py --audio random       # 生成后混入随机音频
"""

import argparse
import os
import sys
import time
import json
import subprocess
import tempfile
import random

# ── 路径配置 ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RANDOM_PROMPT = os.path.join(_SCRIPT_DIR, "random_glitch_prompt.py")
_BUDDY_CLOUD = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Programs", "WorkBuddy",
    "resources", "app.asar.unpacked",
    "resources", "builtin-skills",
    "buddy-multimodal-generation", "scripts", "buddy-cloud.py"
)
if not os.path.isfile(_BUDDY_CLOUD):
    # 兼容非标准安装：允许用环境变量显式指定，而不是回退到某台机器的绝对路径
    _override = os.environ.get("WORKBUDDY_CLOUD_SCRIPT", "")
    if _override and os.path.isfile(_override):
        _BUDDY_CLOUD = _override
    else:
        raise FileNotFoundError(
            "找不到 WorkBuddy 的 buddy-cloud.py。\n"
            "已尝试默认安装位置：%s\n"
            "如需自定义，请设置环境变量 WORKBUDDY_CLOUD_SCRIPT 指向该脚本。" % _BUDDY_CLOUD
        )
_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")
_TIMEOUT = 600  # 10 分钟超时


def generate_prompt(seed: int = None) -> str:
    """调用 random_glitch_prompt.py 生成 prompt"""
    cmd = [sys.executable, _RANDOM_PROMPT]
    if seed is not None:
        cmd.append(str(seed))

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=10
    )
    prompt = result.stdout.strip()
    if not prompt:
        raise RuntimeError(f"生成 prompt 失败: {result.stderr}")
    return prompt


def get_cloud_token(cli_token: str = "") -> str:
    """获取云端服务凭证

    优先级：cli_token > BUDDY_CLOUD_TOKEN > CLOUD_TOKEN
    返回非空 token，若均无则抛异常。
    """
    token = cli_token or os.environ.get("BUDDY_CLOUD_TOKEN", "") or os.environ.get("CLOUD_TOKEN", "")
    if not token:
        raise RuntimeError(
            "未找到认证 token。请通过以下任一方式提供：\n"
            "  1. 命令行参数:  python cloud_generator.py --token <your_token>\n"
            "  2. 环境变量:     set BUDY_CLOUD_TOKEN=<your_token>\n"
            "  3. 在 WorkBuddy 内运行（自动注入）"
        )
    return token


def _run_buddy_cloud(prompt: str, token: str) -> dict:
    """调用 buddy-cloud.py video 子命令，返回解析后的 JSON dict。

    使用 --token-stdin 将 token 通过 stdin 传入（最安全，不暴露在进程列表）。
    """
    if not os.path.isfile(_BUDDY_CLOUD):
        raise FileNotFoundError(f"buddy-cloud.py 不存在: {_BUDDY_CLOUD}")

    cmd = [
        sys.executable, _BUDDY_CLOUD,
        "video", prompt,
        "--token-stdin",
    ]
    print(f"[Cloud] 执行: {sys.executable} ... video '<prompt>' --token-stdin")

    proc = subprocess.run(
        cmd,
        input=token,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT + 30,
    )

    # buddy-cloud.py 成功时 stdout 输出 JSON；失败时 stderr 也有信息
    if proc.returncode != 0:
        err = proc.stdout.strip() or proc.stderr.strip()
        raise RuntimeError(f"buddy-cloud.py 失败（返回码 {proc.returncode}）: {err[:500]}")

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        # 尝试从混合输出中提取 JSON
        import re
        match = re.search(r'\{.*\}', proc.stdout, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise RuntimeError(f"无法解析 buddy-cloud.py 输出: {proc.stdout[:500]}")


def generate_video(prompt: str, token: str, output_path: str) -> bool:
    """调用 buddy-cloud.py 生成视频"""
    print(f"[Cloud] 生成中...\n  Prompt: {prompt[:80]}...\n")

    try:
        data = _run_buddy_cloud(prompt, token)
    except Exception as e:
        print(f"[Cloud] 错误: {e}")
        return False

    # 提取 result_url（兼容多种字段名）
    result_url = ""
    for key in ("result_url", "ResultUrl", "ResultVideoUrl"):
        val = data.get(key, "")
        if val:
            result_url = val
            break
    # 若 result_url 是列表（多图/多视频），取第一个
    if isinstance(result_url, list) and result_url:
        result_url = result_url[0]

    if not result_url:
        print(f"[Cloud] 错误: 响应中未找到 result_url。完整响应: {json.dumps(data, ensure_ascii=False)[:500]}")
        return False

    # 下载视频
    print(f"[Cloud] 下载视频: {result_url}")
    download_cmd = ["curl", "-sS", "-L", "-o", output_path, result_url]
    ret = subprocess.run(download_cmd, timeout=120)
    if ret.returncode != 0:
        print("[Cloud] curl 下载失败，尝试 PowerShell...")
        ps_cmd = f'Invoke-WebRequest -Uri "{result_url}" -OutFile "{output_path}"'
        ret = subprocess.run(["powershell", "-Command", ps_cmd], timeout=120)

    if os.path.isfile(output_path) and os.path.getsize(output_path) > 1024:
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"[Cloud] ✓ 视频已保存: {output_path} ({size_mb:.1f} MB)")
        return True
    else:
        print("[Cloud] ✗ 下载失败或文件过小")
        return False



def mux_audio(video_path: str, audio_arg: str, output_path: str = None) -> bool:
    """用 ffmpeg 混入音频"""
    if not audio_arg:
        return True  # 无需音频

    # 确定音频文件
    if audio_arg == "random":
        sounds_dir = os.path.join(_SCRIPT_DIR, "sfx", "sounds")
        if not os.path.isdir(sounds_dir):
            print(f"[Audio] 警告: 音效目录不存在: {sounds_dir}")
            return False
        audio_files = [f for f in os.listdir(sounds_dir) if f.endswith((".mp3", ".wav", ".ogg"))]
        if not audio_files:
            print(f"[Audio] 警告: 未找到音效文件")
            return False
        audio_path = os.path.join(sounds_dir, random.choice(audio_files))
        print(f"[Audio] 随机音效: {os.path.basename(audio_path)}")
    else:
        audio_path = audio_arg
        if not os.path.isfile(audio_path):
            print(f"[Audio] 错误: 音频文件不存在: {audio_path}")
            return False

    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_audio{ext}"

    # 用 ffmpeg 合成
    ffmpeg_exe = "ffmpeg"
    cmd = [
        ffmpeg_exe, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path
    ]

    print(f"[Audio] 合成音频: {os.path.basename(audio_path)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode == 0 and os.path.isfile(output_path):
        print(f"[Audio] ✓ 已合成: {output_path}")
        # 替换原文件
        os.replace(output_path, video_path)
        return True
    else:
        print(f"[Audio] ✗ 合成失败: {result.stderr[-200:]}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="云端视频生成器（glitch-video-generator 模式）"
    )
    parser.add_argument("seed", type=int, nargs="?", default=None,
                        help="随机种子（可选）")
    parser.add_argument("--prompt", type=str, default=None,
                        help="自定义提示词（替代随机生成）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--audio", type=str, default=None,
                        help="音频文件或 'random'")
    parser.add_argument("--token", type=str, default="",
                        help="认证 token（也可通过 BUDDY_CLOUD_TOKEN 环境变量设置）")
    args = parser.parse_args()

    # 生成 prompt
    if args.prompt:
        prompt = args.prompt
        print(f"[Prompt] 使用自定义提示词")
    else:
        prompt = generate_prompt(args.seed)
        print(f"[Prompt] 随机生成（seed={args.seed}）")

    print(f"\n{'='*60}")
    print(prompt)
    print(f"{'='*60}\n")

    # 获取 token
    try:
        token = get_cloud_token(args.token)
    except RuntimeError as e:
        print(f"[Cloud] 错误: {e}")
        return 1

    # 确定输出路径
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    if args.output:
        output_path = args.output
    else:
        timestamp = int(time.time())
        output_path = os.path.join(_OUTPUT_DIR, f"cloud_video_{timestamp}.mp4")

    # 生成视频
    ok = generate_video(prompt, token, output_path)
    if not ok:
        print("[Cloud] ✗ 视频生成失败")
        return 1

    # 合成音频
    if args.audio:
        mux_audio(output_path, args.audio)

    print(f"\n✨ 完成！视频已保存至：{output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
