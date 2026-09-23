#!/usr/bin/env python3
"""
audio_mixer.py — 视频音频合成工具

功能：
1. 从 sfx/sounds/ 随机选取音效
2. 用 ffmpeg 将音频混入视频
3. 支持音量调整

用法（作为模块）：
  from audio_mixer import mux_audio
  mux_audio("video.mp4", "random", 1.0)
"""

import os
import random
import subprocess
import sys


def find_ffmpeg() -> str:
    """查找 ffmpeg 可执行文件（FFMPEG_BIN 环境变量优先）"""
    # 0) 显式环境变量（最高优先级）：FFMPEG_BIN 可指向任意 ffmpeg 二进制
    env_bin = os.environ.get("FFMPEG_BIN", "").strip()
    if env_bin:
        try:
            subprocess.run([env_bin, "-version"], capture_output=True, timeout=5)
            return env_bin
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
    # 1) 项目自带 essentials build（与 ffprobe 配套，优先于 PATH 上的散装 ffmpeg）
    bundled = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ffmpeg-2026-06-08-git-6028720d70-essentials_build", "bin", "ffmpeg.exe")
    # 2) 常见位置
    # 2026-09-23 去本地化：删除已失效的 C:/ffmpeg/bin/ffmpeg.exe 硬编码候选；
    # PATH 上的 ffmpeg（含 pip 安装的 imageio-ffmpeg shim）由 "ffmpeg" 候选覆盖。
    candidates = [
        bundled,
        "ffmpeg",
        os.path.join(os.environ.get("PROGRAMFILES", ""), "ffmpeg", "bin", "ffmpeg.exe"),
    ]
    for cand in candidates:
        if not cand:
            continue
        try:
            subprocess.run([cand, "-version"], capture_output=True, timeout=5)
            return cand
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    return "ffmpeg"  #  fallback 到 PATH


def find_audio_file(audio_arg: str, script_dir: str = None) -> str:
    """根据参数查找音频文件
    
    Args:
        audio_arg: 文件路径，或 "random"
        script_dir: video_generator.py 所在目录
    
    Returns:
        音频文件完整路径，或 None
    """
    if not audio_arg:
        return None

    if audio_arg == "random":
        if script_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        sounds_dir = os.path.join(script_dir, "sfx", "sounds")
        if not os.path.isdir(sounds_dir):
            print(f"[Audio] 警告: 音效目录不存在: {sounds_dir}")
            return None
        audio_files = [
            f for f in os.listdir(sounds_dir)
            if f.lower().endswith((".mp3", ".wav", ".ogg", ".m4a", ".aac"))
        ]
        if not audio_files:
            print(f"[Audio] 警告: 未找到音效文件")
            return None
        chosen = random.choice(audio_files)
        audio_path = os.path.join(sounds_dir, chosen)
        print(f"[Audio] 随机音效: {chosen}")
        return audio_path
    else:
        # 指定文件路径
        if not os.path.isfile(audio_arg):
            print(f"[Audio] 错误: 音频文件不存在: {audio_arg}")
            return None
        return audio_arg


def mux_audio(
    video_path: str,
    audio_arg: str,
    volume: float = 1.0,
    script_dir: str = None
) -> bool:
    """将音频混入视频
    
    Args:
        video_path: 输入视频路径（也会被覆盖为输出）
        audio_arg: 音频文件路径，或 "random"
        volume: 音量倍率（1.0 = 原音量）
        script_dir: video_generator.py 所在目录
    
    Returns:
        bool: 是否成功
    """
    if not audio_arg:
        return True  # 无需音频

    audio_path = find_audio_file(audio_arg, script_dir)
    if not audio_path:
        return False

    ffmpeg_exe = find_ffmpeg()
    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_temp{ext}"

    cmd = [
        ffmpeg_exe, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
    ]

    # 音量调整
    if volume != 1.0:
        cmd.extend(["-af", f"volume={volume}"])

    cmd.append(temp_output)

    print(f"[Audio] 合成音频: {os.path.basename(audio_path)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        print(f"[Audio] ✗ 合成超时")
        return False

    if result.returncode == 0 and os.path.isfile(temp_output):
        # 替换原文件
        if os.path.isfile(video_path):
            os.remove(video_path)
        os.replace(temp_output, video_path)
        size_mb = os.path.getsize(video_path) / 1024 / 1024
        print(f"[Audio] ✓ 音频已合成 ({size_mb:.1f} MB)")
        return True
    else:
        print(f"[Audio] ✗ 合成失败: {result.stderr[-300:]}")
        if os.path.isfile(temp_output):
            os.remove(temp_output)
        return False


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="视频音频合成工具")
    parser.add_argument("video", help="输入视频文件")
    parser.add_argument("--audio", required=True,
                        help="音频文件或 'random'")
    parser.add_argument("--volume", type=float, default=1.0,
                        help="音量倍率（默认 1.0）")
    args = parser.parse_args()

    ok = mux_audio(args.video, args.audio, args.volume)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    import argparse
    main()
