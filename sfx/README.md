# SFX — 免费音效下载器

从 18 个免注册音效网站批量下载随机免费音效。

## 支持的音效源 (18 个)

SoundGator / Pixabay / SoundJay / Mixkit / Myinstants / Zapsplat / Freesound / SoundBible / 等

## 功能

- **批量下载** — 随机从 18 个源下载免费音效
- **反爬系统** — 25 种浏览器身份轮换 (Chrome/Firefox/Safari/Edge/Android)、Playwright 无头浏览器、stealth.min.js 注入
- **GUI 界面** — 勾选数据源、实时进度显示
- **音效浏览器** — 内置播放器预览已下载音效

## 快速开始

```bash
# 安装依赖
pip install playwright urllib3 requests
playwright install chromium

# CLI 下载
python dl_sounds.py

# GUI 模式
python gui.py

# 浏览已下载的音效
python musicplayer.py
```

## 目录结构

```
sfx/
├── dl_sounds.py       # CLI 核心下载逻辑
├── sfx_utils.py       # 反爬增强工具
├── gui.py             # Tkinter 图形界面
├── musicplayer.py     # 音效浏览器/播放器
├── sounds/            # 353 个已下载音效
├── sources/           # 18 个数据源配置
└── stealth.min.js     # Playwright 反检测脚本
```

## 许可

内部项目，全哥创建。下载的音效版权归各源网站所有。
