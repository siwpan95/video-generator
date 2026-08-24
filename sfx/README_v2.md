# SFX 音效下载器 — 关键词搜索版

在原版（完全随机下载）基础上新增了**关键词搜索下载**和**本地音效库管理**功能。

## 新增命令行参数

```
python dl_sounds.py --keyword "bell"           # 搜索并下载 "bell" 相关音效
python dl_sounds.py -k "explosion" -n 5       # 下载 5 个包含 "explosion" 的音效
python dl_sounds.py -k "footstep" --source pixabay  # 从 Pixabay 搜索 "footstep"
python dl_sounds.py --search-local "bell"         # 搜索本地库
python dl_sounds.py --list                       # 列出本地库所有音效
python dl_sounds.py -n 20                        # 随机下载 20 个（原功能）
```

## 功能说明

### 1. 关键词搜索下载 (`-k` / `--keyword`)
- 优先使用 **Pixabay** 搜索（最可靠）
- 失败则尝试 **FindSounds**、**SoundGator**、**Komori**
- 所有源都失败则 fallback 到随机下载

### 2. 本地音效库管理
- 第一次运行 `--list` 会自动构建索引（`sounds/_index.json`）
- `--search-local "keyword"` 在本地库中按文件名关键词搜索
- 下载新音效后会自动更新索引

### 3. 指定数据源 (`--source`)
支持的数据源名称：`pixabay`, `findsounds`, `komori`, `soundgator`, 以及原有的 18 个随机下载源。

## 目录结构

```
sfx/
├── dl_sounds.py       # 主入口（已修改，支持关键词搜索）
├── sfx_utils.py       # 反爬增强工具（无需修改）
├── gui.py             # Tkinter 图形界面（未修改）
├── musicplayer.py     # 音效浏览器/播放器（未修改）
├── sounds/            # 已下载音效（290+ 个）
│   └── _index.json   # 本地索引（自动生成）
├── sources/           # 18 个数据源配置（未修改）
└── stealth.min.js     # Playwright 反检测脚本（未修改）
```

## 已知问题 / TODO

1. **Pixabay 搜索可能失败** — 可能被 Cloudflare 拦截（`"Just a moment"`），需要等待更久或重试
2. **FindSounds 下载不稳定** — 连接可能被远端重置，需要增加重试逻辑
3. **`_findsounds_scrape_by_keyword()` 未注册到调度器** — 目前只启用了 Pixabay，可手动在 `scrape_by_keyword()` 中启用其他源
4. **部分函数位置混乱** — 新增的 `*_scrape_by_keyword()` 函数定义在文件后半部分，模块加载顺序需要注意

## 手动测试

```bash
# 1. 测试参数解析
python dl_sounds.py --help

# 2. 查看本地库
python dl_sounds.py --list

# 3. 搜索本地库
python dl_sounds.py --search-local "bell"

# 4. 关键词搜索下载（可能因网络问题失败，属正常）
python dl_sounds.py -k "bell" -n 1

# 5. 原版随机下载（不受影响）
python dl_sounds.py -n 3
```

## 依赖

```bash
pip install playwright urllib3 requests
playwright install chromium
```

## 修改记录

- 添加 `argparse` 参数解析
- 新增 `scrape_by_keyword()` 调度函数
- 新增 `_pixabay_scrape_by_keyword()` 函数
- 新增 `build_index()`, `load_index()`, `search_local()`, `list_local()` 本地库管理函数
- 修改 `_download_one()` 支持 `keyword` 参数
- 修改 `main()` 支持新的命令行参数
