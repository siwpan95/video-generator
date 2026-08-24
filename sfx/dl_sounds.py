"""Download free sound effects from sites that require NO registration.

  数据源: SoundGator / SFXLibrary / Komori / BigSoundBank / PacDV / Pixabay /
          SoundEffectsPlus / SoundJay / QuickSounds / Mixkit / Myinstants /
          AudioMicro / FindSounds / PartnersInRhyme / Zapsplat / WavSource /
          Sampleswap / 99Sounds
  数据源配置按 JSON 拆分: sources/ 目录下每个数据源一个 JSON

  【新功能】关键词搜索下载:
    python dl_sounds.py --keyword "explosion"
    python dl_sounds.py -k "bell" -n 5
    python dl_sounds.py -k "footstep" --source pixabay

  【本地库管理】:
    python dl_sounds.py --search-local "bell"      # 搜索本地已下载音效
    python dl_sounds.py --list                      # 列出本地库

  【原功能】随机下载:
    python dl_sounds.py              # 随机下载 NUM_SOUNDS 个
    python dl_sounds.py -n 20       # 下载 20 个

  数据源配置按 JSON 拆分: sources/ 目录下每个数据源一个 JSON
  CLI: python dl_sounds.py [options]
  GUI: python gui.py (Tkinter 图形界面)
  已移除: Freesound API, Pixabay API（需注册）
  已移除: Chosic（音乐播放列表生成器，无下载选项）
  已移除: OrangeFreeSounds, SoundsCrate（Cloudflare WAF 硬封锁，Playwright 也超时）
"""

import os
import sys
import json
import time
import random
import re
import argparse
import atexit
import urllib.request
import urllib.error
import urllib.parse
import sfx_utils

# 程序退出时自动关闭 Playwright 浏览器
atexit.register(sfx_utils.close_pw)

# ── 配置 ──────────────────────────────────────────────────
SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
SOURCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources")
NUM_SOUNDS = 10  # 下载几个音效（默认）
INDEX_FILE = os.path.join(SOUNDS_DIR, "_index.json")

os.makedirs(SOUNDS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
#  工具函数（委托到 sfx_utils 的增强版本）
# ═══════════════════════════════════════════════════════════

_fetch = sfx_utils.enhanced_fetch           # 下载音频文件用（urllib + 身份轮换）
_pw_init = sfx_utils.enhanced_pw_init
_fetch_page = sfx_utils.enhanced_fetch_page         # Playwright 抓页面
_fetch_page_scroll = sfx_utils.enhanced_fetch_page_scroll  # Playwright + 滚动加载


def _save_file(filepath, data):
    with open(filepath, "wb") as f:
        f.write(data)


def _safe_filename(label, src_name, fallback_id):
    """把 label 转成安全的文件名（不含扩展名）"""
    name = ""
    if label:
        name = re.sub(r'[^a-zA-Z0-9\-\s]', '', label)
        name = re.sub(r'\s+', '_', name).strip('-_').lower()
    if not name:
        name = f"{src_name.lower()}_{fallback_id}"

    count = _used_names.get(name, 0)
    _used_names[name] = count + 1
    if count > 0:
        name = f"{name}_{count}"

    return name


_used_names = {}


# ═══════════════════════════════════════════════════════════
#  本地音效库索引管理
# ═══════════════════════════════════════════════════════════

def build_index(verbose=False):
    """扫描 sounds/ 目录，构建索引并保存为 _index.json

    索引格式:
    {
        "version": 1,
        "updated": "2026-06-28T20:00:00",
        "sounds": [
            {"file": "bell_01.mp3", "path": "sounds/bell_01.mp3",
             "size": 12345, "source": "pixabay", "tags": ["bell", "notification"]}
        ]
    }
    """
    import datetime
    index = {"version": 1, "updated": datetime.datetime.now().isoformat(), "sounds": []}
    if not os.path.isdir(SOUNDS_DIR):
        return index

    for fname in os.listdir(SOUNDS_DIR):
        if not fname.lower().endswith((".mp3", ".wav", ".ogg")):
            continue
        fpath = os.path.join(SOUNDS_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        # 从文件名推断标签
        name_no_ext = os.path.splitext(fname)[0]
        tags = [t.strip() for t in name_no_ext.replace("_", " ").split() if t.strip()]
        # 推断来源（文件名格式: {src}_{name} 或 {name}_{src}）
        source = "unknown"
        for src_key in ["pixabay", "komori", "soundgator", "mixkit", "myinstants",
                         "sfxlibrary", "bigsoundbank", "pacdv", "soundjay",
                         "quicksounds", "audiomicro", "findsounds", "partnersinrhyme",
                         "zapsplat", "wavsource", "sampleswap", "99sounds", "soundeffectsplus"]:
            if src_key in fname.lower():
                source = src_key
                break

        entry = {
            "file": fname,
            "path": os.path.join("sounds", fname),
            "size": os.path.getsize(fpath),
            "source": source,
            "tags": tags,
        }
        index["sounds"].append(entry)

    # 保存
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"  ✓ 索引已更新: {len(index['sounds'])} 个音效 → {INDEX_FILE}")
    return index


def load_index():
    """加载本地索引，如果不存在则构建"""
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return build_index()


def search_local(keyword, verbose=True):
    """在本地音效库中搜索包含关键词的音效

    Args:
        keyword: 搜索关键词（空格分隔多个关键词，AND 关系）
        verbose: 是否打印结果

    Returns:
        list of matched entries
    """
    index = load_index()
    if not index or not index.get("sounds"):
        if verbose:
            print("  ⚠ 本地库为空，请先下载音效")
        return []

    kws = [k.strip().lower() for k in keyword.split() if k.strip()]
    results = []
    for entry in index["sounds"]:
        text = (entry["file"] + " " + " ".join(entry.get("tags", []))).lower()
        if all(kw in text for kw in kws):
            results.append(entry)

    if verbose:
        print(f"\n  本地搜索 '{keyword}': 找到 {len(results)} 个匹配\n")
        if results:
            for i, entry in enumerate(results[:20]):  # 最多显示 20 个
                size_kb = entry["size"] / 1024
                print(f"  [{i+1:3d}] {entry['file']}  ({size_kb:.1f} KB) [{entry['source']}]")
            if len(results) > 20:
                print(f"  ... 还有 {len(results) - 20} 个结果（使用 --list 查看全部）")
        else:
            print(f"  ✗ 未找到包含 '{keyword}' 的音效")
            print(f"  提示: 运行 python dl_sounds.py --list 查看所有已下载音效")
    return results


def list_local(verbose=True):
    """列出本地音效库中的所有音效"""
    index = load_index()
    if not index or not index.get("sounds"):
        if verbose:
            print("  ⚠ 本地库为空，请先下载音效")
        return []

    entries = index["sounds"]
    if verbose:
        print(f"\n  本地音效库: 共 {len(entries)} 个\n")
        # 按来源分组
        by_source = {}
        for e in entries:
            src = e.get("source", "unknown")
            by_source.setdefault(src, []).append(e)

        for src in sorted(by_source.keys()):
            print(f"  [{src}] {len(by_source[src])} 个")
            for e in by_source[src][:5]:  # 每个来源最多显示 5 个
                size_kb = e["size"] / 1024
                print(f"    - {e['file']} ({size_kb:.1f} KB)")
            if len(by_source[src]) > 5:
                print(f"    ... 还有 {len(by_source[src]) - 5} 个")
        print()
    return entries


# ── 源统计 ──────────────────────────────────────────────────
_source_stats = {}  # {src_name: {"ok": 0, "fail": 0}}
MAX_SCRAPE_RETRIES = 3  # 每个下载槽最多尝试几个不同数据源


# ═══════════════════════════════════════════════════════════
#  加载 JSON 配置
# ═══════════════════════════════════════════════════════════

def _load_json_config(name):
    """从 sources/{name}.json 加载配置"""
    path = os.path.join(SOURCES_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════
#  数据源 1: SoundGator
# ═══════════════════════════════════════════════════════════

_soundgator_cfg = None
_soundgator_known_ids = set()


def _soundgator_scrape_one():
    """从 SoundGator 随机抓取一个音效的 mp3 下载链接（Playwright + 内部重试）"""
    global _soundgator_cfg
    if _soundgator_cfg is None:
        _soundgator_cfg = _load_json_config("soundgator")

    cats = list(_soundgator_cfg["categories"])
    random.shuffle(cats)

    for cat_url in cats[:4]:
        status, text = _fetch_page(cat_url, timeout=25)
        if status != 200 or len(text) < 3000:
            continue

        products = re.findall(r'/product/(\d+)-([^/"]+)-sound-effect', text)
        if not products:
            continue

        candidates = [(pid, name) for pid, name in products if pid not in _soundgator_known_ids]
        if not candidates:
            continue

        pid, name = random.choice(candidates)
        _soundgator_known_ids.add(pid)

        prod_url = f"https://www.soundgator.com/product/{pid}-{name}-sound-effect"
        status2, text2 = _fetch_page(prod_url, timeout=25)
        if status2 != 200 or len(text2) < 2000:
            continue

        mp3 = re.search(r'https?://[^"\s]+?\.mp3', text2)
        if mp3:
            return mp3.group(0), name
        audio = re.search(r'data-audio=["\']([^"\']+)["\']', text2)
        if audio:
            return audio.group(1), name

    return None, None


# ═══════════════════════════════════════════════════════════
#  数据源 2: SFXLibrary
# ═══════════════════════════════════════════════════════════

_sfxlibrary_cfg = None
_sfxlibrary_known_paths = set()


def _sfxlibrary_scrape_one():
    """从 SFXLibrary 随机抓取一个音效的 mp3 下载链接（Playwright + 内部重试）"""
    global _sfxlibrary_cfg
    if _sfxlibrary_cfg is None:
        _sfxlibrary_cfg = _load_json_config("sfxlibrary")

    pages = list(_sfxlibrary_cfg["listing_pages"])
    random.shuffle(pages)

    for page_url in pages[:4]:
        status, text = _fetch_page(page_url, timeout=25)
        if status != 200 or len(text) < 3000:
            continue

        sound_links = re.findall(r'href="(\.?/sound/[^"]+)"', text)
        if not sound_links:
            continue

        candidates = [l for l in sound_links if l not in _sfxlibrary_known_paths]
        if not candidates:
            continue

        # 试最多 3 个详情页
        random.shuffle(candidates)
        for detail_path in candidates[:3]:
            _sfxlibrary_known_paths.add(detail_path)

            detail_url = f"https://www.sfxlibrary.com{detail_path.lstrip('.')}"
            status2, text2 = _fetch_page(detail_url, timeout=25)
            if status2 != 200 or len(text2) < 2000:
                continue

            mp3_src = re.search(r'src="([^"]+\.mp3)"', text2)
            mp3_plain = re.search(r'https?://[^"\s]+\.mp3', text2)
            mp3_url = None
            if mp3_src:
                mp3_url = mp3_src.group(1)
                if not mp3_url.startswith("http"):
                    mp3_url = f"https://www.sfxlibrary.com/{mp3_url.lstrip('/')}"
            elif mp3_plain:
                mp3_url = mp3_plain.group(0)

            if mp3_url:
                return mp3_url, detail_path.split("/")[-1]

    return None, None


# ═══════════════════════════════════════════════════════════
#  数据源 3: 小森平免费音效 (taira-komori.net)
# ═══════════════════════════════════════════════════════════

_komori_cfg = None
_komori_sound_names = {}
_komori_used = set()


def _komori_scrape_sound_names(page_name):
    """从分类页提取所有音效文件名"""
    if page_name in _komori_sound_names:
        return _komori_sound_names[page_name]

    url = f"https://taira-komori.net/{page_name}"
    status, text = _fetch_page(url)
    if status != 200:
        return []
    matches = re.findall(r'href="(sound_os2/[^/]+/([^"/]+?)\.mp3)"', text)
    if not matches:
        raw_names = re.findall(r'([a-zA-Z0-9_]+)\.mp3', text)
        matches = [(None, n) for n in raw_names]

    _komori_sound_names[page_name] = matches
    return matches


def _komori_scrape_one():
    """从小森平随机抓取一个音效的 mp3 下载链接（Playwright + 内部重试）"""
    global _komori_cfg
    if _komori_cfg is None:
        _komori_cfg = _load_json_config("komori")

    cats = list(_komori_cfg["category_pages"])
    random.shuffle(cats)

    for entry in cats[:5]:
        page_name = entry["file"]
        matches = _komori_scrape_sound_names(page_name)
        if not matches:
            continue

        available = [(rel_path, name) for rel_path, name in matches
                     if (page_name, name) not in _komori_used]
        if not available:
            continue

        rel_path, name = random.choice(available)
        _komori_used.add((page_name, name))

        if rel_path:
            url = f"https://taira-komori.net/{rel_path}"
        else:
            category = page_name.replace("cn.html", "")
            url = f"https://taira-komori.net/sound/{category}/{name}.mp3"
        return url, name

    return None, None


# ═══════════════════════════════════════════════════════════
#  数据源 4: BigSoundBank
# ═══════════════════════════════════════════════════════════

_bigsoundbank_cfg = None
_bigsoundbank_known_ids = set()


def _bigsoundbank_scrape_one():
    """从 BigSoundBank 随机抓取一个音效的 mp3 下载链接（Playwright + 内部重试）"""
    global _bigsoundbank_cfg
    if _bigsoundbank_cfg is None:
        _bigsoundbank_cfg = _load_json_config("bigsoundbank")

    cat_ids = list(_bigsoundbank_cfg["category_ids"])
    random.shuffle(cat_ids)

    for cat_id in cat_ids[:5]:
        cat_url = f"{_bigsoundbank_cfg['base_url']}/search?CatID={cat_id}"
        status, text = _fetch_page(cat_url, timeout=25)
        if status != 200 or len(text) < 3000:
            continue

        ids = re.findall(r'/([^/"]+)-s(\d+)\.html', text)
        if not ids:
            continue

        candidates = [(slug, sid) for slug, sid in ids if sid not in _bigsoundbank_known_ids]
        if not candidates:
            continue

        slug, sid = random.choice(candidates)
        _bigsoundbank_known_ids.add(sid)

        mp3_url = _bigsoundbank_cfg["mp3_url_template"].format(id=sid)
        return mp3_url, slug

    return None, None


# ═══════════════════════════════════════════════════════════
#  数据源 5: PacDV
# ═══════════════════════════════════════════════════════════

_pacdv_cfg = None
_pacdv_known_urls = set()


def _pacdv_scrape_one():
    """从 PacDV 随机抓取一个音效的 mp3 下载链接（Playwright + 内部重试）"""
    global _pacdv_cfg
    if _pacdv_cfg is None:
        _pacdv_cfg = _load_json_config("pacdv")

    cats = list(_pacdv_cfg["categories"])
    random.shuffle(cats)

    for cat in cats[:4]:
        cat_url = f"{_pacdv_cfg['base_url']}/{cat['url']}"
        status, text = _fetch_page(cat_url, timeout=25)
        if status != 200 or len(text) < 2000:
            continue

        mp3_matches = re.findall(r'href="([^"]+\.mp3)"', text)
        if not mp3_matches:
            continue

        candidates = [u for u in mp3_matches if u not in _pacdv_known_urls]
        if not candidates:
            continue

        rel_url = random.choice(candidates)
        _pacdv_known_urls.add(rel_url)

        if rel_url.startswith("http"):
            abs_url = rel_url
        else:
            abs_url = f"{_pacdv_cfg['base_url']}/{rel_url.lstrip('/')}"

        label = os.path.splitext(os.path.basename(rel_url))[0]
        return abs_url, label

    return None, None


# ═══════════════════════════════════════════════════════════
#  数据源 6: Pixabay Sound Effects (Playwright 浏览器 + JSON-LD)
# ═══════════════════════════════════════════════════════════

_pixabay_cfg = None
_pixabay_known_ids = set()


def _pixabay_extract_detail_links(html):
    """从搜索页 HTML 中提取音效详情页链接列表"""
    # 匹配 /sound-effects/slug-name-123456/ 格式
    links = re.findall(
        r'href=["\'](/sound-effects/[a-z][a-z0-9_]*-[a-z0-9_]*-\d+/)["\']',
        html,
    )
    # 去重并过滤已知的
    seen = set()
    unique = []
    for link in links:
        # 从 URL 提取 ID
        id_m = re.search(r"-(\d+)/?$", link)
        if not id_m:
            continue
        sid = id_m.group(1)
        if sid in _pixabay_known_ids or sid in seen:
            continue
        seen.add(sid)
        unique.append(link)
    return unique


def _pixabay_extract_content_url(html):
    """从详情页 HTML 的 JSON-LD 中提取 AudioObject 的 contentUrl"""
    ld_blocks = re.findall(
        r'<script type="application/ld\+json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    for block in ld_blocks:
        try:
            data = json.loads(block)
            # 可能是单个对象或数组
            items = data if isinstance(data, list) else [data]
            for item in items:
                if item.get("@type") == "AudioObject":
                    url = item.get("contentUrl", "")
                    name = item.get("name", "")
                    if url:
                        return url, name
        except (json.JSONDecodeError, TypeError):
            continue
    return None, None


def _pixabay_scrape_one(keyword=None):
    """从 Pixabay Sound Effects 抓取一个音效的 mp3 下载链接

    策略 (Playwright + 滚动加载 + 内部重试):
    1. 随机选搜索词 → Playwright 打开搜索页 + 滚动 3 次触发无限加载
    2. 提取更多详情页链接（滚动后可见更多卡片）
    3. 详情页 → 从 JSON-LD 提取 AudioObject.contentUrl

    Args:
        keyword: 搜索关键词（None=从 search_slugs 随机选）
    """
    global _pixabay_cfg
    if _pixabay_cfg is None:
        _pixabay_cfg = _load_json_config("pixabay")

    # 关键词模式：用关键词构建搜索 URL
    if keyword:
        return _pixabay_scrape_by_keyword(keyword)

    slugs = list(_pixabay_cfg["search_slugs"])
    random.shuffle(slugs)

    for slug in slugs[:4]:  # 内部重试：最多试 4 个不同搜索词
        page = random.randint(1, 3)
        if page == 1:
            search_url = f"{_pixabay_cfg['base_url']}/sound-effects/search/{slug}/"
        else:
            search_url = f"{_pixabay_cfg['base_url']}/sound-effects/search/{slug}/?pagi={page}"

        # 滚动 3 次触发无限加载，让更多音效卡片出现
        status, text = _fetch_page_scroll(search_url, timeout=35,
                                           scroll_count=3, min_length=3000)
        if status != 200 or "Just a moment" in text:
            continue

        detail_links = _pixabay_extract_detail_links(text)
        if not detail_links:
            continue

        # 试最多 3 个不同的详情页
        random.shuffle(detail_links)
        for detail_path in detail_links[:3]:
            id_m = re.search(r"-(\d+)/?$", detail_path)
            if id_m:
                sid = id_m.group(1)
                if sid in _pixabay_known_ids:
                    continue
                _pixabay_known_ids.add(sid)

            detail_url = f"{_pixabay_cfg['base_url']}{detail_path}"
            status2, text2 = _fetch_page(detail_url, timeout=25)
            if status2 != 200 or "Just a moment" in text2 or len(text2) < 3000:
                continue

            mp3_url, name = _pixabay_extract_content_url(text2)
            if mp3_url:
                return mp3_url, name

    return None, None


def _pixabay_scrape_by_keyword(keyword):
    """用关键词从 Pixabay 搜索并抓取一个音效

    Args:
        keyword: 搜索关键词（如 "bell", "explosion"）

    Returns:
        (mp3_url, name) or (None, None)
    """
    global _pixabay_cfg, _pixabay_known_ids
    if _pixabay_cfg is None:
        _pixabay_cfg = _load_json_config("pixabay")

    # 用关键词构建搜索 URL（支持中英文）
    # Pixabay 搜索 URL 格式: /sound-effects/search/{keyword}/
    kw_encoded = urllib.parse.quote(keyword)
    search_url = f"{_pixabay_cfg['base_url']}/sound-effects/search/{kw_encoded}/"

    # 滚动加载更多结果
    status, text = _fetch_page_scroll(search_url, timeout=35,
                                       scroll_count=5, min_length=3000)
    if status != 200 or "Just a moment" in text:
        # 重试不同分页
        for page in range(2, 5):
            page_url = f"{_pixabay_cfg['base_url']}/sound-effects/search/{kw_encoded}/?pagi={page}"
            status, text = _fetch_page_scroll(page_url, timeout=35,
                                               scroll_count=3, min_length=3000)
            if status == 200 and "Just a moment" not in text:
                break
        else:
            return None, None

    detail_links = _pixabay_extract_detail_links(text)
    if not detail_links:
        return None, None

    # 试最多 5 个详情页
    random.shuffle(detail_links)
    for detail_path in detail_links[:5]:
        id_m = re.search(r"-(\d+)/?$", detail_path)
        if id_m:
            sid = id_m.group(1)
            if sid in _pixabay_known_ids:
                continue
            _pixabay_known_ids.add(sid)

        detail_url = f"{_pixabay_cfg['base_url']}{detail_path}"
        status2, text2 = _fetch_page(detail_url, timeout=25)
        if status2 != 200 or "Just a moment" in text2 or len(text2) < 3000:
            continue

        mp3_url, name = _pixabay_extract_content_url(text2)
        if mp3_url:
            return mp3_url, name

    return None, None



# ═══════════════════════════════════════════════════════════
#  数据源 7: SoundEffectsPlus
# ═══════════════════════════════════════════════════════════

_seplus_cfg = None
_seplus_known_fids = set()


def _seplus_scrape_one():
    """从 SoundEffectsPlus 随机抓取一个音效的下载链接（Playwright + 内部重试）"""
    global _seplus_cfg
    if _seplus_cfg is None:
        _seplus_cfg = _load_json_config("soundeffectsplus")

    cats = list(_seplus_cfg["categories"])
    random.shuffle(cats)
    max_pages = _seplus_cfg["max_pages_per_category"]

    for cat in cats[:4]:
        for _ in range(3):
            page = random.randint(1, max_pages)
            if page == 1:
                cat_url = f"{_seplus_cfg['base_url']}/{cat['url']}"
            else:
                cat_url = f"{_seplus_cfg['base_url']}/{cat['url']}{page}/"

            status, text = _fetch_page(cat_url, timeout=25)
            if status != 200 or len(text) < 3000:
                continue

            products = re.findall(r'''href\s*=\s*["'](/product/[^"']+/)["']''', text)
            if not products:
                products = re.findall(r'/product/([^/"\s]+)/', text)
                products = [f"/product/{p}/" for p in set(products)]

            candidates = [p for p in products if p not in _seplus_known_fids]
            if not candidates:
                continue

            product_url = f"{_seplus_cfg['base_url']}{random.choice(candidates)}"
            _seplus_known_fids.add(product_url)

            status2, text2 = _fetch_page(product_url, timeout=25)
            if status2 != 200 or len(text2) < 2000:
                continue

            m = re.search(r'download\.php\?fid=(\d+)', text2)
            if not m:
                continue

            fid = m.group(1)
            dl_url = f"{_seplus_cfg['base_url']}/download.php?fid={fid}"
            label = product_url.rstrip("/").rsplit("/", 1)[-1]
            return dl_url, label, "zip"

    return None, None


# ═══════════════════════════════════════════════════════════
#  数据源 8: SoundJay
# ═══════════════════════════════════════════════════════════

_soundjay_cfg = None
_soundjay_known_urls = set()


def _soundjay_scrape_one():
    """从 SoundJay 随机抓取一个音效的 mp3 下载链接（Playwright + 内部重试）"""
    global _soundjay_cfg
    if _soundjay_cfg is None:
        _soundjay_cfg = _load_json_config("soundjay")

    pages = list(_soundjay_cfg["category_pages"])
    random.shuffle(pages)

    for page_path in pages[:4]:
        page_url = f"{_soundjay_cfg['base_url']}/{page_path}"
        status, text = _fetch_page(page_url, timeout=25)
        if status != 200 or len(text) < 2000:
            continue

        mp3s = re.findall(r'["\']([^"\']+\.mp3)["\']', text)
        if not mp3s:
            continue

        candidates = [u for u in mp3s if u not in _soundjay_known_urls]
        if not candidates:
            continue

        rel_url = random.choice(candidates)
        _soundjay_known_urls.add(rel_url)

        if rel_url.startswith("http"):
            abs_url = rel_url
        else:
            abs_url = f"{_soundjay_cfg['base_url']}/{rel_url.lstrip('/')}"

        label = os.path.splitext(os.path.basename(rel_url))[0]
        return abs_url, label

    return None, None


# ═══════════════════════════════════════════════════════════
#  数据源 9: QuickSounds
# ═══════════════════════════════════════════════════════════

_quicksounds_cfg = None
_quicksounds_known_ids = set()


def _quicksounds_scrape_one():
    """从 QuickSounds 随机抓取一个音效的 mp3 下载链接

    策略：
    1. 随机生成 sound_id（已知范围 29000~29300）
    2. 访问 /sound/{id} 详情页
    3. 提取 /uploads/tracks/{name}.mp3 下载直链
    4. 下载时需要 Referer 头（指向详情页）
    """
    global _quicksounds_cfg
    if _quicksounds_cfg is None:
        _quicksounds_cfg = _load_json_config("quicksounds")

    lo, hi = _quicksounds_cfg["sound_id_range"]
    attempts = 0
    while attempts < 15:
        attempts += 1
        sound_id = random.randint(lo, hi)
        if sound_id in _quicksounds_known_ids:
            continue
        _quicksounds_known_ids.add(sound_id)

        detail_url = f"{_quicksounds_cfg['base_url']}/sound/{sound_id}"
        status, text = _fetch_page(detail_url, timeout=20)
        if status != 200 or len(text) < 5000:
            continue

        # 提取 mp3 链接，优先 /uploads/tracks/ 模式
        mp3s = re.findall(r'["\']([^"\']*\.mp3[^"\']*)["\']', text)
        mp3_url = None
        for m in mp3s:
            if "/uploads/tracks/" in m:
                mp3_url = m if m.startswith("http") else f"{_quicksounds_cfg['base_url']}{m}"
                break
        if not mp3_url and mp3s:
            # 回退：取第一个合法 mp3
            for m in mp3s:
                try:
                    candidate = m if m.startswith("http") else f"{_quicksounds_cfg['base_url']}{m.lstrip('/')}"
                    mp3_url = candidate
                    break
                except Exception:
                    continue

        if not mp3_url:
            continue

        # 从标题提取名称
        title_m = re.search(r'<title>([^<]+)</title>', text)
        label = title_m.group(1).replace(" - QuickSounds.com", "").strip() if title_m else f"qs_{sound_id}"
        return mp3_url, label, "mp3", detail_url

    return None, None


# ═══════════════════════════════════════════════════════════
#  数据源 10: Mixkit (WAV 全量音效)
# ═══════════════════════════════════════════════════════════

_mixkit_cfg = None
_mixkit_known_ids = set()


def _mixkit_scrape_one():
    """从 Mixkit 随机抓取一个音效的 WAV 下载链接（Playwright + 滚动 + 内部重试）"""
    global _mixkit_cfg
    if _mixkit_cfg is None:
        _mixkit_cfg = _load_json_config("mixkit")

    slugs = list(_mixkit_cfg["category_slugs"])
    random.shuffle(slugs)

    for slug in slugs[:4]:  # 内部重试
        cat_url = f"{_mixkit_cfg['base_url']}/free-sound-effects/{slug}/"

        # 滚动加载更多音效卡片
        status, text = _fetch_page_scroll(cat_url, timeout=30,
                                           scroll_count=3, min_length=5000)
        if status != 200:
            continue

        id_matches = re.findall(
            r'data-algolia-analytics-item-id=["\'](\d+)["\']', text,
        )
        if not id_matches:
            continue

        candidates = [sid for sid in id_matches if sid not in _mixkit_known_ids]
        if not candidates:
            continue

        sound_id = random.choice(candidates)
        _mixkit_known_ids.add(sound_id)

        name = f"mixkit_{sound_id}"
        titles = re.findall(r'class="[^"]*meta-links__link[^"]*"[^>]*>([^<]+)<', text)
        ids_in_order = re.findall(r'data-algolia-analytics-item-id=["\'](\d+)["\']', text)
        try:
            idx = ids_in_order.index(sound_id)
            if idx < len(titles):
                name = titles[idx]
        except (ValueError, IndexError):
            pass

        wav_url = _mixkit_cfg["cdn_url_template"].format(id=sound_id)
        return wav_url, name, "wav", cat_url

    return None, None


# ═══════════════════════════════════════════════════════════
#  数据源 11: Myinstants (meme sound buttons)
# ═══════════════════════════════════════════════════════════

_myinstants_cfg = None
_myinstants_known_uuids = set()


def _myinstants_scrape_one():
    """从 Myinstants 随机抓取一个音效按钮的 mp3 下载链接（Playwright + 内部重试）"""
    global _myinstants_cfg
    if _myinstants_cfg is None:
        _myinstants_cfg = _load_json_config("myinstants")

    pages = list(_myinstants_cfg["listing_pages"])
    random.shuffle(pages)
    for listing_url in pages[:4]:
        status, text = _fetch_page(listing_url, timeout=25)
        if status != 200 or len(text) < 3000:
            continue

        matches = re.findall(
            r'/memesounds/([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})/([^/"\s]+)/',
            text,
        )
        if not matches:
            continue

        candidates = [(uid, slug) for uid, slug in matches if uid not in _myinstants_known_uuids]
        if not candidates:
            continue

        uid, slug = random.choice(candidates)
        _myinstants_known_uuids.add(uid)

        mp3_url = f"{_myinstants_cfg['base_url']}/AllSounds/{uid}.mp3"
        return mp3_url, slug, "mp3", listing_url

    return None, None


# ── AudioMicro ──────────────────────────────────────────

_audiomicro_cfg = None
_audiomicro_known_hashes: set = set()


def _audiomicro_scrape_one():
    """随机分类 → 提取下载hash和标题 → /tracks/download/{hash}/wav（Playwright + 内部重试）"""
    global _audiomicro_cfg, _audiomicro_known_hashes
    if _audiomicro_cfg is None:
        _audiomicro_cfg = _load_json_config("audiomicro")

    cats = list(_audiomicro_cfg["categories"])
    random.shuffle(cats)
    max_pages = _audiomicro_cfg.get("pages_per_category", 3)

    for cat in cats[:4]:
        for _ in range(3):
            page_num = random.randint(1, max_pages)
            page_url = f"{_audiomicro_cfg['base_url']}{cat}"
            if page_num > 1:
                page_url += f"/page-{page_num}"

            status, text = _fetch_page(page_url, timeout=25)
            if status != 200 or len(text) < 3000:
                continue

            titles = re.findall(r'class="track-title"[^>]*>\s*(.*?)\s*</span>', text)
            hashes = re.findall(r'/tracks/download/([a-f0-9]+)/wav', text)

            if not hashes:
                continue

            pairs = []
            for i, h in enumerate(hashes):
                label = titles[i] if i < len(titles) else f"audio{h[:8]}"
                pairs.append((h, label))

            candidates = [(h, lb) for h, lb in pairs if h not in _audiomicro_known_hashes]
            if not candidates:
                continue

            track_hash, label = random.choice(candidates)
            _audiomicro_known_hashes.add(track_hash)

            dl_url = f"{_audiomicro_cfg['base_url']}/tracks/download/{track_hash}/wav"
            return dl_url, label, "wav", page_url

    return None, None


# ── FindSounds ──────────────────────────────────────────

_findsounds_cfg = None
_findsounds_known_urls: set = set()


def _findsounds_scrape_one():
    """随机关键词搜索 → 提取结果中的 mp3/wav 链接 → 直接下载（Playwright + 内部重试）"""
    global _findsounds_cfg, _findsounds_known_urls
    if _findsounds_cfg is None:
        _findsounds_cfg = _load_json_config("findsounds")

    keywords = list(_findsounds_cfg["keywords"])
    random.shuffle(keywords)

    for keyword in keywords[:6]:
        page_start = random.randint(0, 5) * 10
        if page_start == 0:
            url = f"{_findsounds_cfg['base_url']}/ISAPI/search.dll?keywords={urllib.parse.quote(keyword)}"
        else:
            url = f"{_findsounds_cfg['base_url']}/ISAPI/search.dll?start={page_start + 1}&keywords={urllib.parse.quote(keyword)}"

        status, text = _fetch_page(url, timeout=25)
        if status != 200 or len(text) < 500:
            continue

        audio_links = re.findall(r'(https?://[^\s"\'<>]+\.(?:mp3|wav))', text, re.IGNORECASE)
        if not audio_links:
            continue

        candidates = [l for l in audio_links if l not in _findsounds_known_urls]
        if not candidates:
            continue

        mp3_url = random.choice(candidates)
        _findsounds_known_urls.add(mp3_url)

        fname = mp3_url.rsplit("/", 1)[-1]
        label = urllib.parse.unquote(fname.rsplit(".", 1)[0])[:80]
        ext = fname.rsplit(".", 1)[-1].lower()

        return mp3_url, label, ext, url

    return None, None


# ── PartnersInRhyme ──────────────────────────────────────
# 需要 Playwright (urllib 被 Cloudflare 403)
# 导航结构: /pir/PIRsfx.shtml → /soundfx/{Cat}.shtml → /soundfx/{Cat}_sounds/{name}.shtml

_pir_cfg = None
_pir_cat_pages: list = []     # 缓存分类页链接
_pir_known_urls: set = set()  # 已访问的详情页


def _pir_scrape_one():
    """入口页 → 分类页 → 详情页 → 提取下载链接"""
    global _pir_cfg, _pir_cat_pages, _pir_known_urls
    if _pir_cfg is None:
        _pir_cfg = _load_json_config("partnersinrhyme")

    # ── Step 1: 从入口页获取/缓存分类页列表 ──
    if not _pir_cat_pages:
        index_url = f"{_pir_cfg['base_url']}{_pir_cfg['index_page']}"
        status, text = _fetch_page(index_url, timeout=40, cf_retries=4)
        if status != 200:
            return None, None

        # 提取分类页: /soundfx/{Name}.shtml (不含 _sounds 子目录的)
        cats = re.findall(
            r'''href\s*=\s*["'](/soundfx/([^"']+)\.shtml)["']''',
            text,
        )
        for link, _ in cats:
            if "_sounds/" not in link and link not in _pir_cat_pages:
                _pir_cat_pages.append(link)

        if not _pir_cat_pages:
            return None, None

    # ── Step 2: 随机选一个分类页 ──
    cat_path = random.choice(_pir_cat_pages)
    cat_url = f"{_pir_cfg['base_url']}{cat_path}"
    cat_name = cat_path.rsplit("/", 1)[-1].replace(".shtml", "")

    status, text = _fetch_page(cat_url, timeout=40, cf_retries=4)
    if status != 200:
        return None, None

    # ── Step 3: 从分类页提取详情页链接 ──
    # 详情页格式: /soundfx/{CatName}_sounds/{sound}.shtml
    detail_pattern = rf'''href\s*=\s*["']({re.escape(f"/soundfx/")}[\w_]+/\w+\.shtml)["']'''
    sound_pages = re.findall(detail_pattern, text)
    if not sound_pages:
        # 回退: 所有 soundfx 下的 shtml 链接（排除分类页自身）
        all_links = re.findall(
            r'''href\s*=\s*["'](/soundfx/([^"']+\.shtml))["']''',
            text,
        )
        sound_pages = [link for link, _ in all_links if link != cat_path]

    candidates = [sp for sp in sound_pages if sp not in _pir_known_urls]
    if not candidates:
        return None, None

    sp = random.choice(candidates)
    _pir_known_urls.add(sp)

    # ── Step 4: 访问详情页，提取下载链接 ──
    if sp.startswith("http"):
        detail_url = sp
    else:
        detail_url = f"{_pir_cfg['base_url']}/{sp.lstrip('/')}"

    status2, text2 = _fetch_page(detail_url, timeout=40, cf_retries=4)
    if status2 != 200:
        return None, None

    label = sp.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    # 策略1: href/src 属性中的 mp3/wav 直链
    audio_links = re.findall(
        r'''(?:href|src)\s*=\s*["']([^"']*\.(?:mp3|wav))["']''',
        text2, re.IGNORECASE,
    )
    for link in audio_links:
        if link.startswith("http"):
            url = link
        elif link.startswith("/"):
            url = f"{_pir_cfg['base_url']}{link}"
        else:
            url = f"{detail_url.rsplit('/', 1)[0]}/{link}"
        ext = "mp3" if link.lower().endswith(".mp3") else "wav"
        return url, label, ext, detail_url

    # 策略2: 全页文本匹配 http 开头的音频 URL
    dl_wav = re.search(r'(https?://[^\s"\']+\.wav)', text2, re.IGNORECASE)
    dl_mp3 = re.search(r'(https?://[^\s"\']+\.mp3)', text2, re.IGNORECASE)

    # 策略3: 回退 — href 中包含 download/mp3/wav 关键词
    if not dl_mp3 and not dl_wav:
        btn = re.search(
            r'''(?:href|src)\s*=\s*["']([^"']*(?:download|mp3|wav|\.wav|\.mp3)[^"']*)["']''',
            text2, re.IGNORECASE,
        )
        if btn:
            dl_url = btn.group(1)
            if not dl_url.startswith("http"):
                dl_url = f"{_pir_cfg['base_url']}/{dl_url.lstrip('/')}"
            if ".mp3" in dl_url.lower():
                return dl_url, label, "mp3", detail_url
            elif ".wav" in dl_url.lower():
                return dl_url, label, "wav", detail_url

    if dl_wav:
        return dl_wav.group(1), label, "wav", detail_url
    elif dl_mp3:
        return dl_mp3.group(1), label, "mp3", detail_url

    return None, None





# ═══════════════════════════════════════════════════════════
#  数据源 15: Zapsplat
# ═══════════════════════════════════════════════════════════

_zapsplat_cfg = None
_zapsplat_known_ids = set()


def _zapsplat_scrape_one():
    """从 Zapsplat 抓取一个免费音效的下载链接（Playwright + 内部重试）"""
    global _zapsplat_cfg
    if _zapsplat_cfg is None:
        _zapsplat_cfg = _load_json_config("zapsplat")

    cats = list(_zapsplat_cfg["categories"])
    random.shuffle(cats)
    max_pages = _zapsplat_cfg.get("max_pages", 5)

    for cat in cats[:4]:  # 内部重试
        for _ in range(3):
            page = random.randint(1, max_pages)
            if page == 1:
                cat_url = f"{_zapsplat_cfg['base_url']}{cat}"
            else:
                cat_url = f"{_zapsplat_cfg['base_url']}{cat}page/{page}/"

            status, text = _fetch_page(cat_url, timeout=30)
            if status != 200 or len(text) < 5000:
                continue

            ids = re.findall(r'data-sound-id=["\'](\d+)["\']', text)
            if not ids:
                continue

            candidates = [sid for sid in ids if sid not in _zapsplat_known_ids]
            if not candidates:
                continue

            sound_id = random.choice(candidates)
            _zapsplat_known_ids.add(sound_id)

            mp3_url = _zapsplat_cfg["cdn_template"].format(id=sound_id)
            return mp3_url, f"zapsplat_{sound_id}", "mp3", cat_url

    return None, None


# ═══════════════════════════════════════════════════════════
#  数据源 16: WavSource
# ═══════════════════════════════════════════════════════════

_wavsource_cfg = None
_wavsource_known_urls = set()


def _wavsource_scrape_one():
    """从 WavSource 抓取一个 WAV 音效直链（Playwright + 内部重试）"""
    global _wavsource_cfg
    if _wavsource_cfg is None:
        _wavsource_cfg = _load_json_config("wavsource")

    cats = list(_wavsource_cfg["categories"])
    random.shuffle(cats)

    for cat in cats[:4]:
        cat_url = f"{_wavsource_cfg['base_url']}/{cat}"
        status, text = _fetch_page(cat_url, timeout=25)
        if status != 200 or len(text) < 1000:
            continue

        wavs = re.findall(r'href=["\']([^"\']+\.wav)["\']', text)
        if not wavs:
            continue

        candidates = [w for w in wavs if w not in _wavsource_known_urls]
        if not candidates:
            continue

        rel = random.choice(candidates)
        _wavsource_known_urls.add(rel)

        if rel.startswith("http"):
            url = rel
        else:
            url = f"{_wavsource_cfg['base_url']}/{rel.lstrip('/')}"

        label = os.path.splitext(os.path.basename(rel))[0]
        return url, label, "wav"

    return None, None


# ═══════════════════════════════════════════════════════════
#  数据源 17: Sampleswap
# ═══════════════════════════════════════════════════════════

_sampleswap_cfg = None
_sampleswap_known_urls = set()


def _sampleswap_scrape_one():
    """从 Sampleswap 抓取免费音频样本（Playwright + 内部重试）"""
    global _sampleswap_cfg
    if _sampleswap_cfg is None:
        _sampleswap_cfg = _load_json_config("sampleswap")

    pages = list(_sampleswap_cfg["listing_pages"])
    random.shuffle(pages)

    for listing_url in pages[:4]:
        status, text = _fetch_page(listing_url, timeout=25)
        if status != 200 or len(text) < 2000:
            continue

        mp3s = re.findall(r'href=["\']([^"\']+\.mp3)["\']', text)
        if not mp3s:
            continue

        candidates = [m for m in mp3s if m not in _sampleswap_known_urls]
        if not candidates:
            continue

        rel = random.choice(candidates)
        _sampleswap_known_urls.add(rel)

        if rel.startswith("http"):
            url = rel
        elif rel.startswith("/"):
            url = f"{_sampleswap_cfg['base_url']}{rel}"
        else:
            url = f"{listing_url.rsplit('/', 1)[0]}/{rel}"

        label = os.path.splitext(os.path.basename(rel))[0][:80]
        return url, label, "mp3"

    return None, None


# ═══════════════════════════════════════════════════════════
#  数据源 18: 99Sounds
# ═══════════════════════════════════════════════════════════

_99sounds_cfg = None
_99sounds_known_urls = set()


def _99sounds_scrape_one():
    """从 99Sounds 抓取样音效包中的独立音频（Playwright + 内部重试）"""
    global _99sounds_cfg
    if _99sounds_cfg is None:
        _99sounds_cfg = _load_json_config("99sounds")

    packs = list(_99sounds_cfg["packs"])
    random.shuffle(packs)

    for pack in packs[:4]:
        pack_url = f"{_99sounds_cfg['base_url']}/{pack}/"
        status, text = _fetch_page(pack_url, timeout=25)
        if status != 200 or len(text) < 5000:
            continue

        audio_matches = re.findall(
            r'href=["\']([^"\']*\.(?:mp3|wav))["\']',
            text, re.IGNORECASE,
        )
        if not audio_matches:
            continue

        candidates = [a for a in audio_matches if a not in _99sounds_known_urls]
        if not candidates:
            continue

        link = random.choice(candidates)
        _99sounds_known_urls.add(link)

        if link.startswith("http"):
            url = link
        else:
            url = f"{_99sounds_cfg['base_url']}/{link.lstrip('/')}"

        ext = "wav" if link.lower().endswith(".wav") else "mp3"
        label = os.path.splitext(os.path.basename(link))[0][:80]
        return url, label, ext

    return None, None


# ═══════════════════════════════════════════════════════════
#  下载执行
# ═══════════════════════════════════════════════════════════

SOURCES = [
    ("SoundGator",         _soundgator_scrape_one),
    ("SFXLibrary",         _sfxlibrary_scrape_one),
    ("Komori",             _komori_scrape_one),
    ("BigSoundBank",       _bigsoundbank_scrape_one),
    ("PacDV",              _pacdv_scrape_one),
    ("Pixabay",            _pixabay_scrape_one),
    ("SoundEffectsPlus",   _seplus_scrape_one),
    ("SoundJay",           _soundjay_scrape_one),
    ("QuickSounds",        _quicksounds_scrape_one),
    ("Mixkit",             _mixkit_scrape_one),
    ("Myinstants",         _myinstants_scrape_one),
    ("AudioMicro",         _audiomicro_scrape_one),
    ("FindSounds",         _findsounds_scrape_one),
    ("PartnersInRhyme",    _pir_scrape_one),
    ("Zapsplat",           _zapsplat_scrape_one),
    ("WavSource",          _wavsource_scrape_one),
    ("Sampleswap",         _sampleswap_scrape_one),
    ("99Sounds",           _99sounds_scrape_one),
]


# ═════════════════════════════════════════════════════════
#  关键词搜索下载（新功能）
# ═════════════════════════════════════════════════════════


def scrape_by_keyword(keyword, src_name=None, verbose=True):
    """用关键词从数据源搜索并抓取一个音效

    优先使用支持搜索的数据源，失败则 fallback 到随机下载+本地筛选。

    Args:
        keyword: 搜索关键词
        src_name: 指定数据源名称（None=自动选择）
        verbose: 是否打印进度

    Returns:
        (url, name, ext, referer) or None
    """
    # ── 动态构建支持关键词搜索的源列表（调用时所有函数均已定义）──
    _kw_sources = []
    # Pixabay（始终可用）
    _kw_sources.append(("Pixabay", _pixabay_scrape_by_keyword))
    # FindSounds（如果已定义）
    if '_findsounds_scrape_by_keyword' in globals():
        _kw_sources.append(("FindSounds", _findsounds_scrape_by_keyword))
    # SoundGator（如果已定义）
    if '_soundgator_scrape_by_keyword' in globals():
        _kw_sources.append(("SoundGator", _soundgator_scrape_by_keyword))
    # Komori（如果已定义）
    if '_komori_scrape_by_keyword' in globals():
        _kw_sources.append(("Komori", _komori_scrape_by_keyword))

    if verbose:
        print(f"\n  🔍 关键词搜索: '{keyword}'")

    # ── 1. 如果指定了数据源，只用那个源 ──
    if src_name:
        for name, fn in _kw_sources:
            if name.lower() == src_name.lower():
                if verbose:
                    print(f"  指定数据源: {name}")
                result = fn(keyword)
                if result and result[0]:
                    return result
                else:
                    if verbose:
                        print(f"  ✗ {name} 未找到匹配")
                    return None
        if verbose:
            print(f"  ⚠ 未知数据源: {src_name}，使用自动选择")
        # fallback 到自动选择

    # ── 2. 自动选择：依次尝试支持搜索的源 ──
    for name, fn in _kw_sources:
        if verbose:
            print(f"  [{name}] 搜索中...", end="")
        try:
            result = fn(keyword)
            if result and result[0]:
                if verbose:
                    print(f" ✓ 找到: {result[1]}")
                return result
            else:
                if verbose:
                    print(" 未找到")
        except Exception as e:
            if verbose:
                print(f" 错误: {e}")

    # ── 3. Fallback: 随机下载一批，然后在本地库搜索 ──
    if verbose:
        print(f"  ⚠ 所有搜索源均未找到，尝试随机下载...")
    return None


def _findsounds_scrape_by_keyword(keyword):
    """用关键词从 FindSounds 搜索并抓取一个音效"""
    global _findsounds_cfg, _findsounds_known_urls
    if _findsounds_cfg is None:
        _findsounds_cfg = _load_json_config("findsounds")

    # FindSounds 支持 URL 搜索
    # 格式: /ISAPI/search.dll?keywords={keyword}&start={start}
    page_start = 0
    url = f"{_findsounds_cfg['base_url']}/ISAPI/search.dll?keywords={urllib.parse.quote(keyword)}"

    status, text = _fetch_page(url, timeout=25)
    if status != 200 or len(text) < 500:
        return None

    audio_links = re.findall(r'(https?://[^\s"\'<>]+\.(?:mp3|wav))', text, re.IGNORECASE)
    if not audio_links:
        return None

    candidates = [l for l in audio_links if l not in _findsounds_known_urls]
    if not candidates:
        return None

    mp3_url = random.choice(candidates)
    _findsounds_known_urls.add(mp3_url)

    fname = mp3_url.rsplit("/", 1)[-1]
    label = urllib.parse.unquote(fname.rsplit(".", 1)[0])[:80]
    ext = fname.rsplit(".", 1)[-1].lower()

    return mp3_url, label, ext, url


def _soundgator_scrape_by_keyword(keyword):
    """用关键词从 SoundGator 搜索（在分类页中筛选包含关键词的音效）"""
    global _soundgator_cfg, _soundgator_known_ids
    if _soundgator_cfg is None:
        _soundgator_cfg = _load_json_config("soundgator")

    cats = list(_soundgator_cfg["categories"])
    random.shuffle(cats)

    keyword_lower = keyword.lower()
    for cat_url in cats[:6]:
        status, text = _fetch_page(cat_url, timeout=25)
        if status != 200 or len(text) < 3000:
            continue

        # 提取产品链接和名称
        products = re.findall(r'/product/(\d+)-([^/"]+)-sound-effect', text)
        if not products:
            continue

        # 筛选包含关键词的产品
        matched = [(pid, name) for pid, name in products
                    if keyword_lower in name.lower() and pid not in _soundgator_known_ids]

        if not matched:
            continue

        pid, name = random.choice(matched)
        _soundgator_known_ids.add(pid)

        prod_url = f"https://www.soundgator.com/product/{pid}-{name}-sound-effect"
        status2, text2 = _fetch_page(prod_url, timeout=25)
        if status2 != 200 or len(text2) < 2000:
            continue

        mp3 = re.search(r'https?://[^"\s]+?\.mp3', text2)
        if mp3:
            return mp3.group(0), name

    return None


def _komori_scrape_by_keyword(keyword):
    """用关键词从小森平搜索音效（在分类页中筛选）"""
    global _komori_cfg, _komori_used
    if _komori_cfg is None:
        _komori_cfg = _load_json_config("komori")

    keyword_lower = keyword.lower()
    cats = list(_komori_cfg["category_pages"])
    random.shuffle(cats)

    for entry in cats[:8]:
        page_name = entry["file"]
        matches = _komori_scrape_sound_names(page_name)
        if not matches:
            continue

        # 筛选包含关键词的音效
        available = [(rel_path, name) for rel_path, name in matches
                     if keyword_lower in name.lower()
                     and (page_name, name) not in _komori_used]

        if not available:
            continue

        rel_path, name = random.choice(available)
        _komori_used.add((page_name, name))

        if rel_path:
            url = f"https://taira-komori.net/{rel_path}"
        else:
            category = page_name.replace("cn.html", "")
            url = f"https://taira-komori.net/sound/{category}/{name}.mp3"
        return url, name

    return None


def _download_one(idx, keyword=None):
    """每次选一个数据源下载，失败时重试其他源（最多 MAX_SCRAPE_RETRIES 次）

    Args:
        idx: 下载序号（用于文件名）
        keyword: 搜索关键词（None=随机下载模式）
    """
    # ── 关键词模式：用 scrape_by_keyword() ──
    if keyword:
        result = scrape_by_keyword(keyword, verbose=True)
        if result and result[0]:
            url, label = result[0], result[1]
            ext = result[2] if len(result) > 2 else "mp3"
            referer = result[3] if len(result) > 3 else None

            src_name = "KeywordSearch"
            status, audio = _fetch(url, referer=referer)
            if status == 200 and audio:
                fname = _safe_filename(label, src_name, idx + 1)
                filepath = os.path.join(SOUNDS_DIR, f"{fname}.{ext}")
                _save_file(filepath, audio)
                print(f"  [{idx + 1}] [Keyword:{keyword}] -> {fname}.{ext}")
                # 更新索引
                build_index(verbose=False)
                return True
            else:
                print(f"  [{idx + 1}] [Keyword:{keyword}] (HTTP {status})")
                return False
        else:
            print(f"  [{idx + 1}] [Keyword:{keyword}] (未找到)")
            return False

    # ── 随机模式（原有逻辑）──
    tried = set()
    for attempt in range(MAX_SCRAPE_RETRIES):
        # 排除已尝试过的源
        available = [(n, f) for n, f in SOURCES if n not in tried]
        if not available:
            break

        src_name, scrape_fn = random.choice(available)
        tried.add(src_name)

        if attempt > 0:
            print(f"  [{idx + 1}] [{src_name}] (retry {attempt})", end="")
        else:
            print(f"  [{idx + 1}] [{src_name}]", end="")

        try:
            result = scrape_fn()
            if not result or not result[0]:
                print(" (no url)")
                # 确保统计记录存在
                _source_stats.setdefault(src_name, {"ok": 0, "fail": 0})
                _source_stats[src_name]["fail"] += 1
                continue

            # 支持两元组 / 三元组 / 四元组 (url, label, ext, referer)
            referer = None
            if len(result) == 4:
                url, label, ext, referer = result
            elif len(result) == 3:
                url, label, ext = result
            else:
                url, label = result
                ext = "mp3"

            if not url:
                print(" (no url)")
                _source_stats.setdefault(src_name, {"ok": 0, "fail": 0})
                _source_stats[src_name]["fail"] += 1
                continue

            status, audio = _fetch(url, referer=referer)
            if status == 200 and audio:
                fname = _safe_filename(label, src_name, idx + 1)
                filepath = os.path.join(SOUNDS_DIR, f"{fname}.{ext}")
                _save_file(filepath, audio)
                print(f" -> {fname}.{ext}")
                _source_stats.setdefault(src_name, {"ok": 0, "fail": 0})
                _source_stats[src_name]["ok"] += 1
                return True
            else:
                print(f" (HTTP {status})")
                _source_stats.setdefault(src_name, {"ok": 0, "fail": 0})
                _source_stats[src_name]["fail"] += 1
        except Exception as e:
            print(f" err:{e}")
            _source_stats.setdefault(src_name, {"ok": 0, "fail": 0})
            _source_stats[src_name]["fail"] += 1

    # 所有重试都失败
    print(f"  [{idx + 1}] [---]  all retries exhausted")
    return False


def main():
    """主函数：解析命令行参数并调度"""
    # ── 参数解析 ──
    parser = argparse.ArgumentParser(
        description="SFX 音效下载器 — 支持关键词搜索和本地库管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python dl_sounds.py --keyword "explosion"          # 搜索并下载 "explosion" 相关音效
  python dl_sounds.py -k "bell" -n 5                # 下载 5 个包含 "bell" 的音效
  python dl_sounds.py -k "footstep" --source pixabay  # 从 Pixabay 搜索 "footstep"
  python dl_sounds.py --search-local "bell"            # 搜索本地库
  python dl_sounds.py --list                              # 列出本地库所有音效
  python dl_sounds.py -n 20                               # 随机下载 20 个（原功能）
"""
    )
    parser.add_argument("-k", "--keyword", type=str, default=None,
                        help="搜索关键词（如 'explosion', 'bell', 'footstep'）")
    parser.add_argument("-s", "--search-local", type=str, default=None,
                        help="搜索本地音效库（按文件名关键词）")
    parser.add_argument("-l", "--list", action="store_true",
                        help="列出本地音效库中的所有音效")
    parser.add_argument("-n", "--num", type=int, default=NUM_SOUNDS,
                        help=f"下载数量（默认 {NUM_SOUNDS}）")
    parser.add_argument("--source", type=str, default=None,
                        help="指定数据源（pixabay, findsounds, komori, soundgator, 等）")

    args = parser.parse_args()

    # ── 列出本地库 ──
    if args.list:
        list_local(verbose=True)
        return

    # ── 搜索本地库 ──
    if args.search_local:
        results = search_local(args.search_local, verbose=True)
        if not results:
            print(f"\n  提示: 可运行 python dl_sounds.py -k \"{args.search_local}\" 从网络搜索并下载")
        return

    # ── 自动发现 sources/ 下的 JSON 并显示数据源列表 ──
    source_names = []
    if os.path.isdir(SOURCES_DIR):
        for f in sorted(os.listdir(SOURCES_DIR)):
            if f.endswith(".json"):
                source_names.append(f.replace(".json", ""))

    source_display = " | ".join(source_names) if source_names else "SoundGator | SFXLibrary | Komori"

    print("=" * 60)
    print("  免费音效下载器（无需注册、无需 API Key）")
    if args.keyword:
        print(f"  模式: 关键词搜索  [ {args.keyword} ]")
        if args.source:
            print(f"  指定数据源: {args.source}")
    else:
        print(f"  模式: 随机下载")
    print(f"  数据源: {len(SOURCES)} 个  |  每槽重试: {MAX_SCRAPE_RETRIES} 次")
    print(f"  保存路径: {SOUNDS_DIR}")
    print("=" * 60)

    target = max(1, args.num)
    print(f"\n  目标: {target} 个\n")

    ok = 0
    for i in range(target):
        if args.keyword:
            if _download_one(i, keyword=args.keyword):
                ok += 1
        else:
            if _download_one(i):
                ok += 1
        time.sleep(0.3)

    # ── 源统计 ──
    print(f"\n  {'─' * 50}")
    if args.keyword:
        print(f"  关键词搜索完成！成功下载 {ok}/{target} 个音效")
        if ok > 0:
            print(f"  更新本地索引...")
            build_index(verbose=True)
            print(f"\n  搜索本地库: python dl_sounds.py -s \"{args.keyword}\"")
    else:
        print(f"  随机下载完成！成功下载 {ok}/{target} 个音效")
        if ok > 0:
            print(f"  更新本地索引...")
            build_index(verbose=True)

    if _source_stats:
        print(f"\n  ── 数据源统计 ──")
        for src_name in sorted(_source_stats.keys()):
            st = _source_stats[src_name]
            total = st["ok"] + st["fail"]
            rate = st["ok"] / total * 100 if total > 0 else 0
            bar = "█" * int(rate / 10) + "░" * (10 - int(rate / 10))
            print(f"  {src_name:<20s} {bar} {st['ok']}/{total} ({rate:.0f}%)")

    sfx_utils.close_pw()


if __name__ == "__main__":
    main()
