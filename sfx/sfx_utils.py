"""
SFX 反爬增强工具模块
借鉴 WebCopier v5.4 的反爬体系，轻量适配到音效下载器。
提供：浏览器身份池轮换、增强 HTTP 请求、增强 Playwright 浏览器。
"""

import os
import time
import random
import ssl
import urllib.request
import urllib.error

# ── stealth.min.js 路径 ──
_STEALTH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stealth.min.js")
_HAS_STEALTH = os.path.exists(_STEALTH_PATH)

# ═══════════════════════════════════════════════════════════
#  浏览器身份池（25 种，覆盖主流浏览器/系统/版本）
# ═══════════════════════════════════════════════════════════
_IDENTITY_POOL = [
    # ── Chrome 131-125 Windows ──
    {"id": "chrome-131-win", "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
     "sc": '"Chromium";v="131", "Google Chrome";v="131", "Not?A_Brand";v="24"', "sm": "?0", "sp": '"Windows"'},
    {"id": "chrome-130-win", "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
     "sc": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"', "sm": "?0", "sp": '"Windows"'},
    {"id": "chrome-129-win", "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
     "sc": '"Chromium";v="129", "Google Chrome";v="129", "Not?A_Brand";v="8"', "sm": "?0", "sp": '"Windows"'},
    {"id": "chrome-128-win", "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
     "sc": '"Chromium";v="128", "Google Chrome";v="128", "Not?A_Brand";v="24"', "sm": "?0", "sp": '"Windows"'},
    {"id": "chrome-127-win", "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
     "sc": '"Chromium";v="127", "Google Chrome";v="127", "Not?A_Brand";v="99"', "sm": "?0", "sp": '"Windows"'},
    {"id": "chrome-126-win", "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
     "sc": '"Chromium";v="126", "Google Chrome";v="126", "Not?A_Brand";v="99"', "sm": "?0", "sp": '"Windows"'},
    {"id": "chrome-125-win", "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
     "sc": '"Chromium";v="125", "Google Chrome";v="125", "Not?A_Brand";v="99"', "sm": "?0", "sp": '"Windows"'},
    # ── Chrome macOS ──
    {"id": "chrome-131-mac", "ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
     "sc": '"Chromium";v="131", "Google Chrome";v="131", "Not?A_Brand";v="24"', "sm": "?0", "sp": '"macOS"'},
    {"id": "chrome-129-mac", "ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
     "sc": '"Chromium";v="129", "Google Chrome";v="129", "Not?A_Brand";v="8"', "sm": "?0", "sp": '"macOS"'},
    # ── Edge Windows ──
    {"id": "edge-131-win", "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
     "sc": '"Chromium";v="131", "Microsoft Edge";v="131", "Not?A_Brand";v="24"', "sm": "?0", "sp": '"Windows"'},
    {"id": "edge-129-win", "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0",
     "sc": '"Chromium";v="129", "Microsoft Edge";v="129", "Not?A_Brand";v="8"', "sm": "?0", "sp": '"Windows"'},
    {"id": "edge-127-win", "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0",
     "sc": '"Chromium";v="127", "Microsoft Edge";v="127", "Not?A_Brand";v="99"', "sm": "?0", "sp": '"Windows"'},
    # ── Firefox Windows ──
    {"id": "ff-136-win", "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0", "sc": "", "sm": "", "sp": ""},
    {"id": "ff-135-win", "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0", "sc": "", "sm": "", "sp": ""},
    {"id": "ff-133-win", "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0", "sc": "", "sm": "", "sp": ""},
    {"id": "ff-132-win", "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0", "sc": "", "sm": "", "sp": ""},
    # ── Firefox macOS ──
    {"id": "ff-135-mac", "ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:135.0) Gecko/20100101 Firefox/135.0", "sc": "", "sm": "", "sp": ""},
    # ── Safari macOS ──
    {"id": "safari-18_3", "ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15",
     "sc": "", "sm": "", "sp": ""},
    {"id": "safari-18_2", "ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
     "sc": "", "sm": "", "sp": ""},
    {"id": "safari-17_6", "ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15",
     "sc": "", "sm": "", "sp": ""},
    # ── Chrome Android ──
    {"id": "chrome-131-android", "ua": "Mozilla/5.0 (Linux; Android 14; Pixel 8 Pro Build/UQ1A.240105.004) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.135 Mobile Safari/537.36",
     "sc": '"Chromium";v="131", "Google Chrome";v="131", "Not?A_Brand";v="24"', "sm": "?1", "sp": '"Android"'},
    {"id": "chrome-130-android", "ua": "Mozilla/5.0 (Linux; Android 14; SM-S908B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.6723.86 Mobile Safari/537.36",
     "sc": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"', "sm": "?1", "sp": '"Android"'},
    # ── Chrome Linux ──
    {"id": "chrome-131-linux", "ua": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
     "sc": '"Chromium";v="131", "Google Chrome";v="131", "Not?A_Brand";v="24"', "sm": "?0", "sp": '"Linux"'},
    {"id": "chrome-128-linux", "ua": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
     "sc": '"Chromium";v="128", "Google Chrome";v="128", "Not?A_Brand";v="24"', "sm": "?0", "sp": '"Linux"'},
]

_rotation_index = 0


def get_identity():
    """按轮转顺序获取一个浏览器身份"""
    global _rotation_index
    ident = _IDENTITY_POOL[_rotation_index % len(_IDENTITY_POOL)]
    _rotation_index += 1
    return dict(ident)


def get_random_identity():
    """随机获取一个浏览器身份"""
    return dict(random.choice(_IDENTITY_POOL))


# ═══════════════════════════════════════════════════════════
#  增强 HTTP 请求头构建
# ═══════════════════════════════════════════════════════════

def build_headers(identity=None, referer=None):
    """基于浏览器身份构建完整 HTTP 请求头（包含 sec-ch-ua 指纹头）"""
    if identity is None:
        identity = get_identity()

    headers = {
        "User-Agent": identity["ua"],
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.7"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9,zh-TW;q=0.8,en;q=0.7,ja;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
        "DNT": "1",
    }

    # Chrome/Edge 专有：sec-ch-ua 指纹头
    sc = identity.get("sc", "")
    if sc:
        headers["sec-ch-ua"] = sc
        headers["sec-ch-ua-mobile"] = identity.get("sm", "?0")
        headers["sec-ch-ua-platform"] = identity.get("sp", '"Windows"')

    if referer:
        headers["Referer"] = referer

    return headers


# ═══════════════════════════════════════════════════════════
#  增强 HTTP 请求（带身份轮换 + 指数退避）
# ═══════════════════════════════════════════════════════════

_ssl_ctx = ssl.create_default_context()


def enhanced_fetch(url, retries=5, referer=None):
    """HTTP GET → (status, data_bytes)

    相比原始 _fetch() 的增强：
    - 每次重试切换浏览器身份（UA + sec-ch-ua 指纹头）
    - 指数退避（1s → 2s → 4s → 8s → 16s）
    - 反爬状态码（403/429/503）额外等待
    - 默认 5 次重试（原 2 次）
    """
    last_err = None
    for attempt in range(retries + 1):
        # 每次请求/重试使用不同身份
        identity = get_random_identity() if attempt > 0 else get_identity()
        headers = build_headers(identity=identity, referer=referer)
        req = urllib.request.Request(url)
        for key, value in headers.items():
            req.add_header(key, value)

        try:
            with urllib.request.urlopen(req, timeout=15, context=_ssl_ctx) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (403, 429, 503, 502, 418):
                # 反爬/限流状态码：等待更久 + 切换身份重试
                wait = min(5 * (2 ** attempt), 30)
                time.sleep(wait + random.uniform(0, 3))
                continue
            elif e.code >= 500:
                time.sleep(min(2 ** attempt, 10))
                continue
            return e.code, b""
        except Exception as e:
            last_err = e
            # 指数退避：1s, 2s, 4s, 8s, 16s
            backoff = min(1 * (2 ** attempt), 30)
            time.sleep(backoff + random.uniform(0, 2))

    if last_err:
        raise last_err
    raise RuntimeError(f"Failed to fetch: {url}")


# ═══════════════════════════════════════════════════════════
#  增强 Playwright 浏览器（stealth.js + 反检测参数 + 资源拦截）
# ═══════════════════════════════════════════════════════════

from playwright.sync_api import sync_playwright

_pw_playwright = None
_pw_browser = None
_pw_context = None
_pw_page_count = 0
_PW_MAX_PAGES = 50  # 每 50 个页面后重启浏览器，防止内存泄漏

# 资源类型拦截：屏蔽图片/媒体/字体，提速 3-5 倍
_BLOCKED_RESOURCE_TYPES = {"image", "media", "font"}


def _block_route(route):
    try:
        if route.request.resource_type in _BLOCKED_RESOURCE_TYPES:
            route.abort()
        else:
            route.continue_()
    except Exception:
        # 浏览器已断开（EPIPE 等），尝试 abort 兜底
        try:
            route.abort()
        except Exception:
            pass


def enhanced_pw_init():
    """启动反检测 Chromium + 共享 context（首次调用自动 init）

    相比原始 _pw_init() 的增强：
    - stealth.min.js 全量反检测注入
    - 更多浏览器启动参数（隐藏自动化特征）
    - 资源路由拦截（屏蔽图片/媒体/字体，显著提速）
    - 固定 viewport 尺寸
    - 自动隐藏 webdriver 属性
    - 浏览器崩溃时自动重连
    - 每 50 个页面后自动重启浏览器，防止内存泄漏
    """
    global _pw_playwright, _pw_browser, _pw_context, _pw_page_count
    if _pw_browser is not None:
        # 定期重启浏览器，防止内存泄漏导致 EPIPE
        if _pw_page_count >= _PW_MAX_PAGES:
            close_pw()
        else:
            return

    # 先清理旧连接
    close_pw()

    _pw_playwright = sync_playwright().start()
    _pw_browser = _pw_playwright.chromium.launch(
        headless=True,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,site-per-process",
            "--disable-web-security",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-infobars",
            "--disable-dev-shm-usage",
            "--disable-browser-side-navigation",
            "--disable-gpu",
            "--disable-extensions",
            "--disable-background-networking",
            "--disable-sync",
            "--disable-default-apps",
            "--hide-scrollbars",
            "--metrics-recording-only",
            "--mute-audio",
            "--no-first-run",
            "--safebrowsing-disable-auto-update",
        ],
    )
    _pw_context = _pw_browser.new_context(
        user_agent=get_random_identity()["ua"],
        viewport={"width": 1920, "height": 1080},
        locale="zh-CN",
    )

    # 注入 stealth.min.js（防检测核心）
    if _HAS_STEALTH:
        _pw_context.add_init_script(path=_STEALTH_PATH)

    # 资源路由拦截：提速 3-5 倍
    _pw_context.route("**/*", _block_route)


# ═══════════════════════════════════════════════════════════
#  Cloudflare 挑战页面检测
# ═══════════════════════════════════════════════════════════

_CF_CHALLENGE_TEXTS = [
    "Just a moment",
    "Checking your browser",
    "Please wait",
    "Please enable cookies",
    "DDoS protection",
    "cf-browser-verification",
    "cf-challenge-running",
    "cf-please-wait",
    "Attention Required!",
    "请稍候",
    "正在检查您的浏览器",
    "请启用 Cookie",
    "One more step",
    "Cloudflare",
]


def _is_cf_challenge(html):
    """检测页面是否为 Cloudflare 挑战页"""
    if not html or len(html) < 100:
        return True
    for text in _CF_CHALLENGE_TEXTS:
        if text in html:
            return True
    return False


def enhanced_fetch_page(url, timeout=30, cf_retries=3):
    """Playwright 获取 HTML 页面的文本内容 → (status, text)

    相比原始 _fetch_page() 的增强：
    - 使用 domcontentloaded（非 networkidle），避免广告挂起
    - stealth.min.js 已注入，降低触发 CF 挑战的概率
    - 资源拦截已启用，页面加载速度 3-5 倍提升
    - 更精细的 CF 挑战等待：检测 #challenge-running 等 DOM 元素
    - 每次重试前重新加载页面
    - 浏览器崩溃时自动标记为需重连
    """
    global _pw_browser, _pw_context, _pw_page_count
    page = None
    try:
        enhanced_pw_init()
        page = _pw_context.new_page()
        _pw_page_count += 1

        for attempt in range(cf_retries + 1):
            try:
                # domcontentloaded 比 networkidle 快很多，且不会因广告挂起
                page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
            except Exception:
                pass

            # 等待 JS 执行（CF 挑战、XHR 加载等）
            time.sleep(2 + attempt)

            try:
                page.wait_for_load_state("load", timeout=10000)
            except Exception:
                pass

            # 额外等待（让 CF 挑战完成、让 AJAX 加载完）
            time.sleep(1)

            try:
                html = page.content()
            except Exception:
                # 浏览器断开，标记失效
                _pw_browser = None
                _pw_context = None
                return 0, ""

            if not _is_cf_challenge(html):
                return 200, html

            # 仍被 CF 拦截，等待后重试
            if attempt < cf_retries:
                wait = 3 * (attempt + 1)
                time.sleep(wait)
                try:
                    page.reload(timeout=timeout * 1000, wait_until="domcontentloaded")
                except Exception:
                    pass
                time.sleep(2)

        return (403 if html else 0), html
    except Exception:
        # 浏览器可能崩溃，标记失效以便下次重连
        _pw_browser = None
        _pw_context = None
        return 0, ""
    finally:
        if page:
            try:
                page.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════
#  增强 Playwright 页面获取（带滚动加载 + 选择器等待）
# ═══════════════════════════════════════════════════════════

def enhanced_fetch_page_scroll(url, timeout=30, cf_retries=3,
                                scroll_count=0, scroll_delay=0.8,
                                wait_selector=None, min_length=2000):
    """Playwright 获取 HTML + 滚动触发懒加载 + 可选等待选择器 → (status, text)

    相比 enhanced_fetch_page() 的增强：
    - scroll_count: 向下滚动的次数（触发无限滚动 / 懒加载）
    - scroll_delay: 每次滚动后的等待间隔
    - wait_selector: CSS 选择器，等待该元素出现后再返回（超时 8s 不报错）
    - min_length: 最小内容长度，低于此值视为无效页面
    - 浏览器崩溃时自动标记为需重连
    """
    global _pw_browser, _pw_context, _pw_page_count
    page = None
    try:
        enhanced_pw_init()
        page = _pw_context.new_page()
        _pw_page_count += 1

        for attempt in range(cf_retries + 1):
            try:
                page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
            except Exception:
                pass

            time.sleep(2 + attempt)

            try:
                page.wait_for_load_state("load", timeout=10000)
            except Exception:
                pass

            # ── 滚动加载 ──
            if scroll_count > 0:
                for s in range(scroll_count):
                    try:
                        page.evaluate("window.scrollBy(0, window.innerHeight)")
                        time.sleep(scroll_delay)
                    except Exception:
                        break
                # 滚完后等 JS 渲染
                time.sleep(1)

            # ── 等待选择器 ──
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=8000)
                except Exception:
                    pass

            try:
                html = page.content()
            except Exception:
                # 浏览器断开，标记失效
                _pw_browser = None
                _pw_context = None
                return 0, ""

            if not _is_cf_challenge(html) and len(html) >= min_length:
                return 200, html

            if attempt < cf_retries:
                time.sleep(3 * (attempt + 1))
                try:
                    page.reload(timeout=timeout * 1000, wait_until="domcontentloaded")
                except Exception:
                    pass
                time.sleep(2)

        return (403 if html and len(html) > 0 else 0), html
    except Exception:
        # 浏览器可能崩溃，标记失效以便下次重连
        _pw_browser = None
        _pw_context = None
        return 0, ""
    finally:
        if page:
            try:
                page.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════
#  关闭 Playwright
# ═══════════════════════════════════════════════════════════

def close_pw():
    """关闭 Playwright 浏览器（程序退出时调用）"""
    global _pw_playwright, _pw_browser, _pw_context, _pw_page_count
    try:
        if _pw_context:
            _pw_context.close()
    except Exception:
        pass
    try:
        if _pw_browser:
            # 使用 force 确保即使管道断开也能关闭
            _pw_browser.close()
    except Exception:
        pass
    try:
        if _pw_playwright:
            _pw_playwright.stop()
    except Exception:
        pass
    finally:
        _pw_context = None
        _pw_browser = None
        _pw_playwright = None
        _pw_page_count = 0
