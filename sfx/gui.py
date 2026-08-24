"""音效下载器 - Tkinter 图形界面"""

import os
import sys
import time
import random
import signal
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox

# 导入核心模块（同目录）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from dl_sounds import (
    SOURCES as ALL_SOURCES,
    SOUNDS_DIR, NUM_SOUNDS,
    _download_one, _fetch, _safe_filename, _save_file,
)
import dl_sounds
import sfx_utils


class SfxDownloaderGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("免费音效下载器")
        self.root.geometry("680x580")
        self.root.resizable(True, True)
        self._running = False
        self._stop_flag = False

        # 窗口关闭 / Ctrl+C 时优雅退出
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        signal.signal(signal.SIGINT, self._on_sigint)

        # 数据源勾选状态
        self._src_vars = {}
        self._build_ui()

    def _on_close(self):
        """窗口关闭时清理 Playwright 并退出"""
        self._stop_flag = True
        self._running = False
        try:
            sfx_utils.close_pw()
        except Exception:
            pass
        self.root.destroy()

    def _on_sigint(self, signum, frame):
        """Ctrl+C 时优雅退出"""
        self._on_close()

    def _build_ui(self):
        root = self.root

        # ── 顶部：标题 + 路径 ──
        top_frame = ttk.Frame(root, padding=8)
        top_frame.pack(fill=tk.X)
        ttk.Label(top_frame, text="免费音效下载器", font=("", 16, "bold")).pack(side=tk.LEFT)
        ttk.Label(top_frame, text=f" 保存至: {SOUNDS_DIR}", foreground="gray").pack(side=tk.RIGHT)

        # ── 数量选择 ──
        ctrl_frame = ttk.Frame(root, padding=(8, 4))
        ctrl_frame.pack(fill=tk.X)

        ttk.Label(ctrl_frame, text="下载数量:").pack(side=tk.LEFT)
        self._num_var = tk.IntVar(value=NUM_SOUNDS)
        num_spin = ttk.Spinbox(
            ctrl_frame, from_=1, to=200, width=6,
            textvariable=self._num_var,
        )
        num_spin.pack(side=tk.LEFT, padx=4)

        ttk.Separator(ctrl_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Label(ctrl_frame, text="保存路径:").pack(side=tk.LEFT)
        self._path_var = tk.StringVar(value=SOUNDS_DIR)
        path_entry = ttk.Entry(ctrl_frame, textvariable=self._path_var, width=40)
        path_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl_frame, text="浏览...", command=self._browse_dir, width=7).pack(side=tk.LEFT)

        # ── 数据源选择（多列） ──
        src_frame = ttk.LabelFrame(root, text="数据源 (全选/取消)", padding=6)
        src_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # 点击标题栏切换全选/取消
        src_frame.bind("<Button-1>", lambda e: self._toggle_all())

        inner = ttk.Frame(src_frame)
        inner.pack(fill=tk.BOTH, expand=True)

        cols = 3
        for i, (name, _) in enumerate(ALL_SOURCES):
            r, c = divmod(i, cols)
            var = tk.BooleanVar(value=True)
            self._src_vars[name] = var
            cb = ttk.Checkbutton(inner, text=name, variable=var)
            cb.grid(row=r, column=c, sticky=tk.W, padx=8, pady=2, ipadx=4)

        # ── 按钮 ──
        btn_frame = ttk.Frame(root, padding=8)
        btn_frame.pack(fill=tk.X)

        self._start_btn = ttk.Button(btn_frame, text="▶ 开始下载", command=self._start, width=14)
        self._start_btn.pack(side=tk.LEFT, padx=4)

        self._stop_btn = ttk.Button(btn_frame, text="⏹ 停止", command=self._stop, width=10, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT, padx=4)

        self._open_btn = ttk.Button(btn_frame, text="📂 打开文件夹", command=self._open_folder, width=12)
        self._open_btn.pack(side=tk.RIGHT, padx=4)

        # ── 日志区域 ──
        log_frame = ttk.LabelFrame(root, text="日志", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self._log = scrolledtext.ScrolledText(log_frame, height=12, state=tk.DISABLED,
                                               font=("Consolas", 9), wrap=tk.WORD)
        self._log.pack(fill=tk.BOTH, expand=True)
        self._log.tag_config("ok", foreground="green")
        self._log.tag_config("err", foreground="red")
        self._log.tag_config("warn", foreground="orange")

    def _toggle_all(self):
        all_on = all(v.get() for v in self._src_vars.values())
        for v in self._src_vars.values():
            v.set(not all_on)

    def _browse_dir(self):
        d = filedialog.askdirectory(initialdir=self._path_var.get(), mustexist=False)
        if d:
            self._path_var.set(d)

    def _open_folder(self):
        p = self._path_var.get()
        if os.path.isdir(p):
            os.startfile(p) if sys.platform == "win32" else None

    def _log_msg(self, msg, tag=None):
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, f"{msg}\n", tag or "")
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)

    def _get_active_sources(self):
        """返回用户勾选的数据源列表"""
        active = []
        for name, fn in ALL_SOURCES:
            if self._src_vars.get(name, tk.BooleanVar()).get():
                active.append((name, fn))
        return active

    def _start(self):
        active = self._get_active_sources()
        if not active:
            messagebox.showwarning("提示", "请至少选择一个数据源")
            return

        target = max(1, self._num_var.get())

        # 确保输出目录存在
        out_dir = self._path_var.get()
        os.makedirs(out_dir, exist_ok=True)

        self._running = True
        self._stop_flag = False
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)

        # 清空旧日志
        self._log.config(state=tk.NORMAL)
        self._log.delete("1.0", tk.END)
        self._log.config(state=tk.DISABLED)

        self._log_msg(f"目标: {target} 个 | 目录: {out_dir}")
        self._log_msg(f"数据源: {', '.join(n for n, _ in active)}\n")

        t = threading.Thread(target=self._run_download, args=(active, target, out_dir), daemon=True)
        t.start()

    def _stop(self):
        self._stop_flag = True
        self._log_msg("\n⏹ 正在停止...", "warn")

    def _run_download(self, sources, target, out_dir):
        ok = 0
        fail = 0

        # 临时替换全局 SOURCES 和 SOUNDS_DIR
        old_sources = dl_sounds.SOURCES
        old_dir = dl_sounds.SOUNDS_DIR
        dl_sounds.SOURCES = sources
        dl_sounds.SOUNDS_DIR = out_dir

        try:
            for i in range(target):
                if self._stop_flag:
                    break

                # _download_one 内部自行选择数据源 + 重试（最多 MAX_SCRAPE_RETRIES 次）
                success = _download_one(i)

                if success:
                    ok += 1
                    self.root.after(0, lambda n=i+1: self._log_msg(f"  [{n}] ✓ 成功", "ok"))
                else:
                    fail += 1
                    self.root.after(0, lambda n=i+1: self._log_msg(f"  [{n}] ✗ 失败", "err"))

                time.sleep(0.3)
        finally:
            dl_sounds.SOURCES = old_sources
            dl_sounds.SOUNDS_DIR = old_dir

        total = ok + fail
        self._log_msg(f"\n{'='*40}")
        self._log_msg(f"完成! 成功 {ok}/{total} 个, 失败 {fail} 个",
                      "ok" if fail == 0 else "warn")

        # 恢复按钮状态
        self.root.after(0, self._on_done)

    def _on_done(self):
        self._running = False
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)

    def run(self):
        self.root.mainloop()


def main():
    app = SfxDownloaderGUI()
    app.run()


if __name__ == "__main__":
    main()
