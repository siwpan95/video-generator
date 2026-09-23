import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import shutil
import tempfile
import threading

from playsound3 import playsound

# 尝试导入 pydub（用于转换不支持的格式）
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

CONFIG_FILE = os.path.expanduser("~/.musicplayer_config.json")
# 2026-09-23 去本地化：音效目录改由 环境变量 SOUNDS_DIR → 本脚本同级 sounds/ 推导
# （原 D:\allin\t0l\sfx\sounds 已不存在）
SOUNDS_DIR = os.environ.get("SOUNDS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
TEMP_DIR = os.path.join(tempfile.gettempdir(), "sound_preview_converted")
AUDIO_EXTENSIONS = (
    '.mp3', '.mp2', '.wav', '.wave', '.ogg', '.oga', '.flac',
    '.mid', '.midi', '.mod', '.xm', '.it', '.s3m',
    '.669', '.med', '.mtm', '.okt', '.stm', '.ult', '.far', '.amf',
    '.aiff', '.aif', '.aifc', '.au', '.snd', '.voc',
    '.m4a', '.aac', '.mp4', '.m4b', '.wma', '.opus', '.spx'
)


def load_config():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return {"theme": "dark"}


def save_config(config):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False)
    except:
        pass


def convert_to_playable(file_path):
    """用 pydub 将音频转成标准 PCM WAV 临时文件"""
    if not PYDUB_AVAILABLE:
        return None
    os.makedirs(TEMP_DIR, exist_ok=True)
    import hashlib
    h = hashlib.md5(file_path.encode()).hexdigest()[:8]
    safe_name = os.path.splitext(os.path.basename(file_path))[0]
    temp_path = os.path.join(TEMP_DIR, f"{h}_{safe_name}.wav")
    if os.path.exists(temp_path):
        return temp_path
    try:
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(44100).set_channels(2).set_sample_width(2)
        audio.export(temp_path, format="wav")
        return temp_path
    except Exception:
        return None


def cleanup_temp():
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception:
            pass


def gui_mode():
    root = tk.Tk()
    root.title("🔊 音效试听器")
    root.geometry("520x420")
    root.resizable(False, False)

    # 配置
    config = load_config()
    current_theme = config.get("theme", "dark")

    # 播放器状态
    is_playing: bool = False
    playlist: list[str] = []
    current_index: int = -1
    loop_mode: str = "none"
    converted_map: dict[str, str] = {}

    # 播放线程控制
    _active: bool = False
    _thread: threading.Thread | None = None

    # 主题
    themes = {
        "dark": {"bg": "#2b2b2b", "fg": "white", "select": "#5a5a5a", "button": "#4a4a4a"},
        "light": {"bg": "#f0f0f0", "fg": "black", "select": "#c0c0c0", "button": "#e0e0e0"}
    }

    def get_color(key):
        return themes[current_theme][key]

    def apply_theme():
        color = get_color("bg")
        root.configure(bg=color)
        for widget in root.winfo_children():
            try:
                widget.configure(bg=color)
            except:
                pass

    def toggle_theme():
        nonlocal current_theme
        current_theme = "light" if current_theme == "dark" else "dark"
        apply_theme()
        save_config({"theme": current_theme})
        theme_btn.config(text="🌙" if current_theme == "dark" else "☀️")
        playlist_listbox.configure(
            bg=get_color("bg").replace("#2b2b2b", "#3a3a3a").replace("#f0f0f0", "#ffffff"),
            fg=get_color("fg"),
            selectbackground=get_color("select"),
            selectforeground=get_color("fg")
        )
        for widget in main_frame.winfo_children():
            try:
                widget.configure(bg=get_color("bg"))
            except:
                pass

    # ---- 文件扫描与列表 ----

    def scan_sounds_dir():
        audio_files = []
        if os.path.exists(SOUNDS_DIR):
            for root_dir, dirs, files in os.walk(SOUNDS_DIR):
                for file in files:
                    if file.lower().endswith(AUDIO_EXTENSIONS) and not file.startswith('._'):
                        audio_files.append(os.path.join(root_dir, file))
        return sorted(audio_files, key=lambda f: os.path.basename(f).lower())

    def resolve_path(file_path):
        if file_path in converted_map:
            return converted_map[file_path]
        return file_path

    def can_play(file_path):
        """检测文件是否存在且可访问"""
        actual = file_path
        # pydub 转换
        if PYDUB_AVAILABLE:
            converted = convert_to_playable(file_path)
            if converted:
                converted_map[file_path] = converted
                actual = converted
        return os.path.isfile(actual)

    def rebuild_playlist(files):
        nonlocal playlist, current_index
        old_playing = None
        if current_index >= 0 and current_index < len(playlist):
            old_playing = playlist[current_index]
        playlist = []
        skipped = 0
        playlist_listbox.delete(0, tk.END)
        for f in files:
            if f not in playlist and can_play(f):
                playlist.append(f)
                playlist_listbox.insert(tk.END, os.path.basename(f))
            else:
                skipped += 1
        if not playlist:
            current_index = -1
            file_label.config(text="未找到可播放文件")
            play_btn.config(state="disabled")
            status_label.config(text=f"目录中无可播放音频（{skipped}个跳过）")
            return
        msg = f"共 {len(playlist)} 个"
        if skipped > 0:
            msg += f"（跳过 {skipped} 个）"
        status_label.config(text=msg)
        try:
            new_idx = playlist.index(old_playing)
            current_index = new_idx
        except (ValueError, TypeError):
            current_index = 0
        ready_current()

    def refresh_playlist():
        new_files = scan_sounds_dir()
        nonlocal playlist
        if set(new_files) == set(playlist):
            return
        rebuild_playlist(new_files)

    def auto_refresh():
        if root.winfo_exists():
            refresh_playlist()
            root.after(5000, auto_refresh)

    def manual_refresh():
        refresh_playlist()
        status_label.config(text=f"已刷新，共 {len(playlist)} 个")
        if current_index >= 0:
            file_label.config(text=os.path.basename(playlist[current_index]))

    # ---- 播放控制 ----

    def ready_current():
        """将当前索引的文件准备好显示"""
        if current_index < 0 or current_index >= len(playlist):
            return
        file_path = playlist[current_index]
        file_label.config(text=os.path.basename(file_path), foreground=get_color("fg"))
        playlist_listbox.selection_clear(0, tk.END)
        playlist_listbox.selection_set(current_index)
        playlist_listbox.see(current_index)
        play_btn.config(state="normal")
        update_mode_label()

    def _play_thread(path):
        """在子线程中调用 playsound"""
        nonlocal _active
        try:
            playsound(path)
        except Exception as e:
            root.after(0, lambda: status_label.config(text=f"播放失败: {e}"))
            root.after(0, lambda: set_ui_idle())
            return
        # 播放完毕后回调主线程
        if _active:
            root.after(0, _on_play_end)

    def _on_play_end():
        """播放自然结束后的处理"""
        nonlocal is_playing, _active
        _active = False
        if loop_mode == "one":
            # 单曲循环：重新播放
            start_playback()
        elif loop_mode == "all" or current_index < len(playlist) - 1:
            # 列表循环/顺序下一首
            play_next()
        else:
            # 播放完毕
            status_label.config(text="播放完成")
            is_playing = False
            play_btn.config(text="▶")

    def set_ui_playing():
        nonlocal is_playing
        is_playing = True
        play_btn.config(text="⏹")

    def set_ui_idle():
        nonlocal is_playing, _active
        is_playing = False
        _active = False
        play_btn.config(text="▶")

    def start_playback():
        """开始/切换播放当前文件"""
        nonlocal _active, _thread, is_playing
        if current_index < 0 or current_index >= len(playlist):
            status_label.config(text="没有可播放的文件")
            return
        file_path = playlist[current_index]
        actual = resolve_path(file_path)
        if not os.path.isfile(actual):
            status_label.config(text=f"文件不存在: {os.path.basename(file_path)}")
            return
        # 标记激活并启动播放线程
        _active = True
        _thread = threading.Thread(target=_play_thread, args=(actual,), daemon=True)
        _thread.start()
        set_ui_playing()
        status_label.config(text=f"▶ {os.path.basename(file_path)}")
        ready_current()

    def stop_playback():
        """停止播放（标记取消，线程自然结束后不触发回调）"""
        nonlocal _active, is_playing
        _active = False
        is_playing = False
        play_btn.config(text="▶")
        status_label.config(text="已停止")

    def play_music():
        """▶ / ⏹ 按钮：播放或停止"""
        if not playlist:
            status_label.config(text="没有可播放的文件")
            return
        if is_playing:
            stop_playback()
        else:
            start_playback()

    def play_next():
        nonlocal current_index
        if not playlist:
            return
        current_index = (current_index + 1) % len(playlist)
        if is_playing:
            start_playback()

    def play_prev():
        nonlocal current_index
        if not playlist:
            return
        current_index = (current_index - 1) % len(playlist)
        if is_playing:
            start_playback()

    def toggle_loop():
        nonlocal loop_mode
        modes = {"none": ("🔁 列表", "all"), "all": ("🔂 单曲", "one"), "one": ("➡️ 关闭", "none")}
        btn_text, loop_mode = modes[loop_mode]
        loop_btn.config(text=btn_text)
        update_mode_label()
        status_label.config(text=f"循环模式: {btn_text}")

    def update_mode_label():
        mode_texts = {"none": "➡️", "all": "🔁", "one": "🔂"}
        mode_label.config(text=f" {mode_texts[loop_mode]} ")

    # ---- 播放列表交互 ----

    def on_playlist_double_click(event):
        nonlocal current_index
        sel = playlist_listbox.curselection()
        if sel:
            current_index = sel[0]
            start_playback()

    def on_playlist_select(event):
        nonlocal current_index
        sel = playlist_listbox.curselection()
        if sel:
            current_index = sel[0]
            ready_current()

    def show_context_menu(event):
        nonlocal current_index
        sel = playlist_listbox.curselection()
        if sel:
            current_index = sel[0]
            playlist_listbox.selection_clear(0, tk.END)
            playlist_listbox.selection_set(current_index)
        context_menu.tk_popup(event.x_root, event.y_root)

    def play_selected():
        if current_index >= 0:
            start_playback()

    def remove_selected():
        nonlocal current_index
        sel = playlist_listbox.curselection()
        if sel:
            idx = sel[0]
            removed = playlist.pop(idx)
            playlist_listbox.delete(idx)
            converted_map.pop(removed, None)
            if current_index == idx:
                if playlist:
                    current_index = min(idx, len(playlist) - 1)
                    ready_current()
                else:
                    current_index = -1
                    file_label.config(text="未选择文件")
                    play_btn.config(state="disabled")
            elif current_index > idx:
                current_index -= 1

    def clear_playlist():
        nonlocal current_index, is_playing, playlist
        playlist = []
        current_index = -1
        is_playing = False
        _active = False
        playlist_listbox.delete(0, tk.END)
        file_label.config(text="未选择文件")
        play_btn.config(state="disabled")
        play_btn.config(text="▶")
        status_label.config(text="播放列表已清空")

    # ---- 快捷键 ----

    def show_shortcuts():
        shortcuts_win = tk.Toplevel(root)
        shortcuts_win.title("快捷键")
        shortcuts_win.geometry("280x240")
        shortcuts_win.resizable(False, False)
        shortcuts_win.transient(root)
        shortcuts_win.grab_set()

        frame = ttk.Frame(shortcuts_win, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        shortcuts = [
            ("空格键", "播放/停止"),
            ("→", "下一首"),
            ("←", "上一首"),
            ("L", "切换循环模式"),
            ("T", "切换主题"),
            ("双击", "播放选中文件"),
            ("右键", "操作菜单"),
        ]

        ttk.Label(frame, text="⌨️ 快捷键", font=("Arial", 14, "bold")).pack(pady=(0, 15))
        for key, desc in shortcuts:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=f"{key}", font=("Arial", 10, "bold"), width=12).pack(side=tk.LEFT)
            ttk.Label(row, text=desc).pack(side=tk.LEFT)

        ttk.Button(shortcuts_win, text="关闭", command=shortcuts_win.destroy).pack(pady=10)

    # ---- UI 构建 ----

    # 右键菜单
    context_menu = tk.Menu(root, tearoff=0)
    context_menu.add_command(label="▶ 播放", command=play_selected)
    context_menu.add_command(label="🗑️ 删除", command=remove_selected)
    context_menu.add_separator()
    context_menu.add_command(label="🗑️ 清空列表", command=clear_playlist)

    style = ttk.Style()
    style.theme_use('clam')

    # 主容器
    main_frame = ttk.Frame(root, padding="15")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 顶部工具栏
    toolbar = ttk.Frame(main_frame)
    toolbar.pack(fill=tk.X, pady=(0, 10))

    title_label = ttk.Label(toolbar, text="🔊 音效试听器", font=("Arial", 16, "bold"))
    title_label.pack(side=tk.LEFT)

    theme_btn = ttk.Button(toolbar, text="🌙" if current_theme == "dark" else "☀️",
                           command=toggle_theme, width=3)
    theme_btn.pack(side=tk.RIGHT, padx=2)

    shortcuts_btn = ttk.Button(toolbar, text="⌨️", command=show_shortcuts, width=3)
    shortcuts_btn.pack(side=tk.RIGHT, padx=2)

    # 当前文件
    file_frame = ttk.Frame(main_frame)
    file_frame.pack(fill=tk.X, pady=5)
    ttk.Label(file_frame, text="当前:").pack(side=tk.LEFT)
    file_label = ttk.Label(file_frame, text="未选择文件", foreground="#888888")
    file_label.pack(side=tk.LEFT, padx=10)

    # 播放列表
    list_frame = ttk.Frame(main_frame)
    list_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    scrollbar = ttk.Scrollbar(list_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    playlist_listbox = tk.Listbox(list_frame, bg="#3a3a3a", fg="white",
                                   selectbackground="#5a5a5a", selectforeground="white",
                                   font=("Arial", 9), height=8, yscrollcommand=scrollbar.set,
                                   activestyle="none")
    playlist_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=playlist_listbox.yview)
    playlist_listbox.bind('<<ListboxSelect>>', on_playlist_select)
    playlist_listbox.bind('<Double-Button-1>', on_playlist_double_click)
    playlist_listbox.bind('<Button-3>', show_context_menu)

    # 按钮行
    btn_row = ttk.Frame(main_frame)
    btn_row.pack(fill=tk.X, pady=5)

    ttk.Button(btn_row, text="🔄 刷新列表", command=manual_refresh).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_row, text="🗑️ 清空", command=clear_playlist).pack(side=tk.LEFT, padx=2)

    # 控制按钮
    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(pady=10)

    loop_btn = ttk.Button(btn_frame, text="🔁 列表", command=toggle_loop, width=8)
    loop_btn.pack(side=tk.LEFT, padx=5)

    prev_btn = ttk.Button(btn_frame, text="⏮", command=play_prev, width=4)
    prev_btn.pack(side=tk.LEFT, padx=3)

    play_btn = ttk.Button(btn_frame, text="▶", command=play_music, state="disabled", width=5)
    play_btn.pack(side=tk.LEFT, padx=2)

    next_btn = ttk.Button(btn_frame, text="⏭", command=play_next, width=4)
    next_btn.pack(side=tk.LEFT, padx=3)

    # 状态栏
    status_frame = ttk.Frame(main_frame)
    status_frame.pack(fill=tk.X, pady=5)

    status_label = ttk.Label(status_frame, text="正在扫描...", font=("Arial", 9))
    status_label.pack(side=tk.LEFT)

    mode_label = ttk.Label(status_frame, text=" ➡️ ", font=("Arial", 9))
    mode_label.pack(side=tk.RIGHT)

    # ---- 启动初始化 ----

    initial_files = scan_sounds_dir()
    if initial_files:
        rebuild_playlist(initial_files)
    else:
        status_label.config(text=f"目录无音频文件")

    root.after(5000, auto_refresh)

    # 快捷键
    root.bind("<space>", lambda e: play_music())
    root.bind("<Right>", lambda e: play_next())
    root.bind("<Left>", lambda e: play_prev())
    root.bind("l", lambda e: toggle_loop())
    root.bind("t", lambda e: toggle_theme())
    root.bind("s", lambda e: stop_playback())

    # 系统托盘
    try:
        import pystray
        from PIL import Image, ImageDraw

        def create_icon_image():
            img = Image.new('RGB', (64, 64), color='#2b2b2b')
            draw = ImageDraw.Draw(img)
            draw.ellipse((12, 12, 52, 52), fill='#4a9eff')
            draw.polygon([(26, 22), (26, 42), (42, 32)], fill='white')
            return img

        icon_image = create_icon_image()

        def on_tray_click(icon, event):
            root.after(0, root.deiconify)

        def on_tray_exit(icon, event):
            root.after(0, on_closing)

        tray_menu = pystray.Menu(
            pystray.MenuItem("显示", on_tray_click),
            pystray.MenuItem("退出", on_tray_exit)
        )

        tray_icon = pystray.Icon("sound_preview", icon_image, "音效试听器", tray_menu)

        def minimize_to_tray():
            root.withdraw()
            tray_icon.run_detached()

        def on_closing():
            _active = False
            stop_playback()
            cleanup_temp()
            try:
                tray_icon.stop()
            except:
                pass
            root.destroy()

        ttk.Button(toolbar, text="➖", command=minimize_to_tray, width=3).pack(side=tk.RIGHT, padx=2)

    except ImportError:
        def on_closing():
            _active = False
            stop_playback()
            cleanup_temp()
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    apply_theme()
    root.mainloop()


if __name__ == "__main__":
    gui_mode()
