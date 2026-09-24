"""fix_syntax.py — 修复 video_generator.py 中的 f-string 语法错误（用法：python fix_syntax.py [目标文件]）"""
# 2026-09-23 去本地化：目标文件改由 命令行参数 → 本脚本同级目录 推导
# （原路径 所在的 原路径已不存在）
import sys
from pathlib import Path

TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 else (Path(__file__).resolve().parent / "video_generator.py")
if not TARGET.is_file():
    raise SystemExit(f"[错误] 找不到目标文件：{TARGET}\n  用法：python fix_syntax.py [video_generator.py 路径]")

with open(TARGET, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到所有包含 print(f" 的行，检查是否有未终止的 f-string
# 问题表现：某行以 print(f" 结尾（后面是换行），下一行是字符串内容
fixed = 0
i = 0
while i < len(lines):
    line = lines[i]
    # 检查这一行是否以 print(f" 结尾（即 f-string 没有正确关闭）
    stripped = line.rstrip('\n')
    if stripped.endswith('f"') or stripped.endswith('f"'):
        # 这一行以 f" 结尾，说明 f-string 没有关闭
        # 下一行应该是 f-string 的内容
        if i + 1 < len(lines):
            # 合并两行
            next_line = lines[i+1]
            # 去掉下一行的换行，合并
            merged = stripped + next_line
            lines[i] = merged
            del lines[i+1]
            fixed += 1
            # 不增加 i，继续检查这一行
            continue
    i += 1

if fixed > 0:
    with open(TARGET, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f'OK: 修复了 {fixed} 处 f-string 语法错误')
else:
    print('未找到需要修复的 f-string 错误')
