"""
formula_eval.py — 用 asteval 安全求值 draw JSON 里的公式

自包含、可选依赖：asteval 不在时整体降级为原有逻辑。

为什么用 asteval 而不是 eval()？
  - 联网调研 (asteval PyPI/GitHub)：asteval 基于 Python ast 模块，
    对用户输入做"受限解释器"，禁止 open/import/getattr 等危险操作，
    比裸 eval() 安全得多。
  - 强调数学表达式，自动导入 numpy 多数函数（sin/cos/sqrt…）。
  - 支持 if 表达式、条件、循环等，能求值 draw 的初始化公式
    （如 "1 if (0.2 < (x-0.5)**2 + (y-0.5)**2 < 0.25) else 0"）。

提供的函数：
  - is_available()
  - eval_grid_expr(expr, x, y) -> (H,W) float
  - eval_color_expr(expr, v)   -> (H,W) float   （v ∈ [0,1]）
"""

from typing import Optional

import ast
import re

import numpy as np

try:
    from asteval import Interpreter
    _HAS_ASTEVAL = True
except Exception:  # noqa: BLE001
    Interpreter = None
    _HAS_ASTEVAL = False


def is_available() -> bool:
    """asteval 是否可用"""
    return _HAS_ASTEVAL


# ── 向量化 if/else 转换 ──────────────────────────────────────────────
# asteval 不会把 `A if 数组条件 else B` 向量化（返回标量 NaN），
# 因此把顶层 `A if C else B` 改写成 `where(C, A, B)`，由 numpy 逐元素求值。
_IF_RE = re.compile(
    r"^\s*(?P<a>.+?)\s+if\s+(?P<c>.+?)\s+else\s+(?P<b>.+?)\s*$",
    re.DOTALL,
)


def _vectorize_if(expr: str) -> str:
    """把顶层 `A if C else B` 转成 `where(C, A, B)`（仅一层，覆盖绝大多数 draw 公式）"""
    m = _IF_RE.match(expr)
    if m:
        return f"where(({m.group('c')}), ({m.group('a')}), ({m.group('b')}))"
    return expr


def _expand_chained_comparisons(expr: str) -> str:
    """把链式比较（asteval 不会逐元素求值）展开为显式 `&` 组合。

    asteval 对 `a < x < b` 这类链式比较返回标量 NaN，而非逐元素布尔数组，
    导致 `1 if (0.2 < X < 0.25) else 0` 之类公式退化成标量。
    用 ast 把 `a OP1 b OP2 c` 重写为 `(a OP1 b) & (b OP2 c) & ...`，
    `&` 在 numpy 下是逐元素的（已验证 `(0.2 < X) & (X < 0.25)` 返回正确网格）。
    任何异常都安全降级为原字符串（保持既有行为）。
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return expr

    class _ChainedExpander(ast.NodeTransformer):
        def visit_Compare(self, node):  # noqa: N802
            self.generic_visit(node)
            if len(node.ops) <= 1:
                return node
            parts = []
            prev = node.left
            for op, comp in zip(node.ops, node.comparators):
                parts.append(
                    ast.Compare(left=prev, ops=[op], comparators=[comp])
                )
                prev = comp
            comb = parts[0]
            for p in parts[1:]:
                comb = ast.BinOp(left=comb, op=ast.BitAnd(), right=p)
            return ast.copy_location(comb, node)

    try:
        new_tree = _ChainedExpander().visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)
    except Exception:  # noqa: BLE001
        return expr


def _preprocess(expr: str) -> str:
    """组合预处理：先展开链式比较，再做向量化 if 转换。"""
    return _vectorize_if(_expand_chained_comparisons(expr))


def _new_interp(v: Optional[np.ndarray] = None,
                   x: Optional[np.ndarray] = None,
                   y: Optional[np.ndarray] = None) -> "Interpreter":
    """构造一个带常用符号表的解释器"""
    ae = Interpreter()
    syms = {
        "pi": np.pi,
        "v": v if v is not None else 0.0,
        "x": x if x is not None else 0.0,
        "y": y if y is not None else 0.0,
        # draw 颜色表达式里常见的相位/偏移常数（JSON 常未给，给默认 0）
        "p1": 1.0, "p2": 2.0, "p3": 3.0,
        "d1": 0.0, "d2": 0.0, "d3": 0.0,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "sqrt": np.sqrt, "exp": np.exp, "abs": np.abs,
        "log": np.log, "pow": np.power,
        "clip": lambda a, lo, hi: np.clip(a, lo, hi),
        "where": np.where,
        "np": np,
    }
    ae.symtable.update(syms)
    return ae


def eval_grid_expr(expr: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """在 (x,y) 网格上求值公式，返回与 x 同形的 float 数组

    x, y: (H,W) 数组（一般已归一化到 [0,1]）。
    公式可含 if 表达式（经 _vectorize_if 改写为向量化的 where）；
    链式比较（如 `0.2 < X < 0.25`）经 _expand_chained_comparisons 展开为
    显式 `&` 组合，确保 asteval 逐元素求值，而非退化成标量 NaN。
    """
    if not _HAS_ASTEVAL:
        raise RuntimeError("asteval 不可用")
    ae = _new_interp(x=x, y=y)
    ae("__r = " + _preprocess(expr))
    return np.asarray(ae.symtable.get("__r"), dtype=np.float64)


def eval_color_expr(expr: str, v: np.ndarray) -> np.ndarray:
    """在 v 网格上求值颜色通道表达式，返回 (H,W) float（约 [0,1]）

    与 eval_grid_expr 同样经 _preprocess（链式比较展开 + 向量化 if）。
    """
    if not _HAS_ASTEVAL:
        raise RuntimeError("asteval 不可用")
    ae = _new_interp(v=v)
    ae("__r = " + _preprocess(expr))
    return np.asarray(ae.symtable.get("__r"), dtype=np.float64)


if __name__ == "__main__":
    if not is_available():
        print("[formula_eval] asteval 不可用，降级路径（本自测需安装 asteval）")
    else:
        ys, xs = np.mgrid[0:5, 0:5] / 4.0
        r = eval_grid_expr(
            "1 if (0.2 < (x-0.5)**2 + (y-0.5)**2 < 0.25) else 0", xs, ys)
        print("grid expr ->\n", r.astype(int))
        v = np.linspace(0, 1, 9).reshape(3, 3)
        c = eval_color_expr("0.5+0.5*cos(2*pi*(v*p1+d1))", v)
        print("color expr range:", round(float(c.min()), 3), round(float(c.max()), 3))
        print("OK")
