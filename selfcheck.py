#!/usr/bin/env python3
"""video-generator — 无头自检入口。

只 import 不碰 OpenGL / ffmpeg / 音频设备的纯逻辑模块：
effect_model、effect_loader、formula_eval、noise_field、random_composer。
不 import video_generator / gl_renderer / pw_renderer / draw_renderer（会拉
numba JIT、GL 上下文或 ffmpeg 子进程）。

验证点：
  1. 效果模型 + 注册表：draw/pw 两种 JSON 字段映射、类别与标签推导、
     目录加载与坏文件跳过、注册表按来源/类别过滤与随机取样。
  2. 公式求值（asteval）：环带掩码与直接 numpy 求值逐元素一致、
     链式比较展开、向量化 if、三角函数颜色表达式。
  3. 噪声场：Perlin/Simplex/Worley 值域 [0,1]、同种子可复现、换种子相异。
  4. 随机组合：同种子组合完全可复现、换种子相异，且图层/参数/时长约束成立。

用法：python selfcheck.py   （成功退出 0，失败打印 FAIL 并退出 1）
"""

import contextlib
import io
import json
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)  # 保证导入的是本目录模块

_FAILS = []
_REAL_REGISTRY = None  # 真实效果注册表（9685 个 JSON，加载一次复用）


def check(name, fn):
    try:
        detail = fn()
        if detail is False:
            raise AssertionError("检查返回 False")
        print(f"  [ok] {name}" + (f" — {detail}" if isinstance(detail, str) else ""))
    except Exception as exc:  # noqa: BLE001 - 自检需要兜住一切失败
        _FAILS.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")


def quiet(fn, *args, **kwargs):
    """吞掉被调模块的 print（loader 会打印加载统计），保持自检输出干净。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════
#  1. 效果模型 + 注册表
# ═══════════════════════════════════════════════════════════════
_DRAW_JSON = {
    "name": "BZReaction_Super_Threshold",
    "grid": {"width": 32, "height": 48},
    "evolution": {"rule": "bz-oregonator"},
    "color": {"red": "sin(2*pi*v)", "green": "1-v", "blue": "v"},
    "postprocess": {"effect": "vignette"},
}
_PW_JSON = {
    "id": "pw-rain",
    "name": "粒子雨",
    "category": "粒子",
    "params": {
        "layers": [{"type": "particle"}, {"type": "wave"}],
        "audio_modulation": {"band": "low"},
    },
    "tags": ["rain", "dark"],
}


def _write_effect_fixtures(tmpdir):
    draw_dir = os.path.join(tmpdir, "draw")
    pw_dir = os.path.join(tmpdir, "pw")
    os.makedirs(draw_dir)
    os.makedirs(pw_dir)

    with open(os.path.join(draw_dir, "bz.json"), "w", encoding="utf-8") as f:
        json.dump(_DRAW_JSON, f)
    with open(os.path.join(draw_dir, "waver.json"), "w", encoding="utf-8") as f:
        json.dump({"name": "WaveEquation_Core", "grid": {"width": 16, "height": 16}}, f)
    # 坏 JSON：必须被跳过而不是让加载整体失败
    with open(os.path.join(draw_dir, "broken.json"), "w", encoding="utf-8") as f:
        f.write("{ this is not json ")
    # 顶层是 list 的文件（如 themes.json）：pw 目录里必须跳过
    with open(os.path.join(pw_dir, "themes.json"), "w", encoding="utf-8") as f:
        json.dump([{"id": "t1"}], f)
    with open(os.path.join(pw_dir, "1_rain.json"), "w", encoding="utf-8") as f:
        json.dump(_PW_JSON, f)
    return draw_dir, pw_dir


def check_effect_registry():
    global _REAL_REGISTRY

    from effect_loader import EffectRegistry, get_default_registry, load_draw_effects
    from effect_model import EffectConfig

    # ── draw JSON -> EffectConfig 字段映射 ──
    cfg = EffectConfig.from_draw_json("effects/draw/bz.json", _DRAW_JSON)
    if (cfg.draw_grid_w, cfg.draw_grid_h) != (32, 48):
        raise AssertionError(f"grid 映射错误：{cfg.draw_grid_w}x{cfg.draw_grid_h}")
    if cfg.draw_rule != "bz-oregonator":
        raise AssertionError(f"rule 映射错误：{cfg.draw_rule}")
    if cfg.draw_postprocess != "vignette":
        raise AssertionError(f"postprocess 映射错误：{cfg.draw_postprocess}")
    if (cfg.draw_color_red, cfg.draw_color_green, cfg.draw_color_blue) != ("sin(2*pi*v)", "1-v", "v"):
        raise AssertionError("颜色表达式映射错误")
    if cfg.category != "反应扩散":
        raise AssertionError(f"类别推导错误：{cfg.category}")
    if cfg.tags != ["super", "threshold"]:
        raise AssertionError(f"标签推导错误：{cfg.tags}")
    if cfg._guess_palette() != "Rainbow":
        raise AssertionError(f"调色板推导错误：{cfg._guess_palette()}")
    summary = cfg.summary()
    if "[DRAW]" not in summary or "32x48" not in summary:
        raise AssertionError(f"draw 摘要错误：{summary}")

    # ── pw JSON -> EffectConfig 字段映射 ──
    pw = EffectConfig.from_pw_json("effects/pw/1_rain.json", _PW_JSON)
    if pw.effect_id != "pw-rain" or pw.category != "粒子":
        raise AssertionError("pw id/category 映射错误")
    if len(pw.pw_layers) != 2 or pw.pw_layers[0]["type"] != "particle":
        raise AssertionError(f"pw layers 映射错误：{pw.pw_layers}")
    if pw.pw_has_audio is not True:
        raise AssertionError("audio_modulation 应置 pw_has_audio=True")
    if "[PW]" not in pw.summary() or "particle" not in pw.summary():
        raise AssertionError(f"pw 摘要错误：{pw.summary()}")

    # ── 目录加载：坏文件跳过、list 文件跳过 ──
    tmpdir = tempfile.mkdtemp(prefix="vg_selfcheck_")
    try:
        draw_dir, pw_dir = _write_effect_fixtures(tmpdir)
        draw_effects = quiet(load_draw_effects, draw_dir)
        if len(draw_effects) != 2:
            raise AssertionError(f"draw 目录应加载 2 个（跳过坏 JSON），实际 {len(draw_effects)}")

        reg = EffectRegistry()
        quiet(reg.load_all, draw_dir, pw_dir)
        if (len(reg.all()), len(reg.draw_effects), len(reg.pw_effects)) != (3, 2, 1):
            raise AssertionError(
                f"注册表计数错误：all={len(reg.all())} draw={len(reg.draw_effects)} pw={len(reg.pw_effects)}"
            )
        if len(reg.by_source("draw")) != 2 or len(reg.by_source("pw")) != 1:
            raise AssertionError("by_source 过滤错误")
        if [e.category for e in reg.by_category("反应扩散")] != ["反应扩散"]:
            raise AssertionError("by_category 过滤错误")
        if [e.category for e in reg.by_categories(["粒子"])] != ["粒子"]:
            raise AssertionError("by_categories 过滤错误")
        picked = reg.random_one(source="pw")
        if picked is None or picked.effect_id != "pw-rain":
            raise AssertionError("random_one(source='pw') 未选中唯一 pw 效果")
        if reg.random_one(source="draw") is None:
            raise AssertionError("random_one(source='draw') 返回 None")
        many = reg.random_many(2, source="draw")
        if len(many) != 2 or len({id(e) for e in many}) != 2:
            raise AssertionError("random_many 应返回 2 个互不相同的 draw 效果")
        if len(reg.random_many(99)) != 3:
            raise AssertionError("random_many 超过总数时应截断到 3")
    finally:
        for name in os.listdir(tmpdir):
            p = os.path.join(tmpdir, name)
            if os.path.isdir(p):
                for sub in os.listdir(p):
                    os.remove(os.path.join(p, sub))
                os.rmdir(p)
            else:
                os.remove(p)
        os.rmdir(tmpdir)

    # ── 真实 effects/ 目录（9685 个 JSON）必须能整体加载 ──
    _REAL_REGISTRY = quiet(get_default_registry)
    total = len(_REAL_REGISTRY.all())
    if total < 1000:
        raise AssertionError(f"真实效果注册表加载过少：{total}")
    if total != len(_REAL_REGISTRY.draw_effects) + len(_REAL_REGISTRY.pw_effects):
        raise AssertionError("all() 计数与两来源之和不一致")
    for e in _REAL_REGISTRY.all()[:200]:
        if not e.name or not e.effect_id:
            raise AssertionError(f"真实效果存在空 name/effect_id：{e!r}")

    return (
        f"draw/pw 字段映射与类别标签推导正确；坏 JSON 与 list 文件被跳过；"
        f"真实 effects/ 加载 {total} 个（draw {len(_REAL_REGISTRY.draw_effects)} + "
        f"pw {len(_REAL_REGISTRY.pw_effects)}）"
    )


# ═══════════════════════════════════════════════════════════════
#  2. 公式求值（asteval 受限解释器）
# ═══════════════════════════════════════════════════════════════
def check_formula_eval():
    import numpy as np

    import formula_eval as fe

    if not fe.is_available():
        raise RuntimeError("asteval 不可用（项目声明依赖缺失）")

    # 环带掩码：asteval 求值结果必须与直接 numpy 求值逐元素相等
    xs, ys = np.meshgrid(np.linspace(0.0, 1.0, 11), np.linspace(0.0, 1.0, 11))
    expr = "1 if (0.2 < (x-0.5)**2 + (y-0.5)**2 < 0.25) else 0"
    got = fe.eval_grid_expr(expr, xs, ys)
    r2 = (xs - 0.5) ** 2 + (ys - 0.5) ** 2
    expected = np.where((r2 > 0.2) & (r2 < 0.25), 1.0, 0.0)
    if got.shape != xs.shape:
        raise AssertionError(f"求值结果形状错误：{got.shape} != {xs.shape}")
    if not np.array_equal(got, expected):
        raise AssertionError(f"环带掩码与 numpy 参考解不一致（差异 {int((got != expected).sum())} 格）")
    if expected.sum() == 0:
        raise AssertionError("测试网格未覆盖环带，检查用例失效")

    # 链式比较必须逐元素展开（否则 asteval 返回标量 NaN）
    ring_center = fe.eval_grid_expr("1 if 0.2 < (x-0.5)**2 + (y-0.5)**2 < 0.25 else 0", xs, ys)
    if not np.array_equal(ring_center, expected):
        raise AssertionError("链式比较未被逐元素展开")

    # 颜色表达式：0.5+0.5*cos(2*pi*v) 在 v=0/0.5/1 处应为 1/0/1（约）
    v = np.array([[0.0, 0.25], [0.5, 1.0]])
    color = fe.eval_color_expr("0.5+0.5*cos(2*pi*v)", v)
    if not (abs(color[0][0] - 1.0) < 1e-9 and abs(color[1][0] - 0.0) < 1e-9
            and abs(color[1][1] - 1.0) < 1e-9):
        raise AssertionError(f"cos 颜色表达式求值错误：{color.tolist()}")
    if not np.all((color >= -1e-9) & (color <= 1.0 + 1e-9)):
        raise AssertionError(f"颜色表达式越界：{color.tolist()}")

    # 常用数学函数 + clip + 逐元素运算
    radial = fe.eval_grid_expr("sqrt((x-0.5)**2+(y-0.5)**2)", xs, ys)
    if not np.allclose(radial, np.sqrt(r2)):
        raise AssertionError("sqrt 表达式与 numpy 参考解不一致")
    clipped = fe.eval_grid_expr("clip(x*2, 0, 0.5)", xs, ys)
    if not np.allclose(clipped, np.clip(xs * 2, 0, 0.5)):
        raise AssertionError("clip 表达式与 numpy 参考解不一致")

    # 非法公式必须抛错或返回非有限值——绝不能伪装成一张合法网格。
    # （实测契约：asteval 把错误记进解释器而不抛，本函数返回 nan；此处锁住该降级
    #   行为不被静默改成 0/finite，调用方 draw_renderer 才有机会发现公式写错。
    #   只测未定义符号：语法错误那条 asteval 会直接写真实 stderr，污染自检输出。）
    try:
        result = fe.eval_grid_expr("nonexistent_name + 1", xs, ys)
    except Exception:
        pass
    else:
        if np.all(np.isfinite(result)):
            raise AssertionError(f"非法公式返回了有限网格，错误被静默吞掉：{result!r}")

    return "环带掩码/链式比较/向量化 if/sqrt/clip 与 numpy 参考解逐元素一致，非法公式不产出有限网格"


# ═══════════════════════════════════════════════════════════════
#  3. 噪声场
# ═══════════════════════════════════════════════════════════════
def check_noise_field():
    import numpy as np

    import noise_field as nf

    # Perlin：值域 [0,1]、非常量、同种子逐位复现、换种子相异
    a = nf.PerlinNoise(seed=42).render_field(32, 32, t=1.5, scale=8.0)
    b = nf.PerlinNoise(seed=42).render_field(32, 32, t=1.5, scale=8.0)
    c = nf.PerlinNoise(seed=43).render_field(32, 32, t=1.5, scale=8.0)
    if not np.array_equal(a, b):
        raise AssertionError("Perlin 同种子结果不可复现")
    if np.array_equal(a, c):
        raise AssertionError("Perlin 不同种子产生了相同场")
    if not (np.all(a >= 0.0) and np.all(a <= 1.0)):
        raise AssertionError(f"Perlin 值域越界：[{a.min()}, {a.max()}]")
    if float(a.std()) < 1e-6:
        raise AssertionError("Perlin 场是常量，噪声实现可能失效")
    if not (np.all(np.isfinite(a))):
        raise AssertionError("Perlin 场含 NaN/Inf")

    # 时间维度必须真的动起来（同一 seed、不同 t 不应完全相同）
    a_t2 = nf.PerlinNoise(seed=42).render_field(32, 32, t=2.5, scale=8.0)
    if np.array_equal(a, a_t2):
        raise AssertionError("Perlin 场不随 t 变化（时间维度失效）")

    # fbm 与 Simplex / Worley 的契约：值域 [0,1] 且非常量
    xs, ys = np.meshgrid(np.linspace(0, 8, 24), np.linspace(0, 8, 24))
    fbm = nf.PerlinNoise(seed=7).fbm(xs, ys, octaves=4)
    if not (np.all(fbm >= 0.0) and np.all(fbm <= 1.0)) or float(fbm.std()) < 1e-6:
        raise AssertionError(f"fbm 值域/方差异常：[{fbm.min()}, {fbm.max()}] std={fbm.std()}")

    for cls_name, gen in (
        ("SimplexNoise", lambda: nf.SimplexNoise(seed=3).render_field(24, 24, t=0.5)),
        ("WorleyNoise", lambda: nf.WorleyNoise(seed=3).render_field(24, 24, t=0.5)),
    ):
        field = gen()
        if field.shape != (24, 24):
            raise AssertionError(f"{cls_name} 形状错误：{field.shape}")
        if not (np.all(field >= 0.0) and np.all(field <= 1.0)):
            raise AssertionError(f"{cls_name} 值域越界：[{field.min()}, {field.max()}]")

    return "Perlin/Simplex/Worley/fbm 值域 [0,1]、Perlin 同种子复现且随 t 演进、换种子相异"


# ═══════════════════════════════════════════════════════════════
#  4. 随机组合（compose）
# ═══════════════════════════════════════════════════════════════
def check_random_composer():
    from random_composer import DEFAULT_DURATION, RandomComposer, VideoComposition

    if _REAL_REGISTRY is None:
        raise RuntimeError("真实效果注册表未加载（第 1 项检查失败导致）")
    rc = RandomComposer(_REAL_REGISTRY)

    comp_a = rc.compose(seed=7)
    summary_a = comp_a.summary()
    comp_b = rc.compose(seed=7)
    if comp_a.summary() != comp_b.summary() or comp_a.description != comp_b.description:
        raise AssertionError("同种子两次 compose 结果不一致（不可复现）")

    comp_c = rc.compose(seed=8)
    if comp_c.summary() == summary_a:
        raise AssertionError("不同种子产生了完全相同的组合")

    # 结构约束：尺寸/帧率/时长/主题都在允许集合内
    if comp_a.width not in (512, 768, 1024) or comp_a.height not in (512, 768, 1024):
        raise AssertionError(f"尺寸越界：{comp_a.width}x{comp_a.height}")
    if comp_a.fps not in (24, 30, 60):
        raise AssertionError(f"帧率越界：{comp_a.fps}")
    if comp_a.duration not in (5, 10, 15, 30):
        raise AssertionError(f"时长越界：{comp_a.duration}")
    if comp_a.theme not in RandomComposer.THEMES:
        raise AssertionError(f"主题越界：{comp_a.theme}")
    if comp_a.global_postprocess not in RandomComposer.POSTPROCESSES:
        raise AssertionError(f"后处理越界：{comp_a.global_postprocess}")

    # 图层约束：1~4 层，效果确实来自注册表，时间窗不超出视频时长
    if not (1 <= len(comp_a.layers) <= 4):
        raise AssertionError(f"图层数越界：{len(comp_a.layers)}")
    registry_ids = {id(e) for e in _REAL_REGISTRY.all()}
    for layer in comp_a.layers:
        if id(layer.effect) not in registry_ids:
            raise AssertionError(f"图层效果不在注册表内：{layer.effect.name}")
        if layer.blend_mode not in RandomComposer.BLEND_MODES:
            raise AssertionError(f"混合模式越界：{layer.blend_mode}")
        if not (0.0 <= layer.opacity <= 1.0):
            raise AssertionError(f"不透明度越界：{layer.opacity}")
        if layer.start_time < 0.0:
            raise AssertionError(f"起始时间为负：{layer.start_time}")
        end = comp_a.duration if layer.duration is None else layer.start_time + layer.duration
        if end > comp_a.duration + 1e-9:
            raise AssertionError(f"图层超出视频时长：{layer.start_time}+{layer.duration} > {comp_a.duration}")
    if not comp_a.description:
        raise AssertionError("组合缺少自动生成的描述")

    # 空注册表必须优雅返回空组合（不能抛异常）
    from effect_loader import EffectRegistry
    empty_comp = RandomComposer(EffectRegistry()).compose(seed=1)
    if not isinstance(empty_comp, VideoComposition) or empty_comp.layers:
        raise AssertionError("空注册表应返回无图层的组合")

    if DEFAULT_DURATION <= 0:
        raise AssertionError("DEFAULT_DURATION 异常")

    return (
        f"同种子可复现、换种子相异；{len(comp_a.layers)} 图层全部来自注册表、"
        f"尺寸/帧率/时长/时间窗约束成立；空注册表优雅降级"
    )


def main():
    print("video-generator 无头自检（不启动 GL / ffmpeg / 音频设备）")
    check("效果模型与注册表", check_effect_registry)
    check("公式求值（asteval）", check_formula_eval)
    check("噪声场", check_noise_field)
    check("随机视频组合", check_random_composer)

    if _FAILS:
        print(f"FAIL: {len(_FAILS)} 项检查未通过")
        for f in _FAILS:
            print(f"  - {f}")
        return 1

    total = len(_REAL_REGISTRY.all()) if _REAL_REGISTRY is not None else 0
    print(
        "selfcheck ok: effect JSON 映射与注册表（真实 effects/ "
        f"{total} 个）+ asteval 公式求值与 numpy 参考解逐元素一致 + "
        "Perlin/Simplex/Worley 噪声场 + 随机组合可复现与结构约束"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
