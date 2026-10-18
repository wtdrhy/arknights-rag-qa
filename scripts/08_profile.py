"""
08_profile.py —— 通用数据画像（Data Profiling）

============================================================
为什么要有这个脚本
============================================================
数据处理阶段最耗成本的事，是「人（或 LLM）反复打开原始数据看一眼长什么样」。
每看一次原文，就意味着把大段原始文本读进上下文 —— 既慢又贵，还容易以偏概全
（只看了 1 个样本就以为全部如此，本项目已经因此被打脸两次）。

正确做法：让代码把数据的「全貌」压缩成一张固定大小的报表，
人只看这张表。这就是数据科学里的 **data profiling**（数据画像）。

业界同类工具：
    pandas-profiling / ydata-profiling   自动生成 HTML 数据报告
    Great Expectations / pandera         数据契约与断言校验
    polars.describe()                    列级统计
本脚本是它们的轻量版：零第三方依赖（纯标准库），输出纯文本，
专门针对「嵌套 JSON」设计 —— 因为我们的解析结果是嵌套结构，
pandas 对嵌套字段基本无能为力。

============================================================
用法
============================================================
    python scripts/08_profile.py data/processed/operators.jsonl
    python scripts/08_profile.py data/processed/operators.jsonl --top 8
    python scripts/08_profile.py data/processed/operators.jsonl --deep 面板

参数：
    --top N     取值分布显示前 N 个（默认 5）
    --deep 字段 对某个嵌套字段逐层展开统计
    --rows N    只处理前 N 行（快速试跑）

============================================================
输出说明（怎么看这张表）
============================================================
    缺失率    = 空值 / 总行数。>0 不代表有问题，要结合领域规则判断
    唯一值    = 去重后的取值个数
    取值分布  只给低基数字段（唯一值 <= 30）显示，高基数字段只给示例
    常量字段  全表只有一个取值 —— 通常是解析失败的特征，值得警惕
    长尾告警  数值字段的 max / 中位数 > 50 —— 说明分布极度右偏，
              后续如果按固定阈值切块/分箱，会被那一个巨值带偏
"""

import io
import json
import os
import sys
from collections import Counter, defaultdict


# ============================================================
# 工具函数
# ============================================================

def is_empty(v) -> bool:
    """判断一个值是否算「空」。空列表/空字典/空字符串都算。"""
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    if isinstance(v, (list, tuple, dict)) and len(v) == 0:
        return True
    return False


def kind_of(v) -> str:
    """值的类型标签，比 type() 更适合数据分析。"""
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def flatten(record: dict, prefix: str = "") -> dict:
    """
    把嵌套结构摊平成「路径 -> 值」。

    {"面板": {"levels": {"精2_满级": {"生命": 3000}}}}
    ->
    {"面板.levels.精2_满级.生命": 3000}

    为什么要摊平：嵌套字段里藏着解析失败（比如 面板 存在但 面板.levels 是空的），
    只看顶层字段的缺失率完全看不出来。
    """
    out = {}
    for k, v in record.items():
        path = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            # 字典再往下一层；但列表里的字典不再展开（否则路径会爆炸）
            out.update(flatten(v, path))
        else:
            out[path] = v
    return out


def summarize(values: list) -> dict:
    """对一个字段的所有取值做统计摘要。"""
    n = len(values)
    non_empty = [v for v in values if not is_empty(v)]
    kinds = Counter(kind_of(v) for v in non_empty)

    info = {
        "总数": n,
        "空值": n - len(non_empty),
        "缺失率": (n - len(non_empty)) / n if n else 0.0,
        "类型": kinds.most_common(2),
        "类型混用": len(kinds) > 1,
    }

    # 列表字段：统计长度
    if kinds and list(kinds)[0] == "list":
        lens = [len(v) for v in non_empty]
        if lens:
            lens_sorted = sorted(lens)
            info["列表长度"] = {
                "min": lens_sorted[0],
                "中位": lens_sorted[len(lens_sorted) // 2],
                "max": lens_sorted[-1],
                "平均": round(sum(lens) / len(lens), 1),
            }
        # 列表里的字典，统计其键名覆盖情况
        keys = Counter()
        for v in non_empty:
            for item in v:
                if isinstance(item, dict):
                    keys.update(item.keys())
        if keys:
            info["元素键名"] = keys.most_common(8)
        return info

    # 数值字段
    nums = [v for v in non_empty if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(nums) == len(non_empty) and nums:
        nums_sorted = sorted(nums)
        info["数值"] = {
            "min": nums_sorted[0],
            "中位": nums_sorted[len(nums_sorted) // 2],
            "max": nums_sorted[-1],
            "均值": round(sum(nums) / len(nums), 1),
        }
        med = nums_sorted[len(nums_sorted) // 2] or 1
        info["长尾告警"] = nums_sorted[-1] / med > 50
        info["取值分布"] = Counter(nums).most_common(8)
        return info

    # 字符串 / 混合：按基数决定给分布还是给示例
    strs = [str(v) for v in non_empty]
    uniq = Counter(strs)
    info["唯一值"] = len(uniq)
    if 0 < len(uniq) <= 30:
        info["取值分布"] = uniq.most_common(10)
    else:
        info["示例"] = [s[:40] for s in strs[:3]]
    return info


# ============================================================
# 主流程
# ============================================================

def main():
    args = [a for a in sys.argv[1:]]
    if not args or args[0].startswith("-"):
        print(__doc__)
        return

    path = args[0]
    top_n = 5
    deep = None
    row_limit = None

    for i, a in enumerate(args):
        if a == "--top" and i + 1 < len(args):
            top_n = int(args[i + 1])
        elif a == "--deep" and i + 1 < len(args):
            deep = args[i + 1]
        elif a == "--rows" and i + 1 < len(args):
            row_limit = int(args[i + 1])

    if not os.path.exists(path):
        print(f"文件不存在：{path}")
        return

    rows = []
    with io.open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  第 {line_no} 行不是合法 JSON，已跳过：{e}")
            if row_limit and len(rows) >= row_limit:
                break

    if not rows:
        print("没有读到任何数据")
        return

    print(f"数据文件：{os.path.abspath(path)}")
    print(f"记录数：{len(rows)}")
    print()

    # ---------- 第一层：顶层字段画像 ----------
    print("=" * 74)
    print("字段画像（顶层）")
    print("=" * 74)
    print(f"{'字段':<16}{'类型':<8}{'缺失率':>8}{'唯一值':>8}   摘要")
    print("-" * 74)

    first_level = defaultdict(list)
    for r in rows:
        flat = flatten(r)
        for k, v in flat.items():
            first_level[k.split(".")[0]].append(v)

    profiles = {}
    for field in rows[0].keys():
        vals = [r.get(field) for r in rows]
        info = summarize(vals)
        profiles[field] = info

        kinds = info["类型"]
        kind = kinds[0][0] if kinds else "-"
        if info["类型混用"]:
            kind += "*"  # 类型不统一，往往是解析有分支

        miss = f"{info['缺失率'] * 100:.1f}%"
        uniq = info.get("唯一值", "-")

        # 摘要行：按类型挑最有信息量的那个
        if "取值分布" in info and info.get("唯一值", 99) <= 30:
            dist = ", ".join(f"{k}:{c}" for k, c in info["取值分布"][:top_n])
            summary = dist[:56]
        elif "数值" in info:
            nv = info["数值"]
            summary = f"min={nv['min']} 中位={nv['中位']} max={nv['max']}"
            if info.get("长尾告警"):
                summary += "  ⚠长尾"
        elif "列表长度" in info:
            ll = info["列表长度"]
            summary = f"长度 min={ll['min']} 中位={ll['中位']} max={ll['max']}"
        else:
            summary = " | ".join(info.get("示例", []))[:56]

        flag = "  ⚠类型混用" if info["类型混用"] else ""
        print(f"{field:<16}{kind:<9}{miss:>8}{str(uniq):>8}   {summary}{flag}")

    # ---------- 告警 ----------
    print()
    print("=" * 74)
    print("自动告警")
    print("=" * 74)
    warn = 0

    for field, info in profiles.items():
        # 1) 常量字段：全表一个值，通常意味着解析没生效
        if info.get("唯一值") == 1 and info["缺失率"] < 1:
            print(f"  ⚠ [{field}] 全表只有一个取值 —— 疑似解析未生效或该字段无区分度")
            warn += 1
        # 2) 高缺失
        if info["缺失率"] > 0.2:
            print(f"  ⚠ [{field}] 缺失率 {info['缺失率'] * 100:.1f}% —— 需结合领域规则判断是否可接受")
            warn += 1
        # 3) 类型混用
        if info["类型混用"]:
            print(f"  ⚠ [{field}] 类型不统一 {info['类型']} —— 数据存在多种形态，下游要分别处理")
            warn += 1
        # 4) 空列表：字段存在但里面没东西（完整度报告看不出来的那类 bug）
        if info.get("列表长度") and info["列表长度"]["max"] == 0:
            print(f"  ⚠ [{field}] 有值但列表恒为空 —— 典型「模板识别到了、参数没抽出来」")
            warn += 1

    if not warn:
        print("  无告警")
    else:
        print(f"\n  共 {warn} 条告警。注意：告警 ≠ bug，")
        print("  必须先用 src/parser/profile.py 的领域豁免规则过一遍再下结论。")

    # ---------- 可选：深挖某个嵌套字段 ----------
    if deep:
        print()
        print("=" * 74)
        print(f"深挖字段：{deep}")
        print("=" * 74)
        sub = defaultdict(list)
        for r in rows:
            v = r.get(deep)
            if isinstance(v, dict):
                for k, vv in flatten(v).items():
                    sub[k].append(vv)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        for k, vv in flatten(item).items():
                            sub[k].append(vv)
        if not sub:
            print(f"  字段 {deep} 不是 dict / list[dict]，无法深挖")
        else:
            print(f"{'子路径':<28}{'缺失率':>8}   摘要")
            print("-" * 74)
            for k in sorted(sub):
                info = summarize(sub[k])
                miss = f"{info['缺失率'] * 100:.1f}%"
                if "取值分布" in info and info.get("唯一值", 99) <= 30:
                    s = ", ".join(f"{a}:{b}" for a, b in info["取值分布"][:top_n])[:40]
                elif "数值" in info:
                    nv = info["数值"]
                    s = f"min={nv['min']} 中位={nv['中位']} max={nv['max']}"
                elif "列表长度" in info:
                    ll = info["列表长度"]
                    s = f"长度 {ll['min']}~{ll['max']}"
                else:
                    s = " | ".join(info.get("示例", []))[:40]
                print(f"{k:<30}{miss:>8}   {s}")

    print()
    print("提示：把上面这张表整体贴给协作者即可，不需要再贴原始数据。")


if __name__ == "__main__":
    main()
