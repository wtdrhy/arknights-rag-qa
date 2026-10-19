"""
09_probe_template.py —— 模板结构探针

============================================================
解决什么问题
============================================================
写解析器的第一步是「知道这个模板有哪些参数、值长什么样」。
以前靠打开单个页面肉眼看，问题是：
  1. 单页可能是特例，看完以为全部如此 → 以偏概全（本项目已被打脸两次）
  2. 原始页面动辄几万字符，全看一遍极耗成本

本脚本把「某个模板在全站所有页面里的参数结构」摊成一张表，
写解析器前跑一次，参数名、取值形态、是否为嵌套模板，一目了然。

============================================================
用法
============================================================
    python scripts/09_probe_template.py CharinfoV2
    python scripts/09_probe_template.py 敌人信息/common2 --type 敌人
    python scripts/09_probe_template.py 属性 --sample 泥岩      # 看某个页面里的原始值
    python scripts/09_probe_template.py CharinfoV2 --top 60
    python scripts/09_probe_template.py CharinfoV2 --grep 势力   # 只显示含关键词的参数

参数：
    --type  页面类型（用哪个标识模板筛选页面，默认「干员页面名」）
    --sample 页名    打印该页里此模板的完整原始片段
    --top N          最多显示 N 个参数（默认 40）
    --grep 关键词    只显示参数名含关键词的行
    --val 关键词     只显示「参数值」含关键词的行（用来找某个字段藏在哪个参数里）

============================================================
输出怎么看
============================================================
    页数   有多少个页面含这个参数 —— 接近总页数才是通用字段
    样例   取 2 个真实值。值里带 {{ }} 说明是嵌套模板，解析时要二次处理
    嵌套   该参数的值里含模板的比例
"""

import io
import json
import os
import re
import sys
from collections import Counter, defaultdict


def iter_templates(wikitext: str, name: str):
    """
    抓取 wikitext 里所有 name 模板的参数块。

    不用 mwparserfromhell 是为了让本脚本零依赖（勘探阶段随时能跑），
    代价是遇到深层嵌套会截断 —— 对「看参数名清单」这个目的足够用。
    """
    for m in re.finditer(r"\{\{\s*" + re.escape(name) + r"\s*[\|\n]", wikitext):
        start = m.end() - 1  # 指向 | 或 \n
        i = start
        # depth = 当前未闭合的模板层数。已吃掉 {{，故处于 1 层内。
        # 坑：初值若写成 2，则首个 }} 只把它降到 1 而不结束，
        #     扫描会一直往后吞，把后面的整个章节都当成参数 —— 曾污染出 814 个假参数。
        depth = 1
        while i < len(wikitext) and depth > 0:
            c = wikitext[i]
            if c == "{" and wikitext[i + 1:i + 2] == "{":
                depth += 1
                i += 2
                continue
            if c == "}" and wikitext[i + 1:i + 2] == "}":
                depth -= 1
                i += 2
                continue
            i += 1
        yield wikitext[start:i - 2]


def parse_params(block: str) -> dict:
    """
    把参数块拆成 {参数名: 原始值}。只按「顶层」的 | 切分。

    坑（本项目亲历）：模板名后面可能跟换行而不是 |，
        {{CharinfoV2\n|干员名=泥岩\n|...}}
    早期版本把「模板名后第一段」当成参数名处理，一旦是换行开头就整体错位，
    结果 463 个页面只解析出 1 页的参数。正确做法是：
    先按顶层 | 切成片段，第一段（模板名）直接丢弃，其余按 = 拆键值。
    """
    parts = []
    buf = []
    depth = 0
    i = 0
    while i < len(block):
        if block[i:i + 2] == "{{":
            depth += 1
            buf.append("{{")
            i += 2
            continue
        if block[i:i + 2] == "}}":
            depth -= 1
            buf.append("}}")
            i += 2
            continue
        if block[i] == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(block[i])
        i += 1
    parts.append("".join(buf))

    out = {}
    # parts[0] 是模板名（也可能是空串或换行），丢弃
    for idx, seg in enumerate(parts[1:], start=1):
        s = seg.strip()
        if not s:
            continue
        if "=" in s:
            k, _, v = s.partition("=")
            out[k.strip()] = v.strip()
        else:
            out[str(idx)] = s
    return out


def main():
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        print(__doc__)
        return

    tpl_name = args[0]
    page_type = "干员页面名"
    sample = None
    top_n = 40
    grep = None
    valgrep = None

    for i, a in enumerate(args):
        if a == "--type" and i + 1 < len(args):
            page_type = args[i + 1]
        elif a == "--sample" and i + 1 < len(args):
            sample = args[i + 1]
        elif a == "--top" and i + 1 < len(args):
            top_n = int(args[i + 1])
        elif a == "--grep" and i + 1 < len(args):
            grep = args[i + 1]
        elif a == "--val" and i + 1 < len(args):
            valgrep = args[i + 1]

    path = os.path.join("data", "raw", "pages.jsonl")
    if not os.path.exists(path):
        print(f"找不到 {path}，请先跑 01_fetch.py")
        return

    pages = {}
    hit_pages = {}
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            title = d.get("title", "")
            content = d.get("content") or ""
            if sample:
                pages[title] = content
            if page_type and page_type not in content:
                continue
            if tpl_name in content:
                hit_pages[title] = content

    if sample:
        if sample not in pages:
            print(f"页面不存在：{sample}")
            return
        c = pages[sample]
        i = c.find("{{" + tpl_name)
        if i < 0:
            print(f"{sample} 里没有模板 {tpl_name}")
            return
        print(f"=== {sample} 中 【{tpl_name}】 原始片段 ===")
        print(c[i:i + 1500])
        return

    if not hit_pages:
        print(f"没有任何（{page_type} 类型的）页面含模板 {tpl_name}")
        print("提示：用 --type 换一个页面标识模板，或确认模板名是否带命名空间前缀")
        return

    # 统计参数
    param_pages = Counter()
    param_total = Counter()
    param_examples = defaultdict(list)
    param_nested = Counter()
    param_valhit = Counter()
    param_nonempty = Counter()

    for title, content in hit_pages.items():
        seen = set()
        for block in iter_templates(content, tpl_name):
            for k, v in parse_params(block).items():
                if not k:
                    continue
                param_total[k] += 1
                seen.add(k)
                if v.strip():
                    param_nonempty[k] += 1
                if len(param_examples[k]) < 2 and v.strip():
                    param_examples[k].append(v.strip().replace("\n", " / ")[:46])
                if "{{" in v:
                    param_nested[k] += 1
                if valgrep and valgrep in v:
                    param_valhit[k] += 1
        for k in seen:
            param_pages[k] += 1

    total_pages = len(hit_pages)
    print(f"模板：{tpl_name}")
    print(f"命中页面：{total_pages} 个（筛选条件：页面含 {page_type}）")
    print()
    print("=" * 88)
    print("参数清单（按覆盖页数排序）")
    print("=" * 88)
    print(f"{'参数名':<26}{'页数':>6}{'非空':>7}{'嵌套':>6}   样例")
    print("-" * 88)

    shown = 0
    skipped = 0
    for k, p in sorted(param_pages.items(), key=lambda x: (-x[1], x[0])):
        if grep and grep not in k:
            continue
        if valgrep and param_valhit[k] == 0:
            skipped += 1
            continue
        if shown >= top_n:
            print(f"  ...（还有 {len(param_pages) - shown} 个参数，用 --top 调大）")
            break
        ex = " | ".join(param_examples.get(k, []))[:48]
        nested = f"{param_nested[k]}/{param_total[k]}" if param_nested[k] else "-"
        ne = param_nonempty[k]
        print(f"{k[:25]:<26}{p:>6}{ne:>7}{nested:>7}   {ex}")
        shown += 1

    print()
    print("怎么看：")
    print("  页数   = 参数出现的页面数（不代表有值！）")
    print("  非空   = 值不为空的页面数 —— 这才是字段的真实可用性，务必看这列")
    print("  两者差距大 → 该参数只是模板的固定占位，多数页面为空，别拿它当主键")
    print("  嵌套比例高 → 值是模板，需要二次解析或清洗")
    print("  想找某个字段藏在哪个参数里 → 加 --val 关键词")


if __name__ == "__main__":
    main()
