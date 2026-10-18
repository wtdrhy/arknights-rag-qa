# -*- coding: utf-8 -*-
"""
字段缺失诊断：自动区分「数据本来就这样」和「解析器漏了」
=========================================================
运行：python scripts/05_diagnose.py

为什么要有这个脚本？
    只看「完整度 95.2%」这种数字，无法判断剩下 4.8% 是什么问题。
    它可能意味着：
      A. 游戏里本来就没有这个数据（1 星干员没技能）→ 不用管
      B. wiki 编辑者没填（源数据缺失）            → 不用管
      C. 我们的解析器没覆盖到某个模板变体         → 必须修

    人工一个个看太慢，而且每次改解析器都要重看一遍。
    这个脚本把判断过程自动化，三层判定：

      第 1 层  领域规则（profile.EXEMPTION_RULES）
              → 命中就是「数据固有特性」，豁免
      第 2 层  原始溯源（回 data/raw 查 wikitext）
              页面里根本没有对应模板 → 「源数据未填写」，豁免
      第 3 层  页面里有模板但没解析出来 → 「解析器待修」，这才是 bug

    只有第 3 层需要人（或我）介入。
"""

import io
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser.profile import (  # noqa: E402
    EXEMPTION_RULES,
    FIELD_PARAMS,
    FIELD_TEMPLATES,
    KNOWN_TEMPLATES,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed", "operators.jsonl")
RAW = os.path.join(ROOT, "data", "raw", "pages.jsonl")
OUT = os.path.join(ROOT, "data", "processed", "anomalies.json")

CHECK_FIELDS = ["星级", "职业", "分支", "特性", "面板",
                "精英化材料", "技能升级材料", "模组", "天赋", "技能",
                "档案", "势力", "档案文本",
                "攻击范围", "潜能", "后勤技能", "相关道具", "悖论模拟", "干员密录"]

# 提取模板名。注意用 (?=[|\n}]) 前瞻而不是直接匹配分隔符 ——
# 因为 wiki 里大量写成 {{模板名\n|参数=值}}（模板名后是换行不是竖线），
# 直接匹配 [|}] 会漏掉这一类。这是实际踩过的坑。
RE_TPL_NAME = re.compile(r"\{\{\s*([^{\}|\n]{1,40}?)\s*(?=[|\n}])")


def load_jsonl(path):
    with io.open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def template_names(content):
    """提取一个页面里出现过的所有模板名（去重）"""
    return set(n.strip() for n in RE_TPL_NAME.findall(content or ""))


def main():
    if not os.path.exists(PROCESSED):
        print("找不到解析结果，请先运行：python scripts/04_parse_operators.py")
        return

    records = load_jsonl(PROCESSED)
    print(f"读取解析结果：{len(records)} 个干员\n")

    # ---------- 第 1 层：套用领域豁免规则 ----------
    exempt = defaultdict(list)      # field -> [(名称, 理由)]
    pending = defaultdict(list)     # field -> [record]  需要进一步溯源的

    for field in CHECK_FIELDS:
        rules = [r for r in EXEMPTION_RULES if r["field"] == field]
        for r in records:
            if r.get(field):        # 字段有值，没问题
                continue
            hit = None
            for rule in rules:
                try:
                    if rule["when"](r):
                        hit = rule["reason"]
                        break
                except Exception:
                    continue
            if hit:
                exempt[field].append((r.get("名称", "?"), hit, r.get("星级"),
                                      r.get("干员类别")))
            else:
                pending[field].append(r)

    # ---------- 第 2 层：对未豁免项做原始溯源 ----------
    need_titles = set()
    for field, rs in pending.items():
        for r in rs:
            need_titles.add(r["名称"])

    raw_map = {}
    if need_titles and os.path.exists(RAW):
        with io.open(RAW, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("title") in need_titles:
                    raw_map[d["title"]] = d.get("content") or ""
        print(f"已加载 {len(raw_map)}/{len(need_titles)} 个待查页面的原始 wikitext\n")

    source_missing = defaultdict(list)   # wiki 没写
    parser_bug = defaultdict(list)       # 我们的锅
    unknown_tpl = Counter()              # 疑似未覆盖的新模板

    for field, rs in pending.items():
        wanted = set(FIELD_TEMPLATES.get(field, []))
        for r in rs:
            name = r["名称"]
            content = raw_map.get(name)
            if content is None:
                parser_bug[field].append((name, "原始数据里找不到该页"))
                continue
            names = template_names(content)
            if names & wanted:
                # 页面里明明有这个模板，却没解析出来 → 先怀疑解析器。
                # 但还要再确认一层：**模板有 ≠ 参数有值**。
                # 模板往往带一堆固定占位参数（实测：所属国家 出现于 463 页，
                # 只有 411 页真的有值），值为空属于 wiki 没填，不能赖解析器。
                keys = FIELD_PARAMS.get(field)
                if keys:
                    has_value = False
                    for k in keys:
                        m = re.search(r"\|\s*" + re.escape(k) + r"\s*=\s*([^\|\}\n]*)", content)
                        if m and m.group(1).strip():
                            has_value = True
                            break
                    if not has_value:
                        source_missing[field].append(
                            (name, r.get("星级"), r.get("干员类别")))
                        continue
                parser_bug[field].append(
                    (name, "页面含模板 " + "/".join(sorted(names & wanted))))
            else:
                source_missing[field].append((name, r.get("星级"), r.get("干员类别")))
                # 顺便收集该页面里我们不认识的模板名，用于发现新变体
                for n in names:
                    if n not in KNOWN_TEMPLATES and n.isprintable():
                        unknown_tpl[n] += 1

    # ---------- 输出报告 ----------
    print("=" * 66)
    print("诊断结论")
    print("=" * 66)

    total_exempt = sum(len(v) for v in exempt.values())
    total_src = sum(len(v) for v in source_missing.values())
    total_bug = sum(len(v) for v in parser_bug.values())

    print("\n【A】数据固有特性（游戏机制使然，不需要修）")
    print("-" * 66)
    if total_exempt:
        by_reason = defaultdict(list)
        for field, items in exempt.items():
            for name, reason, star, cat in items:
                by_reason[(field, reason)].append(name)
        for (field, reason), names in sorted(by_reason.items(),
                                             key=lambda x: -len(x[1])):
            print(f"  {field:<8} {len(names):>3} 个  <- {reason}")
            print(f"           例：{names[:6]}")
    else:
        print("  无")

    print("\n【B】源数据未填写（wiki 页面里本来就没有该模板）")
    print("-" * 66)
    if total_src:
        for field, items in source_missing.items():
            if not items:
                continue
            stars = Counter(i[1] for i in items)
            print(f"  {field:<8} {len(items):>3} 个   星级分布 {dict(sorted(stars.items()))}")
            print(f"           例：{[i[0] for i in items[:6]]}")
    else:
        print("  无")

    print("\n【C】解析器待修（页面里有数据但没解析出来 —— 这是真 bug）")
    print("-" * 66)
    if total_bug:
        for field, items in parser_bug.items():
            if not items:
                continue
            print(f"  {field:<8} {len(items):>3} 个")
            for name, why in items[:10]:
                print(f"           - {name}  （{why}）")
            if len(items) > 10:
                print(f"           ... 还有 {len(items) - 10} 个")
    else:
        print("  无 —— 解析器已覆盖所有存在的模板")

    print("\n【D】疑似未识别的新模板变体（出现次数 >= 3 才列出）")
    print("-" * 66)
    cand = [(n, c) for n, c in unknown_tpl.most_common() if c >= 3]
    if cand:
        for n, c in cand[:20]:
            print(f"  {n:<32} 出现于 {c} 个页面")
        print("  提示：若某个名字看着像上面缺失字段的同义模板，就要补进解析器")
    else:
        print("  无")

    print("\n" + "=" * 66)
    print(f"汇总：数据固有 {total_exempt}  |  源数据缺失 {total_src}  |  待修 {total_bug}")
    print("=" * 66)

    # ---------- 落盘，方便逐条查看细节 ----------
    payload = {
        "固有豁免": {f: [{"名称": a, "理由": b} for a, b, c, d in v]
                     for f, v in exempt.items()},
        "源数据缺失": {f: [{"名称": a, "星级": b, "类别": c} for a, b, c in v]
                       for f, v in source_missing.items()},
        "解析器待修": {f: [{"名称": a, "原因": b} for a, b in v]
                       for f, v in parser_bug.items()},
        "疑似新模板": unknown_tpl.most_common(50),
    }
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n明细已写入：{OUT}")


if __name__ == "__main__":
    main()
