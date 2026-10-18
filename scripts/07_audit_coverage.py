"""
覆盖率审计：自动发现「页面里有、但解析器没提取」的内容板块

为什么要有这个脚本？
------------------------------------------------------------------
第 4 步返工过三次，每次都是补一个"刚想起来"的板块：
档案 → 势力 → 攻击范围 → 潜能 → 语音 → 密录……

根源是：解析器是按"我知道有什么"写的，而不是按"数据里有什么"写的。
Wiki 是玩家共建的，模板体系横跨好几年、新旧混杂，靠记忆穷举必然漏。

这个脚本把流程反过来：
    不看解析器写了什么，只看数据里有什么，
    然后自动算出「哪些板块还没被覆盖」。

以后扩展新页面类型（关卡 / 敌人 / 道具）时，先跑这个脚本，
它能告诉你那类页面一共有多少板块、哪些已经有字段、哪些还没有。

用法：
    python scripts/07_audit_coverage.py            # 审计干员页
    python scripts/07_audit_coverage.py --type 关卡  # 审计关卡页
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser.profile import KNOWN_TEMPLATES  # noqa: E402

RAW = ROOT / "data" / "raw" / "pages.jsonl"

# 纯装饰性模板：只管排版，不含任何信息，审计时忽略
# 判定标准：次/页 >= 3 且不含语义内容
DECORATIVE = {
    "color", "术语", "*", "+", "++", "**", "±", "变动数值lite", "修正",
    "异常效果", "popup", "popup/range", "mdi", "fa", "ja", "道具图标",
    "ads/mobile", "ads/normal", "pathnav2", "spineId", "spine",
    "FULLPAGENAME", "BASEPAGENAME", "参阅", "参阅三",
    "干员导航", "关卡导航", "敌人导航", "道具导航", "剧情导航",
    "Cbox2", "Navigator", "Widget:Data_Char", "Widget:Data_Audio",
    "Widget:Data_Image", "mdi",
}

# 页面类型判定：不同类型页面用不同的标识模板
PAGE_TYPES = {
    "干员": lambda c: "干员页面名" in c,
    "关卡": lambda c: "普通关卡信息" in c,
    "敌人": lambda c: "敌人信息/common2" in c,
    "道具": lambda c: "道具信息" in c,
}


def scan(page_type: str = "干员"):
    pred = PAGE_TYPES.get(page_type, PAGE_TYPES["干员"])

    tpl_pages: Counter = Counter()
    tpl_total: Counter = Counter()
    sections: Counter = Counter()
    n_pages = 0

    with RAW.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            content = rec.get("content") or ""
            if not pred(content):
                continue
            n_pages += 1

            seen = set()
            for m in re.finditer(r"\{\{\s*([^{\}|\n]{1,40}?)\s*(?=[|\n}])", content):
                name = m.group(1).strip()
                if not name or name.startswith("#"):
                    continue
                tpl_total[name] += 1
                seen.add(name)
            for name in seen:
                tpl_pages[name] += 1

            for m in re.finditer(r"^==+[^=\n]+==+", content, re.M):
                s = m.group(0).strip("= ").strip()
                if s:
                    sections[s] += 1

    return n_pages, tpl_pages, tpl_total, sections


def main():
    page_type = "干员"
    if "--type" in sys.argv:
        page_type = sys.argv[sys.argv.index("--type") + 1]

    n_pages, tpl_pages, tpl_total, sections = scan(page_type)

    print(f"页面类型：{page_type}    命中页数：{n_pages}")
    print()

    # ---------------- 1. 内容板块（章节）全覆盖清单 ----------------
    print("=" * 68)
    print("一、内容板块清单（页面章节，这就是「一共有多少东西」的答案）")
    print("=" * 68)
    print(f"  {'章节':<20}{'页数':>7}{'占比':>9}")
    for s, n in sections.most_common(30):
        print(f"  {s:<20}{n:>7}{n / n_pages * 100:>8.0f}%")

    # ---------------- 2. 未覆盖模板 ----------------
    print()
    print("=" * 68)
    print("二、未被解析器覆盖的模板（按覆盖页数排序）")
    print("=" * 68)
    print("  这些模板出现在页面里，但不在 KNOWN_TEMPLATES 清单中 → 可能是遗漏")
    print()
    uncovered = []
    for name, pages in tpl_pages.items():
        if name in KNOWN_TEMPLATES or name in DECORATIVE:
            continue
        if pages < max(3, n_pages * 0.02):  # 太罕见的不报警
            continue
        uncovered.append((name, pages, tpl_total[name],
                          tpl_total[name] / pages))
    uncovered.sort(key=lambda x: -x[1])

    if not uncovered:
        print("  无 —— 所有高频模板都已被覆盖")
    else:
        print(f"  {'模板名':<28}{'覆盖页':>7}{'总次数':>8}{'次/页':>8}  判定")
        for name, pages, total, ratio in uncovered[:25]:
            kind = "装饰性" if ratio >= 3 else "标识性 ← 需检查"
            print(f"  {name:<28}{pages:>7}{total:>8}{ratio:>8.1f}  {kind}")

    # ---------------- 3. 装饰性但被当成标识的 ----------------
    print()
    print("=" * 68)
    print("三、自检：被引入 KNOWN_TEMPLATES 但其实是装饰性的模板")
    print("=" * 68)
    wrong = [(n, tpl_pages[n], tpl_total[n] / tpl_pages[n])
             for n in KNOWN_TEMPLATES
             if n in tpl_pages and tpl_total[n] / tpl_pages[n] >= 3]
    if wrong:
        for n, p, r in wrong:
            print(f"  !! {n:<24} 覆盖 {p} 页，次/页 {r:.1f} —— 它一页重复多次，"
                  f"对分类无价值")
        print("     注：材料消耗这类虽重复多次，但它是字段的唯一载体，"
              "属例外（分类看次/页，提取看覆盖页数）")
    else:
        print("  无异常")

    print()
    print("=" * 68)
    print("四、结论")
    print("=" * 68)
    print(f"  已覆盖模板：{len(KNOWN_TEMPLATES)} 个")
    print(f"  待检查模板：{len(uncovered)} 个")
    print(f"  内容板块  ：{len(sections)} 类")
    print()
    print("  下一步：把「标识性 ← 需检查」里确认有价值的模板")
    print("          补进 src/parser/wikitext.py，并同步更新 profile.py")


if __name__ == "__main__":
    main()
