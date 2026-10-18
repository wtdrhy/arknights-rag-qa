"""
第 3 步：数据勘探（Data Recon）
==============================
目的：在写任何解析代码之前，先搞清楚数据到底长什么样。

核心技巧 —— 用「出现次数 / 覆盖页数」的比值区分两类模板：
  比值 ≈ 1   → 标识性模板，一个页面只用一次，用来识别页面类型（如 {{干员页面名}}）
  比值 >> 1  → 装饰性模板，一页里重复出现，是排版用的（如 {{color}}、{{术语}}）
只有标识性模板能用来给页面分类。

运行：
    python scripts/02_explore.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import mwparserfromhell
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

PAGES_FILE = Path(__file__).parent.parent / "data" / "raw" / "pages.jsonl"


def count_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    if not PAGES_FILE.exists():
        print(f"找不到数据文件 {PAGES_FILE}，请先运行 scripts/01_fetch.py")
        return

    total_pages = count_lines(PAGES_FILE)
    print(f"数据文件：{PAGES_FILE}")
    print(f"总页数：{total_pages}\n")

    occurrences = Counter()   # 模板在所有页面里的总出现次数
    page_counts = Counter()   # 有多少个页面包含该模板
    char_total = 0
    lengths = []

    # 用已知的标识模板确认各类页面规模
    known_markers = {
        "干员页面名": 0,
        "CharinfoV2": 0,
        "道具信息": 0,
        "敌人信息/common2": 0,
        "普通关卡信息": 0,
        "关卡材料掉落": 0,
        "活动信息": 0,
    }
    redirect_count = 0   # #redirect 页面，纯垃圾
    short_count = 0      # 内容过短的页面，可疑

    with open(PAGES_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_pages, desc="解析中", unit="page"):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            content = record.get("content") or ""

            # 重定向页只有一行 #redirect，是垃圾数据，不参与模板统计
            if content.lstrip().lower().startswith("#redirect"):
                redirect_count += 1
                continue
            if len(content) < 100:
                short_count += 1

            char_total += len(content)
            lengths.append(len(content))

            try:
                code = mwparserfromhell.parse(content)
            except Exception:
                continue

            seen_on_this_page = set()
            for template in code.filter_templates():
                name = str(template.name).strip()
                if not name:
                    continue
                occurrences[name] += 1
                seen_on_this_page.add(name)

            for name in seen_on_this_page:
                page_counts[name] += 1
                if name in known_markers:
                    known_markers[name] += 1

    # ---------- 基础统计 ----------
    lengths.sort()
    print("=" * 72)
    print("基础统计")
    print("=" * 72)
    print(f"  总字符数   : {char_total / 1024 / 1024:.1f} MB")
    print(f"  平均长度   : {char_total / max(len(lengths), 1):.0f} 字符")
    print(f"  中位数     : {lengths[len(lengths) // 2] if lengths else 0} 字符")
    print(f"  最短 / 最长: {lengths[0] if lengths else 0} / {lengths[-1] if lengths else 0} 字符")
    print(f"  重定向页   : {redirect_count} 页（已排除，纯垃圾数据）")
    print(f"  超短页<100 : {short_count} 页（需要人工看看是什么）")

    print()
    print("=" * 72)
    print("已知标识模板命中页数")
    print("=" * 72)
    for name, count in known_markers.items():
        print(f"  {name:<16} {count:>6} 页")

    # ---------- 模板频次排行 ----------
    print()
    print("=" * 72)
    print("模板排行（按覆盖页数排序，前 40）")
    print("=" * 72)
    print(f"{'模板名':<28}{'覆盖页数':>9}{'总次数':>10}{'次/页':>8}   类型判断")
    print("-" * 72)

    for name, pcount in page_counts.most_common(40):
        occ = occurrences[name]
        ratio = occ / pcount
        if ratio <= 1.5:
            kind = "标识性"   # 一页基本只用一次 → 可用来分类
        elif ratio <= 4:
            kind = "半标识"
        else:
            kind = "装饰性"   # 一页重复很多次 → 排版用
        display = name[:26]
        print(f"{display:<28}{pcount:>9}{occ:>10}{ratio:>8.1f}   {kind}")

    print()
    print("提示：把上面「标识性」的模板挑出来，就是第 4 步做页面分类的依据。")


if __name__ == "__main__":
    main()
