"""
第 4 步：批量解析干员页  ->  data/processed/operators.jsonl

用法：
    python scripts/04_parse_operators.py

这一半的价值不在「跑通」，而在**最后打印的质量报告**。
真实项目里，解析器写完必须做完整性校验：
先看有多少页缺字段、缺的是哪些，再回头修解析器。
跳过这步直接入库，等 Agent 答错时才发现数据缺失，排查成本要高十倍。
"""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser.wikitext import is_operator_page, parse_operator  # noqa: E402

RAW = os.path.join("data", "raw", "pages.jsonl")
OUT_DIR = os.path.join("data", "processed")
OUT = os.path.join(OUT_DIR, "operators.jsonl")

# 需要统计完整度的字段
CHECK_FIELDS = [
    "星级", "职业", "分支", "特性", "面板",
    "精英化材料", "技能升级材料", "模组", "天赋", "技能",
    "档案", "势力", "档案文本",
    "攻击范围", "潜能", "后勤技能", "相关道具", "悖论模拟", "干员密录",
]


def load_operators():
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if is_operator_page(rec.get("content") or ""):
                yield rec["title"], rec.get("content") or ""


def main():
    if not os.path.exists(RAW):
        print(f"找不到原始数据：{RAW}\n请先运行 python scripts/01_fetch.py")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    results = []
    errors = []
    total = 0

    print(f"读取原始数据：{RAW}")
    for title, content in load_operators():
        total += 1
        rec = parse_operator(title, content)
        if "_error" in rec:
            errors.append((title, rec["_error"]))
        results.append(rec)

    with open(OUT, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------------- 质量报告 ----------------
    print(f"\n解析完成：{total} 个干员页，失败 {len(errors)} 个")
    print(f"输出文件：{os.path.abspath(OUT)}\n")

    # 完整度的分母必须是「本来就该有这个字段的干员」。
    # 例如 1 星小车本来就没技能、异格干员升变不需要精英化材料 ——
    # 把它们算进分母，会把「数据固有特性」误报成「解析缺失」。
    # 豁免规则统一在 src/parser/profile.py 维护，改那里即可。
    from src.parser.profile import EXEMPTION_RULES

    normal_all = [r for r in results if r.get("干员类别") == "常规"]
    cat = Counter(r.get("干员类别", "?") for r in results)
    alt = sum(1 for r in results if r.get("是否异格"))
    print("=" * 62)
    print(f"干员类别分布：{dict(cat)}    其中异格/升变干员 {alt} 个")
    print(f"字段完整度（分母已剔除「数据本就缺失」的干员）")
    print("=" * 62)
    for field in CHECK_FIELDS:
        rules = [r for r in EXEMPTION_RULES if r["field"] == field]
        denom = []
        for r in normal_all:
            exempted = False
            for rule in rules:
                try:
                    if rule["when"](r):
                        exempted = True
                        break
                except Exception:
                    continue
            if not exempted:
                denom.append(r)
        ok = sum(1 for r in denom if r.get(field))
        pct = ok / len(denom) * 100 if denom else 0
        bar = "#" * int(pct / 4)
        print(f"  {field:<14} {ok:>4}/{len(denom):<4} {pct:5.1f}%  {bar}")

    # 材料总量（用来判断解析是否真的抽出了数据，而不是空壳）
    mat_count = 0
    mat_names = Counter()
    for r in results:
        for _, items in (r.get("精英化材料") or {}).items():
            for it in items:
                mat_count += 1
                mat_names[it["name"]] += 1
    print(f"\n精英化材料条目数：{mat_count}，涉及材料种类：{len(mat_names)}")
    print(f"出现最多的材料 TOP5：{mat_names.most_common(5)}")

    # ---------------- 定点验证 ----------------
    print("\n" + "=" * 62)
    print("定点验证：泥岩精英化二材料")
    print("=" * 62)
    hit = next((r for r in results if r.get("名称") == "泥岩"), None)
    if hit:
        for k, items in (hit.get("精英化材料") or {}).items():
            s = "，".join(f"{i['name']}x{i['amount']}" for i in items)
            print(f"  {k}: {s}")
        print(f"\n  职业/分支：{hit.get('职业')} / {hit.get('分支')}")
        print(f"  星级：{hit.get('星级')}（wiki 原始值 {hit.get('稀有度原始值')}，0-based 已 +1）")
        print(f"  干员类别：{hit.get('干员类别')}（序号 {hit.get('干员序号')}）")
        lv = (hit.get("面板") or {}).get("levels", {})
        if lv:
            print(f"  面板等级档位：{list(lv.keys())}")
        # 注意：天赋条目数 = 各解锁条件下的描述条数，同一天赋在
        # 精英0/精英1/精英2/模组2级/模组3级 下会有多条，不等于天赋种类数
        print(f"  模组数：{len(hit.get('模组') or [])}  天赋条目数：{len(hit.get('天赋') or [])}  技能数：{len(hit.get('技能') or [])}")
        for sk in (hit.get("技能") or []):
            print(f"    - 技能：{sk['名称']}（{sk['回复类型']}/{sk['触发类型']}，{len(sk['等级'])} 个等级档位）")
    else:
        print("  未找到泥岩，请检查数据")

    # ---------------- 全量交叉校验 ----------------
    print("\n" + "=" * 62)
    print("全量交叉校验")
    print("=" * 62)
    stars = Counter(r.get("星级") for r in results if r.get("星级"))
    print(f"  星级分布（应为 1~6）：{sorted(stars.items())}")
    sk_num = Counter(len(r.get("技能") or []) for r in results)
    print(f"  技能数分布：{sorted(sk_num.items())}")
    print("  -> 若出现大量 1~2 技能且星级偏高，说明仍有模板没覆盖到")

    by_cat = {}
    for r in results:
        by_cat.setdefault(r.get("干员类别", "?"), []).append(r["名称"])
    print(f"\n  类别分布：常规 {len(by_cat.get('常规', []))}  "
          f"特勤 {len(by_cat.get('特勤', []))}  "
          f"特殊模式 {len(by_cat.get('特殊模式', []))}")
    if by_cat.get("特勤"):
        print(f"  特勤干员（靠集成战略任务解锁，无材料消耗）：{by_cat['特勤']}")
    if by_cat.get("特殊模式"):
        print(f"  特殊模式名单：{by_cat['特殊模式'][:30]}")

    # 按游戏机制，各星级「应有」的技能数。
    # 1~2 星是支援小车，没有技能；3 星 1 个；4~5 星 2 个；6 星 3 个。
    # 用这张表做校验，才能区分「真漏解析」和「本来就没有」。
    EXPECT_SKILL = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 3}
    bad = []
    for r in normal_all:
        want = EXPECT_SKILL.get(r.get("星级"))
        got = len(r.get("技能") or [])
        if want is not None and got < want:
            bad.append((r["名称"], r.get("星级"), want, got))
    if bad:
        print(f"\n  技能数少于星级预期的常规干员（{len(bad)} 个，需排查）：")
        for name, star, want, got in bad[:15]:
            print(f"    {name}  {star}星  预期{want} 实际{got}")
    else:
        print("\n  所有常规干员的技能数均符合星级预期")

    if errors:
        print("\n" + "=" * 62)
        print("解析失败的页面（前 10 个，需要回头看是什么变体）")
        print("=" * 62)
        for t, e in errors[:10]:
            print(f"  {t}: {e}")

    print("\n提示：完整度低的字段不一定是 bug，先跑诊断脚本看分类：")
    print("      python scripts/05_diagnose.py")


if __name__ == "__main__":
    main()
