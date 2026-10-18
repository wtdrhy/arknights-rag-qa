"""
第 5 步：把解析结果写进 SQLite（ETL 的 Load 阶段）。

用法：
    python scripts/10_build_db.py

产出：
    data/arknights.db

可重复运行（每次 DROP 重建），所以改了解析器之后重跑一条命令就能刷新。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import Counter  # noqa: E402

from src.parser.wikitext import residual_templates  # noqa: E402
from src.store.db import (  # noqa: E402
    build,
    connect,
    elite_materials,
    table_counts,
)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "arknights.db"
PROCESSED = ROOT / "data" / "processed"


def bar(n: int, total: int, width: int = 24) -> str:
    return "#" * int(n / total * width) if total else ""


def scan_residual(conn) -> tuple[int, Counter]:
    """全库扫描：所有 TEXT 列里残留的 {{模板}}。

    为什么要"全表自动扫"而不是手写几张表：
        上一版只查了 skill_levels / talents / archive_texts / modules，
        结果 voices 里 168 条被截断的台词、stories 里 2 篇空正文全部漏网 ——
        **检查项本身也会有 bug**，而且它一旦漏了，你就以为数据是干净的。
    原则：检查要覆盖"全部"，而不是覆盖"你上次想到的那几张表"。
    """
    total = 0
    names: Counter = Counter()
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    for t in tables:
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info([{t}])")]
        for c in cols:
            sql = f"SELECT [{c}] FROM [{t}] WHERE [{c}] LIKE ?"
            try:
                rows = conn.execute(sql, ("%{{%",)).fetchall()
            except Exception:
                continue      # 数值列 LIKE 会报错，无所谓，跳过
            for (v,) in rows:
                if not isinstance(v, str):
                    continue
                hit = residual_templates(v)
                if hit:
                    total += 1
                    names.update(hit)
    return total, names


def main() -> None:
    for f in ("operators.jsonl", "voices.jsonl", "stories.jsonl"):
        if not (PROCESSED / f).exists():
            print(f"[错误] 缺少 {PROCESSED / f}")
            print("       请先跑：python scripts/04_parse_operators.py 和 06_parse_subpages.py")
            return

    print("建库中...")
    stat = build(DB_PATH, PROCESSED)

    conn = connect(DB_PATH)

    print()
    print("=" * 68)
    print("数据库表行数")
    print("=" * 68)
    counts = table_counts(conn)
    for name, n in counts:
        print(f"  {name:<18} {n:>8,} 行")
    print(f"\n  数据库文件：{DB_PATH}")
    size = DB_PATH.stat().st_size / 1024 / 1024
    print(f"  文件大小  ：{size:.2f} MB")

    # ---------------- 未映射字段告警 ----------------
    unmapped = stat.pop("_unmapped_stat_keys", {})
    skipped_v = stat.pop("_skipped_voices", [])
    if unmapped:
        print()
        print("=" * 68)
        print("!! 未识别的面板属性字段（需要补进 _STAT_MAP）")
        print("=" * 68)
        for k, v in sorted(unmapped.items(), key=lambda x: -x[1])[:10]:
            print(f"  {k:<16} 出现 {v} 次")
    if skipped_v:
        print(f"\n  !! {len(skipped_v)} 个语音页没匹配到干员：{skipped_v[:5]}")

    # ---------------- 定点验证 ----------------
    print()
    print("=" * 68)
    print("定点验证：泥岩")
    print("=" * 68)
    print("  【精英化材料】")
    for r in elite_materials(conn, "泥岩"):
        print(f"    {r['stage']}: {r['material']} x{r['amount']}")

    print()
    print("  【满级面板】")
    row = conn.execute("""
        SELECT ls.* FROM level_stats ls JOIN operators o ON o.id = ls.operator_id
        WHERE o.name='泥岩' AND ls.level_key='精2_满级'
    """).fetchone()
    if row:
        print(f"    生命 {row['hp']}  攻击 {row['atk']}  防御 {row['defense']}  法抗 {row['res']}")

    print()
    print("  【技能升级材料 · 三技能专精三】（验证 skill_idx=3 / mastery_lv=3）")
    for r in conn.execute("""
        SELECT material, amount FROM skill_materials sm
        JOIN operators o ON o.id = sm.operator_id
        WHERE o.name='泥岩' AND sm.skill_idx=3 AND sm.mastery_lv=3
    """):
        print(f"    {r['material']} x{r['amount']}")

    print()
    print("  【势力】")
    for r in conn.execute("""
        SELECT kind, value FROM factions f JOIN operators o ON o.id=f.operator_id
        WHERE o.name='泥岩'
    """):
        print(f"    {r['kind']}: {r['value']}")

    # ---------------- 跨表校验 ----------------
    print()
    print("=" * 68)
    print("跨表校验（这几项不对就说明建库有问题）")
    print("=" * 68)
    checks = [
        ("干员总数",
         "SELECT COUNT(*) FROM operators", 463),
        ("势力归属条数",
         "SELECT COUNT(*) FROM factions", None),
        ("精英化材料条目",
         "SELECT COUNT(*) FROM elite_materials", 3185),
        ("语音条目",
         "SELECT COUNT(*) FROM voices", None),
        ("泥岩技能数",
         "SELECT COUNT(*) FROM skills s JOIN operators o ON o.id=s.operator_id WHERE o.name='泥岩'", 3),
        ("泥岩模组数（不含证章）",
         "SELECT COUNT(*) FROM modules m JOIN operators o ON o.id=m.operator_id "
         "WHERE o.name='泥岩' AND m.name NOT LIKE '%证章%'", 2),
        ("金额归一化检查：龙门币应 >= 10000",
         "SELECT MIN(amount) FROM elite_materials WHERE material='龙门币'", None),
        ("材料总表里出现 NULL 数量（应为 0）",
         "SELECT COUNT(*) FROM elite_materials WHERE amount IS NULL", 0),
        # 文本字段里残留 {{模板}} 说明清洗函数没覆盖到某个装饰模板。
        # 这是最容易漏的一类 bug：字段有值、长度正常，但内容里夹着脏标记，
        # 完整度统计看不出来，只有进向量库检索时才会变成噪声。
        # 走 scan_residual() 全表自动扫，不再手写表名。
        ("文本字段残留 {{模板}} 标记（应为 0）", "@residual@", 0),
    ]
    residual_names: Counter = Counter()
    for label, sql, expect in checks:
        if sql == "@residual@":
            got, residual_names = scan_residual(conn)
        else:
            got = conn.execute(sql).fetchone()[0]
        if expect is None:
            print(f"  {label:<34} {got:,}")
        else:
            ok = "OK " if got == expect else "!! "
            print(f"  {ok}{label:<32} {got:,}  (期望 {expect:,})")

    if residual_names:
        print("\n  残留模板名 TOP（把名字补进 src/parser/wikitext.py 的规则表）：")
        for name, n in residual_names.most_common(15):
            print(f"    {name:<28} {n} 次")

    # ---------------- 演示一条真实 SQL ----------------
    print()
    print("=" * 68)
    print("试一条 SQL：精英化消耗固源岩最多的前 5 个干员")
    print("=" * 68)
    print("  SELECT o.name, e.stage, e.amount")
    print("  FROM elite_materials e JOIN operators o ON o.id=e.operator_id")
    print("  WHERE e.material='固源岩' ORDER BY e.amount DESC LIMIT 5")
    print()
    for r in conn.execute("""
        SELECT o.name, e.stage, e.amount
        FROM elite_materials e JOIN operators o ON o.id=e.operator_id
        WHERE e.material='固源岩' ORDER BY e.amount DESC LIMIT 5
    """):
        print(f"    {r['name']:<12} {r['stage']}  固源岩 x{r['amount']}")

    conn.close()
    print()
    print("下一步：python scripts/11_query_demo.py  （看各类真实问题怎么变成 SQL）")


if __name__ == "__main__":
    main()
