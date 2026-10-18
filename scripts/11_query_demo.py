"""
第 5 步配套：把真实问题变成 SQL 的演示。

作用有两个：
  1. 验证建库正确 —— 数值查询是确定性的，答案对不对一眼能看出来
  2. 给第 7 步的 Agent 攒 few-shot 样例 ——
     Text-to-SQL 最有效的提示方式就是「问题 → SQL」的成对示例，
     这个文件跑出来的内容就是这个库的示例库雏形。

用法：
    python scripts/11_query_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.store.db import (  # noqa: E402
    connect,
    elite_materials,
    faction_members,
    profile_of,
    skill_desc,
    stat_at,
    total_materials,
    voice_lines,
)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "arknights.db"


def show(title: str, sql: str = "", note: str = "") -> None:
    print()
    print("-" * 70)
    print(f"Q: {title}")
    if sql:
        print(f"SQL: {sql.strip()}")
    if note:
        print(f"注 : {note}")


def main() -> None:
    if not DB_PATH.exists():
        print("数据库不存在，先跑：python scripts/10_build_db.py")
        return
    conn = connect(DB_PATH)

    # ---------------- 1. 单条精确查询 ----------------
    show("泥岩精二需要哪些材料？", """
SELECT stage, material, amount
FROM elite_materials e JOIN operators o ON o.id=e.operator_id
WHERE o.name='泥岩' AND e.stage='精2'
ORDER BY amount DESC
""")
    for r in elite_materials(conn, "泥岩", "精2"):
        print(f"    {r['material']:<10} x{r['amount']:,}")

    # ---------------- 2. 多跳汇总（跨 3 张表）----------------
    show("把泥岩精二 + 两个模组都拉满三级，一共要多少材料？（前 8）", """
-- UNION ALL 把精英化材料、模组材料两条来源合并，再按材料名 SUM
SELECT material, SUM(amount) FROM ( ... ) GROUP BY material
""", "这就是「多子问题拆解」的底座：一次 SQL 出结果，不需要 LLM 做算术")
    for r in total_materials(conn, "泥岩")[:8]:
        print(f"    {r['材料']:<12} {r['总数量']:>10,}")

    # ---------------- 3. 聚合统计 ----------------
    show("罗德岛有多少干员？按星级分布是怎样的？", """
SELECT o.rarity, COUNT(*) FROM factions f JOIN operators o ON o.id=f.operator_id
WHERE f.value='罗德岛' GROUP BY o.rarity ORDER BY o.rarity DESC
""")
    for r in conn.execute("""
        SELECT o.rarity, COUNT(*) AS n
        FROM factions f JOIN operators o ON o.id=f.operator_id
        WHERE f.value='罗德岛' GROUP BY o.rarity ORDER BY o.rarity DESC
    """):
        print(f"    {r['rarity']}星: {r['n']} 人")
    n = conn.execute("SELECT COUNT(*) FROM factions WHERE value='罗德岛'").fetchone()[0]
    print(f"    合计: {n} 人")

    # ---------------- 4. 档案类 ----------------
    show("泥岩是什么种族？出身哪里？生日？", """
SELECT gender, race, birthplace, birthday, height FROM profiles p
JOIN operators o ON o.id=p.operator_id WHERE o.name='泥岩'
""")
    p = profile_of(conn, "泥岩")
    if p:
        print(f"    性别 {p['gender']}  种族 {p['race']}  出身 {p['birthplace']}")
        print(f"    生日 {p['birthday']}  身高 {p['height']}  战斗经验 {p['combat_exp']}")

    # ---------------- 5. 面板数值 ----------------
    show("泥岩精二满级的防御力是多少？", """
SELECT level_key, hp, atk, defense, res FROM level_stats ls
JOIN operators o ON o.id=ls.operator_id
WHERE o.name='泥岩' AND ls.level_key='精2_满级'
""")
    s = stat_at(conn, "泥岩")
    if s:
        print(f"    生命 {s['hp']}  攻击 {s['atk']}  防御 {s['defense']}  法抗 {s['res']}")

    show("全站精二满级防御力最高的 5 个干员", """
SELECT o.name, ls.defense FROM level_stats ls JOIN operators o ON o.id=ls.operator_id
WHERE ls.level_key='精2_满级' ORDER BY ls.defense DESC LIMIT 5
""")
    for r in conn.execute("""
        SELECT o.name, ls.defense FROM level_stats ls JOIN operators o ON o.id=ls.operator_id
        WHERE ls.level_key='精2_满级' ORDER BY ls.defense DESC LIMIT 5
    """):
        print(f"    {r['name']:<12} 防御 {r['defense']}")

    # ---------------- 6. 技能机制 ----------------
    show("泥岩的三技能「秽壤的血脉」在各等级下效果如何？（抽 3 档）", """
SELECT sl.level, sl.desc, sl.init_sp, sl.sp_cost, sl.duration
FROM skill_levels sl JOIN skills s ON s.id=sl.skill_id
JOIN operators o ON o.id=s.operator_id
WHERE o.name='泥岩' AND s.name LIKE '%秽壤%' AND sl.level IN ('1','7','专精3')
""", "注意：等级档位是 1~7 + 专精1/2/3，不是 1~10")
    for r in skill_desc(conn, "泥岩", "秽壤"):
        if r["等级"] in ("1", "7", "专精3"):
            print(f"    {r['等级']:>2}级: {r['描述']}")
            print(f"          初始{r['init_sp']} 消耗{r['sp_cost']} 持续{r['duration']}")

    show("泥岩三技能专精三需要什么材料？", """
SELECT material, amount FROM skill_materials sm JOIN operators o ON o.id=sm.operator_id
WHERE o.name='泥岩' AND sm.skill_idx=3 AND sm.mastery_lv=3
""", "skill_idx=第几个技能，mastery_lv=专精等级（wiki 里的「三10」= 3号技能 + 10级）")
    for r in conn.execute("""
        SELECT material, amount FROM skill_materials sm JOIN operators o ON o.id=sm.operator_id
        WHERE o.name='泥岩' AND sm.skill_idx=3 AND sm.mastery_lv=3
    """):
        print(f"    {r['material']} x{r['amount']}")

    # ---------------- 7. 语音（结构化 + 语义的边界）----------------
    show("泥岩的「交谈1」说了什么？", """
SELECT title, line FROM voices v JOIN operators o ON o.id=v.operator_id
WHERE o.name='泥岩' AND v.title='交谈1'
""", "这条是精确查询；但「泥岩喜欢什么」要靠向量检索，因为答案散在多句台词里")
    for r in voice_lines(conn, "泥岩", "交谈1"):
        print(f"    【{r['title']}】{r['line'][:60]}...")

    show("泥岩所有带「信赖」的语音有几条？", """
SELECT COUNT(*) FROM voices v JOIN operators o ON o.id=v.operator_id
WHERE o.name='泥岩' AND v.title LIKE '%信赖%'
""")
    n = conn.execute("""
        SELECT COUNT(*) FROM voices v JOIN operators o ON o.id=v.operator_id
        WHERE o.name='泥岩' AND v.title LIKE '%信赖%'
    """).fetchone()[0]
    print(f"    {n} 条")

    # ---------------- 8. 阵营 ----------------
    show("彩虹小队有哪些干员？", """
SELECT o.name, o.rarity, o.profession FROM factions f
JOIN operators o ON o.id=f.operator_id WHERE f.value='彩虹小队'
""")
    for r in faction_members(conn, "彩虹小队"):
        print(f"    {r['name']:<12} {r['rarity']}星 {r['profession']}")

    # ---------------- 9. 边界：这类问题 SQL 答不了 ----------------
    show("【对比】哪些问题 SQL 答不了，必须交给向量检索？", note="""
    「泥岩是个什么样的人？」「她和可露希尔关系如何？」「她为什么离开卡兹戴尔？」
    —— 这些答案散落在档案正文、密录剧情、语音台词里，没有固定字段，
       第 6 步会把这些长文本切块进 Chroma，走语义检索。
    判断标准一句话：能写成 WHERE 的走 SQL，只能"读明白"的走向量。""")

    conn.close()
    print()


if __name__ == "__main__":
    main()
