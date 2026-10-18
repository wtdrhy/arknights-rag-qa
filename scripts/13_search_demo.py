"""
第 6 步演示：语义检索长什么样，以及它和 SQL 的分工边界

用法：
    python scripts/13_search_demo.py
    python scripts/13_search_demo.py --q "泥岩和可露希尔是什么关系"

【这一脚本想说明的核心】
    上一节的 11_query_demo.py 演示了「能写成 WHERE 的问题」→ SQL
    这一节演示「只能读明白的问题」→ 向量检索

    判断标准一句话：
        能写成 WHERE 的走 SQL（准确、可验算、零幻觉）
        只能"读明白"的走向量（模糊、语义匹配）
    第 7 步的 Agent 就是在做这个路由判断。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.store.vector import (  # noqa: E402
    clean_text_for_show,
    get_collection,
    search,
)

CHROMA_DIR = ROOT / "data" / "chroma"

# 演示用的问题，标注了「为什么 SQL 答不了」
DEMOS = [
    {
        "q": "泥岩是个什么样的人？她的性格如何？",
        "why": "性格散落在档案正文 + 几十条台词里，没有「性格」这个字段",
        "op": "泥岩",
    },
    {
        "q": "泥岩和可露希尔之间发生了什么？",
        "why": "人物关系是自然语言描述的，数据库里没有关系表",
        "op": "泥岩",
    },
    {
        "q": "为什么泥岩不愿意以真面目示人？",
        "why": "答案是一句台词，只能靠语义匹配找到",
        "op": "泥岩",
    },
]


def show(title: str, res: list[dict], n: int = 3, width: int = 150) -> None:
    for r in res[:n]:
        m = r["metadata"]
        print(f"    [{m['source_label']}·{m['operator']}] score={r['score']:.3f}")
        print(f"      {clean_text_for_show(r['text'], width)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", help="自定义问题，只跑这一条")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    if not CHROMA_DIR.exists():
        print(f"找不到 {CHROMA_DIR}，请先跑 python scripts/12_build_vector.py")
        return

    col = get_collection(CHROMA_DIR)
    print(f"Chroma 集合内块数：{col.count():,}")

    if args.q:
        print(f"\nQ: {args.q}")
        show("", search(col, args.q, k=args.k), n=args.k)
        return

    # ---------------- 1) SQL 答不了的三类问题 ----------------
    print()
    print("=" * 72)
    print("一、这些 SQL 答不了 —— 答案散在长文本里，只能靠语义匹配")
    print("=" * 72)
    for d in DEMOS:
        print(f"\n  Q: {d['q']}")
        print(f"     为什么 SQL 答不了：{d['why']}")
        show("", search(col, d["q"], k=args.k), n=args.k)

    # ---------------- 2) 元数据过滤的价值 ----------------
    print()
    print("=" * 72)
    print("二、同一个问题，加不加元数据过滤的区别")
    print("=" * 72)
    q = "她的过去经历了什么？"
    print(f"\n  Q: {q}   （刻意用了歧义主语「她」）\n")
    print("  【全库海搜】—— 会混进各种干员的过去")
    show("", search(col, q, k=3), n=3, width=110)
    print("\n  【限定 operator=泥岩】—— 范围一缩，噪声立刻消失")
    show("", search(col, q, k=3, operator="泥岩"), n=3, width=110)

    print("\n  这就是 metadata 过滤（pre-filter）的价值：")
    print("  Agent 先从问题里识别出干员名 → 变成 where 条件 → 再语义检索。")
    print("  比「全库搜完再按干员名筛选」快，也更准。")

    # ---------------- 3) 按来源过滤 ----------------
    print()
    print("=" * 72)
    print("三、按来源过滤：不同问题该看不同的资料")
    print("=" * 72)
    q2 = "能天使"
    for src, label in (("voice", "语音台词"), ("archive", "人员档案"),
                       ("story", "干员密录"), ("skill", "技能")):
        print(f"\n  【{label}】{q2}")
        show("", search(col, q2, k=2, source=src, operator=q2), n=2, width=120)

    # ---------------- 4) 边界对比 ----------------
    print()
    print("=" * 72)
    print("四、同一件事，两条轨道各答一半")
    print("=" * 72)
    print("  Q: 泥岩值得培养吗？")
    print("""
    SQL 负责的部分（准确、可验算）：
        星级 6 / 职业 重装 / 精二满级防御 602（全站第 12）
        精二材料：龙门币18万 + 炽合金块x5 + 重装双芯片x4 + 晶体电子单元x4
   向量负责的部分（语义、需要 LLM 总结）：
        档案里她的战斗定位、台词里她的作战风格、密录里她的剧情表现
    Agent 要做的事：识别这是一个「复合问题」→ 拆成两个子问题 → 分别查 → 汇总
    """)


if __name__ == "__main__":
    main()
