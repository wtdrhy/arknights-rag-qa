"""第 7 步收尾：Agent 路由器的端到端演示。

【这一步在 RAG 全景里的位置】
    用户提问
      ↓
    Query Routing（本脚本）—— 判断走 SQL / 向量 / 两者
      ↓
    Retrieval（已建好的两条轨道）
      ↓
    Generation（第 8 步接 LLM）

【一个容易被忽略的能力：拒答】
    不是所有问题都有答案。比如"泥岩和可露希尔是什么关系"，
    全库其实只有一处提到两人（泥岩的一句台词），
    如果检索不到共现证据，正确的做法是**明确说"资料里没有记载"**，
    而不是让 LLM 用两个人的档案拼一段看起来合理的剧情。

    后者就是幻觉的主要来源。业界叫 abstention / hallucination control，
    是生产级 RAG 和 demo 级 RAG 的分水岭。

    本脚本会在关系类问题上自动做这个判断。
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retriever.hybrid import HybridRetriever          # noqa: E402
from src.retriever.router import EntityMatcher, route     # noqa: E402
from src.store.vector import clean_text_for_show          # noqa: E402

DB_PATH = ROOT / "data" / "arknights.db"
CHROMA_DIR = ROOT / "data" / "chroma"

DEMOS = [
    "泥岩精二需要哪些材料？",
    "泥岩是个什么样的人？",
    "泥岩和可露希尔是什么关系？",
    "能天使和德克萨斯是什么关系？",
    "泥岩值得培养吗？",
    "哪些干员的技能可以让敌人晕眩？",
]


def run_sql(conn: sqlite3.Connection, sql: str, params: tuple, limit: int = 8) -> None:
    try:
        rows = conn.execute(sql, params).fetchall()
    except Exception as e:
        print(f"      SQL 执行失败：{e}")
        return
    if not rows:
        print("      （无结果）")
        return
    for r in rows[:limit]:
        print("      " + "  ".join(str(x) for x in r if x is not None))
    if len(rows) > limit:
        print(f"      …共 {len(rows)} 行")


def answer(conn, hr: HybridRetriever, matcher: EntityMatcher,
           question: str, k: int = 3) -> None:
    r = route(question, matcher)

    print()
    print("=" * 74)
    print(f"Q: {question}")
    print("=" * 74)
    print(f"  路由：{r.explain()}")
    if r.entities:
        forms = r.operator or []
        extra = f"（含异格形态 {len(forms) - 1} 个）" if len(forms) > 1 else ""
        print(f"  识别到实体：{' / '.join(r.entities)}{extra}")

    # ---------------- 轨道一：SQL ----------------
    if r.sql and r.intent in ("sql", "hybrid"):
        print()
        print("  【SQL 轨道】精确、可验算")
        print(f"    {r.sql}")
        run_sql(conn, r.sql, r.sql_params)

    # ---------------- 轨道二：向量 + 稀疏 ----------------
    if r.intent in ("semantic", "hybrid"):
        print()
        print("  【检索轨道】混合检索（向量 + BM25 + 共现重排）")
        out = hr.search(question, k=k, entities=r.entities,
                        operator=r.operator, source=r.source)
        if not out["results"] and r.source:
            # 兜底：按资料类型过滤后一条都没有（比如问"故事"但这位干员
            # 没有密录），就放开资料限制重来。宁可噪声大一点，
            # 也不要空手去喂 LLM —— 空 context 时 LLM 最容易编。
            print("    （按资料类型过滤后无结果，已自动放开资料限制重试）")
            out = hr.search(question, k=k, entities=r.entities,
                            operator=r.operator, source=None)
        for i, hit in enumerate(out["results"], 1):
            m = hit["metadata"]
            dense = f"向量#{hit['dense_rank']}" if hit["dense_rank"] else "向量—"
            sparse = f"字面#{hit['sparse_rank']}" if hit["sparse_rank"] else "字面—"
            tag = f" [共现{hit['cooccur']}]" if hit.get("cooccur", 0) >= 2 else ""
            print(f"    {i}. [{m['source_label']}·{m['operator']}] "
                  f"{dense} {sparse} {hit['score']:.4f}{tag}")
            print(f"       {clean_text_for_show(hit['text'], 110)}")

        # ---------------- 可答性判定 ----------------
        if len(r.entities) >= 2 and not out["has_cooccurrence"]:
            print()
            print("    ⚠ 前 k 条里没有任何一段同时提到这两个人。")
            print("      正确行为：告诉用户「资料里没有记载两人有直接互动」，")
            print("      而不是用两人的档案各拼一句编出一段关系。—— 这就是幻觉的来源。")
            print("      若要强行回答，应降级为「分别介绍两人」，并声明是推断。")

    # ---------------- 第 8 步的占位 ----------------
    print()
    print("  【下一步】以上内容会作为 context 拼进 prompt 交给 LLM 总结（第 8 步）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", help="只问一个问题")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    print("装载检索器…")
    import time
    t0 = time.time()
    hr = HybridRetriever(CHROMA_DIR)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    matcher = EntityMatcher(conn)
    print(f"就绪（{time.time() - t0:.1f}s）· 已知干员名 {len(matcher.names)} 个")

    qs = [args.q] if args.q else DEMOS
    for q in qs:
        answer(conn, hr, matcher, q, k=args.k)

    if not args.q:
        print()
        print("=" * 74)
        print("试着自己问：python scripts/15_agent_router.py --q \"你想问的话\"")
        print("=" * 74)


if __name__ == "__main__":
    main()
