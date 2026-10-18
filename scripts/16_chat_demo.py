"""第 8 步收尾：端到端问答 —— 从一句话到一句带引用的回答。

【完整链路（本项目第一次把 RAG 三个字母全跑通）】

    问题
      ↓  ① Query Routing      src/retriever/router.py    规则判断走哪条轨道
      ↓  ② SQL 轨道            SQLite                    精确值，不经过 LLM 改写
      ↓  ③ 语义轨道            src/retriever/hybrid.py   三路召回 + RRF
      ↓  ④ Augmentation       src/generator/prompt.py    编号、截断、拒答分支
      ↓  ⑤ Generation         src/generator/llm.py       DeepSeek 出答案
      ↓  ⑥ Citation           本脚本                     [n] 映射回 chunk 来源
    回答 + 引用清单

【第一个建议：先跑 --mock】
    python scripts/16_chat_demo.py --mock
不花一分钱，验证 prompt 组装、证据编号、拒答分支都对。
确认无误后再去掉 --mock 真调 API。

【⑥ 为什么必须有引用溯源】
没有引用，你无法判断回答是"查到的"还是"编的"。
有了引用，任何一句结论都能点回原始 chunk 核对。
这是 RAG 系统能被信任的最低要求，也是简历上很值钱的一行。
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator.llm import LLMClient                       # noqa: E402
from src.generator.prompt import (                            # noqa: E402
    build_messages, render_citations, refusal_answer,
)
from src.retriever.hybrid import HybridRetriever              # noqa: E402
from src.retriever.router import EntityMatcher, route         # noqa: E402

DB_PATH = ROOT / "data" / "arknights.db"
CHROMA_DIR = ROOT / "data" / "chroma"

DEMOS = [
    "泥岩精二需要哪些材料？",                 # SQL 轨
    "泥岩是个什么样的人？",                   # 语义轨
    "泥岩和可露希尔是什么关系？",             # 共现路
    "泥岩值得培养吗？",                       # 复合 → 两轨都要
    "给我讲讲主线剧情第一章的大致剧情",        # 剧情（第 10 步补齐）
    # 注意别再用"森西和泥岩"当拒答案例：补进活动剧情后，两人在「品尝魔物」
    # 联动里真的对话了，现在它能答上来。真拒答要用下面这条（全库 0 共现）。
    "史尔特尔和泥岩之间有什么故事？",          # 预期触发降级/拒答
]


def run_sql(conn: sqlite3.Connection, sql: str,
            params: tuple) -> dict | None:
    """执行模板 SQL，返回给 prompt 用的结构。查不到就返回 None。"""
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
    except Exception as e:                                    # noqa: BLE001
        print(f"      SQL 执行失败：{e}")
        return None
    if not rows:
        return None
    columns = [d[0] for d in cur.description] if cur.description else []
    return {"sql": sql, "columns": columns, "rows": [tuple(r) for r in rows]}


def answer(conn, hr: HybridRetriever, matcher: EntityMatcher,
           llm: LLMClient, question: str, k: int = 4,
           show_prompt: bool = False, stream: bool = False) -> None:
    r = route(question, matcher)

    print()
    print("=" * 74)
    print(f"Q: {question}")
    print("=" * 74)
    print(f"  ① 路由：{r.explain()}")
    if r.entities:
        # 把识别到的实体连同词形一起打出来，方便核对别名有没有生效
        shown = []
        for e, g in zip(r.entities, r.entity_groups):
            extra = [t for t in g if t != e]
            shown.append(e + (f"（{'/'.join(extra)}）" if extra else ""))
        print("  ① 实体：" + " / ".join(shown)
              + ("　· 已放开干员过滤" if r.cross_scope else ""))

    # ---------- ② SQL 轨道 ----------
    sql_result = None
    if r.sql and r.intent in ("sql", "hybrid"):
        sql_result = run_sql(conn, r.sql, r.sql_params)
        n = len(sql_result["rows"]) if sql_result else 0
        print(f"  ② SQL 轨道：{'命中 ' + str(n) + ' 行' if n else '无结果'}")

    # ---------- ③ 语义轨道 ----------
    hits: list[dict] = []
    relation_weak = False
    if r.intent in ("semantic", "hybrid"):
        # 关系类单实体 → 放开干员过滤，才能捞到"别人档案里提到 TA"的证据
        ops = None if r.cross_scope else r.operator
        out = hr.search(question, k=k, entities=r.entities,
                        entity_groups=r.entity_groups,
                        operator=ops, source=r.source)
        if not out["results"] and r.source:
            # 按资料类型过滤后空手而归 → 放开限制重来。
            # 空 context 喂给 LLM 是最经典的幻觉诱因。
            print("      （按资料类型过滤后无结果，自动放开限制重试）")
            out = hr.search(question, k=k, entities=r.entities,
                            entity_groups=r.entity_groups,
                            operator=ops, source=None)
        hits = out["results"]
        relation_weak = (len(r.entities) >= 2
                         and not out["has_cooccurrence"] and bool(hits))
        print(f"  ③ 检索轨道：{len(hits)} 条证据"
              + ("　⚠ 无共现证据，将降级为分别介绍" if relation_weak else ""))

    # ---------- ④ Augmentation ----------
    packed = build_messages(question, sql_result=sql_result, hits=hits,
                            relation_weak=relation_weak,
                            sql_kind=r.sql_kind)
    m = packed["meta"]
    print(f"  ④ 组装：SQL {m['n_rows']} 行 · 证据 {m['n_evidence']} 条"
          f" · 丢弃 {m['dropped']} 条"
          + (f" · prompt≈{m.get('prompt_tokens_est', 0)} token"
             if packed["ok"] else ""))

    if not packed["ok"]:
        # 两条轨道都空 → 代码级拒答，根本不调 LLM
        print("  ⑤ 生成：跳过（无证据，代码级拒答）")
        print()
        print("  ── 回答 ──")
        _t = packed["refusal"] or refusal_answer(question)
        print("  " + _t.replace("\n", "\n  "))
        return _pack_result(question, r, sql_result, packed, _t, [], [], 0.0,
                            refused=True)

    if show_prompt:
        print()
        print("  ── prompt 预览 ──")
        for msg in packed["messages"]:
            body = msg["content"][:1200]
            print(f"  <{msg['role']}>")
            print("  " + body.replace("\n", "\n  "))
        print()

    # ---------- ⑤ Generation ----------
    print("  ⑤ 生成中…")
    t0 = time.time()
    if stream and not llm.mock:
        text = llm.chat(packed["messages"], stream=True)
    else:
        text = llm.chat(packed["messages"])
    dt = time.time() - t0
    print(f"     （耗时 {dt:.1f}s）")

    print()
    print("  ── 回答 ──")
    print("  " + (text or "(空回答)").replace("\n", "\n  "))

    # ---------- ⑥ Citation ----------
    idxs = sorted({int(x) for x in re.findall(r"\[(\d+)\]", text or "")})
    cmap = {c["idx"]: c for c in packed["citations"]}
    used = [cmap[i] for i in idxs if i in cmap]
    print()
    print("  ── 引用溯源 ──")
    missing = [i for i in idxs if i not in cmap]
    if used:
        print(render_citations(used))
        if missing:
            print(f"  ⚠ 引用了不存在的编号 {missing} —— 模型幻觉的信号")
    else:
        print("  （回答未标注引用编号）")
        print(render_citations(packed["citations"]))

    # 返回结构化结果。
    # 为什么要 return 而不只是打印：Web 服务（21_serve.py）要用同一份链路。
    # 服务里再抄一遍检索+组装，两边迟早不一致 —— CLI 修了 bug 而 API 没修，
    # 这种事在生产上很常见，而且很难查。
    return _pack_result(question, r, sql_result, packed, text or "", used,
                        missing, dt, refused=False)


def _pack_result(question, r, sql_result, packed, text, used, missing,
                 elapsed, refused) -> dict:
    """把一次问答的中间过程打包成字典，供 API 使用。"""
    return {
        "question": question,
        "intent": r.intent,
        "route_reason": r.explain(),
        "entities": list(r.entities),
        # 每个实体识别到的全部词形（含别名），前端展示用
        "entity_terms": [list(g) for g in r.entity_groups],
        "sql_rows": len(sql_result["rows"]) if sql_result else 0,
        "evidence": packed["citations"],
        "used_citations": [c["idx"] for c in used],
        # 引用了不存在的编号 = 幻觉的强信号，服务层要能拿到
        "missing_citations": missing,
        "meta": packed["meta"],
        "answer": text,
        "refused": refused,
        "elapsed_sec": round(elapsed, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", help="只问一个问题；不给则进入交互模式")
    ap.add_argument("--k", type=int, default=4, help="取几条证据")
    ap.add_argument("--mock", action="store_true",
                    help="不调 API，验证链路（推荐第一次跑这个）")
    ap.add_argument("--show-prompt", action="store_true", help="打印完整 prompt")
    ap.add_argument("--stream", action="store_true", help="流式输出")
    args = ap.parse_args()

    print("装载检索器…")
    t0 = time.time()
    hr = HybridRetriever(CHROMA_DIR)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    matcher = EntityMatcher(conn)
    print(f"就绪（{time.time() - t0:.1f}s）· 已知干员名 {len(matcher.names)} 个")

    llm = LLMClient.from_env(mock=args.mock)

    if args.q:
        answer(conn, hr, matcher, llm, args.q, k=args.k,
               show_prompt=args.show_prompt, stream=args.stream)
        return

    print()
    print("=" * 74)
    print("交互模式：直接输入问题，输入 q 退出。")
    print("=" * 74)
    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("q", "quit", "exit", "退出"):
            break
        answer(conn, hr, matcher, llm, q, k=args.k,
               show_prompt=args.show_prompt, stream=args.stream)


if __name__ == "__main__":
    main()
