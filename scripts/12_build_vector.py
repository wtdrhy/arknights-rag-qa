"""
第 6 步：把长文本切块 → 向量化 → 写进 Chroma

用法：
    python scripts/12_build_vector.py                 # 全量构建（首次约 2~5 分钟）
    python scripts/12_build_vector.py --limit 200     # 只建 200 块，快速验证流程
    python scripts/12_build_vector.py --fresh         # 改了切块策略后必须加，删旧库重建
    python scripts/12_build_vector.py --model BAAI/bge-base-zh-v1.5

产出：
    data/chroma/    Chroma 持久化目录

【这一步做完，RAG 的离线索引就完整了】
    左轨 SQLite  : 数值类问题（材料/面板/技能等级）→ SQL，零幻觉
    右轨 Chroma  : 语义类问题（性格/关系/剧情）    → 向量检索
    第 7 步开始   : Agent 路由 —— 判断问题该走哪条，或两条都要
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser.documents import build_documents, summarize  # noqa: E402
from src.store.db import connect  # noqa: E402
from src.store.vector import (  # noqa: E402
    DEFAULT_MODEL,
    add_documents,
    clean_text_for_show,
    get_collection,
    search,
)

DB_PATH = ROOT / "data" / "arknights.db"
CHROMA_DIR = ROOT / "data" / "chroma"

# ---------------------------------------------------------------------------
# 检索质量探针（golden set）
# ---------------------------------------------------------------------------
# 为什么必须有这个：
#   "建完向量库"和"向量库能用"是两件事。没有量化指标，
#   你改了 chunk size、换了模型，根本不知道是变好还是变坏。
#   这是 retrieval evaluation 的雏形，也是答辩/简历里最硬的一项：
#   「我建了 8 条人工标注的检索探针，Recall@3 = x%」
#
# 每条探针给出「期望命中的干员」和/或「期望命中的来源类型」。
# 关键设计：**检索时不加任何过滤**，全库海搜，然后看期望结果有没有出现在
# top-k 里 —— 这样才能真正考到语义匹配能力。
# （如果检索时就加了 where operator='泥岩'，那是过滤器的功劳，
#   不是 embedding 的功劳，指标会虚高到没有参考价值）
PROBES = [
    {"q": "泥岩为什么离开卡兹戴尔？", "op": "泥岩", "src": None},
    {"q": "泥岩在罗德岛适应得怎么样，她有什么不习惯的地方？", "op": "泥岩", "src": "voice"},
    {"q": "能天使和谁关系最好，她们是什么关系？", "op": "能天使", "src": None},
    {"q": "哪些干员的技能可以让敌人晕眩？", "op": None, "src": "skill",
     "note": "这条是**故意留着的失败样本**，很有教学价值：\n"
             "            「晕眩」是可以在 skill_levels.desc 里 LIKE 出来的字面关键词，\n"
             "            向量检索却把「天赋」排到了前面（U-Official 的天赋确实也晕眩）。\n"
             "            说明：含**明确机制关键词**的问题，纯语义检索容易串字段，\n"
             "            正解是第 7 步的 hybrid retrieval —— 语义召回 + 关键词/SQL 兜底。"},
    {"q": "推进之王在维多利亚经历了什么？", "op": "推进之王", "src": None},
    {"q": "阿米娅成为罗德岛的领袖后有什么烦恼？", "op": "阿米娅", "src": None},
    {"q": "哪个干员的天赋可以提高队友的攻击力？", "op": None, "src": "talent"},
    {"q": "德克萨斯过去是什么身份？", "op": "德克萨斯", "src": None},
]


def _op_match(got: str, want: str) -> bool:
    """干员名匹配，允许异格/升变形态。

    为什么放宽：wiki 里「阿米娅」和「阿米娅(近卫)」、「浊心斯卡蒂」等
    是**同一角色的不同形态**，档案内容高度重叠。用户问「阿米娅」时
    命中「阿米娅(近卫)」的档案不算错，硬判为失败会让指标虚低。
    反过来也成立：标注「阿米娅(近卫)」时命中「阿米娅」同样算对。
    """
    if not want:
        return False
    if got == want:
        return True
    return got.startswith(want + "(") or want.startswith(got + "(")


def _rank(res: list[dict], p: dict) -> int:
    """返回期望结果第一次出现的名次（1-based），没命中返回 0。"""
    for i, r in enumerate(res, 1):
        m = r["metadata"]
        if p["op"] and _op_match(m["operator"], p["op"]):
            return i
        if p["src"] and m["source"] == p["src"]:
            return i
    return 0


def run_probes(col, k: int = 5, show: int = 2) -> dict:
    """跑探针，返回 Recall@1 / Recall@3 / Recall@5 / MRR。

    MRR（Mean Reciprocal Rank）= 各探针「1/首次命中名次」的平均，
    比 Recall 更敏感：命中在第 1 名得 1 分，第 5 名只有 0.2 分。
    """
    hit1 = hit3 = hit5 = 0
    mrr = 0.0
    print()
    print("=" * 70)
    print(f"检索质量探针（{len(PROBES)} 条人工标注，全库海搜 k={k}，不加过滤）")
    print("=" * 70)
    for p in PROBES:
        res = search(col, p["q"], k=k)
        rank = _rank(res, p)
        hit1 += int(0 < rank <= 1)
        hit3 += int(0 < rank <= 3)
        hit5 += int(0 < rank <= 5)
        mrr += (1.0 / rank) if rank else 0.0
        mark = "OK " if rank == 1 else ("~  " if rank else "!! ")
        want = p["op"] or p["src"]
        print(f"\n  {mark}{p['q']}   期望={want}  首次命中第 {rank or '-'} 名")
        for r in res[:show]:
            m = r["metadata"]
            print(f"      [{m['source_label']}·{m['operator']}] "
                  f"score={r['score']:.3f}  {clean_text_for_show(r['text'], 88)}")
        if not rank and p.get("note"):
            print(f"      注：{p['note']}")
    n = len(PROBES)
    return {"Recall@1": hit1 / n, "Recall@3": hit3 / n,
            "Recall@5": hit5 / n, "MRR": mrr / n, "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只建前 N 块，快速验证")
    ap.add_argument("--fresh", action="store_true", help="删掉旧库重建")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="embedding 模型名或本地路径")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--no-probe", action="store_true", help="跳过探针评测")
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"找不到 {DB_PATH}，请先跑 python scripts/10_build_db.py")
        return

    print("读取 SQLite，聚合成检索文档...")
    conn = connect(DB_PATH)
    docs = build_documents(conn)
    conn.close()

    stat = summarize(docs)
    print()
    print("=" * 70)
    print("文档块画像（切块后的形状，先看这个再建库）")
    print("=" * 70)
    for k, v in stat.items():
        if k == "来源分布":
            print(f"  {k}:")
            for s, n in v.items():
                print(f"      {s:<10} {n:>7,} 块")
        else:
            print(f"  {k:<10} {v:,}" if isinstance(v, int) else f"  {k:<10} {v}")

    if args.limit:
        docs = docs[:args.limit]
        print(f"\n  （--limit {args.limit}，只建前 {len(docs):,} 块）")

    print()
    print("=" * 70)
    print(f"写入 Chroma：{CHROMA_DIR}")
    print("=" * 70)
    t0 = time.time()
    col = get_collection(CHROMA_DIR, fresh=args.fresh, model=args.model)
    add_documents(col, docs, batch=args.batch)
    dt = time.time() - t0
    print(f"  耗时 {dt:.1f} 秒（含模型加载），平均 {dt / max(1, len(docs)) * 1000:.1f} ms/块")

    size = sum(f.stat().st_size for f in CHROMA_DIR.rglob("*") if f.is_file())
    print(f"  Chroma 目录大小：{size / 1024 / 1024:.1f} MB")
    print(f"  集合内块数：{col.count():,}")

    if not args.no_probe and not args.limit:
        m = run_probes(col, k=5)
        print()
        print("=" * 70)
        print(f"探针汇总（{m['n']} 条，全库海搜不加过滤）")
        for k2 in ("Recall@1", "Recall@3", "Recall@5", "MRR"):
            print(f"  {k2:<10} {m[k2]:.2%}")
        # 解读必须**看数据说话**，写死一句话会误导：
        #   Recall@1 明显低于 Recall@5 → 召回到了但排不进第一（排序问题）
        #   两者几乎相等           → 相关内容压根没进 top-k（召回问题）
        # 这两种病因的治法完全不同，前者调切块/加 header，后者调检索通道。
        if m["Recall@5"] - m["Recall@1"] > 0.1:
            print("  解读：Recall@1 明显低于 Recall@5 = 相关内容召回了但排不进第一。")
            print("        这是**排序问题**：试着改小 chunk、加强 header、或加重排序(rerank)。")
        elif m["Recall@5"] < 1.0:
            print("  解读：Recall@1 与 Recall@5 几乎相等 = 没命中的那些**压根没进 top-k**。")
            print("        这是**召回问题**：调 chunk size 无用，要靠关键词通道兜底")
            print("        （第 7 步的 hybrid retrieval：向量 + SQL LIKE / BM25 混合）。")
        else:
            print("  解读：全部命中，可以进入下一步。")
        print("  解读：Recall@1 低而 Recall@5 高 = 相关内容召回了但没排第一，")
        print("        这是 chunk 粒度的信号，可以试着改 chunk size 或加强 header。")
        print("=" * 70)

    print()
    print("下一步：python scripts/13_search_demo.py  （看语义检索长什么样）")


if __name__ == "__main__":
    main()
