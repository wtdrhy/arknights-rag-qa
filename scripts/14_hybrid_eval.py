"""第 7 步评测：稠密 vs 稀疏 vs 混合，三条路放在一起比。

【为什么必须做这个对比】
不做对比，你无法判断"加了 BM25 到底有没有用" ——
很可能只是把指标从 87% 变成 88%，却多了一套要维护的索引。
工程决策必须有数据支撑，不能凭感觉。

【指标怎么读】
    Recall@k   前 k 条里有没有正确答案（有没有）
    MRR        正确答案平均排第几（1/rank 的均值，越接近 1 越好）
    共现命中   多实体问题里，前 k 条有没有"同时提到两个实体"的块
               这个指标是本项目特有的，直接衡量"关系类问题"能不能答

【关键提醒】
    探针一律**不加 operator/source 过滤**。
    加了过滤指标会虚高到没有参考价值 —— 那是过滤器的功劳，
    不是检索算法的功劳。真实场景里用户不会先告诉你"限定泥岩"。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retriever.hybrid import HybridRetriever  # noqa: E402

CHROMA_DIR = ROOT / "data" / "chroma"
DB_PATH = ROOT / "data" / "arknights.db"

# ---------------------------------------------------------------------------
# 探针集
#
# 字段说明：
#   q         问题
#   op        期望命中的干员（None = 不限制）
#   src       期望命中的资料类型
#   entities  问题里提到的实体（用于共现加成；也是自动识别能力的输入）
#   cooccur   True = 这道题的正确答案必须同时提到多个实体
# ---------------------------------------------------------------------------
PROBES = [
    {
        "q": "泥岩为什么离开卡兹戴尔？",
        "op": "泥岩", "src": None, "entities": ["泥岩"], "cooccur": False,
    },
    {
        "q": "泥岩和可露希尔之间发生了什么？",
        "op": None, "src": None, "entities": ["泥岩", "可露希尔"], "cooccur": True,
    },
    {
        "q": "能天使和德克萨斯是什么关系？",
        "op": None, "src": None, "entities": ["能天使", "德克萨斯"], "cooccur": True,
    },
    {
        "q": "为什么泥岩不愿意以真面目示人？",
        "op": "泥岩", "src": None, "entities": ["泥岩"], "cooccur": False,
    },
    {
        "q": "哪些干员的技能可以让敌人晕眩？",
        "op": None, "src": "skill", "entities": [], "cooccur": False,
    },
    {
        "q": "推进之王在维多利亚经历了什么？",
        "op": "推进之王", "src": None, "entities": ["推进之王"], "cooccur": False,
    },
    {
        "q": "阿米娅成为罗德岛的领袖后有什么烦恼？",
        "op": "阿米娅", "src": None, "entities": ["阿米娅"], "cooccur": False,
    },
    {
        # 这题没有唯一正确答案，给一个"命中任一个就算对"的期望集合。
        # 注意：op/src 全为空、且 cooccur=False 的探针会永远判不中
        # （hit() 无从判断对错），写探针时务必至少给定一种期望。
        "q": "谁擅长做饭，经常给大家准备料理？",
        "op": ["古米", "森西", "角峰"], "src": None,
        "entities": [], "cooccur": False,
    },
    {
        "q": "凯尔希和博士之间是什么关系？",
        "op": None, "src": None, "entities": ["凯尔希", "博士"], "cooccur": True,
    },
    {
        "q": "泥岩在罗德岛适应得怎么样，她有什么不习惯的地方？",
        "op": "泥岩", "src": "voice", "entities": ["泥岩"], "cooccur": False,
    },
]


def hit(res: list[dict], p: dict) -> int:
    """正确答案第一次出现的名次（1-based），没命中返回 0。"""
    for i, r in enumerate(res, 1):
        m = r["metadata"]
        if p["cooccur"]:
            if r.get("cooccur", 0) >= 2:
                return i
            continue
        if p["op"]:
            ops = p["op"] if isinstance(p["op"], list) else [p["op"]]
            if any(_same_op(m.get("operator"), o) for o in ops):
                return i
        if p["src"] and m.get("source") == p["src"]:
            return i
    return 0


def _same_op(got: str | None, want: str) -> bool:
    """泥岩 应该也命中 泥岩(卫戍协议) 这类异格形态。"""
    if not got:
        return False
    return got == want or got.startswith(want + "(") or want.startswith(got + "(")


def eval_mode(hr: HybridRetriever, mode: str, k: int = 5,
              verbose: bool = False) -> dict:
    """跑一轮评测。

    mode="dense_raw" 是**第 6 步的原始基线**：纯向量，不开共现路。
    留着这一列是为了看清"加了东西到底有没有用"，
    否则对比的是两个都改良过的方案，看不出真实增量。
    """
    real_mode = "dense" if mode == "dense_raw" else mode
    cooc = mode != "dense_raw"

    recall = {1: 0, 3: 0, 5: 0}
    mrr = 0.0
    co_hit = 0
    co_total = 0
    details = []

    for p in PROBES:
        out = hr.search(p["q"], k=k, entities=p["entities"],
                        mode=real_mode, cooc=cooc)
        res = out["results"]
        rank = hit(res, p)
        for kk in (1, 3, 5):
            if 0 < rank <= kk:
                recall[kk] += 1
        mrr += (1.0 / rank) if rank else 0.0

        is_co = p["cooccur"]
        if is_co:
            co_total += 1
            if any(r.get("cooccur", 0) >= 2 for r in res):
                co_hit += 1

        details.append({
            "q": p["q"], "rank": rank, "top": res[0] if res else None,
            "co": is_co,
        })

    n = len(PROBES)
    stat = {
        "Recall@1": recall[1] / n,
        "Recall@3": recall[3] / n,
        "Recall@5": recall[5] / n,
        "MRR": mrr / n,
        "共现命中": (co_hit / co_total) if co_total else float("nan"),
    }

    if verbose:
        for d in details:
            mark = "OK " if d["rank"] else "!! "
            top = d["top"]
            desc = ""
            if top:
                m = top["metadata"]
                desc = f"[{m['source_label']}·{m['operator']}]"
            print(f"  {mark}#{d['rank'] or '-':<2} {d['q'][:28]:<30} {desc}")
    return stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="dense_raw,dense,sparse,hybrid")
    ap.add_argument("--show", action="store_true", help="列出每道题的命中情况")
    args = ap.parse_args()

    print("装载检索器（首次会建 BM25 索引，约 30~60 秒，之后走缓存）…")
    hr = HybridRetriever(CHROMA_DIR)
    print()

    modes = [m.strip() for m in args.modes.split(",")]
    stats = {}
    for m in modes:
        print(f"  —— {m} ——")
        stats[m] = eval_mode(hr, m, verbose=args.show)
        print()

    print("=" * 90)
    print("召回方式对比（探针全部不加 metadata 过滤）")
    print("  dense_raw = 第 6 步原始基线（纯向量）   hybrid = 本步最终方案")
    print("=" * 90)
    header = f"  {'指标':<12}" + "".join(f"{m:>17}" for m in modes)
    print(header)
    print("  " + "-" * (12 + 17 * len(modes)))
    for key in ("Recall@1", "Recall@3", "Recall@5", "MRR", "共现命中"):
        row = f"  {key:<12}"
        for m in modes:
            v = stats[m][key]
            row += f"{v:>15.1%} " if v == v else f"{'—':>16}"
        print(row)

    print()
    print("怎么读这张表（拿首尾两列对比：基线 vs 最终方案）：")
    base, best = modes[0], modes[-1]
    if base != best:
        for key in ("Recall@1", "Recall@3", "Recall@5", "MRR", "共现命中"):
            a, b = stats[base][key], stats[best][key]
            if a != a or b != b:
                continue
            delta = (b - a) * 100
            sign = f"+{delta:.0f}" if delta > 0 else (f"{delta:.0f}" if delta else "±0")
            print(f"  {key:<10} {a:.1%} → {b:.1%}   ({sign} 个百分点)")
        print()
        print("  Recall@3 的涨幅通常比 Recall@1 大 —— 因为混合检索的主要作用")
        print("  是把正确答案'捞进候选集'，而 LLM 只需要它在前 3~5 条里就行。")

    print()
    print("下一步：python scripts/15_agent_router.py")


if __name__ == "__main__":
    main()
