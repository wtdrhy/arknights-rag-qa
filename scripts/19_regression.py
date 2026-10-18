"""第 9 步：回归评测 —— 把散落的指标固化成「一条命令可复现」的测试。

【为什么前面做的还不够】
第 7 步的 Recall@1 80%、MRR 88.3% 是**一次性的手工探针**，
写在 14_hybrid_eval.py 的常量里。它有两个致命问题：

    1. 改了代码不会重跑 → 你不知道有没有退化
    2. 探针是我拍脑袋编的 → 它守护的是「我猜会出问题」的地方，
       而真正出问题的地方（第 8 步的别名）一个都没覆盖到

所以本步做两件事：

    A. 把用例**外部化**到 data/eval_cases.json
       每条带 why 字段，说明它守护哪个真实发生过的 bug。
       用例必须来自实战，不能来自想象 —— 这是评测集唯一的含金量来源。

    B. 跑一次只要一条命令，失败就红
       以后改 router / hybrid / prompt 任意一环，跑一下就知道有没有踩坏别的。

【三层断言，分开看】
    L1 路由层   实体、意图、过滤范围、SQL 可执行性   ← 零成本，每次改代码都跑
    L2 检索层   top1 命中、共现有无                 ← 零成本
    L3 生成层   引用编号有效性、拒答是否触发         ← 要调 API，--with-llm 才跑

为什么要分层：指标只有一个总分时，你只知道"变差了"，不知道去哪修。
路由层红 → 改 router.py；检索层红 → 改 hybrid.py；生成层红 → 改 prompt.py。
定位成本差一个数量级。

【用法】
    python scripts/19_regression.py              # 跑 L1+L2（零成本，秒级）
    python scripts/19_regression.py --show-ok    # 通过的也列出来
    python scripts/19_regression.py --with-llm   # 加跑 L3（要花钱，约 21 次调用）
    python scripts/19_regression.py --save       # 存为基线
    python scripts/19_regression.py --diff       # 与上次基线对比（看新增/修复）

退出码：有用例失败时为 1，可直接挂 CI。
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retriever.hybrid import HybridRetriever               # noqa: E402
from src.retriever.router import EntityMatcher, route          # noqa: E402

DB_PATH = ROOT / "data" / "arknights.db"
CHROMA_DIR = ROOT / "data" / "chroma"
CASES_PATH = ROOT / "data" / "eval_cases.json"
BASELINE_PATH = ROOT / "data" / "regression_baseline.json"


# ---------------------------------------------------------------------------
# 判定器
# ---------------------------------------------------------------------------
def _same_op(got: str | None, want: str) -> bool:
    """泥岩 应该也命中 泥岩(卫戍协议) 这类异格形态。"""
    if not got:
        return False
    return got == want or got.startswith(want + "(") or want.startswith(got + "(")


def check_route(r, exp: dict, conn: sqlite3.Connection) -> list[str]:
    """路由层断言。返回失败原因列表，空列表 = 通过。"""
    errs: list[str] = []

    if "entities" in exp and r.entities != exp["entities"]:
        errs.append(f"实体 {r.entities or '[]'} ≠ 期望 {exp['entities']}")

    if "intent" in exp:
        want = exp["intent"]
        want = [want] if isinstance(want, str) else list(want)
        if r.intent not in want:
            errs.append(f"意图 {r.intent} ∉ 期望 {want}")

    if "cross_scope" in exp and r.cross_scope != exp["cross_scope"]:
        errs.append(f"cross_scope={r.cross_scope} ≠ 期望 {exp['cross_scope']}")

    if "source" in exp and r.source != exp["source"]:
        errs.append(f"资料类型 {r.source} ≠ 期望 {exp['source']}")

    # 守"汇总类问题必须走聚合 SQL"。这条比行数更能说明问题：
    # 返回 41 行明细也能凑够行数，但那是错的 —— 用户要的是总数。
    if "sql_kind" in exp and r.sql_kind != exp["sql_kind"]:
        errs.append(f"sql_kind={r.sql_kind or '空'} ≠ 期望 {exp['sql_kind']}")

    # sql_rows_min 是最硬的一条：模板选错或参数喂错，这里立刻红。
    # 0 表示只要求 SQL 能执行（不要求有数据）。
    if "sql_rows_min" in exp:
        need = exp["sql_rows_min"]
        if not r.sql:
            errs.append(f"未生成 SQL，期望至少 {need} 行")
        else:
            try:
                n = len(conn.execute(r.sql, r.sql_params).fetchall())
            except Exception as e:                            # noqa: BLE001
                errs.append(f"SQL 执行失败：{str(e)[:60]}")
            else:
                if n < need:
                    errs.append(f"SQL 返回 {n} 行 < 期望 {need} 行")
    return errs


def check_retrieve(res: list[dict], exp: dict) -> list[str]:
    """检索层断言。"""
    errs: list[str] = []
    top = res[0] if res else None
    m = (top or {}).get("metadata", {}) or {}

    if "op" in exp:
        want = exp["op"]
        want = [want] if isinstance(want, str) else list(want)
        got = m.get("operator")
        if not any(_same_op(got, o) for o in want):
            errs.append(f"top1 干员「{got}」不匹配期望 {want}")

    if "src" in exp:
        got = m.get("source")
        if got != exp["src"]:
            errs.append(f"top1 来源「{got}」≠ 期望「{exp['src']}」")

    # 共现是最容易悄悄退化的一项：它不报错、不空结果，
    # 只是把真证据判成"无共现"，然后让 LLM 降级回答。
    if "cooccur" in exp:
        has = any(x.get("cooccur", 0) >= 2 for x in res)
        if has != exp["cooccur"]:
            tone = "有" if has else "无"
            errs.append(f"top3 共现为「{tone}」，期望「{'有' if exp['cooccur'] else '无'}」")
    return errs


def check_generate(text: str, packed: dict) -> list[str]:
    """生成层断言：只查一件事 —— 引用编号是不是真的存在。

    为什么只查这个：答案「好不好」需要 LLM-as-judge 或人工，成本高且不稳定。
    但「引用了不存在的 [7]」是**客观可判**的，而且它是幻觉的强信号：
    模型编了一段话，然后给它安一个来源编号。这个必须零容忍。

    另外空回答**不能算通过**：模型没说话不等于它说得对。
    不加这条的话，网络抖动导致的空响应会被静默判绿，评测就白做了。
    """
    errs: list[str] = []
    if not packed.get("ok"):
        return errs                      # 代码级拒答，本来就没调 LLM
    if not (text or "").strip():
        errs.append("LLM 返回空回答（视为失败：没说话 ≠ 说得对）")
        return errs
    idxs = {int(x) for x in re.findall(r"\[(\d+)\]", text or "")}
    valid = {c["idx"] for c in packed.get("citations", [])}
    ghost = sorted(i for i in idxs if i not in valid)
    if ghost:
        errs.append(f"引用了不存在的编号 {ghost}（幻觉信号）")
    return errs


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def load_cases() -> list[dict]:
    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    return data.get("cases") or data if isinstance(data, dict) else data


def search_like_online(hr, r, k: int = 3):
    """复现 16_chat_demo 的检索调用方式。

    为什么必须复现而不是自己拼参数：
    评测要测的是**线上真实行为**。自己写一套更"理想"的调用，
    指标会好看，但测的不是用户实际会遇到的那条路径 —— 那叫自欺欺人。
    """
    ops = None if r.cross_scope else r.operator
    out = hr.search(r.question, k=k, entities=r.entities,
                    entity_groups=r.entity_groups,
                    operator=ops, source=r.source)
    if not out["results"] and r.source:
        out = hr.search(r.question, k=k, entities=r.entities,
                        entity_groups=r.entity_groups,
                        operator=ops, source=None)
        out["relaxed"] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true", help="加跑 L3 生成层（要调 API）")
    ap.add_argument("--show-ok", action="store_true", help="通过的用例也列出")
    ap.add_argument("--save", action="store_true", help="把本次结果存为基线")
    ap.add_argument("--diff", action="store_true", help="与基线对比，只看变化")
    ap.add_argument("--cat", help="只跑某一分类，如「别名归一」")
    args = ap.parse_args()

    cases = load_cases()
    if args.cat:
        cases = [c for c in cases if c.get("cat") == args.cat]
        if not cases:
            print(f"没有分类为「{args.cat}」的用例")
            return 1

    print(f"用例 {len(cases)} 条 · 来自 {CASES_PATH.relative_to(ROOT)}")

    if args.with_llm:
        print("装载 LLM（L3）…")
        from src.generator.llm import LLMClient
        from src.generator.prompt import build_messages
        llm = LLMClient.from_env()
    else:
        print("装载检索器（L1+L2，不调 API）…")

    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    matcher = EntityMatcher(conn)
    hr = HybridRetriever(CHROMA_DIR)
    print(f"就绪（{time.time() - t0:.1f}s）\n")

    results: list[dict] = []
    n_llm = 0                     # L3 实际调用次数，跑完要打印，防止静默跳过
    for c in cases:
        q = c["q"]
        r = route(q, matcher)
        out = search_like_online(hr, r)
        res = out["results"]

        errs: list[str] = []
        if "expect_route" in c:
            errs += [f"[路由] {e}" for e in check_route(r, c["expect_route"], conn)]
        if "expect_retrieve" in c:
            errs += [f"[检索] {e}" for e in check_retrieve(res, c["expect_retrieve"])]

        answered = ""
        if args.with_llm:
            sql_res = None
            if r.sql and r.intent in ("sql", "hybrid"):
                try:
                    cur = conn.execute(r.sql, r.sql_params)
                    rows = cur.fetchall()
                    if rows:
                        sql_res = {"sql": r.sql,
                                   "columns": [d[0] for d in cur.description],
                                   "rows": rows}
                except Exception:                              # noqa: BLE001
                    pass
            weak = (len(r.entities) >= 2 and not out["has_cooccurrence"]
                    and bool(res))
            packed = build_messages(q, sql_result=sql_res, hits=res,
                                    relation_weak=weak, sql_kind=r.sql_kind)
            if packed["ok"]:
                answered = llm.chat(packed["messages"])
                n_llm += 1
                errs += [f"[生成] {e}" for e in check_generate(answered, packed)]

        # 已知局限不计分，但要显式提醒，不能悄悄跳过
        known = bool(c.get("known_limitation"))
        passed = (not errs) or known

        results.append({"id": c["id"], "q": q, "cat": c.get("cat", ""),
                        "passed": passed, "known": known, "errs": errs})

        mark = ("-- " if known else ("OK " if passed else "NG "))
        if not passed or args.show_ok or known:
            print(f"  {mark}{c['id']:<10} {q[:26]:<28} {c.get('cat', '')}")
            for e in errs:
                print(f"          └─ {e}")
            if known and not errs:
                print("          └─ 已知局限，已按设计不计分")
            if known and errs:
                print("          └─ 已知局限：不计入总分")

    # ---------- 汇总 ----------
    scored = [x for x in results if not x["known"]]
    n_ok = sum(1 for x in scored if x["passed"])
    n_all = len(scored)

    print()
    print("=" * 74)
    print(f"总计  {n_ok}/{n_all}" +
          (f"（另有 {len(results) - n_all} 条已知局限不计分）"
           if len(results) != n_all else ""))
    print("=" * 74)

    cats: dict[str, list[bool]] = {}
    for x in results:
        if x["known"]:
            continue
        cats.setdefault(x["cat"], []).append(x["passed"])
    for cat, ps in sorted(cats.items()):
        bar = "█" * sum(ps) + "░" * (len(ps) - sum(ps))
        print(f"  {cat:<14} {sum(ps)}/{len(ps)}  {bar}")

    # 打印 L3 实际调用次数：如果这个数是 0 或远小于用例数，
    # 说明大部分用例走了拒答分支、L3 等于没测 —— 那种"全绿"没有意义。
    if args.with_llm:
        print(f"\n  L3 实际调用 LLM {n_llm} 次 / {len(cases)} 条用例"
              + ("（其余为代码级拒答，未调 API）" if n_llm < len(cases) else ""))

    # ---------- 基线 ----------
    if args.save:
        BASELINE_PATH.write_text(
            json.dumps({x["id"]: x["passed"] for x in results},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n基线已存 → {BASELINE_PATH.relative_to(ROOT)}")

    if args.diff:
        if not BASELINE_PATH.exists():
            print("\n没有基线，先跑 --save")
        else:
            base = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
            now = {x["id"]: x["passed"] for x in results}
            fixed = [i for i in now if now[i] and not base.get(i, True)]
            broke = [i for i in now if not now[i] and base.get(i, False)]
            print("\n与基线对比：")
            print(f"  新修复 {len(fixed)}：{fixed or '—'}")
            print(f"  新退化 {len(broke)}：{broke or '—'}")
            if broke:
                return 1

    print("\n提示：用例写在 data/eval_cases.json，每条 why 字段说明它守护什么。")
    print("      新增用例时先跑一遍确认期望值，别写「觉得应该对」的值。")
    return 0 if n_ok == n_all else 1


if __name__ == "__main__":
    sys.exit(main())
