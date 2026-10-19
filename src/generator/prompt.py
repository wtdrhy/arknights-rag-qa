"""第 8 步（一）：Prompt 组装 —— RAG 里那个常常被忽略的 "A"。

【这一步在 RAG 全景里的位置】

    用户提问
      ↓
    Query Routing（第 7 步）── 决定走 SQL / 向量 / 两条都走
      ↓
    Retrieval（第 6、7 步）── 拿到 top-k 证据块 + SQL 结果
      ↓
    **Augmentation（本文件）**  ← RAG 的 "A"，就是这里
      ↓
    Generation（src/generator/llm.py）── 调 LLM 出答案

很多人以为 RAG = 检索 + 生成两步。其实中间这段 Augmentation
（证据编号、长度预算、SQL 结果格式化、拒答分支）才是决定效果的地方：

    - 检索到了好证据，但塞不下 → 被截断，等于没检索
    - 检索到了好证据，但没编号 → 模型无法引用，你也无法核实
    - 一条证据都没有，还照样调 LLM → 那就是纯幻觉发生器

【本文件做的四件事】

    1. 证据编号   [1] [2] [3] … 让模型能引用，让你可以溯源
    2. 长度预算   按字符数截断，保证 prompt 不爆、不丢头部的强证据
    3. 分轨格式化 SQL 结果走紧凑表格，文本证据走原文摘录，互不混淆
    4. 拒答分支   证据不足时不生成 prompt，直接返回固定话术（不调 LLM）

【为什么数值类问题要特别处理】
SQL 轨道查出的是**精确值**。LLM 有个坏毛病：看到 "3 个炽合金块"
会顺手帮你"换算"或"估算"成别的数字。所以对这类 prompt 必须明令
禁止重新计算 —— 这条指令写与不写，实测差异非常大。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 系统提示词
# ---------------------------------------------------------------------------
# 写系统提示词的原则：说清「你是谁」「能用什么」「不能用什么」「没得用时怎么办」。
# 不要写「请尽力回答」这种话 —— 它等于鼓励模型编。
SYSTEM_PROMPT = """你是「明日方舟 Wiki 知识助手」，基于罗德岛公开档案资料回答问题。

【你能用的材料】
1. 【数据表】：来自本地结构化数据库的查询结果，是精确值。
2. 【参考资料】：从 Wiki 原文切块检索出来的段落，每条带编号 [1] [2] …

【三条铁律】
1. **只准使用上面两类材料里的信息。** 材料里没写的，一律不许用你自己的
   游戏知识补充。你不确定的，就说不确定。
2. **数值必须照抄【数据表】，禁止重新计算、换算或估算。**
   表里写多少就是多少。即使你认为它算错了，也不要改。
3. **每个关键结论后面标出证据编号**，格式如 [2]。
   一句话综合了多条证据就标多个，如 [1][3]。

【回答风格】
- 中文，简洁，直接给结论，不要复述问题。
- 结构化：多条结论用短句或列表分开，不要写成长段落。
- 长度控制在 200 字以内，除非用户明确要求展开。
"""

# 当证据不足、但又不至于完全无话可说时，追加这条约束
WEAK_EVIDENCE_HINT = """
【特别提醒·证据不充分】
当前资料里，**没有任何一段同时提到这些干员**。
你只能分别介绍每个人各自的档案内容，并且必须在回答开头明确说明：
「资料中没有记载这几位干员之间有直接互动，以下分别介绍各自的情况。」
严禁把两人的档案拼凑成一段互动剧情 —— 那是编造。
"""

# 结果来自聚合 SQL 时追加这条。
# 光靠 SYSTEM_PROMPT 里那句"禁止计算"还不够 —— 模型看到一列数字，
# 仍然会忍不住验算一遍（实测它算出了 8+12+15…=70）。
# 必须明确告诉它：**这一列已经是数据库算好的最终结果**，照抄就行。
AGGREGATE_HINT = """
【特别提醒·数值已由数据库算好】
本次【数据表】里的数量是数据库直接聚合（SUM）得出的**最终值**，
不是待你加工的原始清单。请原样引用，不要再做任何加法、合并或估算。
"""


# 分阶段养成清单（sql_kind="plan"）时追加。
# 清单按"精英化 → 技能 → 专精 → 模组"分阶段，每行标了材料花在哪个阶段；
# 末尾【总计】段是数据库 SUM 好的最终值。
# 第 11 步补丁后等级升满的开销也进了表，旧的"升级消耗不在表内"声明随之作废
# —— 提示词里留着过期的边界说明，会让模型对着有数据的表说"数据缺失"，
#   这类**过期提示**比没有提示更糟：它制造的是假阴性。
PLAN_HINT = """
【特别提醒·分阶段养成清单】
本次【数据表】是一份**分阶段养成清单**，分三层，全部由数据库算好：
① 明细段：每行标明该材料花在哪个具体节点；
② 【小计】段（技能升级2-7级 / 技能N专精 / 模组N）：同一大阶段的合计；
③ 【总计】段：全阶段合计。

阶段顺序就是养成顺序，转述时**不要打乱**：
  【升级】精英0 1→N级  →  精英化1  →  【升级】精英1 1→N级
  →  精英化2  →  【升级】精英2 1→N级
  →  技能升到2~7级  →  各技能专精1~3  →  各模组1~3级
（等级上限随干员星级不同，表里已经是该干员的实际上限。）

转述规则：
- **回答结构固定：先逐阶段清单，最后才给【总计】。**
  不要一上来就甩一个总计（实测模型默认这么做，等于没回答"花在哪个阶段"）。
- **逐阶段列出，一个阶段都不能省**：
  【升级】精英0 → 精英化1 → 【升级】精英1 → 精英化2 → 【升级】精英2
  → 技能升到2~7级 → 技能1/2/3 各自专精 → 模组各自。
- **每个阶段的材料必须写全，禁止用"等""各"省略。**
  用户是照着这张单子刷材料的，漏一项就得再问一次。
- 大阶段（技能升级2-7级、某个技能的全部专精、某个模组的全部等级）
  直接引用【小计】段的一行，不要把逐级的行再抄一遍。
- **凡是跨行合并的数字（如技能2-7级合计）一律引用【小计】段的原值，
  绝对不要自己把几行加起来**——你自己加总容易算错。
- 【总计】段原样引用，不要验算。
- 「经验值」是升级消耗的经验总量，「≈高级作战记录」是数据库
  按 2000 经验/张向上取整算好的张数，两者都直接引用，不要再换算。
- 【总计】里的「龙门币」已包含全部阶段（升级 + 精英化 + 模组），
  不要误以为它只是精英化的花费。
"""


# 只有 SQL 结果、没有文本证据时追加。
NO_CITATION_HINT = """
【特别提醒·本次没有带编号的参考资料】
这次只有【数据表】，没有 [1] [2] 这种编号资料。
**不要编造任何引用编号**，需要说明来源时直接写「根据数据表」即可。
"""

# 完全无证据时的固定话术。注意：这种情况**不调用 LLM**。
def refusal_answer(question: str, entities: list[str] | None = None) -> str:
    """证据为零时的拒答文案。

    为什么不交给 LLM 自由发挥：
    空 context 时模型会调用自己的预训练知识来"补全"，
    那正是幻觉的定义。拒答必须是**代码分支**，不是提示词约束。
    """
    if entities:
        who = "、".join(entities)
        return (f"我在已收录的档案资料里没有找到与「{question}」相关的记载，"
                f"涉及干员：{who}。\n"
                f"可能的原因：该内容不在已抓取的页面范围内，或原始 Wiki 上就没有这一条目。\n"
                f"为避免给出不实信息，我不做推测。你可以换个问法，"
                f"或指定资料类型（档案 / 语音 / 密录 / 技能）。")
    return (f"我在已收录的档案资料里没有找到与「{question}」相关的记载。\n"
            f"为避免给出不实信息，我不做推测。你可以换个问法试试。")


# ---------------------------------------------------------------------------
# 长度预算
# ---------------------------------------------------------------------------
MAX_ROWS = 30            # SQL 明细结果最多给模型看多少行
MAX_EVIDENCE_CHARS = 500  # 单条证据最多多少字
MAX_CONTEXT_CHARS = 6000  # 全部证据的总预算
# 汇总类（GROUP BY）结果的行数上限设得很高，实际上等于不截断。
# 为什么必须区别对待：明细少看几行只是"信息不全"，
# 汇总少看一行，用户拿到的"总共需要多少"就是错的 —— 后者是硬伤。
# 而且汇总表行数天然很少（史尔特尔满练度 34 种材料），根本不占预算。
AGG_ROWS = 500


def _approx_tokens(s: str) -> int:
    """粗估 token 数。中文 1 字 ≈ 0.6~1 token，这里按 0.8 保守估计。

    为什么不用真 tokenizer：聊天模型的 tokenizer 要额外装依赖，
    而 prompt 长度只需要「别爆上下文」这个量级的判断，粗估足够。
    """
    return int(len(s) * 0.8)


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


# ---------------------------------------------------------------------------
# 格式化：SQL 结果
# ---------------------------------------------------------------------------
def fmt_sql_table(sql: str, columns: list[str], rows: list,
                  max_rows: int = MAX_ROWS) -> tuple[str, int]:
    """把 SQL 结果渲染成紧凑 markdown 表格。

    返回 (文本, 被截断掉的行数)。

    用 markdown 表格而不是 CSV：LLM 在预训练里见过海量 markdown 表格，
    对列对齐的理解明显更好；对人来说也更容易核对。
    """
    if not rows:
        return "", 0
    shown = rows[:max_rows]
    cut = len(rows) - len(shown)

    head = "| " + " | ".join(str(c) for c in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join("" if v is None else str(v) for v in r) + " |"
            for r in shown]
    return "\n".join([head, sep] + body), cut


def fmt_plan_table(rows: list) -> str:
    """分阶段养成清单专用渲染：按「阶段」列分组，每组一个小标题 + 一张小表。

    【为什么不复用通用 markdown 表】
    史尔特尔这条有 161 行，平铺成一张表时每行都要重复一遍阶段名。
    实测模型在这种形态下会自己尝试"分组"，结果是：
      ① 把整张表压缩成一个【总计】就交差 —— 用户问的恰恰是各阶段；
      ② 漏掉排在表最前面的等级段（升级经验/龙门币）。
    而"按阶段分组"在代码里就是一次顺序扫描（SQL 已按 ord 排好），
    是**确定性的**；交给模型做就是赌博。

    这跟本项目一以贯之的原则是同一件事：
    能在代码里确定性完成的加工，不要留给模型做。
    """
    if not rows:
        return ""
    out: list[str] = []
    cur = None
    buf: list[str] = []

    def flush() -> None:
        if cur is None:
            return
        out.append(f"**{cur}**")
        out.append("| 材料 | 数量 |")
        out.append("|---|---|")
        out.extend(buf)
        out.append("")

    for r in rows:
        stage = r[0]
        if stage != cur:
            flush()
            cur = stage
            buf = []
        buf.append(f"| {r[1]} | {r[2]} |")
    flush()
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 格式化：文本证据
# ---------------------------------------------------------------------------
def fmt_evidence(hits: list[dict], max_chars: int = MAX_EVIDENCE_CHARS,
                 budget: int = MAX_CONTEXT_CHARS) -> tuple[str, list[dict], int]:
    """把检索结果编成 [1] [2] … 的参考资料块。

    返回 (文本, 引用清单, 因预算被丢弃的条数)。

    chunk 里本来就有 contextual header（【干员：泥岩】【来源：干员密录】），
    第 6 步专门加的，这里直接吃现成的 —— 模型看到 header 就知道这段讲谁、
    出自哪类资料，不容易张冠李戴。
    """
    lines: list[str] = []
    cites: list[dict] = []
    used = 0
    dropped = 0

    for i, h in enumerate(hits, 1):
        meta = h.get("metadata", {}) or {}
        text = _clip(h.get("text", ""), max_chars)
        need = len(text) + 40
        if used + need > budget:
            dropped += len(hits) - i + 1
            break
        used += need
        lines.append(f"[{i}] {text}")
        cites.append({
            "idx": i,
            "operator": meta.get("operator", ""),
            "source": meta.get("source", ""),
            "source_label": meta.get("source_label", ""),
            "id": h.get("id", ""),
            "score": round(float(h.get("score", 0.0)), 4),
            "cooccur": h.get("cooccur", 0),
            "head": text[:60],
        })
    return "\n\n".join(lines), cites, dropped


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------
def build_messages(question: str,
                   sql_result: dict | None = None,
                   hits: list[dict] | None = None,
                   relation_weak: bool = False,
                   sql_kind: str = "") -> dict:
    """把问题 + 证据拼成 OpenAI 消息格式。

    sql_result: {"sql": "...", "columns": [...], "rows": [...]}
    hits:       hybrid.search() 的 results 列表
    sql_kind:   "aggregate" 表示 rows 是聚合结果（禁止截断、提示勿重算）

    返回 {
        "messages": [...],
        "citations": [...],     # 供终端打印引用溯源
        "meta": {...},          # prompt 体积诊断
        "ok": bool,             # False 表示证据不足，不该调 LLM
        "refusal": str | None,
    }
    """
    hits = hits or []
    parts: list[str] = [f"【问题】\n{question}"]
    citations: list[dict] = []
    meta = {"n_rows": 0, "n_evidence": 0, "dropped": 0, "rows_cut": 0}

    # ---- 轨道一：结构化数据 ----
    agg = (sql_kind == "aggregate")
    plan = (sql_kind == "plan")
    if sql_result and sql_result.get("rows"):
        cap = AGG_ROWS if (agg or plan) else MAX_ROWS
        table, cut = fmt_sql_table(sql_result.get("sql", ""),
                                   sql_result.get("columns", []),
                                   sql_result.get("rows", []),
                                   max_rows=cap)
        meta["n_rows"] = len(sql_result["rows"])
        meta["rows_cut"] = cut
        if agg:
            # 汇总值：全量给出，并声明"这是算好的"
            head = ("【数据表·汇总】来自本地数据库，下列数量已由数据库直接聚合，"
                    "是精确最终值，直接引用、不要重新计算：")
            tail = f"\n（共 {meta['n_rows']} 项，已全部列出）"
        elif plan:
            # 分阶段养成清单：行数多但每一行都有用（阶段列是用户要的核心信息），
            # 截断任何一行都等于丢掉"这材料花在哪"的答案，所以同样不截断。
            # 渲染改成按阶段分组的小节（fmt_plan_table），
            # 不再让模型对着一张 161 行的平表自己找阶段。
            table = fmt_plan_table(sql_result.get("rows", [])[:cap])
            head = ("【数据表·分阶段养成清单】来自本地数据库，"
                    "已按培养阶段分组（排序即养成顺序），"
                    "每组末尾的【小计】/【总计】同样是数据库算好的原值：")
            tail = (f"（共 {meta['n_rows']} 行，已全部列出）" if not cut
                    else f"（共 {meta['n_rows']} 行，此处展示前 {cap} 行）")
        else:
            head = "【数据表】来自本地数据库，精确值，直接引用不要重新计算："
            tail = (f"\n（共 {meta['n_rows']} 行，此处展示前 {MAX_ROWS} 行）"
                    if cut else "")
        parts.append(head + "\n" + table + tail)

    # ---- 轨道二：文本证据 ----
    if hits:
        ev, citations, dropped = fmt_evidence(hits)
        meta["n_evidence"] = len(citations)
        meta["dropped"] = dropped
        parts.append("【参考资料】来自 Wiki 原文，每条有编号：\n" + ev)

    # ---- 可答性判定：两条轨道都空 → 拒答，不调 LLM ----
    if meta["n_rows"] == 0 and meta["n_evidence"] == 0:
        return {"messages": [], "citations": [], "meta": meta, "ok": False,
                "refusal": refusal_answer(question)}

    system = SYSTEM_PROMPT
    if relation_weak:
        system += WEAK_EVIDENCE_HINT
    if agg:
        system += AGGREGATE_HINT
    if plan:
        system += PLAN_HINT
    if not citations:
        # 纯 SQL 轨道：根本没有带编号的参考资料。
        # 但 SYSTEM_PROMPT 里写了"每个结论标证据编号"，模型就会硬编一个 [1]
        # —— 实测史尔特尔那条就是这么来的。凭空出现的编号正是幻觉的头号信号，
        # 必须在这里把话说死。
        system += NO_CITATION_HINT

    user = "\n\n".join(parts) + "\n\n【请回答】（记得标注证据编号）"

    meta["prompt_chars"] = len(system) + len(user)
    meta["prompt_tokens_est"] = _approx_tokens(system + user)

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "citations": citations,
        "meta": meta,
        "ok": True,
        "refusal": None,
    }


def render_citations(citations: list[dict]) -> str:
    """把引用清单渲染成一目了然的来源表，供终端打印。"""
    if not citations:
        return "  （无引用）"
    out = []
    for c in citations:
        tag = f" [共现{c['cooccur']}]" if c.get("cooccur", 0) >= 2 else ""
        out.append(f"  [{c['idx']}] {c['source_label'] or c['source']}·"
                   f"{c['operator']}  相似度{c['score']}{tag}")
    return "\n".join(out)
