"""
第 6 步（中）：documents —— 把 SQLite 里的记录，变成"检索用的文档块"。

【为什么要单独这一层，不直接拿表里的内容塞进向量库】
数据库是按**存储**设计的（一条语音一行、一个技能等级一行），
向量库要按**语义**组织（一个 chunk 应该能独立回答一个小问题）。
两者粒度完全不同，直接搬会有三个后果：

1. 语音一条 33 字 → 向量只编码了语气，问"泥岩喜欢什么"永远命中不了
2. 技能一个等级一条 → 10 条描述 90% 重复，检索时互相挤占位置
3. 密录一篇 8000 字 → 超长被截断，后文永远检索不到

所以这一层做的事就是**按语义重新聚合**，再交给 chunking 切块。

【六类文档 + 各自的聚合理由】
| 来源     | 原粒度        | 聚合方式              | 理由 |
|---------|--------------|---------------------|------|
| 档案     | 一条一段      | 一条一块              | 自带标题，语义完整 |
| 语音     | 一条 33 字    | 同干员每 10 条一块     | 短文本要攒够上下文 |
| 密录     | 一篇 8000 字  | 按句切 420 字/重叠 60  | 长文本必切 |
| 悖论模拟 | 一条一段      | 一条一块              | 短且独立 |
| 天赋     | 一阶段一条    | 同天赋跨阶段合并        | 避免重复 |
| 技能     | 一等级一条    | 只取 7 级 + 专精 3      | 10 档描述高度重复 |
| 模组     | 一个模组一条  | 一条一块              | 短且独立 |

【元数据（metadata）为什么重要】
Chroma 支持按 metadata 过滤（where）。有了 operator / source / category
就能做「只在泥岩的资料里搜」「只要档案不要语音」「排除特殊模式干员」。
没有它，检索就只能全库海搜，噪声大且无法解释。
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict

from .chunking import (
    _approx_size,
    chunk_long,
    clean_speaker_tags,
    make_header,
    merge_short,
)

# 短文本合并的档位
VOICE_GROUP = 10          # 每 10 条语音合成一块（≈350 字）
MIN_KEEP = 15             # 少于这个字数的文本不值得单独成块

SOURCE_LABEL = {
    "archive": "人员档案",
    "voice": "语音记录",
    "story": "干员密录",
    "paradox": "悖论模拟",
    "talent": "天赋",
    "skill": "技能",
    "module": "模组",
    # 剧情三类分开：用户问"主线第一章"和"火蓝之心活动"是两种意图，
    # 混在一个 source 里就没法按类型过滤了。
    "lore_main": "主线剧情",
    "lore_side": "支线剧情",
    "lore_set": "故事集",
}

LORE_TYPE2SRC = {"主线": "lore_main", "支线": "lore_side", "剧情": "lore_set"}


# wiki 里的占位符，写进 chunk 就是噪声
_PLACEHOLDERS = {"", "-", "—", "——", "0", "0.0", "None", "无", "null"}


def _s(v) -> str:
    """任何值转字符串再 strip，占位符一律归空。

    为什么需要：SQLite 是弱类型，建库时某些文本列可能混进了数值
    （实测 modules.base_info 就有整数字），直接 .strip() 会抛
    AttributeError: 'int' object has no attribute 'strip'。
    而 "0" / "——" 这类 wiki 占位符进了 chunk 就是纯噪声。
    """
    if v is None:
        return ""
    t = str(v).strip()
    return "" if t in _PLACEHOLDERS else t


def _meta(operator: str, source: str, title: str, rarity, profession,
          category: str, seq: int) -> dict:
    """统一构造 metadata。Chroma 不接受 None，一律给默认值。"""
    return {
        "operator": operator or "",
        "source": source,
        "source_label": SOURCE_LABEL.get(source, source),
        "title": (title or "")[:200],
        "rarity": int(rarity) if rarity else 0,
        "profession": profession or "",
        "category": category or "",
        "seq": seq,
    }


def build_documents(conn: sqlite3.Connection) -> list[dict]:
    """从 SQLite 读出全部长文本，按语义聚合成文档块列表。

    返回 [{"id", "text", "metadata"}, ...]
    """
    conn.row_factory = sqlite3.Row
    # 干员基础信息，供所有来源拼 header
    ops = {r["id"]: r for r in conn.execute(
        "SELECT id, name, rarity, profession, category FROM operators")}

    docs: list[dict] = []
    docs += _docs_archive(conn, ops)
    docs += _docs_voice(conn, ops)
    docs += _docs_story(conn, ops)
    docs += _docs_paradox(conn, ops)
    docs += _docs_talent(conn, ops)
    docs += _docs_skill(conn, ops)
    docs += _docs_module(conn, ops)
    docs += _docs_lore(conn)
    return docs


# ---------------------------------------------------------------------------
# 8) 剧情（主线 / 支线 / 故事集）
# ---------------------------------------------------------------------------
def _chunk_dialog(body: str, size: int = 420, overlap_lines: int = 3) -> list[str]:
    """剧情按**行**切块，而不是按句。

    为什么不用通用的 chunk_long：它是按句切再 "".join 的，
    会把 "临光：……" 这种独占一行的对话全粘成一坨 ——
    "…怎么回事……临光：————你"。检索倒是不受影响，
    但 LLM 读到的上下文没有说话人边界，很容易张冠李戴。

    剧情的每一行本身就是完整语义单元（一句台词或一句旁白），
    按行累积最自然，而且天然保留了换行。
    """
    lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
    if not lines:
        return []
    out: list[str] = []
    cur: list[str] = []
    cur_size = 0
    for ln in lines:
        n = _approx_size(ln)
        if cur and cur_size + n > size:
            out.append("\n".join(cur))
            cur = cur[-overlap_lines:]           # 行级重叠，避免切断一段对白
            cur_size = sum(_approx_size(x) for x in cur)
        cur.append(ln)
        cur_size += n
    if cur:
        out.append("\n".join(cur))
    return [c for c in out if c.strip()]


def _docs_lore(conn) -> list[dict]:
    """剧情切块。

    【为什么要让 operator 字段 = 章节名，而不是干员名】
    检索层支持按 operator 过滤，干员类问题（"泥岩的档案"）会带上
    operator="泥岩"。如果剧情块也把出场角色填进 operator，
    那么每次问干员都会混进一堆剧情片段，把真正的档案挤下去 ——
    **新增数据源把原有检索质量稀释了，这是很典型的回归**。

    所以剧情块的 operator 用章节名。代价是问"泥岩在哪段剧情出场"
    时按人过滤会落空，但正文里本来就有「泥岩：……」这种角色名前缀，
    语义检索照样能命中，够用。

    【header 为什么带上出场角色】
    「临光：阿米娅，带博士离开。」这句的主语在正文里，但一个 chunk
    被切开后可能只剩半句。把整段剧情的出场角色写进 header，
    等于给每个 chunk 都补了一份人物索引。
    """
    out: list[dict] = []
    for r in conn.execute("SELECT * FROM lore ORDER BY id").fetchall():
        src = LORE_TYPE2SRC.get(_s(r["story_type"]), "lore_side")
        body = _s(r["content"])
        if len(body) < MIN_KEEP:
            continue

        chapter = _s(r["chapter"]) or _s(r["group_name"]) or "未分类"
        title = " ".join(x for x in (_s(r["level_code"]), _s(r["level_name"]),
                                     _s(r["part"])) if x)
        speakers = _s(r["speakers"])

        head = f"【章节：{chapter}】【来源：{SOURCE_LABEL[src]}】"
        if title:
            head += f"【关卡：{title}】"
        if speakers:
            head += f"\n出场：{speakers}"

        pieces = _chunk_dialog(body)
        seq = 0
        for i, chunk in enumerate(pieces):
            if len(chunk.strip()) < MIN_KEEP:
                continue
            # 超长兜底，和 _add 里同一个理由：bge 超过 512 token 会静默截断
            subs = (chunk_long(chunk, size=420, overlap=60)
                    if _approx_size(chunk) > MAX_TOKENS else [chunk])
            for j, b in enumerate(subs):
                if len(b.strip()) < MIN_KEEP:
                    continue
                out.append({
                    "id": f"{src}:{r['id']}:{i}.{j}",
                    "text": head + "\n" + b,
                    "metadata": {
                        "operator": chapter,
                        "source": src,
                        "source_label": SOURCE_LABEL[src],
                        "title": (title or chapter)[:200],
                        "rarity": 0,
                        "profession": "",
                        "category": _s(r["story_type"]),
                        "seq": seq,
                        "chapter": chapter,
                    },
                })
                seq += 1
    return out


MAX_TOKENS = 480          # bge 上限 512 token，留 32 给 header


def _add(out: list, docs: list[str], operator: str, source: str, title: str,
         op: sqlite3.Row, prefix: str) -> None:
    """把切好的若干块写进结果集，自动编号。"""
    seq = 0
    for i, chunk in enumerate(docs):
        if len(chunk.strip()) < MIN_KEEP:
            continue
        # 超长兜底：bge 会把超过 512 token 的部分**静默丢弃**（不报错！），
        # 导致这一块的尾部永远检索不到。所以这里再切一次。
        pieces = (chunk_long(chunk, size=420, overlap=60)
                  if _approx_size(chunk) > MAX_TOKENS else [chunk])
        for j, body in enumerate(pieces):
            if len(body.strip()) < MIN_KEEP:
                continue
            out.append({
                "id": f"{source}:{prefix}:{i}.{j}",
                "text": make_header(operator, SOURCE_LABEL[source], title)
                        + "\n" + body,
                "metadata": _meta(operator, source, title, op["rarity"],
                                  op["profession"], op["category"], seq),
            })
            seq += 1


# ---------------------------------------------------------------------------
# 1) 人员档案：一条一块（自带标题，语义完整）
# ---------------------------------------------------------------------------
def _docs_archive(conn, ops) -> list[dict]:
    out: list[dict] = []
    rows = conn.execute("""
        SELECT a.* FROM archive_texts a ORDER BY a.operator_id, a.idx
    """).fetchall()
    for r in rows:
        op = ops.get(r["operator_id"])
        if not op:
            continue
        body = _s(r["content"])
        if len(body) < MIN_KEEP:
            continue
        title = _s(r["title"])
        # 极长档案（如"晋升记录"）按句切，避免超长。
        # 判断用 _approx_size（≈token 数）而不是 len（字符数），
        # 因为纯 ASCII/数字的段落 1600 字符其实只有 400 token，不必切
        pieces = (chunk_long(body, size=420, overlap=60)
                  if _approx_size(body) > MAX_TOKENS else [body])
        _add(out, pieces, op["name"], "archive", title, op,
             f"{r['operator_id']}_{r['idx']}")
    return out


# ---------------------------------------------------------------------------
# 2) 语音：同干员每 10 条合并成一块
# ---------------------------------------------------------------------------
def _docs_voice(conn, ops) -> list[dict]:
    out: list[dict] = []
    by_op: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for r in conn.execute(
            "SELECT * FROM voices ORDER BY operator_id, idx").fetchall():
        by_op[r["operator_id"]].append(r)

    for oid, rows in by_op.items():
        op = ops.get(oid)
        if not op:
            continue
        # 每条带上标题（交谈1 / 信赖提升后交谈2），标题本身就是检索线索
        items = [f"【{_s(r['title'])}】{_s(r['line'])}"
                 for r in rows if _s(r["line"])]
        blocks = merge_short(items, max_size=600, min_size=120)
        _add(out, blocks, op["name"], "voice", f"共{len(rows)}条台词", op, str(oid))
    return out


# ---------------------------------------------------------------------------
# 3) 干员密录：长文本按句切块
# ---------------------------------------------------------------------------
def _docs_story(conn, ops) -> list[dict]:
    out: list[dict] = []
    name_to_op = {op["name"]: op for op in ops.values()}
    for r in conn.execute(
            "SELECT * FROM stories ORDER BY id").fetchall():
        op = name_to_op.get(r["operator"])
        if not op:
            continue
        body = clean_speaker_tags(_s(r["content"]))
        pieces = chunk_long(body, size=420, overlap=60)
        _add(out, pieces, op["name"], "story", r["page"], op,
             f"{op['id']}_{r['id']}")
    return out


# ---------------------------------------------------------------------------
# 4) 悖论模拟：一条一块
# ---------------------------------------------------------------------------
def _docs_paradox(conn, ops) -> list[dict]:
    out: list[dict] = []
    for r in conn.execute("SELECT * FROM paradox").fetchall():
        op = ops.get(r["operator_id"])
        if not op:
            continue
        body = _s(r["description"])
        if len(body) < MIN_KEEP:
            continue
        _add(out, [body], op["name"], "paradox", _s(r["name"]), op,
             f"{r['operator_id']}")
    return out


# ---------------------------------------------------------------------------
# 5) 天赋：同一天赋跨精英阶段合并，避免重复
# ---------------------------------------------------------------------------
def _docs_talent(conn, ops) -> list[dict]:
    out: list[dict] = []
    by_key: dict[tuple, list[sqlite3.Row]] = defaultdict(list)
    for r in conn.execute(
            "SELECT * FROM talents ORDER BY operator_id, idx").fetchall():
        by_key[(r["operator_id"], r["name"])].append(r)

    for (oid, tname), rows in by_key.items():
        op = ops.get(oid)
        if not op:
            continue
        lines = []
        for r in rows:
            cond = _s(r["unlock_cond"])
            eff = _s(r["effect"])
            if not eff:
                continue
            lines.append(f"{cond}：{eff}" if cond else eff)
        if not lines:
            continue
        # 去重（不同阶段效果完全相同时只留一条）
        uniq = list(dict.fromkeys(lines))
        # id 必须稳定：不要用 hash()，Python 的 str hash 带随机种子，
        # 每次运行都会变，重建库时会写出一批 id 不同的重复块
        _add(out, ["\n".join(uniq)], op["name"], "talent", _s(tname), op,
             f"{oid}_{rows[0]['idx']}")
    return out


# ---------------------------------------------------------------------------
# 6) 技能：只取 7 级 + 专精 3，避免 10 档描述互相挤占
# ---------------------------------------------------------------------------
def _docs_skill(conn, ops) -> list[dict]:
    out: list[dict] = []
    # 一次把 1 万条等级描述捞出来分组，不要每个技能查一次库（1012 次往返）
    by_skill: dict[int, dict[str, sqlite3.Row]] = defaultdict(dict)
    for x in conn.execute("SELECT * FROM skill_levels").fetchall():
        by_skill[x["skill_id"]][x["level"]] = x

    rows = conn.execute("""
        SELECT s.*, o.id AS oid
        FROM skills s JOIN operators o ON o.id = s.operator_id
        ORDER BY s.operator_id, s.idx
    """).fetchall()
    for r in rows:
        op = ops.get(r["oid"])
        if not op:
            continue
        lv = by_skill.get(r["id"], {})
        sp, trig = _s(r["sp_type"]), _s(r["trigger_type"])
        parts = [f"技能名：{_s(r['name'])}"]
        if sp or trig:
            parts.append(f"回复方式/触发方式：{sp}{'/' if sp and trig else ''}{trig}")
        for key in ("7", "专精3"):
            if key in lv and lv[key]["desc"]:
                d = lv[key]
                extra = []
                if d["init_sp"] is not None:
                    extra.append(f"初始{d['init_sp']}")
                if d["sp_cost"] is not None:
                    extra.append(f"消耗{d['sp_cost']}")
                if d["duration"] is not None:
                    extra.append(f"持续{d['duration']}")
                tail = f"（{' '.join(extra)}）" if extra else ""
                parts.append(f"{key}级效果：{d['desc']}{tail}")
        body = "\n".join(parts)
        if len(body) < MIN_KEEP:
            continue
        _add(out, [body], op["name"], "skill", _s(r["name"]), op,
             f"{r['oid']}_{r['idx']}")
    return out


# ---------------------------------------------------------------------------
# 7) 模组：特性 + 基础信息
# ---------------------------------------------------------------------------
def _docs_module(conn, ops) -> list[dict]:
    out: list[dict] = []
    for r in conn.execute(
            "SELECT * FROM modules ORDER BY operator_id, idx").fetchall():
        op = ops.get(r["operator_id"])
        if not op:
            continue
        parts = []
        if _s(r["name"]):
            parts.append(f"模组名：{_s(r['name'])}")
        if _s(r["type"]):
            parts.append(f"类型：{_s(r['type'])}")
        for k in ("trait", "trait_add", "base_info"):
            v = _s(r[k])
            if v:
                parts.append(v)
        body = "\n".join(parts)
        if len(body) < MIN_KEEP:
            continue
        _add(out, [body], op["name"], "module", r["name"] or "", op,
             f"{r['operator_id']}_{r['idx']}")
    return out


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def summarize(docs: list[dict]) -> dict:
    """给文档集画像：来源分布、长度分布、干员覆盖数。"""
    from collections import Counter
    by_src = Counter(d["metadata"]["source"] for d in docs)
    lens = sorted(len(d["text"]) for d in docs)
    ops = {d["metadata"]["operator"] for d in docs}
    return {
        "总块数": len(docs),
        "覆盖干员数": len(ops),
        "来源分布": dict(by_src.most_common()),
        "长度_最短": lens[0] if lens else 0,
        "长度_中位": lens[len(lens) // 2] if lens else 0,
        "长度_最长": lens[-1] if lens else 0,
        "字数合计": sum(lens),
    }
