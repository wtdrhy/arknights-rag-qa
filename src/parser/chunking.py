"""
第 6 步（上）：chunking —— 把长文本切成"检索友好"的小块。

【这一步在 RAG 全景里的位置】
    离线索引构建 Indexing = 抓取 → 解析 → 入库(SQL) → **切块 → 向量化 → 写向量库**
    在线检索 Retrieval    = 用户提问 → 向量化 → 近似最近邻搜索 → 拼进 prompt

【为什么必须切块】
embedding 模型有输入长度上限（bge-small-zh 是 512 token ≈ 380~450 汉字）。
一篇密录 8000 字直接塞进去会被截断，只剩开头一小段，后文永远检索不到。

【切块的三个敌人】
1. 切太碎：一句话被劈成两半，语义不完整，检索到也读不懂
2. 切太整：一个 chunk 塞进 3000 字，向量被"平均化"，什么主题都不突出
3. 丢上下文：chunk 里只有正文，检索到之后你不知道它属于哪个干员、哪种资料
   → 解决：把干员名、来源、标题**写进 chunk 正文的开头**（contextual chunk headers）
     实测这对"泥岩的档案里怎么描述她的过去"这类问题的命中率提升很明显

【本项目的六类文档，六套切法】
    短且自带标题（档案/悖论/模组） → 一条一块，不切
    短且极碎（语音台词 30 字）     → 按干员合并若干条成一块
    长叙述（密录 8000 字）        → 按句切 ~450 字，重叠 60 字
    重复度高（技能 10 个等级）     → 只取关键档位，避免稀释语义
"""

from __future__ import annotations

import re

# 句子分隔符：中文句末 + 换行。换行优先，因为 wiki 正文常一行一句
_SENT_SPLIT = re.compile(r"(?<=[。！？…；])\s*|\n+")

# 章节标题行，形如 [name="萨卡兹佣兵"] —— 密录正文里的人物标记，
# 保留它，检索"萨卡兹佣兵说了什么"时能直接命中
# 注意必须带 re.M：密录正文是多行的，[name=...] 出现在每一行的行首，
# 没有 MULTILINE 的话 ^ 只匹配整个字符串的开头，一行都替换不掉
_SPEAKER_RE = re.compile(r"^\s*\[name=\"([^\"]+)\"\]\s*", re.M)


def split_sentences(text: str) -> list[str]:
    """按句切分，保留标点。"""
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text)]
    return [p for p in parts if p]


def _approx_size(s: str) -> int:
    """估算 token 数：中文按 1 字≈1 token，英文/数字按 4 字符≈1 token。

    不用真 tokenizer 是为了零依赖；bge 的中文 tokenizer 大致就是这个比例。
    """
    if not s:
        return 0
    cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    return cjk + (len(s) - cjk) / 4


def chunk_long(text: str, size: int = 420, overlap: int = 60) -> list[str]:
    """长文本按句切块，块间保留 overlap 个"字"的重叠。

    为什么要重叠：
        答案可能刚好跨在切点两侧。没有重叠的话，检索到的 chunk
        永远缺半句话，LLM 读到的上下文是残缺的。
    为什么按句切而不是硬按字数切：
        硬切会把 "泥岩的种族是萨卡兹" 切成 "泥岩的种" + "族是萨卡兹"，
        后者几乎不可能被检索到。
    """
    sents = split_sentences(text)
    if not sents:
        return []
    # 单句就超长（罕见）：按字数硬切兜底
    if _approx_size(sents[0]) > size * 1.5:
        return _hard_split(text, size, overlap)

    chunks: list[str] = []
    cur: list[str] = []
    cur_size = 0
    for s in sents:
        n = _approx_size(s)
        if cur and cur_size + n > size:
            chunks.append("".join(cur))
            # 回退 overlap 个"字"作为下一块的开头
            tail: list[str] = []
            back = 0
            for t in reversed(cur):
                tail.insert(0, t)
                back += _approx_size(t)
                if back >= overlap:
                    break
            cur, cur_size = tail, back
        cur.append(s)
        cur_size += n
    if cur:
        chunks.append("".join(cur))
    return [c for c in chunks if c.strip()]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """没有句子边界时的兜底：按字数硬切。"""
    step = max(1, int(size - overlap))
    out = []
    i = 0
    while i < len(text):
        out.append(text[i:i + int(size)])
        i += step
    return out


def merge_short(items: list[str], max_size: int = 600, min_size: int = 120) -> list[str]:
    """把多条短文本合并成块，避免"一句一块"导致的碎片化。

    为什么短文本也要合并：
        一条语音 30 字，单独成块的话向量几乎只编码了"语气"，
        而用户问的是"泥岩喜欢什么"——需要连续几句台词才构成答案。
    min_size 的意义：攒够 min_size 才收一块，否则继续攒；
    但攒到 max_size 必须断，防止一块过长。
    """
    blocks: list[str] = []
    cur: list[str] = []
    cur_size = 0
    for it in items:
        it = (it or "").strip()
        if not it:
            continue
        cur.append(it)
        cur_size += _approx_size(it)
        if cur_size >= max_size:
            blocks.append("\n".join(cur))
            cur, cur_size = [], 0
    if cur and (cur_size >= min_size or not blocks):
        blocks.append("\n".join(cur))
    elif cur and blocks:
        # 尾巴太短，塞进上一块（宁可略超 max_size，也不留碎片）
        blocks[-1] = blocks[-1] + "\n" + "\n".join(cur)
    return [b for b in blocks if b.strip()]


def make_header(operator: str, source: str, title: str = "") -> str:
    """生成 chunk 头部的上下文标记。

    例：【干员：泥岩】【来源：干员密录】【标题：泥岩/干员密录/1】
    作用：让 embedding 把"这块属于谁、属于哪类资料"也编码进去，
    检索"泥岩的过去"时，即使正文里没出现"泥岩"两字也能命中。
    """
    parts = [f"【干员：{operator}】", f"【来源：{source}】"]
    if title:
        parts.append(f"【标题：{title}】")
    return "".join(parts)


def clean_speaker_tags(text: str) -> str:
    """密录正文里的 [name="XXX"] 转成自然语言「XXX：」，更利于 embedding 理解。"""
    return _SPEAKER_RE.sub(lambda m: f"{m.group(1)}：", text or "")
