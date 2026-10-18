"""稀疏检索：BM25。

【这是什么】
RAG 里有两种"找东西"的方式：

    稠密检索（第 6 步的 Chroma）
        把整段话压成一个向量，比"意思像不像"。
        优点：问"她为什么戴头盔"能找到写盔甲的段落。
        缺点：把"泥岩"这个名字也平均掉了 —— 一个 456 字的块里
              出现一次"可露希尔"，对向量的影响微乎其微。

    稀疏检索（本文件）
        就认字面上有没有这个词。BM25 是 TF-IDF 的现代版本，
        Elasticsearch / Lucene 默认就用的它。
        优点："泥岩""D32钢""炽合金块" 这种专有名词能精确命中。
        缺点：完全不懂语义，问"性格"搜不到"开朗"。

【为什么中文不用分词器】
jieba 会把"可露希尔"切成"可露 / 希尔"或者干脆不认识。
我们改用**字符 2-gram**：把"泥岩和可露希尔"拆成
    ["泥岩", "岩和", "和可", "可露", "露希", "希尔"]
这样"泥岩""希尔"都在索引里，两字人名、三字人名、材料名一视同仁，
而且零依赖（不需要 jieba，不需要额外下载模型）。

【BM25 公式】（了解即可，代码已实现）
                        tf × (k1 + 1)
    score(D, Q) = Σ  IDF(t) × ─────────────────────────
                     t∈Q      tf + k1 × (1 - b + b × |D|/avgdl)

    IDF(t)  这个词越稀有，分越高（"的"到处都是 → 分低；"炽合金块" → 分高）
    tf      这个词在这篇文档里出现几次（有饱和，出现 10 次不等于 10 倍重要）
    |D|/avgdl  长文档惩罚：同样出现一次，短文档比长文档更"命中"
    k1=1.5, b=0.75  业界默认值
"""

from __future__ import annotations

import re
from array import array
from collections import Counter
from pathlib import Path

import numpy as np

# 只保留中日韩文字 + 字母 + 数字，其余（标点、空格、换行）一律丢弃。
# 丢弃标点是必要的：否则 "泥岩。" 和 "泥岩" 会变成两个不同的词。
_KEEP = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7afA-Za-z0-9]")


def tokenize(text: str) -> list[str]:
    """把文本切成 2-gram 词元。

    >>> tokenize("泥岩，重装")
    ['泥岩', '重装']
    """
    chars = _KEEP.findall(text or "")
    if len(chars) < 2:
        # 单字文本（极端情况）退化成 unigram，保证不会因为太短而索引不到
        return chars
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


class BM25:
    """一个够用、可持久化、纯 numpy 的 BM25 索引。

    用法：
        bm = BM25()
        bm.fit(texts)                 # 建索引
        bm.search("泥岩 可露希尔", 50)  # -> [(doc_idx, score), ...]
        bm.save(path) / BM25.load(path)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.postings: list[array] = []   # vocab_id -> 出现过的 doc 编号
        self.tfs: list[array] = []        # vocab_id -> 对应的出现次数
        self.dl: np.ndarray | None = None  # 每篇文档的词元数
        self.avgdl = 0.0
        self.idf: np.ndarray | None = None
        self.n_docs = 0

    # ------------------------------------------------------------------
    # 建索引
    # ------------------------------------------------------------------
    def fit(self, texts: list[str], verbose: bool = True) -> "BM25":
        self.n_docs = len(texts)
        self.dl = np.zeros(self.n_docs, dtype=np.int32)

        # 先临时用 dict[term] -> (list[doc], list[tf])，最后再转成紧凑数组
        tmp_post: dict[str, array] = {}
        tmp_tf: dict[str, array] = {}

        for doc_id, text in enumerate(texts):
            cnt = Counter(tokenize(text))
            self.dl[doc_id] = sum(cnt.values())
            for term, c in cnt.items():
                p = tmp_post.get(term)
                if p is None:
                    p = tmp_post[term] = array("I")   # 4 字节无符号整数
                    t = tmp_tf[term] = array("B")     # 1 字节，tf 上限 255
                else:
                    t = tmp_tf[term]
                p.append(doc_id)
                t.append(c if c < 255 else 255)

        # 固化词表顺序（dict 保持插入顺序，所以这一步是确定性的）
        self.vocab = {t: i for i, t in enumerate(tmp_post)}
        self.postings = [tmp_post[t] for t in self.vocab]
        self.tfs = [tmp_tf[t] for t in self.vocab]

        self.avgdl = float(self.dl.mean()) if self.n_docs else 1.0

        # IDF: log(1 + (N - df + 0.5) / (df + 0.5))
        # 这是 BM25 常用的平滑版本，保证 df 很大时 IDF 也不会变成负数
        df = np.array([len(p) for p in self.postings], dtype=np.float64)
        self.idf = np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

        if verbose:
            print(f"    BM25 索引：{self.n_docs:,} 篇 / 词表 {len(self.vocab):,} "
                  f"/ 平均长度 {self.avgdl:.0f}")
        return self

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(self, query: str, topk: int = 50) -> list[tuple[int, float]]:
        """返回 [(doc_idx, score)]，按分数降序，最多 topk 条。"""
        if self.idf is None:
            raise RuntimeError("请先调用 fit() 或 load() 建索引")

        qc = Counter(tokenize(query))
        if not qc:
            return []

        hits: list[tuple[int, float]] = []
        # 单字查询（比如只输了一个"岩"）切不出 bigram，退化成前缀扫描
        if len(qc) == 1 and len(next(iter(qc))) == 1:
            return self._search_single(query, topk)

        scores = np.zeros(self.n_docs, dtype=np.float32)
        dl = self.dl
        for term, qtf in qc.items():
            tid = self.vocab.get(term)
            if tid is None:
                continue          # 这个词库里压根没出现过
            docs = np.frombuffer(self.postings[tid], dtype=np.int32)
            tf = np.frombuffer(self.tfs[tid], dtype=np.uint8).astype(np.float32)
            if qtf > 1:
                # 查询里重复出现的词，也按同样的饱和规则处理
                qtf = 1 + np.log(qtf)
            # 长文档惩罚项
            norm = 1.0 - self.b + self.b * (dl[docs] / self.avgdl)
            s = self.idf[tid] * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
            np.add.at(scores, docs, s)

        nz = np.nonzero(scores)[0]
        if len(nz) == 0:
            return []
        if len(nz) > topk:
            # argpartition 只排前 topk，比全排快很多（16K 篇差异不明显，
            # 但换成百万篇时这一步是关键）
            part = np.argpartition(-scores[nz], topk)[:topk]
            nz = nz[part]
        order = nz[np.argsort(-scores[nz], kind="stable")]
        return [(int(i), float(scores[i])) for i in order[:topk]]

    def _search_single(self, query: str, topk: int) -> list[tuple[int, float]]:
        """单字查询兜底：找出所有包含该字的 bigram，取它们命中的文档。"""
        ch = _KEEP.findall(query)
        if not ch:
            return []
        ch = ch[0]
        scores: dict[int, float] = {}
        for term, tid in self.vocab.items():
            if ch not in term:
                continue
            docs = np.frombuffer(self.postings[tid], dtype=np.int32)
            for d in docs[:2000]:        # 高频词只取前 2000，避免爆炸
                scores[int(d)] = scores.get(int(d), 0.0) + float(self.idf[tid])
        return sorted(scores.items(), key=lambda x: -x[1])[:topk]

    # ------------------------------------------------------------------
    # 持久化：建索引要几十秒，存下来避免每次重跑
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({
                "k1": self.k1, "b": self.b, "vocab": self.vocab,
                "postings": self.postings, "tfs": self.tfs,
                "dl": self.dl, "avgdl": self.avgdl, "idf": self.idf,
                "n_docs": self.n_docs,
            }, f, protocol=4)

    @classmethod
    def load(cls, path: Path) -> "BM25":
        import pickle
        obj = cls.__new__(cls)
        with Path(path).open("rb") as f:
            d = pickle.load(f)
        obj.__dict__.update(d)
        return obj
