"""混合检索：稠密（向量）+ 稀疏（BM25）两路召回，再融合重排。

【为什么需要两路】
第 6 步只有稠密一路，实测暴露了两个硬伤：

    1. "泥岩和可露希尔是什么关系" —— top50 里 0 个块同时提到两人。
       向量把两个名字各自的语义都编码了，但"两人同时出现"这个
       字面事实被平均掉了。
    2. "哪些技能可以让敌人晕眩" —— 晕眩是能字面匹配到的词，
       纯语义反而串到了天赋字段。

    稀疏路正好补这两个洞：它认字，不认语义。

【融合为什么用 RRF 而不是加权求和】
    稠密分数是余弦相似度（-1~1），稀疏分数是 BM25（0~30+），
    量纲完全不同，加权求和要先归一化，而归一化参数极难调 ——
    换一批数据就失效。

    RRF（Reciprocal Rank Fusion）只看名次：

        score(d) = Σ  1 / (k + rank_i(d))        k 通常取 60
                  i∈路

    一篇文档在两路里都排第 1 → 1/61 + 1/61 = 0.0328
    排第 50 和第 3        → 1/110 + 1/63  = 0.0249

    优点：无量纲、无参数、对单路分数分布不敏感。
    这是 Elasticsearch / Vespa / 主流 RAG 框架的默认做法。

【第三路：共现召回 —— 专门治"多实体关系查询"】
    用户问"A 和 B 是什么关系"时，只有同时提到 A、B 的块才是真证据。

    踩过的两个坑（都是实测出来的，别再走一遍）：

    坑 1：以为融合后加个共现加成分就行。
          加性（+0.02）在单路时够用，两路 RRF 叠加后基线分数翻倍就被稀释；
          改乘性（×1.6）也不够 —— 共现块往往语义上并不相似
          （那句台词讲的是盔甲，不是"关系"），稠密路只给它排到 #186，
          再乘 1.6 也追不上排 #1 的块。

    坑 2：以为 BM25 会顺带把共现块排上来。
          不会。BM25 是词频加权求和："可露希尔"在某篇密录里出现 20 次，
          得分远高于"泥岩 + 可露希尔各出现 1 次"。实测全库唯一那块
          同时提到两人的 chunk，BM25 排不进 top10。

    正确做法：共现**单独成一路**，直接按"同时提到几个实体"检索，
    再参与 RRF。它和稠密/稀疏是并列的证据类型，不是后处理技巧。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..store.vector import (
    DEFAULT_COLLECTION,
    DEFAULT_MODEL,
    get_collection,
    search as dense_search,
)

RRF_K = 60          # RRF 平滑常数，业界惯例 60
POOL = 200          # 每路先召回这么多条，再融合取 top-k

# 三路权重。含义：每一路代表一种**证据类型**，权重即这种证据的可信度。
#
#   dense   1.0  语义相关
#   sparse  0.7  字面精确命中（实测给到 1.0 会压过语义，Recall@1 掉 10 个点）
#   cooccur 1.5  同时提到多个实体 —— 对"关系类问题"是决定性证据，所以最高
# 三路权重。0.7 这个 sparse 值是第 7 步在**只有干员数据**时定的，
# 第 10 步补进 19,797 块剧情后它明显过重，必须重新校准 ——
# 权重不是一次调完就永远对，语料一变就得重扫。
#
# 实测扫描（同一批探针，只改 W_SPARSE）：
#   0.7 → R@1 60%  R@5 100%  MRR 73.7%     ← 旧值，被剧情稀释
#   0.5 → R@1 70%  R@5 100%  MRR 81.7%
#   0.3 → R@1 90%  R@5 100%  MRR 93.3%     ← 采用
#   0.2 → R@1 90%  R@5 100%  MRR 95.0%
#   0.0 → R@1 80%  R@5  80%  MRR 80.0%     ← 关键：去掉 sparse 反而掉 20 个点
#
# 为什么稀释：剧情里角色名高频重复（"阿米娅："出现成千上万次），
# BM25 靠词频打分，会把大量剧情台词块顶到前排，挤掉真正的档案块。
#
# 为什么不能直接归零：最后一行是证据 —— 没有 sparse 时 R@5 从 100% 掉到 80%，
# 说明它捞回了 dense 漏掉的那部分。**稀疏路该被削弱，不该被移除。**
# 0.2 与 0.3 都在最优平台区，取 0.3 更保守：给稀疏路留一点话语权，
# 免得将来语料再变时它彻底失去补位能力。
W_DENSE, W_SPARSE, W_COOC = 1.0, 0.3, 1.5


class HybridRetriever:
    """把 Chroma（稠密）和 BM25（稀疏）绑成一个检索器。

    一次性把 16K 个 chunk 的文本读进内存（约 7MB），
    之后每次检索都不再访问磁盘 —— 这是检索层的常规做法。
    """

    def __init__(self, chroma_dir: Path, name: str = DEFAULT_COLLECTION,
                 model: str = DEFAULT_MODEL, bm25_path: Path | None = None,
                 verbose: bool = True):
        from .lexical import BM25

        self.chroma_dir = Path(chroma_dir)
        self.col = get_collection(self.chroma_dir, name=name, model=model)
        self.bm25_path = Path(bm25_path) if bm25_path else \
            self.chroma_dir.parent / "bm25.pkl"

        # ---- 把 chunk 全部拉出来，建立 idx <-> chroma_id 的映射 ----
        # 必须分批。第 10 步补进剧情后块数从 16,562 涨到 36,259，
        # 一次性 get() 直接报 "too many SQL variables" ——
        # Chroma 底层把 ids 拼进 SQL 的 IN (...)，SQLite 的变量数有上限。
        # 这种 bug 只在数据量跨过某个阈值时才冒出来，小样本测试永远碰不到。
        self.ids, self.texts, self.metas = [], [], []
        offset, step = 0, 5000
        while True:
            got = self.col.get(include=["documents", "metadatas"],
                               limit=step, offset=offset)
            if not got["ids"]:
                break
            self.ids.extend(got["ids"])
            self.texts.extend(got["documents"])
            self.metas.extend(got["metadatas"])
            offset += len(got["ids"])
            if len(got["ids"]) < step:
                break
        self.id2idx = {cid: i for i, cid in enumerate(self.ids)}
        if verbose:
            print(f"    Chroma 载入 {len(self.ids):,} 块")

        # ---- BM25 索引：有缓存就直接用，没有就建 ----
        if self.bm25_path.exists():
            self.bm25 = BM25.load(self.bm25_path)
            if self.bm25.n_docs != len(self.texts):
                # 块数对不上说明 Chroma 重建过而缓存没更新，必须重来
                if verbose:
                    print("    BM25 缓存与当前块数不符，重建…")
                self.bm25 = None
            elif verbose:
                print(f"    BM25 缓存载入（{self.bm25.n_docs:,} 篇）")
            if self.bm25 is None:
                self._build_bm25(verbose)
        else:
            self._build_bm25(verbose)

    def _build_bm25(self, verbose: bool) -> None:
        from .lexical import BM25
        self.bm25 = BM25().fit(self.texts, verbose=verbose)
        self.bm25.save(self.bm25_path)
        if verbose:
            print(f"    已缓存 → {self.bm25_path}")

    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 5,
               operator: str | list[str] | None = None,
               source: str | None = None,
               entities: list[str] | None = None,
               entity_groups: list[list[str]] | None = None,
               mode: str = "hybrid",
               cooc: bool = True,
               pool: int = POOL) -> dict:
        """检索，返回一个带诊断信息的字典。

        operator: 单个干员名，或一组名字（异格干员要搜全部形态时用列表）。
        entities: 实体名列表（旧接口，每个实体一个词）。
        entity_groups: 每个实体的**全部词形**，如 [["泥岩","Mudrock"],
                       ["逻各斯","Logos","ロゴス"]]。共现按「实体」计数，
                       传了它才能让"精英干员Logos"被算作提到逻各斯。
        mode: "dense" | "sparse" | "hybrid"（做评测对比时用）
        """
        entities = [e for e in (entities or []) if e]
        # 分组是核心：共现必须按实体计数，不能按词计数。
        # 一段文本同时出现"泥岩"和"Mudrock"仍然只提到 1 个实体。
        groups = [g for g in (entity_groups or []) if g]
        if not groups and entities:
            groups = [[e] for e in entities]
        flat: list[str] = []
        for g in groups:
            for t in g:
                if t and t not in flat:
                    flat.append(t)
        ops = [operator] if isinstance(operator, str) else operator

        # ---- 稠密路：交给 Chroma，元数据过滤在这里生效（pre-filter）----
        dense_rank: list[tuple[int, float]] = []
        if mode in ("dense", "hybrid"):
            if ops and len(ops) > 1:
                # 多个形态（如 阿米娅 / 阿米娅(近卫)）→ Chroma 的 $in 条件
                cond: dict = {"operator": {"$in": list(ops)}}
                if source:
                    cond = {"$and": [cond, {"source": source}]}
                raw = self.col.query(query_texts=[query], n_results=pool,
                                     where=cond)
                dense_rank = [(self.id2idx[i], 1 - d) for i, d in
                              zip(raw["ids"][0], raw["distances"][0])
                              if i in self.id2idx]
            else:
                # Chroma 的 where 只接受标量，单元素列表要还原成字符串
                res = dense_search(self.col, query, k=pool, source=source,
                                   operator=ops[0] if ops else None)
                dense_rank = [(self.id2idx[r["id"]], r["score"])
                              for r in res if r["id"] in self.id2idx]

        # ---- 稀疏路：BM25 全库打分，再本地套用同样的过滤条件 ----
        sparse_rank: list[tuple[int, float]] = []
        if mode in ("sparse", "hybrid"):
            allowed = self._allowed(ops, source)
            # 稀疏路**只对实体名检索**，不用整句。
            # 原因（实测）：整句里的"之间""发生了什么"这类泛词会主导 BM25
            # 打分，字面第一名叫到无关的"干员密录·凯尔希"。
            # 稀疏路的职责就是精确命中专有名词，语义归稠密路管。
            # 用全部词形（含英文代号/日文名）去查，命中面更大
            lxq = " ".join(flat) if flat else query
            for idx, sc in self.bm25.search(lxq, topk=pool * 3):
                if allowed is not None and idx not in allowed:
                    continue
                sparse_rank.append((idx, sc))
                if len(sparse_rank) >= pool:
                    break

        # ---- 第三路：共现召回 ----
        # 为什么不能靠 BM25 顺带实现：BM25 是词频加权求和，
        # "可露希尔"在某篇里出现 20 次，得分远高于"泥岩+可露希尔各 1 次"。
        # 实测全库唯一那块同时提到两人的 chunk，BM25 排不进 top10。
        # 所以共现必须单独成一路，直接按"同时提到几个实体"检索。
        cooc_rank: list[tuple[int, float]] = []
        if cooc and len(groups) >= 2:
            cooc_rank = self._cooccur_rank(groups, limit=pool)

        # ---- 融合（加权 RRF：每一路代表一种证据类型）----
        lanes = [lst for lst in (dense_rank, sparse_rank, cooc_rank) if lst]
        wts = [w for lst, w in ((dense_rank, W_DENSE), (sparse_rank, W_SPARSE),
                                (cooc_rank, W_COOC)) if lst]
        fused = rrf(lanes, weights=wts)

        order = sorted(fused, key=lambda i: -fused[i])[:k]

        out = [{
            "id": self.ids[i],
            "text": self.texts[i],
            "metadata": self.metas[i],
            "score": fused[i],
            "dense_rank": _rank_of(dense_rank, i),
            "sparse_rank": _rank_of(sparse_rank, i),
            "cooc_rank": _rank_of(cooc_rank, i),
            # 按「实体」计数：同一实体的多个词形只算一次
            "cooccur": sum(1 for g in groups
                           if any(t in self.texts[i] for t in g))
                       if groups else 0,
        } for i in order]

        return {
            "results": out,
            "query": query,
            "entities": entities,
            "entity_groups": groups,
            "mode": mode,
            # 下面这两个字段是给 Agent 做「能不能回答」判断用的
            "has_cooccurrence": any(r["cooccur"] >= 2 and len(groups) >= 2
                                    for r in out),
            "n_dense": len(dense_rank),
            "n_sparse": len(sparse_rank),
        }

    def _cooccur_rank(self, groups: list[list[str]],
                      limit: int = 100) -> list[tuple[int, float]]:
        """找出所有同时提到 >=2 个实体的块，按"提到的实体数"降序。

        groups 里每个元素是一个实体的全部词形；命中任一词形即算提到该实体，
        但同一实体只计 1 分（否则"泥岩"+"Mudrock"会被误判成两个实体）。

        分数用命中的实体数（2, 3, 4…），这样 RRF 只看名次也有区分度。
        次级排序用块长度升序 —— 同样提到两人，更短的块通常更聚焦。
        """
        cands: list[tuple[int, int, int]] = []   # (命中数, 长度, idx)
        for i, t in enumerate(self.texts):
            n = sum(1 for g in groups if any(term in t for term in g))
            if n >= 2:
                cands.append((n, len(t), i))
        if not cands:
            return []
        cands.sort(key=lambda x: (-x[0], x[1]))
        return [(i, float(n)) for n, _ln, i in cands[:limit]]

    def _allowed(self, operator: list[str] | None,
                 source: str | None) -> set[int] | None:
        """本地过滤集合。None 表示不过滤。"""
        if not operator and not source:
            return None
        s = set()
        opset = set(operator) if operator else None
        for i, m in enumerate(self.metas):
            if opset and m.get("operator") not in opset:
                continue
            if source and m.get("source") != source:
                continue
            s.add(i)
        return s


def rrf(rank_lists: list[list[tuple[int, float]]], k: int = RRF_K,
        weights: tuple[float, ...] | None = None) -> dict[int, float]:
    """加权 Reciprocal Rank Fusion：只认名次，不认分数。

        score(d) = Σ  w_i / (k + rank_i(d))

    weights 允许给某一路更高的权重。默认等权（即标准 RRF）。
    """
    if weights is None:
        weights = (1.0,) * len(rank_lists)
    scores: dict[int, float] = defaultdict(float)
    for w, lst in zip(weights, rank_lists):
        for rank, (idx, _score) in enumerate(lst, 1):
            scores[idx] += w / (k + rank)
    return dict(scores)


def _rank_of(lst: list[tuple[int, float]], idx: int) -> int:
    """返回 idx 在这路结果里的名次（1-based），没进榜返回 0。"""
    for i, (j, _) in enumerate(lst, 1):
        if j == idx:
            return i
    return 0
