"""
第 6 步（下）：Chroma 向量库封装 —— embedding + 写入 + 检索

【术语对照】
    embedding（向量化）  ：文本 → 固定维度浮点向量（bge-small-zh 是 512 维）
    vector store（向量库）：存向量 + 支持近似最近邻检索的数据库
    ANN / HNSW           ：近似最近邻算法，Chroma 默认用 HNSW，
                           把 O(n) 的暴力比对降到 O(log n)，百万级也能毫秒返回
    cosine similarity    ：余弦相似度，只看向量夹角不看长度。
                           bge 官方推荐用 cosine（而非 L2 欧氏距离）

【为什么选 bge-small-zh-v1.5】
    - 中文检索榜单长期靠前，且是**专为检索训练**的（不是通用语言模型）
    - small 版只有 24M 参数（约 95MB），4060 上编码 1.5 万块只要 1~2 分钟
    - v1.5 修正了相似度分布，query 侧**不需要**再加指令前缀（v1 需要）
      → 这条很重要：很多人照抄旧教程给 query 加 "Represent this sentence..."，
        在 v1.5 上反而会变差

【国内下载模型的坑】（2024 之后情况变了，别照抄旧教程）
    1. huggingface.co 很多时候**直连就能通**；反倒是部分镜像站会返回
       308 重定向，最终表现为 json.decoder.JSONDecodeError: Expecting value，
       完全看不出是网络问题。→ 先试直连，不要一上来就设 HF_ENDPOINT。
    2. 下载中断会在 ~/.cache/huggingface 留下 **0 字节的 json**，
       之后每次加载都读它并报同样的 JSONDecodeError，极具迷惑性。
       → 本模块内置 _repair_hf_cache() 自动体检并清除重下。
    3. 真下不动就走 ModelScope：pip install modelscope（代码已内置兜底）。
    4. 终极方案：手动下载模型到本地目录，用 --model <本地目录> 指定。
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_COLLECTION = "arknights"


def _prepare_hf_env() -> None:
    """代理环境下自动关掉 Xet 下载通道。

    【坑 · 非常隐蔽】
    huggingface_hub ≥ 1.0 默认用 Xet 存储下载（cas-server.xethub.hf.co）。
    在系统挂了 HTTP 代理（Clash / v2ray 等，见 HTTP_PROXY 环境变量）时，
    Xet 通道可能**静默失败**：文件下载"成功"、不抛异常，但落盘是 0 字节。
    表现就是后续 json.load 报
        JSONDecodeError: Expecting value: line 1 column 1 (char 0)
    看不出任何网络问题的痕迹。

    解法：设置 HF_HUB_DISABLE_XET=1，退回经典 HTTPS 下载（它正常走代理）。
    必须在 import huggingface_hub 之前设好，所以放在模块顶层执行。
    """
    if os.environ.get("HF_HUB_DISABLE_XET"):
        return
    proxy = (os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
             or os.environ.get("http_proxy") or os.environ.get("https_proxy"))
    if proxy:
        os.environ["HF_HUB_DISABLE_XET"] = "1"


_prepare_hf_env()


class BGEEmbedding:
    """把文本编码成向量的 embedding function。

    写成类而不是函数，是为了**懒加载模型**：
    建库脚本可能会先做统计再编码，没必要一上来就花 3 秒加载模型。
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64,
                 device: str | None = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    # Chroma 需要的方法名：__call__。
    #
    # 【坑】参数名**必须**叫 input，不能叫 texts / documents。
    # Chroma 从 0.4.16 起把 EmbeddingFunction 协议的参数名写死为 input，
    # 而且调用时用的是关键字参数 ef(input=...)，不是位置参数。
    # 名字不对会在 get_or_create_collection 阶段直接抛：
    #   ValueError: Expected EmbeddingFunction.__call__ to have the
    #   following signature: odict_keys(['self', 'input']), got ...
    # 这类错误属于「依赖升级导致的破坏性变更」，看报错信息里的
    # "Please note the recent change" 就知道是接口改了，不是我们逻辑错了。
    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.encode(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """查询侧编码。

        Chroma 1.x 在 query 时会**无条件**调用 ef.embed_query(input=...)
        （见 chromadb/api/models/CollectionCommon.py 的 _embed）。
        不定义这个方法，建库能过、一检索就 AttributeError。

        为什么这里不额外加指令前缀：bge-small-zh-v1.5 是**对称检索**模型，
        query 和 document 用同一套编码即可。
        （bge v1 和部分多语模型需要给 query 加 "Represent this sentence..."，
          照抄旧教程给 v1.5 加上反而会变差。）
        将来若换成非对称模型，只需改这一个方法，文档侧不用动。
        """
        return self.encode(input)

    def name(self) -> str:
        return "bge"

    def get_config(self) -> dict:
        """Chroma 1.x 会把 EF 配置持久化进 collection，靠它做一致性校验。"""
        return {"name": self.name(), "model_name": self.model_name,
                "batch_size": self.batch_size}

    @staticmethod
    def build_from_config(config: dict) -> "BGEEmbedding":
        cfg = dict(config or {})
        cfg.pop("name", None)
        return BGEEmbedding(**cfg)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            self._load()
        vecs = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,   # 归一化后 cosine == 内积，检索更稳
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vecs]

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "缺少 sentence-transformers，请先 pip install sentence-transformers"
            ) from e

        if self.device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"

        # 三级兜底：本地路径 → HuggingFace → ModelScope
        # 为什么写这么长：模型下载是这套流程里唯一依赖外网的地方，
        # 而失败信息往往毫无提示性（实测 hf-mirror 会返回 308 重定向，
        # 最终表现是 json.decoder.JSONDecodeError: Expecting value，
        # 完全看不出是网络问题）。把三种来源都试一遍并打印下一步该做什么，
        # 能省掉大量"我到底哪儿错了"的时间。
        errors: list[str] = []

        # ① 本地路径 / 已缓存
        if Path(self.model_name).exists():
            print(f"  加载 embedding 模型：本地目录 {self.model_name}"
                  f"  (device={self.device})")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            return

        # ② HuggingFace（含 HF_ENDPOINT 镜像）
        #    注意：这里**不**预先做缓存体检。缓存体检要删 95MB 再重下，
        #    放在正常路径上会白白拖慢每次运行；只在真的加载失败时才做。
        print(f"  加载 embedding 模型：{self.model_name}  (device={self.device})")
        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
            return
        except Exception as e:
            errors.append(f"HuggingFace 失败：{e}")
            # 失败后做一次缓存体检：网络中断/镜像异常会留下 0 字节的关键文件，
            # 之后每次加载都读它，报错是 JSONDecodeError 而非网络错误，极难定位。
            if _repair_hf_cache(self.model_name):
                print("    缓存已清除，重新拉取…")
                try:
                    self._model = SentenceTransformer(self.model_name,
                                                      device=self.device)
                    return
                except Exception as e2:
                    errors.append(f"HuggingFace 重试失败：{e2}")
            print(f"    HuggingFace 拉取失败（{type(e).__name__}），尝试 ModelScope…")

        # ③ ModelScope（阿里镜像，国内最稳）
        try:
            local = _modelscope_download(self.model_name)
            self._model = SentenceTransformer(local, device=self.device)
            print(f"    已改用 ModelScope 镜像：{local}")
            return
        except Exception as e:
            errors.append(f"ModelScope 失败：{e}")

        raise RuntimeError(
            "模型加载失败，三种来源都试过了：\n"
            + "\n".join(f"    · {x}" for x in errors)
            + "\n\n  可操作的解法（任选一种）："
            "\n    1) 直连试试（实测 huggingface.co 本身可达，反而某些镜像会 308）："
            "\n       PowerShell: Remove-Item Env:HF_ENDPOINT"
            "\n    2) 装 ModelScope 走阿里镜像：pip install modelscope"
            "\n    3) 手动下载模型到本地目录，再用 --model <本地目录> 指定"
            f"\n       当前 HF_ENDPOINT = {os.environ.get('HF_ENDPOINT') or '(未设置，直连 huggingface.co)'}"
        )


# HuggingFace 模型名 → ModelScope 模型名（两边命名不一致时需要映射）
MODELSCOPE_MAP = {
    "BAAI/bge-small-zh-v1.5": "BAAI/bge-small-zh-v1.5",
    "BAAI/bge-base-zh-v1.5": "BAAI/bge-base-zh-v1.5",
    "BAAI/bge-m3": "BAAI/bge-m3",
}


def _repair_hf_cache(model_name: str) -> bool:
    """体检 HF 缓存，发现损坏就清掉该模型的缓存目录。

    返回 True 表示清除过（调用方应重试）。

    【为什么需要】
    下载被中断（镜像 308、断网、Ctrl-C）时，huggingface_hub 会在
    ~/.cache/huggingface/hub/models--XXX--YYY/ 里留下 **0 字节的文件**。
    之后每次加载都直接读这个空文件，抛的是
        json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
    跟网络一点关系都没有，排查时极易误判为代码 bug。

    【判定要"严"：只查真正必需的文件】
    第一版我只要看到 0 字节的 .json 就判定损坏，结果把 adapter_config.json
    这类**本来就是空**的可选文件也算进去，导致每次运行都白删 95MB 缓存重下。
    所以规则收紧为两类：
      A. 必需文件缺失或为空（modules.json / config.json / 权重 / 词表）
      B. 权重类文件为 0 字节（权重文件永远不可能合法为空）

    【安全性】
    只动 HF 缓存目录下**这一个模型**的子目录，且只在确实发现坏文件时才删。
    缓存随时可重新下载，不含任何用户数据。
    """
    import shutil

    # A. 必需文件：元组内任一存在且非空即算通过
    REQUIRED_GROUPS = (
        ("modules.json",),
        ("config.json",),
        ("model.safetensors", "pytorch_model.bin"),
        ("tokenizer.json", "vocab.txt"),
    )
    # B. 权重类后缀：为 0 字节一定不正常
    WEIGHT_SUFFIX = {".bin", ".safetensors", ".model", ".onnx", ".pt", ".h5"}

    root = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    repo = Path(root) / "hub" / ("models--" + model_name.replace("/", "--"))
    if not repo.exists():
        return False

    bad: list[Path] = []
    for snap in (repo / "snapshots").glob("*"):
        if not snap.is_dir():
            continue
        for names in REQUIRED_GROUPS:
            ok = False
            for n in names:
                f = snap / n
                if f.exists() and f.stat().st_size > 0:
                    ok = True
                    break
            if not ok:
                bad.append(snap / names[0])
        for p in snap.rglob("*"):
            if p.is_file() and p.suffix in WEIGHT_SUFFIX and p.stat().st_size == 0:
                bad.append(p)

    if not bad:
        return False

    shown = ", ".join(p.name for p in bad[:3])
    print(f"    检测到损坏的 HF 缓存（{len(bad)} 项：{shown}），将清除重下："
          f"\n      {repo}")
    shutil.rmtree(repo, ignore_errors=True)
    return True


def _modelscope_download(model_name: str) -> str:
    """从 ModelScope 下载模型，返回本地目录路径。"""
    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise RuntimeError("未安装 modelscope（pip install modelscope）") from e
    ms_id = MODELSCOPE_MAP.get(model_name, model_name)
    return snapshot_download(ms_id)


# ---------------------------------------------------------------------------
# 库 / 集合管理
# ---------------------------------------------------------------------------

def get_client(persist_dir: Path):
    import chromadb
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_collection(persist_dir: Path, name: str = DEFAULT_COLLECTION,
                   model: str = DEFAULT_MODEL, fresh: bool = False):
    """取（或重建）一个 collection。

    fresh=True 会先删掉旧的 —— 改了切块策略后必须用它，
    否则新旧 chunk 混在一起，检索结果一会儿新一会儿旧，极难排查。
    """
    client = get_client(persist_dir)
    if fresh:
        try:
            client.delete_collection(name)
        except Exception:
            pass  # 不存在就跳过
    ef = BGEEmbedding(model)
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef,   # type: ignore[arg-type]
    )


def add_documents(col, docs: list[dict], batch: int = 500, verbose: bool = True) -> int:
    """分批写入。分批的原因：一次塞 1.5 万条会显存爆掉且没有进度反馈。"""
    total = 0
    for i in range(0, len(docs), batch):
        part = docs[i:i + batch]
        col.add(
            ids=[d["id"] for d in part],
            documents=[d["text"] for d in part],
            metadatas=[d["metadata"] for d in part],
        )
        total += len(part)
        if verbose:
            print(f"    写入 {total:,}/{len(docs):,} 块", end="\r")
    if verbose:
        print(f"    写入 {total:,}/{len(docs):,} 块    ")
    return total


def search(col, query: str, k: int = 5, where: dict | None = None,
           source: str | None = None, operator: str | None = None) -> list[dict]:
    """语义检索。

    where / source / operator 三个参数是**元数据过滤**（pre-filter），
    先缩范围再做向量比对。这是生产级 RAG 的标配：
    全库海搜噪声大，而且用户问"泥岩"时不该搜出别人的密录。
    """
    cond: dict = {}
    if where:
        cond.update(where)
    if source:
        cond["source"] = source
    if operator:
        cond["operator"] = operator

    # Chroma 要求：单条件用 dict，多条件要包一层 $and
    if len(cond) > 1:
        cond = {"$and": [{k: v} for k, v in cond.items()]}

    res = col.query(
        query_texts=[query],
        n_results=k,
        where=cond or None,
    )
    out = []
    for i in range(len(res["ids"][0])):
        out.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i] if res.get("distances") else None,
            "score": 1 - res["distances"][0][i] if res.get("distances") else None,
        })
    return out


def clean_text_for_show(text: str, width: int = 160) -> str:
    """把 chunk 压成一行便于终端查看（去掉 header 的重复信息）。"""
    one = (text or "").replace("\n", " / ")
    return one[:width] + ("…" if len(one) > width else "")
