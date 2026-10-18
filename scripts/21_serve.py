"""第 10 步（服务化）：把 CLI 问答包装成 HTTP 服务 + 一个能直接用的网页。

【这一步在 RAG 全景里的位置】
它不增加 RAG 的深度，只是把 16 的链路暴露成接口。
真正有价值的是它逼出来的三个工程问题，CLI 里根本遇不到：

    1. **模型单例**
       HybridRetriever() 每次构造都要加载 embedding 模型和 BM25 缓存
       （36,259 篇，pickle 反序列化好几秒）。CLI 跑一次没事，
       Web 服务每请求构造一次必崩。必须用 lifespan 在启动时加载一次。

    2. **并发安全**
       sqlite3 连接默认只能在创建它的线程用。FastAPI 用线程池跑同步函数，
       请求 A 的连接可能被请求 B 拿到。这里每个请求新建连接（开销极小），
       只共享**无状态**的检索器和匹配器。

    3. **降级**
       LLM 超时 / API 报错时，服务不能 500 崩掉，要把错误变成可读的响应。
       检索层的异常和生成层的异常要分开报 —— 前者是系统坏了，
       后者是外部服务抖动，重试策略完全不同。

【为什么用同步 def 而不是 async def】
底层的 Chroma、BM25、sqlite3 全是同步库。写成 async 但不 await 任何东西，
反而会阻塞事件循环。FastAPI 对同步 def 会自动丢进线程池，这才是正确用法。
"""

from __future__ import annotations

import argparse
import io
import sqlite3
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException                    # noqa: E402
from fastapi.responses import HTMLResponse                    # noqa: E402
from pydantic import BaseModel, Field                         # noqa: E402

from src.generator.llm import LLMClient                       # noqa: E402
from src.retriever.hybrid import HybridRetriever              # noqa: E402
from src.retriever.router import EntityMatcher                # noqa: E402

DB_PATH = ROOT / "data" / "arknights.db"
CHROMA_DIR = ROOT / "data" / "chroma"

# 全局单例，由 lifespan 在启动时填充
STATE: dict = {}


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=300)
    k: int = Field(default=4, ge=1, le=10, description="取几条证据")


class AskResponse(BaseModel):
    answer: str
    refused: bool
    intent: str
    route_reason: str
    entities: list[str]
    entity_terms: list[list[str]]
    sql_rows: int
    evidence: list[dict]
    missing_citations: list[int]
    elapsed_sec: float


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------
PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>明日方舟 Wiki 问答</title>
<style>
  :root{--bg:#f7f7f5;--card:#fff;--bd:#e3e1da;--tx:#2c2c2a;--mu:#5f5e5a;--ac:#185fa5}
  *{box-sizing:border-box}
  body{margin:0;padding:32px 16px;background:var(--bg);color:var(--tx);
       font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
  .wrap{max-width:760px;margin:0 auto}
  h1{font-size:19px;font-weight:600;margin:0 0 4px}
  .sub{color:var(--mu);font-size:13px;margin-bottom:20px}
  .row{display:flex;gap:8px}
  input{flex:1;padding:11px 13px;border:1px solid var(--bd);border-radius:8px;
        font-size:15px;background:var(--card);color:var(--tx)}
  input:focus{outline:2px solid var(--ac);outline-offset:-1px}
  button{padding:11px 20px;border:0;border-radius:8px;background:var(--ac);
         color:#fff;font-size:15px;cursor:pointer}
  button:disabled{opacity:.5;cursor:default}
  .card{margin-top:18px;padding:16px 18px;background:var(--card);
        border:1px solid var(--bd);border-radius:10px;white-space:pre-wrap}
  .card.hide{display:none}
  .meta{color:var(--mu);font-size:12px;margin-top:10px;padding-top:10px;
        border-top:1px solid var(--bd)}
  .cite{margin-top:8px;font-size:12px;color:var(--mu);line-height:1.9}
  .warn{color:#a32d2d}
  .eg{margin-top:14px;font-size:13px;color:var(--mu)}
  .eg span{display:inline-block;margin:3px 6px 0 0;padding:4px 9px;
           background:var(--card);border:1px solid var(--bd);border-radius:6px;
           cursor:pointer}
  .eg span:hover{border-color:var(--ac);color:var(--ac)}
</style>
</head>
<body>
<div class="wrap">
  <h1>明日方舟 Wiki 知识问答</h1>
  <div class="sub">基于本地知识库的 RAG 问答 · 每条回答都带引用溯源</div>
  <div class="row">
    <input id="q" placeholder="例如：泥岩和可露希尔是什么关系？" autofocus>
    <button id="go" onclick="ask()">提问</button>
  </div>
  <div class="eg">试试这些：
    <span onclick="set('给我讲讲主线剧情第一章的大致剧情')">第一章剧情</span>
    <span onclick="set('泥岩和可露希尔是什么关系？')">两人关系</span>
    <span onclick="set('史尔特尔满练度总共要多少材料')">材料汇总</span>
    <span onclick="set('史尔特尔和泥岩之间有什么故事？')">会拒答的问题</span>
    <span onclick="set('小羊是谁的昵称')">昵称别名</span>
  </div>
  <div id="out" class="card hide"></div>
</div>
<script>
function set(t){document.getElementById('q').value=t;ask()}
async function ask(){
  const q=document.getElementById('q').value.trim();
  if(!q)return;
  const btn=document.getElementById('go'),out=document.getElementById('out');
  btn.disabled=true;out.className='card';out.textContent='检索与生成中…';
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q})});
    const d=await r.json();
    if(!r.ok){out.innerHTML='<span class="warn">'+esc(d.detail||'出错了')+'</span>';return}
    let h=esc(d.answer);
    if(d.evidence&&d.evidence.length){
      h+='<div class="cite">引用来源：';
      d.evidence.forEach(c=>{h+='<br>['+c.idx+'] '+
        esc(c.source_label||c.source)+'·'+esc(c.operator)+'（相似度 '+c.score+'）'});
      h+='</div>';
    }
    if(d.missing_citations&&d.missing_citations.length){
      h+='<div class="cite warn">⚠ 引用了不存在的编号 '+
         d.missing_citations.join(',')+' —— 幻觉信号</div>';
    }
    h+='<div class="meta">意图 '+esc(d.intent)+' · SQL '+d.sql_rows+' 行 · 证据 '+
        d.evidence.length+' 条 · 耗时 '+d.elapsed_sec+'s</div>';
    out.innerHTML=h;
  }catch(e){out.innerHTML='<span class="warn">请求失败：'+esc(e.message)+'</span>'}
  finally{btn.disabled=false}
}
function esc(s){return (s+'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')ask()});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(title="明日方舟 Wiki 问答", version="1.0")

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # ---- 启动时加载一次。这是服务化最关键的一件事 ----
        t0 = time.time()
        print("装载检索器…")
        STATE["hr"] = HybridRetriever(CHROMA_DIR)
        # 这个 conn 只给 EntityMatcher 建索引用，用完即弃
        _c = sqlite3.connect(DB_PATH)
        _c.row_factory = sqlite3.Row
        STATE["matcher"] = EntityMatcher(_c)
        _c.close()
        STATE["llm"] = LLMClient.from_env()
        print(f"就绪（{time.time() - t0:.1f}s）· 服务已启动")
        yield
        STATE.clear()

    app = FastAPI(title="明日方舟 Wiki 问答", version="1.0", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(PAGE)

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": "hr" in STATE,
            "chunks": len(STATE["hr"].texts) if "hr" in STATE else 0,
            "operators": len(STATE["matcher"].names) if "matcher" in STATE else 0,
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> dict:
        if "hr" not in STATE:
            raise HTTPException(status_code=503, detail="服务还在装载，请稍候")

        # 每个请求新建 sqlite 连接：sqlite3 连接不是线程安全的，
        # 而 FastAPI 会在线程池里跑这个同步函数，不能共享连接。
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            # 16 的 answer() 会往 stdout 打印诊断信息。
            # 在 HTTP 请求里那会污染日志，这里吞掉；
            # 真正要的是它的返回值。
            buf = io.StringIO()
            with redirect_stdout(buf):
                r = _answer(conn, STATE["hr"], STATE["matcher"], STATE["llm"],
                            req.question, k=req.k)
            if r is None:
                raise HTTPException(status_code=500, detail="问答链路未返回结果")
            return r
        except HTTPException:
            raise
        except Exception as e:                                 # noqa: BLE001
            # 检索层挂了和 LLM 挂了要分开看，但都不该让服务 500 崩掉
            raise HTTPException(status_code=500,
                                detail=f"问答失败：{type(e).__name__}: {e}")
        finally:
            conn.close()

    return app


_ANSWER_FN = None


def _load_answer():
    """加载 16_chat_demo.answer。

    为什么用 importlib 而不是 import：文件名以数字开头（16_chat_demo.py），
    不是合法 Python 标识符，正常 import 语法写不出来。
    用 spec_from_file_location 加载时 __name__ 不是 "__main__"，
    所以文件末尾的 main() 不会被执行 —— 正好只拿到 answer 函数。
    """
    global _ANSWER_FN
    if _ANSWER_FN is None:
        import importlib.util
        p = ROOT / "scripts" / "16_chat_demo.py"
        spec = importlib.util.spec_from_file_location("demo16_cli", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ANSWER_FN = mod.answer
    return _ANSWER_FN


def _answer(conn, hr, matcher, llm, question: str, k: int = 4) -> dict | None:
    """复用 16_chat_demo 的 answer()，拿它的结构化返回值。

    不在服务里重写一遍检索+组装，是为了保证 CLI 和 API 走**同一份逻辑** ——
    否则改了 CLI 忘了改 API，两边行为不一致，这种 bug 特别难查。
    """
    return _load_answer()(conn, hr, matcher, llm, question, k=k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    import uvicorn
    # 用字符串导入路径才能在 --reload 下工作
    uvicorn.run("scripts._serve_app:app" if args.reload else create_app(),
                host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
