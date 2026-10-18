"""第 8 步补丁 2：抓取官方重定向 —— Wiki 编辑者维护的权威别名表。

【起因：一次真实翻案】
17 脚本用正文词频挖昵称，"小羊"高频出现在贝娜的档案里（奶奶叫她"小羊羔"），
而艾雅法拉的正文里几乎不出现"小羊"——但玩家口中的"小羊"恰恰指艾雅法拉。
Wiki 官方用重定向页给出了正确答案：

    访问 prts.wiki/小羊 → 自动跳转到 艾雅法拉，页面顶部显示「（重定向自小羊）」

【为什么重定向标题是最权威的别名】
MediaWiki 的重定向页内容只有一行 `#重定向 [[目标页]]`。
编辑者建它的唯一动机，就是把社区常用叫法接到正式条目——
这份映射是**人工维护、显式声明**的，可信度远高于正文词频。

【为什么第 2 步没抓它】
01_fetch 用 apfilterredir=nonredirects 把重定向页过滤掉了。
过滤**内容**是对的（不然向量库混入上万跳转空壳），
但把**标题**也扔了，等于扔掉了别名表。现在单独拉回来。

【本脚本的取法：反向查询，不是全站扫描】
    action=query&prop=redirects&titles=<50个干员名>&rdlimit=500&rdnamespace=0

对每个干员页列出"所有指向它的重定向页标题"。463 个干员 ÷ 50/批 ≈ 10 次请求，
两三分钟跑完 —— 比全站枚举重定向再反查少两个数量级的请求。

【产物】
    data/redirects.json    {"小羊": "艾雅法拉", ...}
router 启动时自动加载，优先级**高于** 17 脚本的正文词频结果
（官方显式声明 > 文本统计推断，这条优先级写在 src/retriever/router.py）。

【用法】
    python scripts/18_fetch_redirects.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prts_client import PRTSClient                     # noqa: E402

DB_PATH = ROOT / "data" / "arknights.db"
RAW_PATH = ROOT / "data" / "raw" / "pages.jsonl"
OUT_PATH = ROOT / "data" / "redirects.json"


def chunked(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def load_local_titles() -> set[str]:
    """本地已收录的全部条目标题（用于校验别名确实指向已收录页面）。"""
    titles: set[str] = set()
    if RAW_PATH.exists():
        with RAW_PATH.open(encoding="utf-8") as f:
            for line in f:
                try:
                    titles.add(json.loads(line).get("title", ""))
                except json.JSONDecodeError:
                    continue
    return titles


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    ops = [r[0] for r in conn.execute(
        "SELECT name FROM operators WHERE name IS NOT NULL ORDER BY id")]
    print(f"干员 {len(ops)} 个，预计 {len(ops) // 50 + 1} 批请求（每批间隔限流）…")

    client = PRTSClient(delay=2.0)
    alias2target: dict[str, str] = {}

    for i, batch in enumerate(chunked(ops, 50), 1):
        cont: dict = {}
        while True:
            params = {
                "action": "query",
                "prop": "redirects",
                "titles": "|".join(batch),
                "rdlimit": 500,
                "rdnamespace": 0,
                "rdprop": "title",
            }
            params.update(cont)
            data = client.query(params)

            for page in data.get("query", {}).get("pages", {}).values():
                target = page.get("title")
                for rd in page.get("redirects", []):
                    alias = rd.get("title", "")
                    if alias and target and alias != target:
                        alias2target[alias] = target

            if "continue" in data:
                cont = data["continue"]
            else:
                break
        print(f"  批次 {i}：累计 {len(alias2target)} 条重定向别名")

    local = load_local_titles()
    print(f"\n本地条目集 {len(local):,} 个")
    kept = {a: t for a, t in alias2target.items() if t in local}
    dropped = len(alias2target) - len(kept)
    print(f"指向本地已收录页面的别名 {len(kept)} 条"
          + (f"（丢弃 {dropped} 条指向未收录页面的）" if dropped else ""))

    # 干员相关的部分单列预览 —— 这是修复"小羊"问题的关键子集
    op_set = set(ops)
    op_hits = {a: t for a, t in kept.items() if t in op_set}
    print(f"\n其中指向干员页的 {len(op_hits)} 条（样例 30 条）：")
    for a, t in sorted(op_hits.items())[:30]:
        print(f"    {a} → {t}")

    OUT_PATH.write_text(
        json.dumps(dict(sorted(kept.items())), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n已写入 {OUT_PATH}")
    print("重启 16_chat_demo.py 即生效（router 会自动加载，无需重建向量库）。")
    print("注意：官方重定向优先级高于 17 脚本的词频候选，"
          "两者冲突时以本文件为准。")


if __name__ == "__main__":
    main()
