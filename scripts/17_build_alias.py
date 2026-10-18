"""第 8 步补丁：干员别名/昵称挖掘。

【为什么要这个脚本】
第 8 步实测暴露了两类「叫法不一致」：

    A. 英文代号：泥岩档案里写"精英干员Logos"，用户问的是"逻各斯"
       → operators 表本来就有 name_en，router 已自动启用，**无需本脚本**
    B. 玩家昵称：用户说"小刻"，正式名是"刻俄柏"
       → Wiki 里没有这个字段，只能挖

【昵称为什么不能全自动】
规则能派生候选（"小"+首字 = 小刻），但会撞上同名异物：
"小车"是支援小车不是干员，"小羊"可能是别的东西。
所以本脚本的做法是：

    派生候选 → **用全库文本验证出现次数** → 排出候选清单 → 人工确认

这是本项目一贯的原则：能用数据验证的，就不要拍脑袋。

【用法】
    python scripts/17_build_alias.py               # 输出候选清单
    python scripts/17_build_alias.py --min 10      # 提高门槛，更保守
确认无误后把生成的文件改名即可启用：
    mv data/alias_candidates.json data/alias.json
（或手动编辑成你要的样子；router 启动时会读它，不存在就跳过）
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data" / "arknights.db"
OUT_PATH = ROOT / "data" / "alias_candidates.json"

# 正文来源表：昵称会出现在台词和故事里，而不是面板字段里
TEXT_TABLES = [("stories", "content"), ("voices", "line"),
               ("archive_texts", "content")]


def _is_cjk(s: str) -> bool:
    return bool(s) and all("\u4e00" <= ch <= "\u9fff" for ch in s)


def gen_candidates(name: str) -> list[str]:
    """从一个正式名派生昵称候选。

    规则来自中文昵称的常见构词法：
        小/阿/老 + 首字或末字      小刻、阿能、老爷子
        三字以上名：首字 + 末字     推进之王 → 推王
    只处理 2~4 字的纯中文名 —— 外文名走 operators.name_en，不用猜。
    """
    base = name.split("(")[0]
    if not (2 <= len(base) <= 4) or not _is_cjk(base):
        return []
    first, last = base[0], base[-1]
    out: set[str] = set()
    for pre in ("小", "阿", "老"):
        out.add(pre + first)
        if last != first:
            out.add(pre + last)
    if len(base) >= 3:
        out.add(first + last)
    return [c for c in out if len(c) >= 2]


def load_corpus(conn: sqlite3.Connection) -> str:
    """把三张正文表拼成一个大字符串，供 str.count 快速统计。"""
    parts: list[str] = []
    for table, col in TEXT_TABLES:
        try:
            rows = conn.execute(
                f"SELECT {col} FROM {table} WHERE {col} IS NOT NULL").fetchall()
        except sqlite3.Error as e:
            print(f"  跳过表 {table}：{e}")
            continue
        parts.extend(r[0] for r in rows)
        print(f"  {table}: {len(rows):,} 条")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=5,
                    help="候选在全库正文里至少出现几次才采纳（默认 5）")
    ap.add_argument("--top", type=int, default=60, help="最多输出多少条")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)

    # ---- 1) 报告已自动启用的外文别名（这类不用人工确认）----
    en_rows = conn.execute(
        "SELECT name, name_en FROM operators "
        "WHERE name IS NOT NULL AND name_en IS NOT NULL AND name_en != ''"
    ).fetchall()
    print(f"已自动启用的英文代号别名：{len(en_rows)} 个（router 直接读表，无需确认）")
    for n, en in en_rows[:5]:
        print(f"    {n} ← {en}")
    print("    …\n")

    # ---- 2) 派生昵称候选 ----
    names = [r[0] for r in conn.execute(
        "SELECT name FROM operators WHERE name IS NOT NULL").fetchall()]
    official: set[str] = set()
    for n in names:
        official.add(n)
        official.add(n.split("(")[0])

    cand2canon: dict[str, set[str]] = defaultdict(set)
    for n in names:
        base = n.split("(")[0]
        for c in gen_candidates(base):
            if c in official:        # 已经是别人的正式名，不占用
                continue
            cand2canon[c].add(base)

    print(f"派生候选 {len(cand2canon)} 个，开始用全库正文验证…")
    corpus = load_corpus(conn)
    print(f"  正文总量 {len(corpus):,} 字符\n")

    results: list[dict] = []
    ambiguous = 0
    for cand, canons in cand2canon.items():
        if len(canons) > 1:          # 多个干员共用同一个昵称 → 无法判定，丢弃
            ambiguous += 1
            continue
        cnt = corpus.count(cand)
        if cnt < args.min:
            continue
        canon = next(iter(canons))
        p = corpus.find(cand)
        ctx = corpus[max(0, p - 25):p + len(cand) + 25].replace("\n", " ")
        results.append({"alias": cand, "canon": canon, "count": cnt,
                        "sample": ctx})

    results.sort(key=lambda x: -x["count"])
    print(f"歧义候选（丢弃）：{ambiguous} 个")
    print(f"通过门槛（>={args.min} 次）：{len(results)} 个\n")
    print("=" * 74)
    print(f"{'昵称':<8}{'对应干员':<12}{'次数':>6}   上下文样例")
    print("=" * 74)
    for r in results[:args.top]:
        print(f"{r['alias']:<8}{r['canon']:<12}{r['count']:>6}   …{r['sample']}…")

    # ---- 3) 落成候选文件 ----
    by_canon: dict[str, list[str]] = defaultdict(list)
    for r in results[:args.top]:
        by_canon[r["canon"]].append(r["alias"])
    OUT_PATH.write_text(
        json.dumps(dict(sorted(by_canon.items())), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print()
    print("=" * 74)
    print(f"候选已写入 {OUT_PATH}")
    print("请人工过一遍上面的「上下文样例」，确认没有同名异物（如 小车/小羊）。")
    print("确认后改名启用：")
    print(f"    mv data/alias_candidates.json data/alias.json")
    print("改名后重启 16_chat_demo.py，实体识别会自动带上这些昵称。")


if __name__ == "__main__":
    main()
