"""
第 4 步（补全）：解析干员子页面 —— 语音记录 + 干员密录正文

输入：data/raw/pages.jsonl
输出：
    data/processed/voices.jsonl   语音（结构化条目）
    data/processed/stories.jsonl  密录正文（清洗后的剧情文本）

为什么单独跑而不是并进 04？
    这两类页面体量大（6.6MB + 10MB）、结构完全不同，
    且归属不同：语音既要结构化入库，也要切块进向量库。
    分开跑便于单独重跑，也便于第 6 步直接消费。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from src.parser.subpages import (  # noqa: E402
    expand_data_refs,
    is_story_page,
    is_voice_page,
    parse_story_page,
    parse_voice_page,
    voice_to_text,
)
from src.parser.wikitext import residual_templates  # noqa: E402

RAW = ROOT / "data" / "raw" / "pages.jsonl"
OUT_DIR = ROOT / "data" / "processed"

# 只保留含信息量大的语音条目，过滤纯语气词
KEEP_VOICE_MIN_LEN = 2


def main() -> None:
    if not RAW.exists():
        print(f"找不到 {RAW}，请先跑 scripts/01_fetch.py")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 第 0 遍：把 /data 子页面捞出来做索引 ----------
    # 少数密录主页只有一行 {{:{{PAGENAME}}/data}}，正文在子页面里。
    # 这里只缓存以 /data 结尾的页（几百 KB），不把 204MB 全读进内存。
    data_pages: dict[str, str] = {}
    with RAW.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            t = rec.get("title") or ""
            if t.endswith("/data"):
                c = rec.get("content")
                if isinstance(c, str):
                    data_pages[t] = c
    print(f"缓存 /data 子页面：{len(data_pages)} 个")

    voices, stories = [], []
    v_chars = s_chars = 0
    total = 0

    with RAW.open(encoding="utf-8") as f:
        for line in tqdm(f, desc="扫描子页面", unit="page"):
            total += 1
            rec = json.loads(line)
            title = rec.get("title") or ""
            content = rec.get("content") or ""

            if is_voice_page(title):
                try:
                    v = parse_voice_page(title, content)
                    if v["条目"]:
                        # 合并文本供第 6 步切块
                        v["合并文本"] = voice_to_text(v)
                        v_chars += len(v["合并文本"])
                        voices.append(v)
                except Exception as e:
                    print(f"  语音解析失败 {title}: {type(e).__name__}: {e}")
            elif is_story_page(title):
                try:
                    content = expand_data_refs(content, title, data_pages)
                    s = parse_story_page(title, content)
                    if s["正文"]:
                        s_chars += s["字符数"]
                        stories.append(s)
                except Exception as e:
                    print(f"  密录解析失败 {title}: {type(e).__name__}: {e}")

    v_path = OUT_DIR / "voices.jsonl"
    s_path = OUT_DIR / "stories.jsonl"

    with v_path.open("w", encoding="utf-8") as f:
        for v in voices:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    with s_path.open("w", encoding="utf-8") as f:
        for s in stories:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print()
    print(f"扫描总页数：{total}")
    print(f"语音记录：{len(voices)} 页 → {v_path}")
    print(f"密录正文：{len(stories)} 页 → {s_path}")
    print()

    # ---------------- 质量报告 ----------------
    print("=" * 62)
    print("语音质量抽查")
    print("=" * 62)
    if voices:
        counts = [len(v["条目"]) for v in voices]
        print(f"  每页条目数：最少 {min(counts)}  最多 {max(counts)}  "
              f"平均 {sum(counts) / len(counts):.1f}")
        print(f"  合并文本总量：{v_chars / 1024 / 1024:.2f} MB")

        # 定点验证：找一个有「交谈1」的干员
        hit = next((v for v in voices
                    if any(it["标题"] == "交谈1" for it in v["条目"])), None)
        if hit:
            print(f"\n  定点验证：{hit['标题']}")
            for it in hit["条目"]:
                if it["标题"] in ("任命助理", "交谈1", "信赖提升后交谈1", "生日"):
                    print(f"    【{it['标题']}】{it['台词'][:70]}")

        # 检查多语是否被正确分离
        multi = sum(1 for v in voices
                    for it in v["条目"] if it.get("其他语言"))
        print(f"\n  含多语台词的条目数：{multi}（应接近条目总数，说明五语都抽到了）")

        empty = [v["标题"] for v in voices if len(v["条目"]) < 5]
        if empty:
            print(f"\n  条目数 <5 的异常页（{len(empty)} 个）：{empty[:8]}")

    print()
    print("=" * 62)
    print("密录正文质量抽查")
    print("=" * 62)
    if stories:
        lens = [s["字符数"] for s in stories]
        lens.sort()
        print(f"  正文长度：最少 {lens[0]}  中位 {lens[len(lens) // 2]}  最长 {lens[-1]}")
        print(f"  正文总量：{s_chars / 1024 / 1024:.2f} MB")

        # 关键：引擎指令必须清干净
        dirty = [s["标题"] for s in stories if "[Dialog]" in s["正文"]
                 or "[Background" in s["正文"] or "[HEADER" in s["正文"]]
        if dirty:
            print(f"\n  !! 仍有引擎指令残留（{len(dirty)} 页）：{dirty[:8]}")
        else:
            print("\n  引擎指令已全部清除")

        hit = next((s for s in stories if "泥岩" in s["标题"]), None)
        if hit:
            print(f"\n  定点验证：{hit['标题']}（{hit['字符数']} 字符）")
            print("   ", hit["正文"][:200].replace("\n", " / "))

        too_short = [s["标题"] for s in stories if s["字符数"] < 200]
        if too_short:
            print(f"\n  正文 <200 字符的页（{len(too_short)} 个）：{too_short[:8]}")

    # ---------------- 残留模板自检（第 6 步切块前的最后一道闸） ----------------
    # 这一步必须做在「进向量库之前」：一旦脏标记进了 chunk，
    # 检索时会被当成语义噪声，而你看不到任何报错。
    print()
    print("=" * 62)
    print("残留 {{模板}} 自检（应为 0；非 0 就把名字补进 wikitext.py 的规则表）")
    print("=" * 62)
    bad_v = [(v["标题"], it["台词"]) for v in voices for it in v["条目"]]
    bad_s = [(s["标题"], s["正文"]) for s in stories]
    hits: dict[str, int] = {}
    for _, txt in bad_v + bad_s:
        for name in residual_templates(txt):
            hits[name] = hits.get(name, 0) + 1
    nv = sum(1 for _, t in bad_v if "{{" in t)
    ns = sum(1 for _, t in bad_s if "{{" in t)
    print(f"  语音台词含 '{{{{' 的条数：{nv}    密录正文含 '{{{{' 的页数：{ns}")
    if hits:
        for k, v in sorted(hits.items(), key=lambda x: -x[1])[:15]:
            print(f"    !! {k:<24} {v} 次")
    else:
        print("    OK 全部干净")


if __name__ == "__main__":
    main()
