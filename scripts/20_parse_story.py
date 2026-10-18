"""第 10 步（补数据）：解析全部剧情文本 —— 主线 / 支线 / 故事集。

【为什么要补这一步】
第 8 步真实问答暴露：问「主线第一章讲了什么」时，系统检索到的全是干员密录，
最后正确拒答。查下来库里确实 0 条剧情 —— 不是检索故障，是**数据范围**没覆盖。
而原生数据其实早就躺在 pages.jsonl 里（第 2 步全站抓取顺带抓下来了），
只是第 4/6 步的解析器只认「干员名/干员密录/N」这一种页面。

【剧情页面的结构】
PRTS 用 `{{剧情模拟器}}` 模板承载剧情，关键字段：

    文本路径=obt/main/level_main_00-10_beg   ← 可反查关卡 id
    剧情类型=主线 | 支线 | 剧情 | 干员密录     ← 只取前三种（密录已有 stories 表）
    剧情分组=黑暗时代·上                      ← 章节 / 活动名

    文本数据=
    [HEADER(key="title_test"...)] 第六关（后）
    [Background(image="bg_cher_5", ...)]
    [Character(name="char_148_nearl_1")]
    [name="临光"]   ————你。
    [Dialog]
    [name="临光"]   制压切尔诺伯格军事据点的整合运动像机器一样高效——

所以解析要做三件事：
    1. 把 `文本数据=` 到模板结束 `}}` 之间的内容抠出来（括号计数，防嵌套截断）
    2. 剔除 [Character] [Delay] [Dialog] 这类**演出指令**，它们对语义检索是纯噪声
    3. 把 [name="临光"]台词 变成「临光：台词」，让角色名进入正文
       —— 这样问「临光在哪段剧情里出场」才能被向量检索命中

【干员密录为什么不在这里处理】
它的分组字段是干员名（剧情分组=泥岩），结构和章节剧情完全不同，
且第 6 步已经解析进 stories 表（394 条）。重复收录只会造成检索重复命中。
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser.wikitext import clean_inline          # noqa: E402

RAW_PATH = ROOT / "data" / "raw" / "pages.jsonl"
DB_PATH = ROOT / "data" / "arknights.db"

# 只收这三类。干员密录在 stories 表里已有，不重复。
WANT_TYPES = ("主线", "支线", "剧情")

# 演出指令：全部剔除。
#
# 第一版我按名字白名单列（Character / Dialog / Delay …），结果实测 188 页仍有残留 ——
# wiki 里还有 BackgroundTween、CharacterCutin、characteraction 这些没见过的，
# 而且 [Delay=0.5] 用的是等号不是括号。
# **白名单在这里是错的做法**：指令是引擎脚本，种类随时会加，
# 而它们的共同特征很稳定 —— 方括号里是一个英文标识符，可选 (...) 或 =值。
# 正文里不会出现这种形态，所以按形态剔除比按名字枚举稳。
CMD_RE = re.compile(r"\[[A-Za-z][A-Za-z0-9_]*\s*(?:\([^)]*\)|=[^\]]*)?\s*\]")
# [name="临光"]  → 说话人。name2= 是同框第二人，也提取出来当出场角色。
SPEAKER_RE = re.compile(r'\[name\d*="([^"]+)"\s*(?:\([^)]*\))?\s*\]')
# 关卡 id：文本路径=obt/main/level_main_00-10_beg → main_00-10
PATH_ID_RE = re.compile(r"文本路径=([^\s|]+)")
LEVEL_ID_RE = re.compile(r"(?:level_)?([a-z0-9_]*(?:main|ro|act)[a-z0-9_\-]*?)(?:_(?:beg|end))?$",
                         re.I)

CHAR_NOISE = re.compile(r"[　\s]+")

# 游戏内占位符 {@xxx}：
#   {@nickname} / {@player} 是玩家名 —— 和第 4 步的 {{DrName}} 同一类问题，
#     不替换成「博士」的话，问"杜宾对博士说了什么"就检索不到。
#   {@nbs} / {@nbsp} 是不换行空格，直接删。
# 未知的 {@...} 一律删除：宁可留空，也不要把一个猜错的名字写进正文。
BRACE_PH_RE = re.compile(r"\{@([^}]*)\}")


def _sub_placeholder(m: re.Match) -> str:
    k = m.group(1).lower()
    if "nick" in k or "player" in k or "dr" in k:
        return "博士"
    return ""


def iter_raw(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_chapter_map(rows) -> dict[str, str]:
    """关卡 id → 所属区域（如 main_00-10 → 序章 黑暗时代·上）。

    为什么需要它：剧情页自己的「剧情分组」只写了章节**名**（黑暗时代·上），
    不含「第几章」。而用户问的是「第一章」。所属区域字段恰好是
    「序章 黑暗时代·上」「第一章 xxx」这种带序号的完整写法，
    用它才能回答"第一章讲了什么"。
    """
    m: dict[str, str] = {}
    for d in rows:
        c = d.get("content", "")
        if "关卡id=" not in c:
            continue
        lid = re.search(r"关卡id=([^|\s\n]+)", c)
        area = re.search(r"所属区域=([^|\n]+)", c)
        if lid and area:
            lid_v = lid.group(1).strip()
            area_v = area.group(1).strip()
            if lid_v and area_v and area_v != "-":
                m[lid_v] = area_v
    return m


def extract_body(content: str) -> str:
    """抠出 `文本数据=` 之后到模板结束之间的原文。

    用括号计数而不是正则非贪婪 —— 模板里还有 {{color|...}} 这类嵌套，
    非贪婪会在第一个 }} 就截断（第 4 步语音解析踩过完全一样的坑）。
    """
    i = content.find("文本数据=")
    if i < 0:
        return ""
    i += len("文本数据=")
    depth = 0
    j = i
    n = len(content)
    while j < n:
        if content.startswith("{{", j):
            depth += 1
            j += 2
            continue
        if content.startswith("}}", j):
            if depth == 0:
                break
            depth -= 1
            j += 2
            continue
        j += 1
    return content[i:j]


def line_to_text(line: str) -> tuple[str, list[str]]:
    """把一行原始剧情转成 (正文, 本行出现的角色名)。

    返回空串表示这行是纯演出指令，应该整行丢掉。
    """
    speakers = [s.strip() for s in SPEAKER_RE.findall(line)]
    # 先摘掉说话人标记，剩下的才是指令/正文
    rest = SPEAKER_RE.sub("", line)
    # 顺序有讲究：先跑 wiki 清洗，再删引擎指令。
    # 反过来的话 [[Player]] 这类 wiki 链接会被指令正则啃掉一半，留一个孤零零的 [。
    rest = clean_inline(rest).strip()
    rest = CMD_RE.sub("", rest)
    rest = BRACE_PH_RE.sub(_sub_placeholder, rest)
    rest = CHAR_NOISE.sub(" ", rest).strip()
    if not rest:
        return "", speakers
    # 有说话人 → "临光：台词"，让角色名进入正文可被检索
    if speakers:
        return f"{speakers[0]}：{rest}", speakers
    return rest, speakers


def parse_story(title: str, content: str, chapter_map: dict[str, str]) -> dict | None:
    m = re.search(r"剧情类型=([^|}\n]+)", content)
    if not m:
        return None
    stype = m.group(1).strip()
    if stype not in WANT_TYPES:
        return None

    grp = re.search(r"剧情分组=([^|}\n]+)", content)
    group = grp.group(1).strip() if grp else ""

    # 章节：优先用带序号的「所属区域」，退回剧情分组
    chapter = ""
    pm = PATH_ID_RE.search(content)
    if pm:
        raw = pm.group(1).strip()
        base = raw.rsplit("/", 1)[-1]          # level_main_00-10_beg
        base = re.sub(r"_(beg|end)$", "", base, flags=re.I)
        base = re.sub(r"^level_", "", base)     # main_00-10
        chapter = chapter_map.get(base, "")

    body = extract_body(content)
    if not body.strip():
        return None

    lines: list[str] = []
    speakers: list[str] = []
    for raw_line in body.split("\n"):
        txt, sp = line_to_text(raw_line)
        for s in sp:
            if s and s not in speakers:
                speakers.append(s)
        if txt:
            lines.append(txt)

    text = "\n".join(lines).strip()
    if len(text) < 30:                          # 太短的碎片没有检索价值
        return None

    # 关卡代号 / 关卡名：标题形如 "0-10 困境/BEG"
    lv_code, lv_name, part = "", "", ""
    head = title.split("/")[0]
    pm2 = re.match(r"^([A-Za-z0-9\-]+)\s+(.+)$", head)
    if pm2:
        lv_code, lv_name = pm2.group(1), pm2.group(2)
    if "/" in title:
        part = title.rsplit("/", 1)[-1].upper()
        if part not in ("BEG", "END"):
            part = ""

    return {
        "page": title,
        "story_type": stype,
        "group": group,
        "chapter": chapter or group,
        "level_code": lv_code,
        "level_name": lv_name,
        "part": {"BEG": "行动前", "END": "行动后"}.get(part, part),
        "speakers": ",".join(speakers[:40]),
        "content": text,
        "char_count": len(text),
    }


DDL = """
CREATE TABLE IF NOT EXISTS lore (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    page        TEXT UNIQUE,
    story_type  TEXT,
    chapter     TEXT,
    group_name  TEXT,
    level_code  TEXT,
    level_name  TEXT,
    part        TEXT,
    speakers    TEXT,
    content     TEXT,
    char_count  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_lore_type ON lore(story_type);
CREATE INDEX IF NOT EXISTS idx_lore_chapter ON lore(chapter);
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 页，调试用")
    args = ap.parse_args()

    if not RAW_PATH.exists():
        print(f"找不到 {RAW_PATH}，请先跑 01_fetch.py")
        sys.exit(1)

    print("第一遍：扫描全文，建立 关卡id → 所属区域 映射 …")
    rows = list(iter_raw(RAW_PATH))
    if args.limit:
        rows = rows[: args.limit]
    chapter_map = build_chapter_map(rows)
    print(f"    映射 {len(chapter_map)} 条")

    print("第二遍：解析剧情页 …")
    parsed: list[dict] = []
    skipped = 0
    for d in rows:
        r = parse_story(d.get("title", ""), d.get("content", ""), chapter_map)
        if r:
            parsed.append(r)
        else:
            skipped += 1
    print(f"    解析成功 {len(parsed)} 条，跳过 {skipped} 条")

    if not parsed:
        print("没有解析到任何剧情，终止（避免建空表）")
        sys.exit(1)

    by_type: dict[str, int] = defaultdict(int)
    chars: dict[str, int] = defaultdict(int)
    for r in parsed:
        by_type[r["story_type"]] += 1
        chars[r["story_type"]] += r["char_count"]
    print()
    print("    按类型：")
    for t, n in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"      {t:<6} {n:>5} 页   {chars[t]/1000:>8.1f}k 字符")

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(DDL)
    # 全量重建：先清空。
    # 不做这一步的话，上一版解析出来的脏数据（比如清洗后过短的碎片）
    # 会一直赖在表里，而且你根本看不出来它是旧的 —— 表只会越攒越多。
    conn.execute("DELETE FROM lore")
    conn.executemany(
        "INSERT OR REPLACE INTO lore "
        "(page, story_type, chapter, group_name, level_code, level_name, "
        " part, speakers, content, char_count) "
        "VALUES (:page,:story_type,:chapter,:group,:level_code,:level_name,"
        ":part,:speakers,:content,:char_count)",
        parsed,
    )
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM lore").fetchone()[0]
    print()
    print(f"入库完成 → lore 表共 {total} 条")

    # ---- 自检：残留标记扫描（第 5 步吃过亏，检查项本身也要被检查）----
    print()
    print("自检：残留演出指令扫描")
    bad = 0
    for page, c in conn.execute("SELECT page, content FROM lore"):
        if re.search(r"\[(Character|Dialog|Delay|Background|HEADER|Blocker)", c):
            bad += 1
            if bad <= 3:
                print(f"    ! {page}")
    print(f"    残留指令的页：{bad}")

    empty_ch = conn.execute(
        "SELECT COUNT(*) FROM lore WHERE chapter='' OR chapter IS NULL").fetchone()[0]
    print(f"    章节为空的页：{empty_ch}")

    brace = conn.execute(
        "SELECT COUNT(*) FROM lore WHERE content LIKE '%{@%'").fetchone()[0]
    print(f"    残留 {{@占位符}} 的页：{brace}")

    print()
    print("章节样例（主线前 8）：")
    for r in conn.execute(
            "SELECT chapter, COUNT(*), SUM(char_count) FROM lore "
            "WHERE story_type='主线' GROUP BY chapter "
            "ORDER BY MIN(id) LIMIT 8"):
        print(f"    {r[0]:<24} {r[1]:>3} 页   {r[2]/1000:>7.1f}k")


if __name__ == "__main__":
    main()
