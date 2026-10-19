"""第 11 步补丁：解析「升级消耗」—— 补齐养成清单里最后一块拼图。

【起因：用户说"数据还是有误"】
满练度清单里只有精英化 / 技能 / 专精 / 模组的材料，
唯独缺了**把等级从 1 练到上限**的开销（经验值 + 龙门币）。
用户要的是：

    1级到50级、精英化1、精英化1级到80级、精英化2、精英化2到90级、技能

缺了等级段，这份清单就是残缺的 —— 而残缺的清单会让用户误以为"就这些"。

【第一反应是去抓，结果发现不用抓】
第一次抓全站时（01_fetch）就顺带存下了这些通用数据页：
    「干员升级数值」  7626 字，三个 <tabber> 标签：精英阶段0/1/2
    「干员等级上限」  372 字，按稀有度给出各阶段等级上限
    「基础/初级/中级/高级作战记录」  经验换算 200/400/1000/2000
这些页面**本来就在 data/raw/pages.jsonl 里**，只是从来没被解析过。

这件事本身值得记一笔：**补数据前先翻本地有没有**。
"数据源缺失"和"我有数据但没解析"是两种完全不同的问题，
后者的修复成本是前者的百分之一。

【数据长什么样】
「干员升级数值」每个精英阶段一张 wikitable：

    !当前等级 !!升级所需经验 !!累积经验 !!升级所需龙门币 !!累计龙门币
    |1||100||0||30||0
    |2||117||100||36||30
    ...
    |50||—||24400||—||26719

「累计」列的语义是**从 1 级升到该等级**的总量，
所以取 level=上限 那一行的 exp_cum / money_cum 就是该阶段的完整开销。
（不是"升下一级"的量 —— 满级行写的是"—"，因为没下一级了。）

【等级上限为什么必须按稀有度取】
精英0 的上限：6星50 / 5星50 / 4星45 / 3星40 / 2星30 / 1星30。
同一个"精英0 升满"对不同星级的干员是不同数字，
所以查询侧必须 JOIN operators.rarity → level_cap 才能算准。

【产物】
    level_up_cost(elite, level, exp_need, exp_cum, money_need, money_cum)
    level_cap(rarity, elite, max_level)
    exp_card(name, exp_value)

【用法】
    python scripts/22_parse_levelcost.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_PATH = ROOT / "data" / "raw" / "pages.jsonl"
DB_PATH = ROOT / "data" / "arknights.db"

# 升级数值页：按 "精英阶段N=" 切分，每段一张表
UP_PAGE = "干员升级数值"
CAP_PAGE = "干员等级上限"
CARD_PAGES = ["基础作战记录", "初级作战记录", "中级作战记录", "高级作战记录"]

# |1||100||0||30||0   —— 最后一列可能带行尾空格
ROW_RE = re.compile(r"^\|(\d+)\|\|(\S+?)\|\|(\d+)\|\|(\S+?)\|\|(\d+)\s*$", re.M)
# |{{材料消耗|中级作战记录}}||1000
CARD_RE = re.compile(r"\|\{\{材料消耗\|(.+?)\}\}\|\|(\d+)")


def load_page(title: str) -> str:
    """从本地原始抓取里取出指定页面的 wikitext。"""
    with open(RAW_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("title") == title:
                return r.get("content") or ""
    return ""


def _int_or_none(s: str):
    """"—" 表示没有下一级（满级行），存 NULL 而不是 0。

    存 0 是有害的：0 会被当成真值参与 SUM/比较，
    NULL 才会被聚合函数正确忽略。
    """
    return None if s in ("—", "-", "", "—") else int(s)


def parse_up_cost(text: str) -> list[tuple]:
    """解析「干员升级数值」，返回 (elite, level, exp_need, exp_cum, money_need, money_cum)。"""
    out: list[tuple] = []
    # 用捕获组切分：['前言', '0', '精英0段', '1', '精英1段', '2', '精英2段']
    parts = re.split(r"^精英阶段(\d)=$", text, flags=re.M)
    for i in range(1, len(parts), 2):
        elite = int(parts[i])
        body = parts[i + 1]
        for lv, exp_need, exp_cum, money_need, money_cum in ROW_RE.findall(body):
            out.append((elite, int(lv),
                        _int_or_none(exp_need), int(exp_cum),
                        _int_or_none(money_need), int(money_cum)))
    return out


def parse_level_cap(text: str) -> list[tuple]:
    """解析「干员等级上限」。

    原文（表头列序 = 6星→1星）：
        !!★6!!★5!!★4!!★3!!★2!!★1
        !精英阶段0 ... |50||50||45||40||30||30
    """
    rarities = [6, 5, 4, 3, 2, 1]
    out: list[tuple] = []
    lines = text.splitlines()
    cur_elite = None
    for ln in lines:
        m = re.search(r"精英阶段(\d)", ln)
        if m:
            cur_elite = int(m.group(1))
            continue
        if cur_elite is None or not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("||")]
        if len(cells) != len(rarities):
            continue
        for rarity, cell in zip(rarities, cells):
            if cell.isdigit():
                out.append((rarity, cur_elite, int(cell)))
        cur_elite = None          # 一行的上限读完就等下一个阶段标题
    return out


def parse_exp_card() -> list[tuple]:
    """解析四张作战记录页的「提供经验」。"""
    out: list[tuple] = []
    for title in CARD_PAGES:
        text = load_page(title)
        if not text:
            print(f"    ! 本地缺页：{title}")
            continue
        m = CARD_RE.search(text)
        if not m:
            print(f"    ! {title} 没找到经验值")
            continue
        out.append((m.group(1).strip(), int(m.group(2))))
    return out


DDL = """
CREATE TABLE IF NOT EXISTS level_up_cost (
    elite      INTEGER NOT NULL,
    level      INTEGER NOT NULL,
    exp_need   INTEGER,
    exp_cum    INTEGER,
    money_need INTEGER,
    money_cum  INTEGER,
    PRIMARY KEY (elite, level)
);
CREATE TABLE IF NOT EXISTS level_cap (
    rarity    INTEGER NOT NULL,
    elite     INTEGER NOT NULL,
    max_level INTEGER,
    PRIMARY KEY (rarity, elite)
);
CREATE TABLE IF NOT EXISTS exp_card (
    name      TEXT PRIMARY KEY,
    exp_value INTEGER
);
"""


def main() -> None:
    if not RAW_PATH.exists():
        print(f"原始数据不存在：{RAW_PATH}")
        return

    print("=" * 60)
    print("解析「升级消耗」（等级经验 + 龙门币）")
    print("=" * 60)

    up_text = load_page(UP_PAGE)
    cap_text = load_page(CAP_PAGE)
    if not up_text or not cap_text:
        print(f"本地缺页：{UP_PAGE if not up_text else CAP_PAGE}")
        print("需要先跑 01_fetch.py 抓取全站。")
        return

    ups = parse_up_cost(up_text)
    caps = parse_level_cap(cap_text)
    cards = parse_exp_card()
    print(f"  升级数值行：{len(ups)}")
    print(f"  等级上限行：{len(caps)}")
    print(f"  作战记录：  {cards}")

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(DDL)
    # 幂等：这三张表是纯字典表，每次全量重建
    conn.execute("DELETE FROM level_up_cost")
    conn.execute("DELETE FROM level_cap")
    conn.execute("DELETE FROM exp_card")
    conn.executemany(
        "INSERT OR REPLACE INTO level_up_cost "
        "(elite, level, exp_need, exp_cum, money_need, money_cum) "
        "VALUES (?,?,?,?,?,?)", ups)
    conn.executemany(
        "INSERT OR REPLACE INTO level_cap (rarity, elite, max_level) "
        "VALUES (?,?,?)", caps)
    conn.executemany(
        "INSERT OR REPLACE INTO exp_card (name, exp_value) VALUES (?,?)", cards)
    conn.commit()

    print()
    print("入库：level_up_cost / level_cap / exp_card")

    # ---- 自检 1：等级必须连续，不能有断层 ----
    print()
    print("自检：等级序列连续性")
    bad = 0
    for elite, n, mn, mx in conn.execute(
            "SELECT elite, COUNT(*), MIN(level), MAX(level) "
            "FROM level_up_cost GROUP BY elite ORDER BY elite"):
        ok = (mx - mn + 1) == n
        bad += 0 if ok else 1
        print(f"    精英{elite}: {mn}~{mx} 共 {n} 行  "
              f"{'连续' if ok else '! 有断层'}")

    # ---- 自检 2：累积列必须单调不减（累积量倒挂说明解析错位）----
    print()
    print("自检：累积经验/龙门币单调性")
    for elite in (0, 1, 2):
        rows = conn.execute(
            "SELECT level, exp_cum, money_cum FROM level_up_cost "
            "WHERE elite=? ORDER BY level", (elite,)).fetchall()
        mono = all(rows[i][1] <= rows[i + 1][1] and rows[i][2] <= rows[i + 1][2]
                   for i in range(len(rows) - 1))
        print(f"    精英{elite}: {'单调不减 OK' if mono else '! 有倒挂'}  "
              f"满级 经验={rows[-1][1]} 龙门币={rows[-1][2]}")

    # ---- 自检 3：等级上限表要覆盖 6 个星级 × 3 个阶段 ----
    n_cap = conn.execute("SELECT COUNT(*) FROM level_cap").fetchone()[0]
    print()
    print(f"    等级上限覆盖：{n_cap} 条"
          f"（6星×3阶段 = 18 为满，部分星级无精英2 属正常）")
    for r in conn.execute(
            "SELECT elite, GROUP_CONCAT(rarity || '星:' || max_level, '  ') "
            "FROM level_cap GROUP BY elite ORDER BY elite"):
        print(f"      精英{r[0]}: {r[1]}")

    # ---- 自检 4：6 星满级开销（用于人工核对）----
    print()
    print("6 星干员各阶段升满开销：")
    for elite in (0, 1, 2):
        row = conn.execute(
            "SELECT c.exp_cum, c.money_cum, cap.max_level FROM level_up_cost c "
            "JOIN level_cap cap ON cap.elite=c.elite AND cap.rarity=6 "
            "WHERE c.elite=? AND c.level=cap.max_level", (elite,)).fetchone()
        if row:
            print(f"    精英{elite} 1→{row[2]}级：经验 {row[0]:>7}  龙门币 {row[1]:>7}")

    conn.close()


if __name__ == "__main__":
    main()
