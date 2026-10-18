"""
SQLite 存储层：建库 + 写入 + 查询封装。

对应 ETL 里的 **L（Load）**：上游解析好的 JSONL 落到关系表里。

为什么这里不用 ORM（SQLAlchemy）：
  本项目查询模式固定、表结构由我们自己定，直接写 SQL 更易读、零依赖，
  也方便在答辩时把一条真实 SQL 贴出来讲。
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from .schema import INDEXES, TABLES, VIEWS

# ---------------------------------------------------------------- 小工具

_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}

# 面板属性字段 → 列名（wiki 里不同分支叫法略有差异）
_STAT_MAP = {
    "生命上限": "hp", "生命": "hp", "最大生命": "hp",
    "攻击": "atk", "攻击力": "atk",
    "防御": "defense", "防御力": "defense",
    "法术抗性": "res", "法抗": "res",
}


def _to_amount(v) -> int | None:
    """把 wiki 写法归一化成整数：180000 / "8万" / "3w" / "1.5k"。

    必须做，否则 SUM(数量) 会把字符串当 0，材料汇总直接算错。
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip().replace(",", "")
    if not s:
        return None
    m = re.match(r"^(-?[\d.]+)\s*([万wW千kK]?)$", s)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2).lower()
    if unit in ("万", "w"):
        num *= 10000
    elif unit in ("千", "k"):
        num *= 1000
    return int(num)


def _jdump(obj) -> str | None:
    if obj in (None, {}, [], ""):
        return None
    return json.dumps(obj, ensure_ascii=False)


def parse_skill_stage(raw: str) -> tuple[int | None, int | None, int, int]:
    """解析技能升级材料的阶段 key。

    wiki 的 key 有两种，含义完全不同，建错表会导致材料查询串号：
        "2" ~ "7"   → 普通技能等级（所有技能共用一套材料）
        "一8" "二9" "三10" → 第 N 个技能 升到 第 M 级（M>=8 即专精）
                            前缀 一/二/三 是「第几个技能」，不是专精等级！
        验证：全站 "一8" 出现 398 页、"三8" 只有 137 页 —— 正好是有 3 个技能的干员数。

    返回 (skill_idx, target_level, is_mastery, mastery_lv)
    """
    raw = (raw or "").strip()
    if raw.isdigit():
        lv = int(raw)
        return (None, lv, 0, 0)
    m = re.match(r"^([一二三四五六])(\d+)$", raw)
    if m:
        idx = _CN_NUM.get(m.group(1))
        lv = int(m.group(2))
        return (idx, lv, 1 if lv >= 8 else 0, lv - 7 if lv >= 8 else 0)
    return (None, None, 0, 0)


def _split_level(key: str) -> tuple[str, int]:
    """把「生命2」「攻击3」拆成 (属性名, 模组等级)。没有数字后缀则视为 1 级。"""
    k = (key or "").strip()
    m = re.match(r"^(.+?)([2-9])$", k)
    if m:
        return m.group(1), int(m.group(2))
    return k, 1


def _material_level(key: str) -> int:
    """「材料消耗」→1，「材料消耗2」→2，「材料消耗3」→3"""
    k = (key or "").strip()
    m = re.match(r"^材料消耗(\d)$", k)
    return int(m.group(1)) if m else 1


def _talent_level(key: str) -> int:
    """「天赋2」→2，「天赋3」→3"""
    m = re.match(r"^天赋(\d)$", (key or "").strip())
    return int(m.group(1)) if m else 1


# ---------------------------------------------------------------- 建库

def connect(db_path: str | Path) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    # 外键约束默认关闭，显式打开，避免脏数据
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    """建表。先 DROP 再 CREATE —— 保证脚本可以重复运行（幂等）。

    坑：DROP 时必须临时关掉外键约束。
    我们打开了 PRAGMA foreign_keys=ON（为了拦脏数据），
    但 DROP TABLE 是按定义顺序执行的，主表 operators 排在子表前面，
    而子表有 REFERENCES operators(id) —— 带约束删主表会直接报错。
    这个 bug 只在「数据库已存在、第二次运行」时才暴露，第一次跑不会遇到。
    """
    conn.execute("PRAGMA foreign_keys = OFF")
    for _, _, ddl in TABLES:
        name = re.search(r"CREATE TABLE (\w+)", ddl).group(1)
        conn.execute(f"DROP TABLE IF EXISTS {name}")
        conn.execute(ddl)
    for _, ddl in VIEWS:
        name = re.search(r"CREATE VIEW (\w+)", ddl).group(1)
        conn.execute(f"DROP VIEW IF EXISTS {name}")
        conn.execute(ddl)
    for idx in INDEXES:
        conn.execute(idx)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.commit()


# ---------------------------------------------------------------- 写入

def load_operators(conn: sqlite3.Connection, jsonl_path: str | Path) -> dict:
    """把 operators.jsonl 写进所有干员相关表。返回统计。"""
    stat: dict[str, int] = {}
    unmapped_stat_keys: dict[str, int] = {}

    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    for r in rows:
        cur = conn.execute(
            """INSERT INTO operators
               (name, name_cn, name_en, name_jp, char_id, rarity, profession,
                sub_profession, position, tags, trait, artist, cv_cn, cv_jp,
                op_seq, category, is_alt, is_upgrade, mode_suffix, obtain,
                release_time, source_url)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                r.get("名称"), r.get("干员名"), r.get("外文名"), r.get("日文名"),
                r.get("干员ID"), r.get("星级"), r.get("职业"), r.get("分支"),
                r.get("位置"), _jdump(r.get("标签")), r.get("特性"),
                r.get("画师"), r.get("中文配音"), r.get("日文配音"),
                r.get("干员序号"), r.get("干员类别"),
                1 if r.get("是否异格") else 0, 1 if r.get("是否升变") else 0,
                r.get("模式后缀") or None, r.get("获得方式"),
                r.get("上线时间"), r.get("来源"),
            ),
        )
        oid = cur.lastrowid
        stat["operators"] = stat.get("operators", 0) + 1

        # --- 势力 ---
        for kind, val in (r.get("势力") or {}).items():
            if val:
                conn.execute(
                    "INSERT INTO factions (operator_id, kind, value) VALUES (?,?,?)",
                    (oid, kind, str(val).strip()),
                )
                stat["factions"] = stat.get("factions", 0) + 1

        # --- 档案（结构化）---
        prof = r.get("档案") or {}
        if prof:
            known = ("性别", "出身地", "生日", "种族", "身高", "战斗经验",
                     "是否感染者", "矿石病感染情况")
            extra = {k: v for k, v in prof.items() if k not in known}
            conn.execute(
                """INSERT INTO profiles (operator_id, gender, birthplace, birthday,
                   race, height, combat_exp, infected, ore_status, extra)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (oid, prof.get("性别"), prof.get("出身地"), prof.get("生日"),
                 prof.get("种族"), prof.get("身高"), prof.get("战斗经验"),
                 prof.get("是否感染者"), prof.get("矿石病感染情况"), _jdump(extra)),
            )
            stat["profiles"] = stat.get("profiles", 0) + 1

        # --- 档案正文 ---
        for i, t in enumerate(r.get("档案文本") or [], 1):
            conn.execute(
                """INSERT INTO archive_texts (operator_id, idx, title, unlock_cond, content)
                   VALUES (?,?,?,?,?)""",
                (oid, i, t.get("标题"), t.get("解锁条件"), t.get("正文")),
            )
            stat["archive_texts"] = stat.get("archive_texts", 0) + 1

        # --- 面板 ---
        panel = r.get("面板") or {}
        if panel:
            conn.execute(
                """INSERT INTO base_stats (operator_id, redeploy, cost, block_count,
                   attack_speed, module_names) VALUES (?,?,?,?,?,?)""",
                (oid, panel.get("再部署时间"), panel.get("部署费用"),
                 panel.get("阻挡数"), panel.get("攻击速度"),
                 _jdump(panel.get("模组名"))),
            )
            stat["base_stats"] = stat.get("base_stats", 0) + 1
            for lv_key, vals in (panel.get("levels") or {}).items():
                cols = {"hp": None, "atk": None, "defense": None, "res": None}
                for k, v in (vals or {}).items():
                    col = _STAT_MAP.get(k)
                    if col:
                        cols[col] = v if isinstance(v, int) else _to_amount(v)
                    else:
                        unmapped_stat_keys[k] = unmapped_stat_keys.get(k, 0) + 1
                conn.execute(
                    """INSERT INTO level_stats (operator_id, level_key, hp, atk, defense, res)
                       VALUES (?,?,?,?,?,?)""",
                    (oid, lv_key, cols["hp"], cols["atk"], cols["defense"], cols["res"]),
                )
                stat["level_stats"] = stat.get("level_stats", 0) + 1

        # --- 精英化材料 ---
        for stage, mats in (r.get("精英化材料") or {}).items():
            for m in mats or []:
                conn.execute(
                    "INSERT INTO elite_materials (operator_id, stage, material, amount) VALUES (?,?,?,?)",
                    (oid, stage, m.get("name"), _to_amount(m.get("amount"))),
                )
                stat["elite_materials"] = stat.get("elite_materials", 0) + 1

        # --- 技能升级材料 ---
        for stage_raw, mats in (r.get("技能升级材料") or {}).items():
            idx, lv, is_m, m_lv = parse_skill_stage(stage_raw)
            for m in mats or []:
                conn.execute(
                    """INSERT INTO skill_materials (operator_id, stage_raw, skill_idx,
                       target_level, is_mastery, mastery_lv, material, amount)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (oid, stage_raw, idx, lv, is_m, m_lv,
                     m.get("name"), _to_amount(m.get("amount"))),
                )
                stat["skill_materials"] = stat.get("skill_materials", 0) + 1

        # --- 模组 ---
        for i, mo in enumerate(r.get("模组") or [], 1):
            cur2 = conn.execute(
                """INSERT INTO modules (operator_id, idx, name, type, trait_add, trait, base_info, extra)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (oid, i, mo.get("名称"), mo.get("类型"),
                 1 if mo.get("特性追加") else 0, mo.get("特性"),
                 mo.get("基础信息"),
                 _jdump({k: v for k, v in mo.items()
                         if k not in ("名称", "类型", "特性追加", "特性", "基础信息",
                                      "数值", "解锁材料", "天赋更新")})),
            )
            mid = cur2.lastrowid
            stat["modules"] = stat.get("modules", 0) + 1
            for k, v in (mo.get("数值") or {}).items():
                stat_key, level = _split_level(k)
                conn.execute(
                    "INSERT INTO module_stats (module_id, level, stat_key, value) VALUES (?,?,?,?)",
                    (mid, level, stat_key, v if isinstance(v, int) else _to_amount(v)),
                )
                stat["module_stats"] = stat.get("module_stats", 0) + 1
            for k, mats in (mo.get("解锁材料") or {}).items():
                level = _material_level(k)
                for m in mats or []:
                    conn.execute(
                        "INSERT INTO module_materials (module_id, level, material, amount) VALUES (?,?,?,?)",
                        (mid, level, m.get("name"), _to_amount(m.get("amount"))),
                    )
                    stat["module_materials"] = stat.get("module_materials", 0) + 1
            for k, txt in (mo.get("天赋更新") or {}).items():
                conn.execute(
                    "INSERT INTO module_talents (module_id, level, talent_key, text) VALUES (?,?,?,?)",
                    (mid, _talent_level(k), k, txt),
                )
                stat["module_talents"] = stat.get("module_talents", 0) + 1

        # --- 天赋 ---
        for i, t in enumerate(r.get("天赋") or [], 1):
            conn.execute(
                "INSERT INTO talents (operator_id, idx, name, unlock_cond, effect) VALUES (?,?,?,?,?)",
                (oid, i, t.get("名称"), t.get("解锁条件"), t.get("效果")),
            )
            stat["talents"] = stat.get("talents", 0) + 1

        # --- 技能 ---
        for i, sk in enumerate(r.get("技能") or [], 1):
            cur3 = conn.execute(
                """INSERT INTO skills (operator_id, idx, name, name_jp, name_en,
                   sp_type, trigger_type, extra) VALUES (?,?,?,?,?,?,?,?)""",
                (oid, i, sk.get("名称"), sk.get("技能名jp"), sk.get("技能名en"),
                 sk.get("回复类型"), sk.get("触发类型"),
                 _jdump({k: v for k, v in sk.items()
                         if k not in ("名称", "技能名jp", "技能名en", "回复类型",
                                      "触发类型", "等级")})),
            )
            sid = cur3.lastrowid
            stat["skills"] = stat.get("skills", 0) + 1
            for lv in sk.get("等级") or []:
                conn.execute(
                    """INSERT INTO skill_levels (skill_id, level, desc, init_sp, sp_cost, duration)
                       VALUES (?,?,?,?,?,?)""",
                    (sid, str(lv.get("等级")), lv.get("描述"),
                     _to_amount(lv.get("初始技力")), _to_amount(lv.get("技力消耗")),
                     _to_amount(lv.get("持续时间"))),
                )
                stat["skill_levels"] = stat.get("skill_levels", 0) + 1

        # --- 攻击范围 ---
        for k, v in (r.get("攻击范围") or {}).items():
            m = re.match(r"精英(\d)范围", k)
            conn.execute(
                "INSERT INTO attack_ranges (operator_id, elite, range_key) VALUES (?,?,?)",
                (oid, int(m.group(1)) if m else 0, v),
            )
            stat["attack_ranges"] = stat.get("attack_ranges", 0) + 1

        # --- 潜能 ---
        for p in r.get("潜能") or []:
            conn.execute(
                "INSERT INTO potentials (operator_id, level, effect) VALUES (?,?,?)",
                (oid, p.get("潜能"), p.get("效果")),
            )
            stat["potentials"] = stat.get("potentials", 0) + 1

        # --- 后勤技能 ---
        for i, b in enumerate(r.get("后勤技能") or [], 1):
            conn.execute(
                """INSERT INTO base_skills (operator_id, idx, name, unlock_stage, extra)
                   VALUES (?,?,?,?,?)""",
                (oid, i, b.get("名称"), b.get("解锁阶段"),
                 _jdump({k: v for k, v in b.items() if k not in ("名称", "解锁阶段")})),
            )
            stat["base_skills"] = stat.get("base_skills", 0) + 1

        # --- 相关道具 ---
        ri = r.get("相关道具") or {}
        if ri:
            conn.execute(
                """INSERT INTO related_items (operator_id, intro, intro_extra, token_use,
                   token_desc, extra) VALUES (?,?,?,?,?,?)""",
                (oid, ri.get("干员简介"), ri.get("干员简介补充"),
                 ri.get("信物用途"), ri.get("信物描述"),
                 _jdump({k: v for k, v in ri.items()
                         if k not in ("干员简介", "干员简介补充", "信物用途", "信物描述")})),
            )
            stat["related_items"] = stat.get("related_items", 0) + 1

        # --- 悖论模拟 ---
        px = r.get("悖论模拟") or {}
        if px:
            conn.execute(
                """INSERT INTO paradox (operator_id, name, description, unlock_elite,
                   unlock_level, rewards) VALUES (?,?,?,?,?,?)""",
                (oid, px.get("名称"), px.get("描述"),
                 px.get("解锁精英化"), px.get("解锁等级"), _jdump(px.get("报酬"))),
            )
            stat["paradox"] = stat.get("paradox", 0) + 1

        # --- 干员密录索引 ---
        for i, s in enumerate(r.get("干员密录") or [], 1):
            conn.execute(
                """INSERT INTO operator_stories (operator_id, idx, story_set, intro,
                   unlock_elite, unlock_level, unlock_trust, page) VALUES (?,?,?,?,?,?,?,?)""",
                (oid, i, s.get("故事集"), s.get("简介"),
                 s.get("解锁精英化"), s.get("解锁等级"), s.get("解锁信赖"), s.get("正文页")),
            )
            stat["operator_stories"] = stat.get("operator_stories", 0) + 1

        # --- 子页面 ---
        for sp in r.get("子页面") or []:
            conn.execute(
                "INSERT INTO subpages (operator_id, page) VALUES (?,?)", (oid, sp))
            stat["subpages"] = stat.get("subpages", 0) + 1

    conn.commit()
    stat["_unmapped_stat_keys"] = unmapped_stat_keys
    return stat


def load_voices(conn: sqlite3.Connection, jsonl_path: str | Path) -> int:
    """写入语音。标题形如「泥岩/语音记录」，用第一段匹配干员名。"""
    names = {row[0]: row[1] for row in conn.execute("SELECT name, id FROM operators")}
    n = 0
    skipped = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = json.loads(line)
            title = v.get("标题") or ""
            op_name = title.split("/")[0]
            oid = names.get(op_name)
            if oid is None:
                skipped.append(op_name)
                continue
            for i, it in enumerate(v.get("条目") or [], 1):
                conn.execute(
                    """INSERT INTO voices (operator_id, idx, title, trigger, line, other_langs)
                       VALUES (?,?,?,?,?,?)""",
                    (oid, it.get("序号") or i, it.get("标题"), it.get("触发类型"),
                     it.get("台词"), _jdump(it.get("其他语言"))),
                )
                n += 1
    conn.commit()
    return n, skipped


def load_stories(conn: sqlite3.Connection, jsonl_path: str | Path) -> tuple[int, list]:
    """写入密录正文。标题形如「泥岩/干员密录/1」。"""
    n = 0
    skipped = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            page = s.get("标题") or ""
            op_name = page.split("/")[0]
            conn.execute(
                "INSERT INTO stories (page, operator, content, char_count) VALUES (?,?,?,?)",
                (page, op_name, s.get("正文"), s.get("字符数")),
            )
            n += 1
    conn.commit()
    return n, skipped


def build(db_path: str | Path, processed_dir: str | Path = "data/processed") -> dict:
    """一键建库：建表 → 写干员 → 写语音 → 写密录。可重复运行。"""
    processed_dir = Path(processed_dir)
    conn = connect(db_path)
    create_tables(conn)
    stat = load_operators(conn, processed_dir / "operators.jsonl")
    nv, skip_v = load_voices(conn, processed_dir / "voices.jsonl")
    ns, skip_s = load_stories(conn, processed_dir / "stories.jsonl")
    stat["voices"] = nv
    stat["stories"] = ns
    stat["_skipped_voices"] = skip_v
    conn.close()
    return stat


# ---------------------------------------------------------------- 查询

def q(conn: sqlite3.Connection, sql: str, params=()) -> list[sqlite3.Row]:
    return conn.execute(sql, params).fetchall()


def elite_materials(conn, name: str, stage: str | None = None) -> list[sqlite3.Row]:
    """某干员精英化材料。stage=None 则返回全部。"""
    if stage:
        return q(conn, """
            SELECT e.stage, e.material, e.amount
            FROM elite_materials e JOIN operators o ON o.id = e.operator_id
            WHERE o.name = ? AND e.stage = ?
            ORDER BY e.amount DESC
        """, (name, stage))
    return q(conn, """
        SELECT e.stage, e.material, e.amount
        FROM elite_materials e JOIN operators o ON o.id = e.operator_id
        WHERE o.name = ? ORDER BY e.stage, e.amount DESC
    """, (name,))


def total_materials(conn, name: str, include_module: bool = True) -> list[sqlite3.Row]:
    """某干员的全部养成材料汇总（精英化 + 模组，按材料名 SUM）。

    这是「把泥岩精二+拉满三级模组，一共需要多少固源岩」这条多跳问题的底座。
    """
    sql = """
        SELECT material AS 材料, SUM(amount) AS 总数量, '精英化+模组' AS 来源
        FROM (
            SELECT e.material, e.amount
            FROM elite_materials e JOIN operators o ON o.id = e.operator_id
            WHERE o.name = ?
    """
    if include_module:
        sql += """
            UNION ALL
            SELECT mm.material, mm.amount
            FROM module_materials mm
            JOIN modules m ON m.id = mm.module_id
            JOIN operators o ON o.id = m.operator_id
            WHERE o.name = ? AND m.name NOT LIKE '%证章%'
        """
    sql += """
        ) GROUP BY material ORDER BY 总数量 DESC
    """
    params = (name, name) if include_module else (name,)
    return q(conn, sql, params)


def faction_members(conn, value: str, kind: str | None = None) -> list[sqlite3.Row]:
    """某势力/阵营的成员列表。「罗德岛有哪些干员」"""
    if kind:
        return q(conn, """
            SELECT o.name, o.rarity, o.profession
            FROM factions f JOIN operators o ON o.id = f.operator_id
            WHERE f.value = ? AND f.kind = ? ORDER BY o.rarity DESC, o.name
        """, (value, kind))
    return q(conn, """
        SELECT o.name, o.rarity, o.profession, f.kind
        FROM factions f JOIN operators o ON o.id = f.operator_id
        WHERE f.value = ? ORDER BY o.rarity DESC, o.name
    """, (value,))


def skill_desc(conn, name: str, skill_name: str | None = None) -> list[sqlite3.Row]:
    """技能描述（含各等级档位）。"""
    if skill_name:
        return q(conn, """
            SELECT s.name AS 技能, sl.level AS 等级, sl.desc AS 描述,
                   sl.init_sp, sl.sp_cost, sl.duration
            FROM skill_levels sl
            JOIN skills s ON s.id = sl.skill_id
            JOIN operators o ON o.id = s.operator_id
            WHERE o.name = ? AND s.name LIKE ?
            ORDER BY s.idx, sl.id   -- sl.id 即插入顺序 = 1~7 级 + 专精一/二/三
        """, (name, f"%{skill_name}%"))
    return q(conn, """
        SELECT s.name AS 技能, s.sp_type, s.trigger_type, COUNT(sl.id) AS 档位数
        FROM skills s
        JOIN operators o ON o.id = s.operator_id
        LEFT JOIN skill_levels sl ON sl.skill_id = s.id
        WHERE o.name = ? GROUP BY s.id ORDER BY s.idx
    """, (name,))


def stat_at(conn, name: str, level_key: str = "精2_满级") -> sqlite3.Row | None:
    """某干员某档位的面板数值。"""
    rows = q(conn, """
        SELECT ls.level_key, ls.hp, ls.atk, ls.defense, ls.res
        FROM level_stats ls JOIN operators o ON o.id = ls.operator_id
        WHERE o.name = ? AND ls.level_key = ?
    """, (name, level_key))
    return rows[0] if rows else None


def voice_lines(conn, name: str, title_kw: str | None = None) -> list[sqlite3.Row]:
    """语音台词。title_kw 支持模糊匹配（如「交谈」「信赖」）。"""
    if title_kw:
        return q(conn, """
            SELECT v.title, v.line, v.trigger
            FROM voices v JOIN operators o ON o.id = v.operator_id
            WHERE o.name = ? AND v.title LIKE ? ORDER BY v.idx
        """, (name, f"%{title_kw}%"))
    return q(conn, """
        SELECT v.title, v.line FROM voices v JOIN operators o ON o.id = v.operator_id
        WHERE o.name = ? ORDER BY v.idx
    """, (name,))


def profile_of(conn, name: str) -> sqlite3.Row | None:
    rows = q(conn, """
        SELECT p.*, o.name FROM profiles p JOIN operators o ON o.id = p.operator_id
        WHERE o.name = ?
    """, (name,))
    return rows[0] if rows else None


def table_counts(conn) -> list[tuple[str, int]]:
    out = []
    for name, _, _ in TABLES:
        n = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        out.append((name, n))
    return out
