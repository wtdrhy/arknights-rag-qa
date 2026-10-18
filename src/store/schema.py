"""
数据库表结构（DDL）。

设计原则：把 JSON 里的嵌套结构拆成关系表，让「泥岩精二要多少固源岩」
这类问题变成一条 SQL，而不是去向量库里做语义相似度。

为什么要拆表（而不是整条 JSON 塞进一个 TEXT 字段）：
  1. 数值查询必须是确定性的 —— SQL 的 WHERE/SUM 是精确匹配，向量检索是概率匹配
  2. 多值字段（势力、材料、技能等级）如果不拆，就只能 LIKE 模糊匹配，又慢又错
  3. 数据库能建索引，几千条数据查询是毫秒级

术语：这个过程叫 **Normalization（规范化）**，
把一个字段里塞了多个值的表拆成主表 + 子表，就是第一范式（1NF）。
"""

# (表名, 说明, DDL)
TABLES: list[tuple[str, str, str]] = [
    (
        "operators",
        "干员主表，一行一个干员（业务主键=名称）",
        """
        CREATE TABLE operators (
            id             INTEGER PRIMARY KEY,
            name           TEXT NOT NULL UNIQUE,   -- 页面标题，唯一（如 泥岩 / Sharp(卫戍协议)）
            name_cn        TEXT,                   -- 中文代号
            name_en        TEXT,
            name_jp        TEXT,
            char_id        TEXT,                   -- wiki 内部 ID，如 char_311_mudrok
            rarity         INTEGER,                -- 星级 1~6（wiki 原始值已 +1）
            profession     TEXT,                   -- 职业：重装/近卫/术师...
            sub_profession TEXT,                   -- 分支：不屈者/尖兵...
            position       TEXT,                   -- 近战位 / 远程位
            tags           TEXT,                   -- JSON 数组
            trait          TEXT,                   -- 特性
            artist         TEXT,
            cv_cn          TEXT,
            cv_jp          TEXT,
            op_seq         INTEGER,                -- 干员序号；-1 = 特殊模式干员
            category       TEXT,                   -- 常规 / 特勤 / 特殊模式
            is_alt         INTEGER,                -- 是否异格形态
            is_upgrade     INTEGER,                -- 是否升变（阿米娅→近卫/医疗）
            mode_suffix    TEXT,                   -- 卫戍协议 / 集成战略 ...
            obtain         TEXT,                   -- 获得方式
            release_time   TEXT,
            source_url     TEXT
        )
        """,
    ),
    (
        "factions",
        "势力/阵营。一个干员可有多个，且分类型（国家/组织/团队）",
        """
        CREATE TABLE factions (
            id          INTEGER PRIMARY KEY,
            operator_id INTEGER NOT NULL REFERENCES operators(id),
            kind        TEXT,   -- 所属国家 / 所属组织 / 所属团队
            value       TEXT    -- 罗德岛 / 莱茵生命 / 彩虹小队 ...
        )
        """,
    ),
    (
        "profiles",
        "档案结构化字段（性别/种族/出身地/生日/身高...），与干员 1:1",
        """
        CREATE TABLE profiles (
            operator_id INTEGER PRIMARY KEY REFERENCES operators(id),
            gender      TEXT,
            birthplace  TEXT,
            birthday    TEXT,
            race        TEXT,
            height      TEXT,
            combat_exp  TEXT,
            infected    TEXT,
            ore_status  TEXT,
            extra       TEXT    -- JSON，兜住上面没列到的字段（如体检六维）
        )
        """,
    ),
    (
        "archive_texts",
        "档案正文（客观履历/临床诊断/档案资料...），带解锁条件",
        """
        CREATE TABLE archive_texts (
            id          INTEGER PRIMARY KEY,
            operator_id INTEGER NOT NULL REFERENCES operators(id),
            idx         INTEGER,
            title       TEXT,
            unlock_cond TEXT,
            content     TEXT
        )
        """,
    ),
    (
        "base_stats",
        "面板基础属性（再部署/费用/阻挡数/攻击间隔），与干员 1:1",
        """
        CREATE TABLE base_stats (
            operator_id  INTEGER PRIMARY KEY REFERENCES operators(id),
            redeploy     TEXT,
            cost         TEXT,
            block_count  TEXT,
            attack_speed TEXT,
            module_names TEXT    -- JSON 数组
        )
        """,
    ),
    (
        "level_stats",
        "面板数值（按等级档位拆行）——「泥岩精二满级防御力」查这里",
        """
        CREATE TABLE level_stats (
            id          INTEGER PRIMARY KEY,
            operator_id INTEGER NOT NULL REFERENCES operators(id),
            level_key   TEXT,   -- 精0_1级 / 精0_满级 / 精1_满级 / 精2_满级
            hp          INTEGER,
            atk         INTEGER,
            defense     INTEGER,
            res         INTEGER
        )
        """,
    ),
    (
        "elite_materials",
        "精英化材料：精1 / 精2 各一组",
        """
        CREATE TABLE elite_materials (
            id          INTEGER PRIMARY KEY,
            operator_id INTEGER NOT NULL REFERENCES operators(id),
            stage       TEXT,   -- 精1 / 精2
            material    TEXT,
            amount      INTEGER
        )
        """,
    ),
    (
        "skill_materials",
        "技能升级材料。stage_raw 形如 2~7 或 一8/二9/三10",
        """
        CREATE TABLE skill_materials (
            id           INTEGER PRIMARY KEY,
            operator_id  INTEGER NOT NULL REFERENCES operators(id),
            stage_raw    TEXT,    -- 原始 key
            skill_idx    INTEGER, -- 第几个技能（1/2/3）；普通等级为 NULL
            target_level INTEGER, -- 目标技能等级 2~10
            is_mastery   INTEGER, -- 是否专精（等级>=8）
            mastery_lv   INTEGER, -- 专精等级 1/2/3
            material     TEXT,
            amount       INTEGER
        )
        """,
    ),
    (
        "modules",
        "模组主表（一个干员可有多个模组）",
        """
        CREATE TABLE modules (
            id          INTEGER PRIMARY KEY,
            operator_id INTEGER NOT NULL REFERENCES operators(id),
            idx         INTEGER,
            name        TEXT,
            type        TEXT,   -- UNY-X / UNY-Y ...
            trait_add   INTEGER,
            trait       TEXT,
            base_info   TEXT,
            extra       TEXT    -- JSON 兜底
        )
        """,
    ),
    (
        "module_stats",
        "模组数值加成，按模组等级 1/2/3 拆行",
        """
        CREATE TABLE module_stats (
            id        INTEGER PRIMARY KEY,
            module_id INTEGER NOT NULL REFERENCES modules(id),
            level     INTEGER,
            stat_key  TEXT,   -- 生命 / 攻击 / 防御
            value     INTEGER
        )
        """,
    ),
    (
        "module_materials",
        "模组解锁/升级材料，按等级 1/2/3 拆行",
        """
        CREATE TABLE module_materials (
            id        INTEGER PRIMARY KEY,
            module_id INTEGER NOT NULL REFERENCES modules(id),
            level     INTEGER,
            material  TEXT,
            amount    INTEGER
        )
        """,
    ),
    (
        "module_talents",
        "模组带来的天赋更新文本",
        """
        CREATE TABLE module_talents (
            id        INTEGER PRIMARY KEY,
            module_id INTEGER NOT NULL REFERENCES modules(id),
            level     INTEGER,
            talent_key TEXT,
            text      TEXT
        )
        """,
    ),
    (
        "talents",
        "天赋（同一天赋在不同解锁条件下各占一行）",
        """
        CREATE TABLE talents (
            id          INTEGER PRIMARY KEY,
            operator_id INTEGER NOT NULL REFERENCES operators(id),
            idx         INTEGER,
            name        TEXT,
            unlock_cond TEXT,
            effect      TEXT
        )
        """,
    ),
    (
        "skills",
        "技能主表",
        """
        CREATE TABLE skills (
            id           INTEGER PRIMARY KEY,
            operator_id  INTEGER NOT NULL REFERENCES operators(id),
            idx          INTEGER,
            name         TEXT,
            name_jp      TEXT,
            name_en      TEXT,
            sp_type      TEXT,   -- 自动回复 / 受击回复
            trigger_type TEXT,   -- 手动触发 / 自动触发
            extra        TEXT
        )
        """,
    ),
    (
        "skill_levels",
        "技能等级档位（1~7 级 + 专精一/二/三，共 10 档）",
        """
        CREATE TABLE skill_levels (
            id       INTEGER PRIMARY KEY,
            skill_id INTEGER NOT NULL REFERENCES skills(id),
            level    TEXT,
            desc     TEXT,
            init_sp  INTEGER,
            sp_cost  INTEGER,
            duration INTEGER
        )
        """,
    ),
    (
        "attack_ranges",
        "攻击范围（精英化后可能变化）",
        """
        CREATE TABLE attack_ranges (
            id          INTEGER PRIMARY KEY,
            operator_id INTEGER NOT NULL REFERENCES operators(id),
            elite       INTEGER,
            range_key   TEXT
        )
        """,
    ),
    (
        "potentials",
        "潜能提升效果",
        """
        CREATE TABLE potentials (
            id          INTEGER PRIMARY KEY,
            operator_id INTEGER NOT NULL REFERENCES operators(id),
            level       INTEGER,
            effect      TEXT
        )
        """,
    ),
    (
        "base_skills",
        "后勤技能（基建技能）",
        """
        CREATE TABLE base_skills (
            id          INTEGER PRIMARY KEY,
            operator_id INTEGER NOT NULL REFERENCES operators(id),
            idx         INTEGER,
            name        TEXT,
            unlock_stage TEXT,
            extra       TEXT
        )
        """,
    ),
    (
        "related_items",
        "干员简介与信物描述",
        """
        CREATE TABLE related_items (
            operator_id INTEGER PRIMARY KEY REFERENCES operators(id),
            intro       TEXT,
            intro_extra TEXT,
            token_use   TEXT,
            token_desc  TEXT,
            extra       TEXT
        )
        """,
    ),
    (
        "paradox",
        "悖论模拟（1:1）",
        """
        CREATE TABLE paradox (
            operator_id  INTEGER PRIMARY KEY REFERENCES operators(id),
            name         TEXT,
            description  TEXT,
            unlock_elite INTEGER,
            unlock_level INTEGER,
            rewards      TEXT   -- JSON 数组
        )
        """,
    ),
    (
        "operator_stories",
        "干员密录索引（正文在 stories 表）",
        """
        CREATE TABLE operator_stories (
            id           INTEGER PRIMARY KEY,
            operator_id  INTEGER NOT NULL REFERENCES operators(id),
            idx          INTEGER,
            story_set    TEXT,
            intro        TEXT,
            unlock_elite INTEGER,
            unlock_level INTEGER,
            unlock_trust INTEGER,
            page         TEXT    -- 正文页标题，关联 stories.title
        )
        """,
    ),
    (
        "voices",
        "语音台词（中文台词 + 多语 JSON）",
        """
        CREATE TABLE voices (
            id          INTEGER PRIMARY KEY,
            operator_id INTEGER NOT NULL REFERENCES operators(id),
            idx         INTEGER,
            title       TEXT,   -- 任命助理 / 交谈1 / 信赖提升后交谈1 ...
            trigger     TEXT,   -- HOME_PLACE ...
            line        TEXT,
            other_langs TEXT    -- JSON：日文/英文/韩文/繁中
        )
        """,
    ),
    (
        "stories",
        "密录正文（长文本，后续也要切块进 Chroma）",
        """
        CREATE TABLE stories (
            id         INTEGER PRIMARY KEY,
            page       TEXT UNIQUE,
            operator   TEXT,   -- 从标题「泥岩/干员密录/1」推出
            content    TEXT,
            char_count INTEGER
        )
        """,
    ),
    (
        "subpages",
        "干员主页引用的子页面",
        """
        CREATE TABLE subpages (
            id          INTEGER PRIMARY KEY,
            operator_id INTEGER NOT NULL REFERENCES operators(id),
            page        TEXT
        )
        """,
    ),
]

# 常用查询要建的索引
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_elite_mat_op    ON elite_materials(operator_id)",
    "CREATE INDEX IF NOT EXISTS idx_elite_mat_name  ON elite_materials(material)",
    "CREATE INDEX IF NOT EXISTS idx_skill_mat_op    ON skill_materials(operator_id)",
    "CREATE INDEX IF NOT EXISTS idx_skill_mat_name  ON skill_materials(material)",
    "CREATE INDEX IF NOT EXISTS idx_faction_value   ON factions(value)",
    "CREATE INDEX IF NOT EXISTS idx_faction_op      ON factions(operator_id)",
    "CREATE INDEX IF NOT EXISTS idx_voice_op        ON voices(operator_id)",
    "CREATE INDEX IF NOT EXISTS idx_voice_title     ON voices(title)",
    "CREATE INDEX IF NOT EXISTS idx_level_stats_op  ON level_stats(operator_id)",
    "CREATE INDEX IF NOT EXISTS idx_skill_levels_sk ON skill_levels(skill_id)",
    "CREATE INDEX IF NOT EXISTS idx_talents_op      ON talents(operator_id)",
    "CREATE INDEX IF NOT EXISTS idx_stories_op      ON stories(operator)",
]

# 从 operators 派生的视图：把「名称 → id」屏蔽掉，方便手写 SQL
VIEWS = [
    (
        "v_elite_materials",
        """
        CREATE VIEW v_elite_materials AS
        SELECT o.name AS 干员, o.rarity AS 星级, e.stage AS 阶段,
               e.material AS 材料, e.amount AS 数量
        FROM elite_materials e JOIN operators o ON o.id = e.operator_id
        """,
    ),
    (
        "v_factions",
        """
        CREATE VIEW v_factions AS
        SELECT o.name AS 干员, f.kind AS 类型, f.value AS 势力
        FROM factions f JOIN operators o ON o.id = f.operator_id
        """,
    ),
    (
        "v_skill_materials",
        """
        CREATE VIEW v_skill_materials AS
        SELECT o.name AS 干员, s.stage_raw AS 阶段, s.skill_idx AS 技能序号,
               s.target_level AS 目标等级, s.is_mastery AS 是否专精,
               s.mastery_lv AS 专精等级, s.material AS 材料, s.amount AS 数量
        FROM skill_materials s JOIN operators o ON o.id = s.operator_id
        """,
    ),
]
