"""查询路由（Query Routing）：决定一个问题该走哪条轨道。

【这是什么】
RAG 系统里有两条轨道（第 4 步就埋好的设计）：

    SQL 轨道    结构化、精确、可验算 —— 问"精二要几个炽合金块"
    向量轨道    语义化、模糊、需总结 —— 问"她为什么戴头盔"

用户不会告诉你该走哪条，所以需要一个**路由器**在查询前先做判断。
这是 Agent 的"大脑"，业界叫 query routing / intent classification。

【本实现为什么用规则而不是 LLM】
    1. 确定性：同样的输入永远同样的输出，方便写单元测试和回归。
    2. 零延迟、零成本：规则匹配是微秒级，LLM 要几百毫秒到几秒。
    3. 可解释：每个判断都能打印出"因为命中了哪个词"。
       线上出问题时这点极其重要 —— LLM 的路由结果你没法 debug。

    第 8 步接 LLM 时，正确的做法是**规则兜底 + LLM 兜规则覆盖不到的**，
    而不是把规则全扔掉。生产系统里规则路由通常吃掉 70%+ 的流量。

【三条输出】
    intent     sql / semantic / hybrid
    entities   从问题里认出的干员名（用于元数据过滤）
    source     该看哪一类资料（档案/语音/密录/技能…）
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# 意图信号词
# ---------------------------------------------------------------------------
# 命中这些词 → 答案大概率在结构化字段里，能写成 WHERE 或 SUM
SQL_SIGNALS = [
    # 数量 / 聚合
    "多少", "几个", "几只", "数量", "总计", "一共", "总和", "合计",
    "排行", "排名", "最高", "最低", "最多", "最少", "最强", "前几", "前5", "前五",
    # 数值字段
    "星级", "稀有度", "攻击力", "攻击 ", "防御力", "防御 ", "生命值", "生命上限",
    "法抗", "法术抗性", "部署费用", "费用", "攻速", "攻击间隔", "阻挡数",
    "满级", "精二", "精一", "精0", "面板", "属性",
    # 材料 / 消耗
    "材料", "消耗", "需要", "要多少", "龙门币", "芯片", "专精", "精英化",
    # 枚举 / 名单
    "有哪些", "有哪些干员", "名单", "列表", "分别是", "属于哪个", "哪些人",
    # 职业 / 分支：这两个是 operators 表的结构化列，走 SQL 比语义检索准得多。
    # （第 9 步评测集发现：问"泥岩是什么职业"时这里缺词，落到了语义轨道，
    #   而 pick_sql 里明明有对应模板 —— 两层信号词不一致，路由层成了短板。）
    "职业", "分支",
]

# 命中这些词 → 答案散在长文本里，必须语义检索
SEMANTIC_SIGNALS = [
    "为什么", "什么原因", "为何", "怎么想", "怎么看", "感受", "想法", "态度",
    "性格", "什么样", "是怎样", "是个怎样", "为人",
    "关系", "之间", "和谁", "交情", "感情",
    "经历", "过去", "故事", "身世", "背景", "遭遇", "发生了什么",
    "剧情", "台词", "说过", "说了什么", "评价", "印象",
]

# 命中这些 → 明确要求"先看哪类资料"
# 注意 dict 是**有序**的，匹配时命中第一个就停 —— 所以更具体的要排在前面。
#
# 第 10 步修的一个真 bug：原来 "剧情" 挂在 story（干员密录）下面，
# 于是问"主线第一章讲了什么"被判定为查干员密录，
# 检索回来一堆密录片段，模型只能说"没有主线剧情内容"。
# 现在把 lore_* 排在 story 前面，"主线剧情"优先命中 lore_main。
SOURCE_HINTS = {
    "lore_main": ["主线", "主线剧情", "章节", "序章", "第几章", "主线故事"],
    "lore_side": ["支线", "支线剧情", "活动剧情", "sidestory", "side story"],
    "lore_set": ["故事集"],
    "voice": ["台词", "语音", "说过", "说了什么", "说话", "交谈"],
    # "剧情" 从这一项里拿掉了：光说"剧情"分不清是密录还是章节剧情，
    # 此时宁可**不加 source 过滤**，让语义检索三类剧情都能命中。
    "story": ["密录", "故事", "发生了什么", "那段"],
    "archive": ["档案", "履历", "客观履历", "档案资料", "晋升记录", "档案里"],
    "skill": ["技能", "技能效果", "专精"],
    "talent": ["天赋"],
    "module": ["模组"],
    "paradox": ["悖论", "模拟"],
}

# 章节匹配用：问"第一章"要能对上库里的"第一章 黑暗时代·下"。
# 序章/终章没有序号，必须单独列 —— 只写"第X章"的话
# "主线剧情序章发生了什么"会匹配不上（实测踩到）。
_CHAPTER_NUM_RE = re.compile(r"第[一二三四五六七八九十百零0-9]+章|序章|终章|尾声")

# 这些词单独出现时不足以定性，但和上面任意一类共现 → 判定为复合问题
COMPOUND_SIGNALS = [
    "值得", "培养", "好用", "强度", "强不强", "怎么样", "推荐", "优先",
    "性价比", "值得练", "该不该",
]

# 命中这些 **且只提到一个干员** → 用户在问"TA 和别人的关系"。
# 此时答案多半写在**别人的档案**里（比如可露希尔档案里提到泥岩），
# 若按 operator=泥岩 过滤，那些证据根本进不了候选集。
# 所以需要放开干员过滤，见 route() 的 cross_scope。
CROSS_SIGNALS = [
    # 注意这里**没有单独的"谁"**。
    # 第 9 步评测集发现：问"小羊是谁的昵称"时孤立的"谁"也会命中，
    # 于是误判成"问关系"、放开干员过滤 —— 但那是问身份，不是问关系。
    # 保留"和谁""跟谁"这类带介词的写法，既能覆盖"泥岩和谁关系好"，
    # 又不会把"X 是谁"这种问身份的句子拉进来。
    "和谁", "跟谁", "关系", "交好", "认识", "互动", "联系",
    "来往", "朋友", "搭档", "同伴", "熟悉",
]


@dataclass
class Route:
    """一次路由的结果。"""
    question: str
    intent: str                       # "sql" | "semantic" | "hybrid"
    entities: list[str] = field(default_factory=list)
    # entity_groups：每个实体的**全部词形**（中文名 + 英文代号 + 日文名 + 昵称）。
    # 为什么需要它：共现判定必须按「实体」计数而不是按「词」计数。
    # 例如泥岩档案里写的是「精英干员Logos」，若只认中文名"逻各斯"，
    # 这段真证据会被判成"无共现"，进而触发降级。
    entity_groups: list[list[str]] = field(default_factory=list)
    operator: list[str] | None = None  # 用于过滤（含异格形态）
    cross_scope: bool = False          # 关系类单实体 → 放开干员过滤
    source: str | None = None
    sql: str | None = None
    sql_params: tuple = ()
    # "aggregate"（汇总，禁止截断）还是 "detail"（明细，可截断）。
    # 区别很关键：汇总表少一行，答案就是错的；明细表截断只是"不全"。
    sql_kind: str = ""
    rewritten: str = ""                # 送给向量检索的改写查询
    reason: list[str] = field(default_factory=list)

    def explain(self) -> str:
        parts = [f"意图={self.intent}"]
        if self.entities:
            parts.append("实体=" + "/".join(self.entities))
        if self.source:
            parts.append("资料=" + self.source)
        parts.append("依据：" + "；".join(self.reason))
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# 实体识别
# ---------------------------------------------------------------------------
class EntityMatcher:
    """从问题里认出干员名。

    做法：把 operators 表里的 463 个名字按**长度倒序**在问题里找，
    长名优先占位，避免"阿米娅(近卫)"被拆成"阿米娅"。

    为什么不用分词/NER：干员名是封闭集合（就 463 个），
    穷举匹配比任何模型都准，而且零依赖、零延迟。
    这是垂直领域 RAG 的常见技巧 —— 有词典就别上模型。
    """

    def __init__(self, conn: sqlite3.Connection):
        rows = conn.execute(
            "SELECT name, name_en, name_jp FROM operators "
            "WHERE name IS NOT NULL").fetchall()
        self.names = sorted({r[0] for r in rows}, key=len, reverse=True)
        # 形态映射：泥岩 -> [泥岩, 泥岩(卫戍协议)...]
        self.variants: dict[str, list[str]] = {}
        for n in self.names:
            base = n.split("(")[0]
            self.variants.setdefault(base, []).append(n)

        # ---- 别名（三层，优先级从低到高，后合并的覆盖前者）----
        # 为什么必须分层（第 8 步实测翻案）：
        #   正文词频挖出的"小羊"指向贝娜（奶奶叫她小羊羔，高频），
        #   但玩家口中的"小羊"是艾雅法拉 —— Wiki 官方重定向页写得很清楚。
        #   结论：官方显式声明 > 文本统计推断，冲突时必须覆盖。
        #
        #   1. 自动外文名   operators.name_en / name_jp（Logos→逻各斯）
        #   2. 人工昵称表   data/alias.json（17 脚本挖候选、人工确认）
        #   3. 官方重定向   data/redirects.json（18 脚本抓取，最权威）
        merged: dict[str, str] = {}
        for _name, en, jp in rows:
            base = _name.split("(")[0]
            for a in (en, jp):
                a = (a or "").strip()
                if len(a) >= 2:            # 单字符别名极易误伤，不收
                    merged.setdefault(a, base)
        self._merge_alias_file(merged, "alias.json")
        self._merge_alias_file(merged, "redirects.json")

        # 只保留指向干员的别名（redirects.json 里还有关卡/道具的重定向，
        # 那些 "1-7" 之类不是实体，混进来会污染元数据过滤）
        official = {n.split("(")[0] for n in self.names}
        merged = {a: c for a, c in merged.items() if c in official}

        # 反查表：别名 -> 中文名，供 match() 扫描
        self.alias2canon: dict[str, str] = dict(merged)
        self.alias_by_canon: dict[str, list[str]] = {}
        for a, canon in self.alias2canon.items():
            self.alias_by_canon.setdefault(canon, []).append(a)

        # 剧情章节名（lore 表，20 脚本产出）。
        # 表不存在时不能崩 —— 别人 clone 下来还没跑 20 也要能用。
        try:
            self.chapters = sorted(
                (r[0] for r in conn.execute(
                    "SELECT DISTINCT chapter FROM lore "
                    "WHERE chapter IS NOT NULL AND chapter <> ''")),
                key=len, reverse=True)
        except sqlite3.Error:
            self.chapters = []

    def match_chapter(self, question: str) -> str | None:
        """在问题里找章节名，"第一章" → 库里的"第一章 黑暗时代·下"。

        为什么要它：剧情块的 operator 字段存的是章节全名，
        靠语义相似度去捞"第一章"很容易被别的章节带偏（每章文本都很像）。
        能精确匹配时一定要精确匹配 —— 这是结构化字段该干的活。
        """
        for ch in self.chapters:
            # 太短的章节名（"剧情"这种脏分组）容易误伤，要求 4 字以上
            if len(ch) >= 4 and ch in question:
                return ch
        m = _CHAPTER_NUM_RE.search(question)
        if m:
            key = m.group(0)
            for ch in self.chapters:
                if ch.startswith(key):
                    return ch
        return None

    def _merge_alias_file(self, merged: dict[str, str], filename: str) -> None:
        """把 data/ 下的一个别名文件合并进 merged，后合并者优先（覆盖前者）。

        支持两种格式（按值的类型自动区分）：
            {"艾雅法拉": ["小羊"]}    canon -> [别名]  （17 脚本产物）
            {"小羊": "艾雅法拉"}       别名 -> canon   （18 脚本产物，官方重定向）

        为什么昵称不能全自动：干员昵称是玩家约定俗成的（小刻、阿能、推王），
        Wiki 里没有结构化字段。规则能派生候选，但必须由人看一眼再采纳，
        否则会匹配到"小车"这类同名异物 —— 详见 scripts/17_build_alias.py。
        """
        p = Path(__file__).resolve().parents[2] / "data" / filename
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                     # noqa: BLE001
            print(f"  [别名] {filename} 解析失败，已忽略")
            return
        n = 0
        for k, v in data.items():
            if isinstance(v, list):                        # canon -> [别名]
                canon = k.split("(")[0]
                for a in v:
                    a = (a or "").strip()
                    if len(a) >= 2 and a != canon:
                        merged[a] = canon
                        n += 1
            elif isinstance(v, str):                       # 别名 -> canon
                canon = v.split("(")[0]
                alias = (k or "").strip()
                if len(alias) >= 2 and alias != canon:
                    merged[alias] = canon
                    n += 1
        if n:
            print(f"  [别名] 从 {filename} 载入 {n} 条")

    def match(self, question: str) -> list[str]:
        """返回按出现位置排序的干员名列表（别名已归一到中文名）。

        英文名匹配走小写（"logos" 也能命中 "Logos"），
        lower() 不改变字符位置，所以命中位置仍然准确。
        """
        q = question.lower()
        pairs = [(n, n) for n in self.names]
        pairs += list(self.alias2canon.items())
        pairs.sort(key=lambda x: -len(x[0]))       # 长词优先占位

        used: list[tuple[int, int]] = []
        found: list[tuple[int, str]] = []
        for term, canon in pairs:
            t = term.lower()
            start = 0
            while True:
                p = q.find(t, start)
                if p < 0:
                    break
                end = p + len(t)
                # 该区间已被更长的名字占据 → 跳过
                if not any(p >= s and p < e for s, e in used):
                    used.append((p, end))
                    found.append((p, canon))
                start = p + 1
        found.sort()

        # 去重保序：一句话里 "Logos" 和 "逻各斯" 都出现，只算一个实体
        out: list[str] = []
        for _p, c in found:
            if c not in out:
                out.append(c)
        return out

    def expand(self, name: str) -> list[str]:
        """把"阿米娅"展开成它所有形态（本体 + 异格）。"""
        base = name.split("(")[0]
        return self.variants.get(base, [name])

    def terms(self, name: str) -> list[str]:
        """返回一个实体的**全部词形**：中文名 + 英文代号 + 日文名 + 昵称。

        共现判定必须按「实体」计数而不是按「词」计数：
        一段文本同时出现 "泥岩" 和 "Mudrock"，那还是只提到一个实体，
        不能算共现。所以词形要分组传出去（Route.entity_groups）。
        """
        base = name.split("(")[0]
        out = [name]
        if base != name:
            out.append(base)
        for a in self.alias_by_canon.get(base, []):
            if a not in out:
                out.append(a)
        return out


# ---------------------------------------------------------------------------
# SQL 模板
# ---------------------------------------------------------------------------
# 第 7 步先用规则模板，第 8 步再让 LLM 学着生成。
# 模板的好处：100% 可执行、结果可验算，不会出现 LLM 编造表名的情况。
SQL_TEMPLATES = {
    "elite_material": (
        "SELECT e.stage, e.material, e.amount FROM elite_materials e "
        "JOIN operators o ON o.id=e.operator_id "
        "WHERE o.name IN ({ph}) ORDER BY e.stage, e.amount DESC"
    ),
    "skill_material": (
        "SELECT s.name, sm.skill_idx, sm.mastery_lv, sm.material, sm.amount "
        "FROM skill_materials sm JOIN operators o ON o.id=sm.operator_id "
        "LEFT JOIN skills s ON s.operator_id=o.id AND s.idx=sm.skill_idx "
        "WHERE o.name IN ({ph}) ORDER BY sm.skill_idx, sm.mastery_lv"
    ),
    "max_stats": (
        "SELECT o.name, ls.level_key, ls.hp, ls.atk, ls.defense, ls.res "
        "FROM level_stats ls JOIN operators o ON o.id=ls.operator_id "
        "WHERE o.name IN ({ph}) AND ls.level_key='精2_满级'"
    ),
    "basic": (
        "SELECT name, rarity, profession, sub_profession, category "
        "FROM operators WHERE name IN ({ph})"
    ),
    "faction_members": (
        "SELECT o.name, o.rarity, o.profession FROM factions f "
        "JOIN operators o ON o.id=f.operator_id WHERE f.value=? "
        "ORDER BY o.rarity DESC"
    ),
    "rarity_rank": (
        "SELECT o.name, ls.defense FROM level_stats ls "
        "JOIN operators o ON o.id=ls.operator_id "
        "WHERE ls.level_key='精2_满级' ORDER BY ls.defense DESC LIMIT 5"
    ),
    # 满练度材料总表：精英化 + 技能（含专精）+ 模组，三表 UNION 后按材料名聚合。
    #
    # 为什么必须有它（第 10 步真实问答暴露）：
    #   "史尔特尔从1级到满练度包括技能模组等级总共需要多少材料"
    #   命中了"技能" → 只查 skill_materials 一张表的 41 行**明细**，
    #   ① 漏了精英化和模组两张表的材料（用户明明说了"包括"）；
    #   ② 41 行被截断到 30 行，模型说"清单不完整"；
    #   ③ 最危险的一条：模型开始自己做加法（8+12+15…=70），
    #      而它数的是一张残缺的表。
    #
    # "加总"这件事 SQL 一条 GROUP BY 就能精确完成，
    # 交给 LLM 数数是把它放在了自己最不擅长、也最不可靠的位置。
    "total_material": (
        "SELECT t.material AS 材料, SUM(t.amount) AS 总数量 FROM (\n"
        "  SELECT e.material, e.amount FROM elite_materials e "
        "JOIN operators o ON o.id=e.operator_id WHERE o.name IN ({ph})\n"
        "  UNION ALL SELECT sm.material, sm.amount FROM skill_materials sm "
        "JOIN operators o ON o.id=sm.operator_id WHERE o.name IN ({ph})\n"
        "  UNION ALL SELECT mm.material, mm.amount FROM module_materials mm "
        "JOIN modules m ON m.id=mm.module_id "
        "JOIN operators o ON o.id=m.operator_id WHERE o.name IN ({ph})\n"
        ") t GROUP BY t.material ORDER BY 总数量 DESC"
    ),
    # 满练度**分阶段养成清单**（第 11 步用户反馈后新增）。
    #
    # 用户看不懂一列揉在一起的"总量"——材料花在哪个阶段完全看不出来。
    # 真实的练度规划是分阶段的：精英化1、精英化2、技能2~7级、
    # 三个技能各自的专精1~3、每个模组的 1~3 级。
    #
    # 输出三段，全部由 SQL 计算，LLM 只转述：
    #   ① 明细段（a/b/c/d 开头）：每行原始归属，信息无损
    #   ② 小计段（e 开头）：按大阶段（技能升级2-7合计 / 每个技能的专精合计 /
    #      每个模组合计）GROUP BY SUM —— 实测模型转述明细时会自己把
    #      "技能2级卷1×5 + 技能3级卷1×5" 合并成 ×10，恰好对但不可靠；
    #      合并这种计算必须由 SQL 做，模型引用【小计】段即可
    #   ③ 总计段（zz）：全阶段 SUM
    #
    # ord 列纯粹为了排序（a 精英化 → b 技能 → c 专精 → d 模组 → e 小计 → zz 总计）。
    # {ph} 出现次数较多（各子查询 + 小计 + 总计段都要按名字过滤），
    # 调用方按 SQL 里的实际出现次数复制参数，不手写数字。
    "train_plan": (
        "SELECT t.阶段 AS 阶段, t.材料 AS 材料, t.数量 AS 数量 FROM (\n"
        "  SELECT CASE e.stage WHEN '精1' THEN 'a1' ELSE 'a2' END AS ord,\n"
        "         ('精英化' || substr(e.stage, 2, 1)) AS 阶段,\n"
        "         e.material AS 材料, e.amount AS 数量\n"
        "  FROM elite_materials e JOIN operators o ON o.id=e.operator_id "
        "WHERE o.name IN ({ph})\n"
        "  UNION ALL SELECT ('b0' || sm.target_level) AS ord,\n"
        "         ('技能升到' || sm.target_level || '级') AS 阶段,\n"
        "         sm.material AS 材料, sm.amount AS 数量\n"
        "  FROM skill_materials sm JOIN operators o ON o.id=sm.operator_id "
        "WHERE o.name IN ({ph}) AND sm.is_mastery=0\n"
        "  UNION ALL SELECT ('c' || sm.skill_idx || '0' || sm.mastery_lv) AS ord,\n"
        "         ('技能' || sm.skill_idx || ' 专精' || sm.mastery_lv) AS 阶段,\n"
        "         sm.material AS 材料, sm.amount AS 数量\n"
        "  FROM skill_materials sm JOIN operators o ON o.id=sm.operator_id "
        "WHERE o.name IN ({ph}) AND sm.is_mastery=1\n"
        "  UNION ALL SELECT ('d' || m.idx || '0' || mm.level) AS ord,\n"
        "         ('模组' || m.idx || ' 升到' || mm.level || '级') AS 阶段,\n"
        "         mm.material AS 材料, mm.amount AS 数量\n"
        "  FROM module_materials mm JOIN modules m ON m.id=mm.module_id "
        "JOIN operators o ON o.id=m.operator_id WHERE o.name IN ({ph})\n"
        "  UNION ALL SELECT 'e0' AS ord, '【小计】技能升级2-7级' AS 阶段,\n"
        "         s.material AS 材料, s.total AS 数量 FROM (\n"
        "    SELECT sm.material, SUM(sm.amount) AS total\n"
        "    FROM skill_materials sm JOIN operators o ON o.id=sm.operator_id "
        "WHERE o.name IN ({ph}) AND sm.is_mastery=0 GROUP BY sm.material\n"
        "  ) s\n"
        "  UNION ALL SELECT ('e' || sm.skill_idx) AS ord,\n"
        "         ('【小计】技能' || sm.skill_idx || '专精') AS 阶段,\n"
        "         sm.material AS 材料, SUM(sm.amount) AS 数量\n"
        "  FROM skill_materials sm JOIN operators o ON o.id=sm.operator_id "
        "WHERE o.name IN ({ph}) AND sm.is_mastery=1 GROUP BY sm.skill_idx, sm.material\n"
        "  UNION ALL SELECT ('e4' || m.idx) AS ord,\n"
        "         ('【小计】模组' || m.idx) AS 阶段,\n"
        "         mm.material AS 材料, SUM(mm.amount) AS 数量\n"
        "  FROM module_materials mm JOIN modules m ON m.id=mm.module_id "
        "JOIN operators o ON o.id=m.operator_id WHERE o.name IN ({ph})\n"
        "  GROUP BY m.idx, mm.material\n"
        "  UNION ALL SELECT 'zz' AS ord, '【总计】全阶段合计' AS 阶段,\n"
        "         g.material AS 材料, g.total AS 数量 FROM (\n"
        "    SELECT u.material AS material, SUM(u.amount) AS total FROM (\n"
        "      SELECT e.material, e.amount FROM elite_materials e "
        "JOIN operators o ON o.id=e.operator_id WHERE o.name IN ({ph})\n"
        "      UNION ALL SELECT sm.material, sm.amount FROM skill_materials sm "
        "JOIN operators o ON o.id=sm.operator_id WHERE o.name IN ({ph})\n"
        "      UNION ALL SELECT mm.material, mm.amount FROM module_materials mm "
        "JOIN modules m ON m.id=mm.module_id "
        "JOIN operators o ON o.id=m.operator_id WHERE o.name IN ({ph})\n"
        "    ) u GROUP BY u.material\n"
        "  ) g\n"
        ") t ORDER BY t.ord"
    ),
}

# 问"总共/合计/满练度"时的信号词。
# 注意顺序：必须在"技能/专精"之前判断 —— 用户要的是总数，不是某张表的明细。
TOTAL_SIGNALS = [
    "总共", "一共", "合计", "总计", "总量", "汇总",
    "满练度", "全部材料", "所有材料", "一共需要", "总共需要",
]


def pick_sql(question: str,
             entities: list[str]) -> tuple[str, tuple, str] | None:
    """根据问题里的关键词选一条模板 SQL。选不出来返回 None。

    返回 (sql, params, kind)。
    kind="aggregate" 表示结果是**已经聚合好的汇总值**，下游禁止截断
    —— 汇总少一行，答案就是错的；而明细截断只是"信息不全"，可接受。
    """
    q = question
    if not entities:
        # 全站排行类（"全站防御最高的干员"）
        if any(w in q for w in ("最高", "最低", "排行", "排名", "最强", "最多", "最少")):
            return SQL_TEMPLATES["rarity_rank"], (), "detail"
        return None

    ph = ",".join("?" * len(entities))
    params = tuple(entities)

    if any(w in q for w in ("材料", "消耗", "需要", "龙门币", "芯片")):
        # 先判"要总数"，再判"要哪张表的明细"。顺序不能反：
        # "满练度总共要多少材料" 也含"技能"二字，
        # 但用户要的是三表合计，不是技能那一列的流水账。
        if any(w in q for w in TOTAL_SIGNALS):
            # 分阶段养成清单（带【总计】行）替代纯汇总：
            # 用户真正想看的是"材料花在哪个阶段"，总量只是顺带的。
            # {ph} 在模板里出现多次，按实际出现次数复制参数。
            tpl = SQL_TEMPLATES["train_plan"]
            n_ph = tpl.count("{ph}")
            return tpl.format(ph=ph), params * n_ph, "plan"
        if any(w in q for w in ("专精", "技能")):
            return SQL_TEMPLATES["skill_material"].format(ph=ph), params, "detail"
        return SQL_TEMPLATES["elite_material"].format(ph=ph), params, "detail"
    if any(w in q for w in ("面板", "满级", "攻击力", "防御", "生命", "法抗", "属性")):
        return SQL_TEMPLATES["max_stats"].format(ph=ph), params, "detail"
    if any(w in q for w in ("星级", "职业", "分支", "稀有度", "是什么职业")):
        return SQL_TEMPLATES["basic"].format(ph=ph), params, "detail"
    return None


# ---------------------------------------------------------------------------
# 主路由函数
# ---------------------------------------------------------------------------
_SQL_RE = re.compile("|".join(map(re.escape, SQL_SIGNALS)))
_SEM_RE = re.compile("|".join(map(re.escape, SEMANTIC_SIGNALS)))


def route(question: str, matcher: EntityMatcher) -> Route:
    """把一句自然语言问题拆成检索指令。"""
    r = Route(question=question, intent="semantic")
    r.entities = matcher.match(question)
    # 注意：**所有**实体的形态都要并进去，不能只展开第一个。
    # 只展开第一个时，"A 和 B 的关系"会被过滤成只剩 A 的资料，
    # B 的块全被排除，共现证据反而没了。
    ops: list[str] = []
    for e in r.entities:
        for v in matcher.expand(e):
            if v not in ops:
                ops.append(v)
    r.operator = ops or None
    r.rewritten = question
    r.entity_groups = [matcher.terms(e) for e in r.entities]

    # ---- 关系类 + 只提到一个人 → 放开干员过滤 ----
    # 问"泥岩和谁关系好"时，证据可能在**别人的档案**里。
    # 按 operator=泥岩 过滤会直接把那些证据挡在候选集外。
    if len(r.entities) == 1 and any(w in question for w in CROSS_SIGNALS):
        r.cross_scope = True
        r.reason.append("关系类单实体 → 放开干员过滤（证据可能在他人档案里）")

    # ---- 1) 资料类型：问题里点名了哪类资料？ ----
    for src, words in SOURCE_HINTS.items():
        if any(w in question for w in words):
            r.source = src
            r.reason.append(f"命中资料词「{next(w for w in words if w in question)}」→ {src}")
            break

    # ---- 1.5) 章节锁定 ----
    # 放在资料类型**之后**：命中章节说明一定在问剧情，此时章节过滤
    # 比资料类型更具体，应当压过前面的判断。
    # 但如果已经判出是支线/故事集（"火蓝之心第三章"），就保留那个结论。
    chapter = matcher.match_chapter(question)
    if chapter:
        # 剧情块的 operator 字段存的就是章节全名，直接拿它当过滤条件
        r.operator = [chapter]
        if not str(r.source or "").startswith("lore"):
            r.source = "lore_main"
        r.cross_scope = False      # 章节过滤就是要收紧，不能放开
        r.reason.append(f"锁定章节「{chapter}」→ 只检索该章剧情")

    # ---- 2) 意图 ----
    sql_hits = [m.group(0) for m in _SQL_RE.finditer(question)]
    sem_hits = [m.group(0) for m in _SEM_RE.finditer(question)]

    if sql_hits and sem_hits:
        r.intent = "hybrid"
        r.reason.append(f"同时命中 SQL 词{sql_hits[:2]}和语义词{sem_hits[:2]}→ 两路都要")
    elif sql_hits:
        r.intent = "sql"
        r.reason.append(f"命中结构化词{sql_hits[:3]}")
    elif sem_hits:
        r.intent = "semantic"
        r.reason.append(f"命中语义词{sem_hits[:3]}")
    else:
        r.intent = "semantic"
        r.reason.append("无明显信号，默认走语义检索（向量能兜底）")

    # 复合信号会强制升级成 hybrid
    comp = [w for w in COMPOUND_SIGNALS if w in question]
    if comp and r.intent != "hybrid":
        r.intent = "hybrid"
        r.reason.append(f"命中复合词{comp[:2]}→ 需要数值+语义两边")

    # ---- 3) SQL 模板 ----
    if r.intent in ("sql", "hybrid"):
        got = pick_sql(question, r.entities)
        if got:
            r.sql, r.sql_params, r.sql_kind = got
            if r.sql_kind == "aggregate":
                r.reason.append("命中汇总词→走聚合 SQL（不截断）")
        elif r.intent == "hybrid" and r.entities:
            # 复合问题（如"值得培养吗"）没命中任何具体模板。
            # 这时不能干脆不查 SQL —— 否则 LLM 只能凭剧情文本谈强度，
            # 那正是幻觉的温床。默认给满级面板当数值依据。
            ph = ",".join("?" * len(r.entities))
            r.sql = SQL_TEMPLATES["max_stats"].format(ph=ph)
            r.sql_params = tuple(r.entities)
            r.sql_kind = "detail"
            r.reason.append("复合问题无匹配模板，默认取满级面板作数值依据")

    return r
