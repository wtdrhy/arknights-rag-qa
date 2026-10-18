"""
wikitext 解析器 —— 第 4 步的核心模块

【这一层的职责】
把 MediaWiki 的原始源码（wikitext）转成结构化的 Python dict，供后面写入 SQLite。

【为什么要写它，而不是直接丢给大模型】
wiki 页面不是 markdown，而是 {{模板名|键=值}} 的嵌套结构。
例如干员「泥岩」的精英化材料长这样：

    {{精英化材料
    |精1={{材料消耗|龙门币|3w}} {{材料消耗|重装芯片|5}}
    |精2={{材料消耗|龙门币|18w}} {{材料消耗|重装双芯片|4}}
    }}

这是**天生结构化**的数据。用解析器提取 → 数值 100% 准确、零幻觉；
如果切成文本块丢进向量库靠语义检索，反而会算错。
这是本项目「数值类问题不走 RAG」设计的底层支撑。

【本模块的三条设计原则】
1. 单个页面解析失败绝不中断整批 —— 错误要被记下来，不能崩
2. 装饰性标记要剥离 —— {{color|#0098DC|+30%}} 应该还原成 "+30%"
3. 数值要归一化 —— "3w" 要变成 30000，否则后面做不了加法
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import mwparserfromhell

# ---------------------------------------------------------------------------
# 1) 装饰性模板：只负责排版/上色/加链接，对语义没有贡献，解析时要剥掉
#    判定标准见第 3 步勘探：这些模板「次/页」远大于 1
# ---------------------------------------------------------------------------
DECOR_TEMPLATES = {
    "color", "术语", "popup", "tooltip", "修正",
    "变动数值", "变动数值lite", "变动数值2",
    "*", "**", "+", "++", "-", "±",
    "FULLPAGENAME", "BASEPAGENAME", "PAGENAME",
    "lang", "ruby", "lj",
    # {{特殊机制|移动创伤}} —— 参数是关键词本身，有语义，取出来当文本
    "特殊机制",
}

# ---------------------------------------------------------------------------
# 1b) 图标类模板：剥掉之后不能留任何东西
#     它们形如 {{mdi|sword}}、{{fa|heart}}，参数里装的是「图标名」不是文本。
#     如果按 DECOR_TEMPLATES 那样"取最后一个参数"，会把图标名残留到正文里，
#     比如技能描述变成「防御力+30% sword」—— 所以必须单独一类，直接清空。
# ---------------------------------------------------------------------------
STRIP_EMPTY_TEMPLATES = {
    "mdi", "fa", "spine", "spineId", "道具图标",
    "#widget:style", "ads/normal", "ads/mobile",
    # 攻击范围类：参数是「范围代号」（3-1 / y-3 / 3-12），渲染出来是一张格子图。
    # 格子图在页面上是图片，落到纯文本里 "3-1" 没有任何语义，留着只会污染 chunk。
    # （真正的攻击范围我们有 attack_ranges 结构化表，不靠这里的文本）
    "攻击范围", "popup/range", "popup/范围",
}

# ---------------------------------------------------------------------------
# 1c) 动态替换模板：在 wiki 上渲染成「随上下文变化的值」，离线场景必须定值化
#     {{DrName}} 渲染的是「当前登录玩家的名字」，wiki 默认显示"博士"。
#     我们的知识库是离线的、没有"当前玩家"，留着 {{DrName|前缀=Dr.}}
#     就是纯噪声（实测 79 个页面、500+ 处），统一替换成"博士"。
#     若原文显式给了位置参数（{{DrName|好奇的博士}}），说明作者想写别的称呼，
#     那就用参数值 —— 规则：有位置参数用参数，无参数用默认。
# ---------------------------------------------------------------------------
SUBST_TEMPLATES = {"DrName": "博士"}

# 匹配「最内层」的 {{xxx}}：内部不能再含 { 或 }
# 这样天然实现「由内向外」剥离，嵌套模板也能正确处理
_TMPL_INNER = re.compile(r"\{\{([^{}]*)\}\}")

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HTML_BR = re.compile(r"<br\s*/?>", re.I)
_HTML_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.S | re.I)
_HTML_TAG = re.compile(r"<[^>]+>")
_WIKI_BOLD = re.compile(r"'{2,}")


def strip_comments(text: str) -> str:
    """去掉 <!-- 这里的内容不会显示在页面上 --> 这类注释"""
    return _HTML_COMMENT.sub("", text or "")


_RESIDUAL_RE = re.compile(r"\{\{\s*([^{}|]+?)\s*(?:\||\}\})")


def residual_templates(text: str) -> list[str]:
    """返回一段文本里**仍然残留**的模板名。

    为什么要有这个函数：
        「字段有值、长度也正常，但内容里夹着 {{xxx}}」是最阴的一类数据 bug ——
        完整度统计看不出来，只有进了向量库、检索时命中一堆脏标记才发现。
        有了它，任何一步都能自检：残留 0 才算干净，非 0 就把模板名打出来补规则。
    """
    return [m.group(1).strip() for m in _RESIDUAL_RE.finditer(text or "")]


def clean_inline(text: str, max_iter: int = 40, strip_unknown: bool = False) -> str:
    """
    剥离装饰性模板，保留可读文本。

    典型转换：
        {{color|#00B0FF|无法}}            -> 无法
        {{术语|ba.shield|护盾}}            -> 护盾
        {{变动数值lite|up|蓝|22%}}         -> 22%
        {{*|30%|{{color|#0098DC|+30%}}}}  -> +30%   （嵌套，取最后一个参数）

    遇到非装饰模板（如 {{模组|...}}）默认原样保留，
    因为我们后面还要从它里面取字段。

    strip_unknown=True 时连非装饰模板一起剥掉，
    用于「整段就是自然语言、不需要再取字段」的场景（如特殊机制说明）。
    """
    s = text or ""
    keep: list[str] = []

    for _ in range(max_iter):
        def _repl(m: re.Match) -> str:
            parts = [p.strip() for p in m.group(1).split("|")]
            name = parts[0]
            if name in STRIP_EMPTY_TEMPLATES:
                return ""
            if name in SUBST_TEMPLATES:
                # 形如 {{DrName|前缀=Dr.}}：位置参数才是作者想写的称呼，
                # 「前缀=」「语言=」这类具名参数是渲染选项，要忽略。
                pos = [p for p in parts[1:] if p and "=" not in p]
                return pos[0] if pos else SUBST_TEMPLATES[name]
            if name in DECOR_TEMPLATES or strip_unknown:
                vals = [p for p in parts[1:] if p]
                if not vals:
                    return ""
                last = vals[-1]
                # {{popup|内容=xxx}} 这类：等号前是元信息键，真正文本在后面
                if "=" in last and not last.startswith("="):
                    last = last.split("=", 1)[1].strip()
                return last
            # 非装饰模板：存起来，换成占位符，避免死循环
            keep.append(m.group(0))
            return f"\x00{len(keep) - 1}\x00"

        new = _TMPL_INNER.sub(_repl, s)
        if new == s:
            break
        s = new

    # 还原被占位符保护起来的模板
    for i in range(len(keep) - 1, -1, -1):
        s = s.replace(f"\x00{i}\x00", keep[i])

    # DrName 还原成"博士"后，可能和原文里本来就有的"博士"撞在一起
    # （wiki 原文本身就写 {{DrName}}博士您前进的道路，因为玩家名是可自定义的）。
    # 不去重会变成"博士博士您前进的道路"，读起来像口吃，也会影响 embedding 质量。
    s = re.sub(r"(博士){2,}", "博士", s)
    return s


def clean_text(text: str, strip_unknown: bool = False) -> str:
    """
    把一段 wikitext 变成干净的自然语言文本。
    用于描述类字段（天赋效果、技能描述、模组特性）。

    strip_unknown=True：连非装饰模板也剥掉，用于纯叙述性长文本。
    """
    s = strip_comments(text)
    s = clean_inline(s, strip_unknown=strip_unknown)
    s = _HTML_BR.sub("\n", s)
    s = _HTML_REF.sub("", s)
    s = _HTML_TAG.sub("", s)
    s = _WIKI_BOLD.sub("", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# ---------------------------------------------------------------------------
# 2) 数值归一化
# ---------------------------------------------------------------------------

def to_num(s):
    """'882' -> 882 ; '1.6' -> 1.6 ; '70s' -> '70s'（转不了就原样返回）"""
    s = (s or "").strip()
    if not s:
        return None
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?\d*\.\d+", s):
        return float(s)
    return s


def parse_amount(s):
    """
    材料数量归一化。wiki 里大额龙门币写作 '3w'、'18w'。
        '3w'  -> 30000
        '18w' -> 180000
        '12'  -> 12
    """
    s = (s or "").strip()
    m = re.fullmatch(r"([\d.]+)\s*[wW]", s)
    if m:
        return int(float(m.group(1)) * 10000)
    return to_num(s)


def parse_materials(value: str) -> list[dict]:
    """
    解析 {{材料消耗|龙门币|3w}} {{材料消耗|重装芯片|5}}
      -> [{'name': '龙门币', 'amount': 30000}, {'name': '重装芯片', 'amount': 5}]
    """
    out = []
    pattern = re.compile(r"\{\{\s*材料消耗\s*\|\s*([^|}]*?)\s*\|\s*([^}]*?)\s*\}\}")
    for m in pattern.finditer(value or ""):
        name = clean_inline(m.group(1)).strip()
        if name:
            out.append({"name": name, "amount": parse_amount(m.group(2))})
    return out


# ---------------------------------------------------------------------------
# 3) 通用小工具
# ---------------------------------------------------------------------------

def params_dict(tpl) -> dict:
    """模板 -> {参数名: 参数值字符串}"""
    d = {}
    for p in tpl.params:
        d[str(p.name).strip()] = str(p.value).strip()
    return d


def group_templates(code) -> dict[str, list]:
    """把所有模板按名字分组。同名模板可能有多个（如多个技能、多个模组）"""
    groups = defaultdict(list)
    for t in code.filter_templates(recursive=True):
        groups[str(t.name).strip()].append(t)
    return groups


def templates_in_order(code, names) -> list:
    """
    按「在页面中出现的先后顺序」取出指定名字的模板。

    为什么需要它：同一页面可能混用新旧两代模板。
    实测 463 个干员页里有 **131 页**同时使用 {{技能2}} 和 {{技能}}，
    如果只取其中一种，就会漏掉技能（泥岩的三技能「岩崩锤」就是这样被漏掉的）。
    而且必须按出现顺序合并，否则技能一二三的顺序会乱。
    """
    wanted = set(names)
    return [t for t in code.filter_templates(recursive=True)
            if str(t.name).strip() in wanted]


# ---------------------------------------------------------------------------
# 4) 干员页解析
# ---------------------------------------------------------------------------

# 属性面板里需要转成数字的字段名
_STAT_KEYS = ("生命上限", "攻击", "防御", "法术抗性")
_LEVEL_RE = re.compile(r"^精英(\d)_(1级|满级)_(.+)$")


def _parse_stats(attr: dict) -> dict:
    """解析 {{属性}} 模板"""
    stats = {
        "再部署时间": attr.get("再部署"),
        "部署费用": attr.get("部署费用"),
        "阻挡数": attr.get("阻挡数"),
        "攻击速度": attr.get("攻击速度"),
        "levels": {},
    }
    for k, v in attr.items():
        m = _LEVEL_RE.match(k)
        if m:
            phase, level, stat = m.group(1), m.group(2), m.group(3)
            stats["levels"].setdefault(f"精{phase}_{level}", {})[stat] = to_num(v)

    # 模组名（属性模板里只给了名字和简短加成，详细数据在 {{模组}} 里）
    modules = []
    for i in range(1, 6):
        nm = attr.get(f"模组{i}名")
        if nm:
            modules.append({"name": nm, "bonus": attr.get(f"模组{i}数据", "")})
    stats["模组名"] = modules
    return stats


def _parse_modules(tpls: list) -> list[dict]:
    """解析 {{模组}} 模板（一个干员通常有 2~3 个模组）"""
    num_key = re.compile(r"^(生命|攻击|防御|法术抗性)(\d)?$")
    modules = []
    for t in tpls:
        d = params_dict(t)
        item = {
            "名称": d.get("名称"),
            "类型": d.get("类型"),
            "特性追加": d.get("特性追加") == "yes",
            "数值": {},
            "天赋更新": {},
            "解锁材料": {},
        }
        for k, v in d.items():
            if num_key.match(k):
                item["数值"][k] = to_num(v)
            elif k.startswith("天赋") and k[2:].isdigit():
                item["天赋更新"][k] = clean_text(v)
            elif k.startswith("解锁材料") or k.startswith("材料"):
                item["解锁材料"][k] = parse_materials(v)
        tr = d.get("特性")
        if tr:
            item["特性"] = clean_text(tr)
        bi = d.get("基础信息")
        if bi:
            item["基础信息"] = clean_text(bi)
        modules.append(item)
    return modules


def _parse_talents(tpls: list) -> list[dict]:
    """解析天赋模板（共三代，参数命名不同，必须都支持）

    {{天赋列表3}}（新）: 天赋1 / 天赋1条件 / 天赋1效果
    {{天赋列表2}}（中）: 同上
    {{天赋列表}} （旧）: 第一天赋1 / 第一天赋1条件 / 第一天赋1效果
                        还可能有 第二天赋N

    踩过的坑：只支持新前缀时，虎狼丸等 9 个页面天赋全解析成空，
    而完整度报告看不出来（字段存在但值为空列表）。
    """
    out = []
    seen = set()
    for t in tpls:
        d = params_dict(t)
        for prefix in ("天赋", "第一天赋", "第二天赋"):
            for i in range(1, 8):
                nm = d.get(f"{prefix}{i}")
                if not nm:
                    continue
                key = (clean_inline(nm), d.get(f"{prefix}{i}条件", ""))
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "名称": clean_inline(nm),
                    "解锁条件": clean_inline(d.get(f"{prefix}{i}条件", "")),
                    "效果": clean_text(d.get(f"{prefix}{i}效果", "")),
                })
    return out


def _parse_skills(tpls: list) -> list[dict]:
    """解析 {{技能2}} / {{技能}}，含 1~7 级与专精一/二/三"""
    out = []
    for t in tpls:
        d = params_dict(t)
        skill = {
            "名称": clean_inline(d.get("技能名", "")),
            "技能名jp": d.get("技能名jp", ""),
            "技能名en": d.get("技能名en", ""),
            "回复类型": d.get("技能类型1", ""),
            "触发类型": d.get("技能类型2", ""),
            "等级": [],
        }
        for i in range(1, 8):
            desc = d.get(f"技能{i}描述")
            if desc:
                skill["等级"].append({
                    "等级": str(i),
                    "描述": clean_text(desc),
                    "初始技力": to_num(d.get(f"技能{i}初始", "")),
                    "技力消耗": to_num(d.get(f"技能{i}消耗", "")),
                    "持续时间": to_num(d.get(f"技能{i}持续", "")),
                })
        for i in range(1, 4):
            desc = d.get(f"技能专精{i}描述")
            if desc:
                skill["等级"].append({
                    "等级": f"专精{i}",
                    "描述": clean_text(desc),
                    "初始技力": to_num(d.get(f"技能专精{i}初始", "")),
                    "技力消耗": to_num(d.get(f"技能专精{i}消耗", "")),
                    "持续时间": to_num(d.get(f"技能专精{i}持续", "")),
                })
        out.append(skill)
    return out


# ---------------------------------------------------------------------------
# 第 4 步补全：此前遗漏的 9 个板块
# 依据：对 463 个干员页做穷举扫描（模板 69 种 + 章节 17 类）后确定，
#       覆盖率均 >= 60%，且都是玩家高频查询的信息。
# ---------------------------------------------------------------------------

def _parse_attack_range(tpls: list) -> dict:
    """{{干员攻击范围}}：精英0/1/2 三档范围（463 页全覆盖）"""
    if not tpls:
        return {}
    d = params_dict(tpls[0])
    return {
        k: clean_inline(d.get(k, ""))
        for k in ("精英0范围", "精英1范围", "精英2范围") if d.get(k)
    }


def _parse_potential(tpls: list) -> list[dict]:
    """{{潜能提升}}：潜能 2~6 的效果（463 页）"""
    if not tpls:
        return []
    d = params_dict(tpls[0])
    out = []
    for i in range(2, 7):
        v = d.get(f"潜能{i}")
        if v:
            out.append({"潜能": i, "效果": clean_inline(v)})
    return out


def _parse_base_skills(tpls: list) -> list[dict]:
    """{{后勤技能}}：基建技能名 + 解锁阶段（461 页）

    参数形如：|后勤技能1-1=执著  |后勤技能1-1阶段=精英0
    同一阶段可能并列多个技能（1-1 / 1-2），所以要遍历 (组, 序号)。
    """
    if not tpls:
        return []
    d = params_dict(tpls[0])
    out = []
    for grp in range(1, 6):
        for idx in range(1, 4):
            name = d.get(f"后勤技能{grp}-{idx}")
            if name:
                out.append({
                    "名称": clean_inline(name),
                    "解锁阶段": clean_inline(d.get(f"后勤技能{grp}-{idx}阶段", "")),
                })
    return out


def _parse_related_items(tpls: list) -> dict:
    """{{相关道具}}：干员简介 + 信物描述（460 页）

    「干员简介」是 wiki 对角色的官方一句话概括，检索价值很高。
    """
    if not tpls:
        return {}
    d = params_dict(tpls[0])
    return {
        k: clean_text(d.get(k, ""))
        for k in ("干员简介", "干员简介补充", "信物用途", "信物描述") if d.get(k)
    }


def _parse_paradox(tpls: list) -> dict:
    """{{悖论模拟}}：解锁条件 + 报酬（312 页）"""
    if not tpls:
        return {}
    d = params_dict(tpls[0])
    out = {
        "名称": clean_inline(d.get("name", "")),
        "描述": clean_text(d.get("description", "")),
        "解锁精英化": to_num(d.get("精英化", "")),
        "解锁等级": to_num(d.get("等级", "")),
        "报酬": [],
    }
    for i in range(1, 6):
        item = d.get(f"报酬内容{i}")
        if item:
            out["报酬"].append({
                "道具": clean_inline(item),
                "数量": to_num(d.get(f"报酬数量{i}", "")),
            })
    return {k: v for k, v in out.items() if v}


def _parse_attack_range_fallback(tpls: list) -> dict:
    """{{攻击范围|3-1}}：卫戍协议干员用的简化写法（16 页）

    常规干员用 {{干员攻击范围}}（分精英0/1/2 三档），
    卫戍协议干员只有一档，写成 {{攻击范围|3-1}}。
    两者都归到「攻击范围」字段，避免同一信息两处存储。
    """
    if not tpls:
        return {}
    d = params_dict(tpls[0])
    # 无名参数：{{攻击范围|3-1}} → 第一个参数名是 "1"
    val = d.get("1") or next(iter(d.values()), "")
    return {"范围": clean_inline(val)} if val else {}


def _parse_ability_range(tpls: list) -> list[dict]:
    """{{能力生效范围}}：特殊能力的作用范围（23 页）"""
    out = []
    for t in tpls:
        d = params_dict(t)
        if d.get("名称") or d.get("范围"):
            out.append({
                "名称": clean_inline(d.get("名称", "")),
                "范围": clean_inline(d.get("范围", "")),
            })
    return out


def _parse_special_mechanics(wikitext: str) -> str:
    """{{特殊机制}}：技能/天赋的特殊规则说明（61 页）

    这类内容模板里只有机制名，说明文字在模板之后的正文里：
        {{特殊机制|部署保护失效}}<br>※场上不存在重构体时，技能不能开启...
    所以要连带取后面的正文，一直取到下一个章节标题为止。
    """
    m = re.search(r"\{\{\s*特殊机制\s*\|([^}]*)\}\}", wikitext)
    if not m:
        return ""
    # 机制名 + 后续正文（到下一个 ==章节== 为止）
    tail = wikitext[m.end():]
    nxt = re.search(r"^==+[^=\n]+==+", tail, re.M)
    body = tail[:nxt.start()] if nxt else tail[:1500]
    # 整段都是叙述文字，用 strip_unknown 把所有模板剥干净
    text = clean_text(m.group(1) + "\n" + body, strip_unknown=True)
    return text.strip()


def _resolve_subpage(raw: str, title: str) -> str:
    """把 {{FULLPAGENAME}}/干员密录/1 还原成完整子页面标题 泥岩/干员密录/1。

    为什么必须还原：clean_inline 会把 {{FULLPAGENAME}} 当装饰模板剥掉，
    只剩 "/干员密录/1" —— 少了干员名，第 6 步就找不到正文页在哪。
    """
    if not raw:
        return ""
    # 先补上干员名，再交给 clean_inline 剥掉其余装饰
    filled = re.sub(r"\{\{\s*FULLPAGENAME\s*\}\}", title or "", raw)
    return clean_inline(filled).strip()


def _parse_story_sets(lists: list, title: str = "") -> list[dict]:
    """{{干员密录/list}}：密录索引（336 页）

    注意：这里只有索引（故事名 + 简介 + 文本页引用），
    正文在子页面「干员名/干员密录/N」里，由 scripts/06 单独解析。

    之所以直接接收 list 模板而不是从 {{干员密录}} 里二次 parse：
    group_templates() 用了 recursive=True，嵌套模板已经被平铺取出来了，
    再手动 parse 一次既多余又容易踩 API 兼容问题。
    """
    out = []
    for lst in lists:
        d = params_dict(lst)
        if not d:
            continue
        item = {
            "故事集": clean_inline(d.get("storySetName", "")),
            "解锁精英化": to_num(d.get("精英化", "")),
            "解锁等级": to_num(d.get("等级", "")),
            "解锁信赖": to_num(d.get("信赖", "")),
            # 形如 {{FULLPAGENAME}}/干员密录/1 → 还原成 泥岩/干员密录/1
            "正文页": _resolve_subpage(d.get("storyTxt1", ""), title),
        }
        intro = clean_text(d.get("storyIntro1", ""))
        if intro:
            item["简介"] = intro
        out.append(item)
    return out


def _parse_alternate(tpls: list) -> dict:
    """{{异格干员}}：原型 / 是否异格（75 页）

    参数：|原型=W   |非异格=1
      - 有「非异格」→ 本人就是原型（W 是原型，耀骑士临光是异格）
      - 无「非异格」→ 本人是异格形态，原型指向基础干员
    """
    if not tpls:
        return {}
    d = params_dict(tpls[0])
    if not d:
        return {}
    is_origin = bool(d.get("非异格"))
    return {
        "原型": clean_inline(d.get("原型", "")),
        "是否原型": is_origin,
        # 「形态」是本字段的核心：无「非异格」参数 = 本人是异格形态
        "形态": "原型" if is_origin else "异格",
    }


def _parse_alt_tasks(tpl: dict, lists: list) -> dict:
    """{{干员异格任务}}：异格任务链（20 页）

    主模板带 |对象干员=耀骑士临光，任务步骤在嵌套的 /list 里。
    """
    out = {"对象干员": clean_inline(tpl.get("对象干员", "")), "任务": []}
    for lst in lists:
        ld = params_dict(lst)
        if not ld:
            continue
        out["任务"].append({
            "精英化": to_num(ld.get("精英化", "")),
            "等级": to_num(ld.get("等级", "")),
            "奖励": clean_inline(ld.get("任务奖励", "")),
        })
    return out if out["任务"] else {}


def _extract_subpages(wikitext: str, title: str = "") -> list[str]:
    """提取页面引用的子页面（transclusion），返回完整标题。

    干员主页里语音、密录正文都只有一行引用：
        {{:{{FULLPAGENAME}}/语音记录}}
        |storyTxt1={{FULLPAGENAME}}/干员密录/1
    正文在那些子页面里 —— 不顺着这条线找过去，
    「泥岩喜欢什么」这类问题永远答不了，因为主页里根本没有答案。
    """
    subs = set()
    for m in re.finditer(r"\{\{:\s*\{\{FULLPAGENAME\}\}/([^\}|]+)\}\}", wikitext):
        subs.add(m.group(1).strip())
    for m in re.finditer(r"\{\{FULLPAGENAME\}\}/([^\}|]+)", wikitext):
        subs.add(m.group(1).strip())
    prefix = f"{title}/" if title else ""
    return sorted(prefix + s for s in subs)


def parse_operator(title: str, wikitext: str) -> dict:
    """
    解析单个干员页面，返回结构化 dict。

    任何一步出错都会记录在 record['_error'] 里，不会中断整批处理。
    """
    record: dict = {
        "名称": title,
        "来源": f"https://prts.wiki/w/{title}",
    }
    try:
        code = mwparserfromhell.parse(strip_comments(wikitext or ""))
        g = group_templates(code)

        # --- 基础信息 ---
        ci_list = g.get("CharinfoV2") or g.get("Charinfo")
        if ci_list:
            ci = params_dict(ci_list[0])
            record.update({
                "干员名": ci.get("干员名") or title,
                "外文名": ci.get("干员外文名", ""),
                "日文名": ci.get("干员名jp", ""),
                "干员ID": ci.get("干员id", ""),
                # 坑：wiki 的「稀有度」是 0-based（0=1星 … 5=6星）。
                # 全站取值只可能是 0~5，泥岩/银灰/能天使这些都是 5。
                # 不 +1 的话，回答「泥岩几星」会答成 5 星。
                "星级": (lambda r: r + 1 if isinstance(r, int) else None)(
                    to_num(ci.get("稀有度", ""))
                ),
                "稀有度原始值": to_num(ci.get("稀有度", "")),
                "职业": ci.get("职业", ""),
                "分支": ci.get("分支", ""),
                "位置": ci.get("位置", ""),
                "标签": ci.get("标签", "").split(),
                "特性": clean_text(ci.get("特性", "")),
                "画师": ci.get("画师", ""),
                "中文配音": ci.get("中文配音", ""),
                "日文配音": ci.get("日文配音", ""),
            })

        # --- 干员类别：常规 / 特勤 / 特殊模式 ---
        # 这三类走完全不同的数值体系，必须分开，否则会污染查询结果。
        #
        #   常规干员  = 干员序号为正数（泥岩 168）
        #               走标准养成：精英化材料、技能升级材料、模组
        #   特殊模式  = 干员序号 -1（卫戍协议 / 集成战略 专属，如 Sharp）
        #               无养成体系
        #   特勤干员  = 序号为正，但页面含 {{特勤干员}} 模板（机械师、电弧）
        #               2026 新增类型：靠集成战略任务解锁精英化，
        #               页面里 {{材料消耗}} 出现 0 次 —— 根本不用材料
        #
        # 另外还有「异格/升变」干员（阿米娅→近卫/医疗）：靠剧情解锁，
        #   同样没有精英化材料，用 {{异格干员/升变}} 模板标识。
        seq = to_num(ci.get("干员序号", ""))
        record["干员序号"] = seq
        if not (isinstance(seq, int) and seq > 0):
            category = "特殊模式"
        elif g.get("特勤干员"):
            category = "特勤"
        else:
            category = "常规"
        record["干员类别"] = category

        # --- 异格 / 升变 判定（两者是不同概念，必须分开存）---
        #   升变 = {{异格干员/升变}}：阿米娅→近卫/医疗，靠剧情解锁，
        #          无精英化材料 —— 只有这一类该被「缺材料」豁免
        #   异格 = {{异格干员}} 且无「非异格=1」：浊心斯卡蒂等，
        #          异格形态但养成体系与常规干员一致，材料照常
        # 早期版本把两者混为一个「是否异格」，导致 38 个异格干员被漏标。
        _alt = _parse_alternate(g.get("异格干员", []))
        record["是否升变"] = bool(g.get("异格干员/升变"))
        record["是否异格"] = record["是否升变"] or (_alt.get("形态") == "异格")
        # 标题括号后缀，如 Sharp(卫戍协议) / 暮落(集成战略)
        m = re.search(r"[(（]([^()（）]+)[)）]\s*$", title)
        record["模式后缀"] = m.group(1) if m else ""

        # --- 获得方式 ---
        if g.get("干员获得方式"):
            ob = params_dict(g["干员获得方式"][0])
            record["获得方式"] = ob.get("获得方式", "")
            record["上线时间"] = ob.get("上线时间", "")

        # --- 势力 / 阵营（CharinfoV2）---
        # 探针实测（python scripts/09_probe_template.py CharinfoV2 --grep 所属）：
        #   所属国家  463 页有此参数，411 页有值  <- 主字段：罗德岛 / 拉特兰 / 维多利亚…
        #   所属组织  463 页有此参数， 82 页有值  <- 细分：罗德岛-精英干员、莱茵生命…
        # 踩过的坑：早期写成「所属势力」—— 该参数并不存在，
        #   只剩 82 个干员靠「所属组织」兜底，完整度暴跌到 17.6% 却看不出原因。
        # 教训：参数名必须用探针按「非空页数」确认，不能凭印象猜。
        record["势力"] = {
            k: clean_inline(ci.get(k, ""))
            for k in ("所属国家", "所属组织", "所属团队", "所属势力")
            if ci.get(k)
        }

        # --- 档案（人员档案set：结构化字段，100% 覆盖）---
        # 这些是「泥岩是什么种族」「谁出身维多利亚」这类问题的答案，
        # 结构化存储后可以走精确查询，不必交给向量检索。
        if g.get("人员档案set"):
            pf = params_dict(g["人员档案set"][0])
            record["档案"] = {
                k: clean_inline(pf.get(k, ""))
                for k in ("性别", "出身地", "生日", "种族", "身高",
                          "战斗经验", "是否感染者", "矿石病感染情况",
                          "物理强度", "战场机动", "生理耐受",
                          "战术规划", "战斗技巧", "源石技艺适应性",
                          "体细胞与源石融合率", "血液源石结晶密度")
                if pf.get(k)
            }

        # --- 档案正文（人员档案：长文本，供第 6 步切块入向量库）---
        # 客观履历 / 临床诊断 / 档案资料 / 晋升记录 等，
        # 「泥岩的过往经历是什么」这类问题只能靠语义检索回答。
        if g.get("人员档案"):
            recs = []
            for t in g["人员档案"]:
                d = params_dict(t)
                for i in range(1, 12):
                    body = d.get(f"档案{i}文本")
                    if body:
                        recs.append({
                            "标题": clean_inline(d.get(f"档案{i}", "")),
                            "解锁条件": clean_inline(d.get(f"档案{i}条件", "")),
                            "正文": clean_text(body),
                        })
            if recs:
                record["档案文本"] = recs

        # --- 面板 ---
        if g.get("属性"):
            record["面板"] = _parse_stats(params_dict(g["属性"][0]))

        # --- 精英化材料 ---
        if g.get("精英化材料"):
            em = g["精英化材料"][0]
            record["精英化材料"] = {
                str(p.name).strip(): parse_materials(str(p.value))
                for p in em.params
            }

        # --- 技能升级材料 ---
        if g.get("技能升级材料"):
            sm = g["技能升级材料"][0]
            record["技能升级材料"] = {
                str(p.name).strip(): parse_materials(str(p.value))
                for p in sm.params
            }

        # --- 模组 / 天赋 / 技能 ---
        # 技能、天赋都存在新旧模板混用，必须用 templates_in_order 全取并按出现顺序合并
        record["模组"] = _parse_modules(g.get("模组", []))
        record["天赋"] = _parse_talents(
            templates_in_order(code, {"天赋列表3", "天赋列表2", "天赋列表"})
        )
        record["技能"] = _parse_skills(templates_in_order(code, {"技能2", "技能"}))

        # --- 补全的 6 个结构化板块 ---
        # 攻击范围有两种写法：常规三档 / 卫戍协议单档，归一到同一字段
        record["攻击范围"] = (_parse_attack_range(g.get("干员攻击范围", []))
                             or _parse_attack_range_fallback(g.get("攻击范围", [])))
        if g.get("能力生效范围"):
            record["能力生效范围"] = _parse_ability_range(g["能力生效范围"])
        special = _parse_special_mechanics(wikitext)
        if special:
            record["特殊机制"] = special
        record["潜能"] = _parse_potential(g.get("潜能提升", []))
        record["后勤技能"] = _parse_base_skills(g.get("后勤技能", []))
        record["相关道具"] = _parse_related_items(g.get("相关道具", []))
        record["悖论模拟"] = _parse_paradox(g.get("悖论模拟", []))
        # list 类模板是嵌套在父模板里的，group_templates(recursive=True) 已平铺取出
        record["干员密录"] = _parse_story_sets(g.get("干员密录/list", []), title)

        # --- 异格关系（原型 ↔ 异格形态互指）---
        # _alt 在上面判类别时已算过，这里直接复用，避免重复解析
        if _alt:
            record["异格"] = _alt
        if g.get("干员异格任务"):
            record["异格任务"] = _parse_alt_tasks(
                params_dict(g["干员异格任务"][0]), g.get("干员异格任务/list", [])
            )

        # --- 子页面引用（语音 / 密录正文的真实位置）---
        subs = _extract_subpages(wikitext, title)
        if subs:
            record["子页面"] = subs

    except Exception as e:  # 单页失败不影响整批
        record["_error"] = f"{type(e).__name__}: {e}"

    return record


def is_operator_page(content: str) -> bool:
    """页面是否干员页（第 3 步勘探确定的标识模板）"""
    return "干员页面名" in (content or "")
