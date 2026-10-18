"""
干员子页面解析器：语音记录 + 干员密录正文

为什么要单独一个模块？
------------------------------------------------------------------
干员主页里这两块内容**只有一行引用**，正文在子页面：

    主页：{{:{{FULLPAGENAME}}/语音记录}}        → 泥岩/语音记录
    主页：storyTxt1={{FULLPAGENAME}}/干员密录/1 → 泥岩/干员密录/1

如果只解析主页，那么「泥岩喜欢什么」「泥岩和可露希尔什么关系」
这类问题永远答不了 —— 因为答案根本不在主页里，全在语音台词中。

体量（实测）：
    语音记录   439 页，平均 15.7k 字符，合计 6.6 MB
    干员密录   396 页，平均 26.1k 字符

去向：
    语音     → 结构化（标题 + 中文台词），进 SQLite 做精确查询
               同时整段合并进 Chroma，供「泥岩的口头禅是什么」等语义问题
    密录正文 → 清洗掉引擎指令后进 Chroma
"""

from __future__ import annotations

import re

from .wikitext import clean_inline, clean_text


# ---------------------------------------------------------------------------
# 0) 花括号配对扫描器 —— 解决「正则遇嵌套就提前收尾」这一整类 bug
# ---------------------------------------------------------------------------

def _match_braces(text: str, start: int) -> int:
    """从 start 处的 {{ 起，找到配对的 }}，返回 }} 的下标；找不到返回 -1。

    为什么不用正则：
        `\\{\\{VoiceData/word\\|(.*?)\\}\\} 的非贪婪 .*? 遇到**第一个** }} 就停。
        而台词里经常嵌着别的模板，例如
            {{VoiceData/word|中文|只要能照亮{{DrName|前缀=Dr.}}的话，就好了。}}
        第一个 }} 属于内层 {{DrName}}，于是整句被截成「只要能照亮{{DrName|前缀=Dr.」
        —— 而且不报错、不为空，是最难发现的静默数据丢失（实测 168 条中招）。
    正确做法：数括号。遇到 {{ 深度 +1，遇到 }} 深度 -1，回到 0 才是真结尾。
    """
    depth = 0
    i = start
    n = len(text)
    while i < n:
        if text.startswith("{{", i):
            depth += 1
            i += 2
        elif text.startswith("}}", i):
            depth -= 1
            if depth == 0:
                return i
            i += 2
        else:
            i += 1
    return -1


def iter_templates(text: str, name: str) -> list[str]:
    """找出所有 {{name|...}}（可含嵌套），返回去掉名字后的内容列表。

    例：iter_templates(s, "VoiceData/word")
        -> ["中文|只要能照亮...", "日文|...", ...]
    """
    out: list[str] = []
    key = name.lower()
    i, n = 0, len(text)
    while i < n:
        j = text.find("{{", i)
        if j < 0:
            break
        k = j + 2
        while k < n and text[k] in " \t\n":
            k += 1
        if text[k:k + len(key)].lower() != key:
            i = j + 2          # 不是我们要的模板，继续往下找
            continue
        end = _match_braces(text, j)
        if end < 0:            # 括号不闭合（页面被截断），放弃剩下的
            break
        body = text[j + 2:end]
        out.append(body[len(key):].lstrip(" \t\n|"))
        i = end + 2
    return out


def _field_end(text: str, start: int) -> int:
    """从 |台词N= 的值起点开始，找到这个值的结束位置。

    普通情况到行尾；但台词里若嵌了跨行模板，行尾时括号还没配平，
    就不能停 —— 一直读到括号配平为止，否则又是一次静默截断。
    """
    depth, i, n = 0, start, len(text)
    while i < n:
        if text.startswith("{{", i):
            depth += 1
            i += 2
        elif text.startswith("}}", i):
            depth -= 1
            i += 2
        elif text[i] == "\n" and depth <= 0:
            return i
        else:
            i += 1
    return n


# ---------------------------------------------------------------------------
# 语音记录
# ---------------------------------------------------------------------------

# {{VoiceData/word|中文|台词}} —— 同一句有中/日/英/韩/繁五语，我们默认取中文。
# 提取走 iter_templates（括号配对），不要再用正则，理由见 _match_braces 的注释。
_WORD_RE = re.compile(r"\{\{\s*VoiceData/word\s*\|\s*([^|}]+?)\s*\|\s*(.*?)\}\}", re.S)

# 语言标签归一化：不同页面写法不一致（中文 / 中文-普通话 / 简体中文）
_LANG_ALIAS = {
    "中文": "中文", "中文-普通话": "中文", "简体中文": "中文", "普通话": "中文",
    "日文": "日文", "日语": "日文",
    "英文": "英文", "英语": "英文",
    "韩文": "韩文", "韩语": "韩文",
    "繁体中文": "繁体中文", "繁中": "繁体中文",
}


def parse_voice_page(title: str, wikitext: str, lang: str = "中文") -> dict:
    """解析「干员名/语音记录」子页面。

    源结构：
        {{VoiceTable|表格标题=语音记录
        |语音key=char_311_mudrok
        |标题1=任命助理
        |台词1={{VoiceData/word|中文|...}}{{VoiceData/word|日文|...}}...
        |语音1=CN_001.wav
        |触发类型1=HOME_PLACE
        }}

    返回：
        {"语音key": ..., "条目": [{"标题","台词","触发类型","其他语言":{...}}]}
    """
    out = {"标题": title, "语音key": "", "条目": []}

    m = re.search(r"\|语音key\s*=\s*([^\n|]*)", wikitext)
    if m:
        out["语音key"] = m.group(1).strip()

    # 标题N / 台词N 成对出现，编号不一定连续（泥岩页缺 15、16、35 等）
    titles = {m.group(1): m.group(2).strip()
              for m in re.finditer(r"\|标题(\d+)\s*=\s*([^\n|]*)", wikitext)}
    # 台词值可能嵌跨行模板，所以用 _field_end 而不是简单的 [^\n]*
    lines = {m.group(1): wikitext[m.end():_field_end(wikitext, m.end())]
             for m in re.finditer(r"\|台词(\d+)\s*=", wikitext)}
    types = {m.group(1): m.group(2).strip()
             for m in re.finditer(r"\|触发类型(\d+)\s*=\s*([^\n|]*)", wikitext)}

    for idx in sorted(titles, key=lambda x: int(x)):
        raw_line = lines.get(idx, "")
        if not raw_line:
            continue

        # 拆出全部语种，默认语种单独成字段，其余留着做多语检索
        by_lang: dict[str, str] = {}
        for body in iter_templates(raw_line, "VoiceData/word"):
            if "|" not in body:
                continue
            lang_raw, line = body.split("|", 1)
            name = _LANG_ALIAS.get(lang_raw.strip())
            if name:
                by_lang[name] = clean_inline(line)

        main = by_lang.get(lang) or by_lang.get("中文") or ""
        if not main and by_lang:
            # 该条目没标中文，退回任意一种，宁可不全也不能丢
            main = next(iter(by_lang.values()))
        if not main:
            # 没有 VoiceData 包装，说明是纯文本台词
            main = clean_inline(raw_line)

        item = {
            "序号": int(idx),
            "标题": clean_inline(titles[idx]),
            "台词": main,
            "触发类型": types.get(idx, ""),
        }
        others = {k: v for k, v in by_lang.items() if v and v != main}
        if others:
            item["其他语言"] = others
        out["条目"].append(item)

    return out


def voice_to_text(vocal: dict, include_title: bool = True) -> str:
    """把语音条目压成一段纯文本，供第 6 步切块入向量库。

    每条带上标题（如「交谈1」「信赖提升后交谈2」），因为这类标题本身
    就是检索线索 —— 用户问「泥岩晋升后说了什么」，靠标题能直接命中。
    """
    parts = []
    for it in vocal.get("条目", []):
        if include_title:
            parts.append(f"【{it['标题']}】{it['台词']}")
        else:
            parts.append(it["台词"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 干员密录正文
# ---------------------------------------------------------------------------

# 剧情脚本里的引擎指令，形如 [Dialog] / [Background(image="x")] / [delay(time=1)]
# 这些是播放器指令不是文本内容，必须剥掉，否则 chunk 里全是噪声
_ENGINE_RE = re.compile(r"^\s*\[[^\]]*\]\s*$", re.M)
_INLINE_ENGINE_RE = re.compile(r"\[[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\]")


# {{:{{PAGENAME}}/data}} —— 少数密录页把正文拆到了同名子页面里
# （实测：梅尔/干员密录/1、魔王/干员密录/1）。
# 主页只剩一行引用，不展开的话抓到的是 25 个字符的模板尾巴，正文全丢。
_DATA_REF_RE = re.compile(r"\{\{:\s*(?:\{\{PAGENAME\}\}|[^}|]+?)\s*/data\s*\}\}")


def expand_data_refs(wikitext: str, title: str, subpages: dict[str, str] | None = None) -> str:
    """把 {{:{{PAGENAME}}/data}} 展开成子页面的真实内容。

    subpages：{页面标题: 页面源码}，由调用方预先建好（只存 /data 页，不占内存）。
    """
    if not subpages:
        return wikitext
    return _DATA_REF_RE.sub(lambda m: subpages.get(title + "/data", ""), wikitext)


def parse_story_page(title: str, wikitext: str) -> dict:
    """解析「干员名/干员密录/N」正文页。

    源结构：{{剧情模拟器|...|文本数据=
        [HEADER(key="title_test", ...)]
        [Dialog]
        [Background(image="bg_barracks",screenadapt="coverall")]
        鲍勃，关于你在上次的信中慷慨提出的那个建议……
        [stopmusic(fadetime=1.5)]
    }}

    返回 {"标题","正文","字符数"}
    """
    # 文本数据= 之后才是正文；取不到就退回整页
    m = re.search(r"\|文本数据\s*=\s*(.*)", wikitext, re.S)
    body = m.group(1) if m else wikitext

    # 去掉模板结尾的 }}
    body = re.sub(r"\}\}\s*$", "", body.strip())

    # 剥引擎指令：先整行，再行内
    body = _ENGINE_RE.sub("", body)
    body = _INLINE_ENGINE_RE.sub("", body)

    text = clean_text(body)
    # 压掉清洗后留下的空行
    text = re.sub(r"\n{2,}", "\n", text).strip()

    return {"标题": title, "正文": text, "字符数": len(text)}


def is_voice_page(title: str) -> bool:
    return title.endswith("/语音记录")


def is_story_page(title: str) -> bool:
    # 排除 /data 子页面：它只是主页正文的存放处，不是独立的一篇密录。
    # 不放过它的话会重复入库（实测：梅尔/干员密录/1 7286 字符，
    # 而 梅尔/干员密录/1/data 7271 字符，几乎逐字重复），
    # 第 6 步切块后同一个故事会有两份 chunk，检索时互相抢位次。
    return "/干员密录/" in title and not title.endswith("/data")
