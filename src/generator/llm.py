"""第 8 步（二）：LLM 客户端 —— 真正把 prompt 送出去的那一步。

【为什么用 OpenAI 兼容协议】
国内主流厂商（DeepSeek / 智谱 / 通义 / 月之暗面）都实现了 OpenAI 的
`/chat/completions` 接口。所以只要用 `openai` 这个 SDK，
换厂商 = 换 base_url + api_key + model 三个字符串，业务代码一行不用改。

这是本项目选型时的重要考虑：**不被某一家绑死**。

【DeepSeek 默认配置】
    base_url  https://api.deepseek.com
    model     deepseek-chat        （DeepSeek-V3 通用对话，便宜且够用）
              另有 deepseek-reasoner（推理模型，慢、贵，本项目不需要）

【三个必须做的工程细节】
1. temperature=0：问答系统要的是稳定复现，不是创意。
   默认 1.0 时同一个问题两次回答可能不一样，你没法做回归测试。
2. 超时 + 重试：网络抖动是常态。但**重试要挑异常**——
   超时/限流/5xx 该重试，401（key 错了）重试一百次也没用，要立刻报错。
3. mock 模式：不花钱也能验证整条链路（prompt 拼得对不对、引用编号通不通）。
   这是本项目强烈推荐的第一跑法。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

# ---- DeepSeek 默认参数（可用环境变量覆盖，方便换厂商）----
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# 可重试的异常名（openai SDK 的类名；用字符串比对，不强依赖具体版本）
RETRYABLE = {
    "APITimeoutError", "APIConnectionError", "RateLimitError",
    "InternalServerError", "APIStatusError", "Timeout",
}


# ---------------------------------------------------------------------------
# .env 加载
# ---------------------------------------------------------------------------
def load_env(root: Path | None = None) -> None:
    """读取项目根目录的 .env（不覆盖已存在的环境变量）。

    优先用 python-dotenv，没装就用手写解析 —— 不让一个可选依赖
    变成跑不起来的理由。
    """
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    env_file = root / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv as _ld          # type: ignore
        _ld(env_file, override=False)
        return
    except Exception:
        pass
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------
class LLMClient:
    """一个 OpenAI 兼容的聊天客户端。"""

    def __init__(self, api_key: str | None = None,
                 base_url: str | None = None,
                 model: str | None = None,
                 temperature: float = 0.0,
                 max_tokens: int = 800,
                 timeout: float = 60.0,
                 max_retries: int = 2,
                 mock: bool = False):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") \
            or os.getenv("LLM_API_KEY") or ""
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") \
            or os.getenv("LLM_BASE_URL") or DEEPSEEK_BASE_URL
        self.model = model or os.getenv("DEEPSEEK_MODEL") \
            or os.getenv("LLM_MODEL") or DEEPSEEK_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.mock = mock

        if self.mock:
            print("  [LLM] mock 模式：不会真的调用 API，用于验证链路")
        elif not self.api_key:
            raise RuntimeError(
                "没找到 API Key。请在项目根目录建 .env 写入：\n"
                "    DEEPSEEK_API_KEY=sk-你的密钥\n"
                "（参考 .env.example）。或者加 --mock 先跑通链路。")
        else:
            print(f"  [LLM] {self.base_url} · model={self.model}")

    @classmethod
    def from_env(cls, mock: bool = False, **kw) -> "LLMClient":
        load_env()
        return cls(mock=mock, **kw)

    # ------------------------------------------------------------------
    def chat(self, messages: list[dict], stream: bool = False,
             temperature: float | None = None) -> str:
        """调用聊天接口，返回回答文本。"""
        if self.mock:
            return mock_reply(messages)

        from openai import OpenAI                      # 延迟导入，省启动时间

        client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                        timeout=self.timeout, max_retries=0)  # 重试自己管
        last_err: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature if temperature is None
                    else temperature,
                    max_tokens=self.max_tokens,
                    stream=stream,
                )
                if stream:
                    chunks = []
                    for part in resp:
                        if part.choices and part.choices[0].delta.content:
                            chunks.append(part.choices[0].delta.content)
                    return "".join(chunks)
                return (resp.choices[0].message.content or "").strip()

            except Exception as e:                      # noqa: BLE001
                last_err = e
                name = type(e).__name__
                if name in ("AuthenticationError", "PermissionDeniedError"):
                    raise RuntimeError(
                        f"API Key 被拒绝（{name}）。检查 .env 里的 "
                        f"DEEPSEEK_API_KEY 是否正确、账户是否余额充足。") from e
                if name not in RETRYABLE and "timeout" not in name.lower():
                    raise
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    print(f"  [LLM] {name}，{wait}s 后重试"
                          f"（{attempt + 1}/{self.max_retries}）…")
                    time.sleep(wait)

        raise RuntimeError(f"LLM 调用失败（已重试 {self.max_retries} 次）：{last_err}")


# ---------------------------------------------------------------------------
# mock 回复
# ---------------------------------------------------------------------------
def mock_reply(messages: list[dict]) -> str:
    """不开 API 时的假回答。

    它**故意不编造内容**，只把收到的证据复述一遍。
    这样一眼就能看出：prompt 里到底塞了什么、引用编号对不对、
    SQL 表格有没有正确进去。真接上 API 后，这段就换成模型输出。
    """
    user = ""
    for m in messages:
        if m.get("role") == "user":
            user = m.get("content", "")
    q = user.split("\n")[1] if "\n" in user else user
    has_sql = "【数据表】" in user
    ev_start = user.find("【参考资料】")
    ev = user[ev_start:ev_start + 400] if ev_start >= 0 else ""

    lines = [f"【MOCK 回答，未调用 API】问题：{q}",
             f"收到的数据：{'有 SQL 数据表' if has_sql else '无 SQL 数据'}，"
             f"{'有文本证据' if ev else '无文本证据'}",
             "",
             "参考资料片段预览：",
             ev or "（空）"]
    return "\n".join(lines)
