"""
PRTS Wiki API 客户端
====================
封装 MediaWiki API，解决三个真实工程问题：
  1. 限流     —— 控制请求间隔，不给对方服务器添麻烦，也避免自己被封
  2. 重试退避 —— 遇到 403（限流）自动等待更久再试，而不是立刻放弃
  3. 批量取   —— 一次请求最多拿 50 个页面，把 1.6 万页压缩到约 330 次请求

这是"能跑"和"能稳定跑完"之间的区别。
"""
import time
import random
import requests

API_URL = "https://prts.wiki/api.php"

# 伪装成浏览器。PRTS 有 WAF，不带 UA 会直接 403
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://prts.wiki/",
}

# MediaWiki 对未登录用户限制：一次最多查询 50 个页面标题
MAX_TITLES_PER_REQUEST = 50


class PRTSClient:
    def __init__(self, delay: float = 2.5, timeout: int = 60, max_retries: int = 6):
        """
        delay       每次请求之间至少间隔的秒数（限流，别调太小）
        timeout     单次请求超时时间
        max_retries 遇到 403/网络错误最多重试几次
        """
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_request_at = 0.0

    def _throttle(self):
        """保证两次请求之间至少间隔 delay 秒"""
        elapsed = time.time() - self._last_request_at
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_at = time.time()

    def _get(self, params: dict) -> dict:
        """发一次请求，带指数退避重试"""
        params = dict(params)
        params.setdefault("format", "json")

        last_error = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self.session.get(API_URL, params=params, timeout=self.timeout)

                # 403 是 PRTS 的限流信号，不是真的拒绝 —— 等更久再重试
                if resp.status_code == 403:
                    wait = self.delay * (2 ** attempt) + random.uniform(0, 1.5)
                    time.sleep(wait)
                    last_error = f"403 限流（第 {attempt + 1} 次，等待 {wait:.1f}s）"
                    continue

                resp.raise_for_status()
                return resp.json()

            except Exception as e:
                last_error = str(e)
                wait = self.delay * (2 ** attempt) + random.uniform(0, 1.0)
                time.sleep(wait)

        raise RuntimeError(f"请求失败，已重试 {self.max_retries} 次：{last_error}")

    def query(self, params: dict) -> dict:
        """发送任意 MediaWiki 查询（list= / prop= 均可），带限流与重试。

        第 8 步补丁新增：18 脚本用 prop=redirects 反查干员页的重定向别名。
        """
        return self._get(params)

    def iter_all_titles(self, namespace: int = 0):
        """
        分页遍历全站条目标题（生成器，边拿边用，不占内存）
        namespace=0 表示只取正式条目，排除"讨论""模板""帮助"等页面
        """
        cont = {}
        while True:
            params = {
                "action": "query",
                "list": "allpages",
                "apnamespace": namespace,
                "aplimit": 500,
                "apfilterredir": "nonredirects",  # 跳过重定向页，少抓很多垃圾
            }
            params.update(cont)

            data = self._get(params)
            for page in data.get("query", {}).get("allpages", []):
                yield page["title"]

            if "continue" in data:
                cont = data["continue"]
            else:
                break

    def fetch_batch(self, titles: list) -> dict:
        """
        批量获取页面原始 wikitext
        返回 {标题: 内容}；不存在的页面值为 None
        """
        if len(titles) > MAX_TITLES_PER_REQUEST:
            raise ValueError(f"一次最多 {MAX_TITLES_PER_REQUEST} 个标题，收到 {len(titles)} 个")

        data = self._get({
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "titles": "|".join(titles),
        })

        result = {}
        for page in data.get("query", {}).get("pages", {}).values():
            title = page.get("title")
            if "revisions" in page:
                # 页面存在：取出正文
                result[title] = page["revisions"][0]["slots"]["main"].get("*", "")
            else:
                # 页面不存在（可能是重定向或已删除）
                result[title] = None
        return result
