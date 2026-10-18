"""
第 2 步：抓取全站 wikitext 到本地
=================================
存储格式用 JSONL（每行一个 JSON），而不是一个大 JSON 文件。
原因：可以边抓边追加，中断后重跑自动跳过已抓的 —— 这就是断点续传。

先用小样本试跑：
    python scripts/01_fetch.py --limit 200
确认没问题再全量：
    python scripts/01_fetch.py
"""
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

# 让脚本能 import 到 src/ 下的模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prts_client import PRTSClient, MAX_TITLES_PER_REQUEST

OUTPUT_FILE = Path(__file__).parent.parent / "data" / "raw" / "pages.jsonl"


def load_finished_titles(path: Path) -> set:
    """读取已抓完的标题，实现断点续传"""
    done = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["title"])
            except json.JSONDecodeError:
                continue  # 跳过写坏的行
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只抓前 N 页（0 表示全量）")
    parser.add_argument("--delay", type=float, default=2.5, help="请求间隔秒数")
    args = parser.parse_args()

    client = PRTSClient(delay=args.delay)

    # --- 1. 拿到全站标题 ---
    print("正在获取全站条目列表...")
    all_titles = list(client.iter_all_titles())
    print(f"全站条目共 {len(all_titles)} 个")

    # --- 2. 断点续传：跳过已抓过的 ---
    done = load_finished_titles(OUTPUT_FILE)
    if done:
        print(f"发现已有 {len(done)} 页，自动跳过")
    todo = [t for t in all_titles if t not in done]

    if args.limit:
        todo = todo[: args.limit]
    print(f"本次待抓取 {len(todo)} 页\n")

    if not todo:
        print("没有需要抓的页面，已全部完成")
        return

    # --- 3. 分批抓取，边抓边写 ---
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ok_count = 0
    miss_count = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        with tqdm(total=len(todo), unit="page", desc="抓取中") as pbar:
            for i in range(0, len(todo), MAX_TITLES_PER_REQUEST):
                batch = todo[i : i + MAX_TITLES_PER_REQUEST]

                pages = client.fetch_batch(batch)
                for title, content in pages.items():
                    if content is None:
                        miss_count += 1
                        continue
                    record = {"title": title, "content": content}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    ok_count += 1

                f.flush()  # 立刻落盘，防止中断丢失这一批
                pbar.update(len(batch))

    print(f"\n完成：成功 {ok_count} 页，缺失 {miss_count} 页")
    print(f"数据文件：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
