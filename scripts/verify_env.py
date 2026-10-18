"""
第 1 步：环境自检
-----------------
确认依赖都装好了、显卡能被 PyTorch 识别。

运行方式：
  在 PyCharm 里右键本文件 -> Run 'verify_env'
  或终端执行： python scripts/verify_env.py
"""
import sys
import importlib

CHECKS = [
    ("requests",             "数据采集：请求 PRTS API"),
    ("mwparserfromhell",     "数据解析：解析 wikitext 模板（核心）"),
    ("tqdm",                 "进度条显示"),
    ("chromadb",             "向量数据库"),
    ("sentence_transformers", "Embedding 模型 BGE-M3"),
    ("openai",               "大模型 API 客户端"),
    ("dotenv",               "读取 .env 里的密钥"),
]


def check_imports():
    print("=" * 62)
    print("依赖检查")
    print("=" * 62)
    all_ok = True
    for module, purpose in CHECKS:
        try:
            mod = importlib.import_module(module)
            ver = getattr(mod, "__version__", "已安装")
            print(f"  [OK]   {module:<22} {ver:<12} {purpose}")
        except ImportError:
            print(f"  [缺失] {module:<22} {'':<12} {purpose}")
            all_ok = False
    return all_ok


def check_torch():
    print()
    print("=" * 62)
    print("PyTorch / CUDA 检查（决定 Embedding 能否用显卡加速）")
    print("=" * 62)
    try:
        import torch
    except ImportError:
        print("  [缺失] torch —— 请先执行：")
        print("         pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False

    print(f"  torch 版本      : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        print(f"  CUDA 可用       : 是")
        print(f"  显卡            : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  显存            : {vram:.1f} GB")
        if vram < 3:
            print("  [提示] 显存偏小，BGE-M3 建议改用 bge-small-zh-v1.5")
    else:
        print("  CUDA 可用       : 否（将用 CPU 跑 Embedding，会慢很多）")
        print("  [提示] 若你有 N 卡，说明装的是 CPU 版 torch，需重装 CUDA 版")
    return cuda_ok


def main():
    print()
    print(f"Python 版本: {sys.version.split()[0]}")
    print()
    ok = check_imports()
    torch_ok = check_torch()

    print()
    print("=" * 62)
    if ok and torch_ok:
        print("环境就绪，可以进入第 2 步：数据采集")
    elif ok:
        print("依赖齐全，但显卡不可用。可以先用 CPU 继续，后续再优化")
    else:
        print("有依赖缺失，请按上面的 [缺失] 项安装后重新运行本脚本")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
