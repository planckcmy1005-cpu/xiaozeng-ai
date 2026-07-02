"""
LoRA 训练数据生成器
从 1075 条转录文本中生成 Q&A 训练对

用法：
  export OPENAI_API_KEY='sk-xxx'
  python generate_training_data.py

可选环境变量：
  OPENAI_BASE_URL  - API 地址 (默认 OpenAI)
  OPENAI_MODEL     - 模型 (默认 gpt-4o-mini)
  BATCH_SIZE       - 每批处理条数 (默认 5)
  MAX_ENTRIES      - 最大处理条数 (默认全部)
  START_FROM       - 从第几条开始 (断点续传)

输出：
  training_data.jsonl - Alpaca 格式训练数据
  training_data_sharegpt.jsonl - ShareGPT 格式训练数据（适用于 LLaMA Factory）
"""

import os
import json
import re
import sys
import time
from pathlib import Path
from openai import OpenAI

import random

# 配置
PROJECT_DIR = Path(__file__).parent.parent
DATA_FILE = PROJECT_DIR / "categorized.json"
OUTPUT_DIR = Path(__file__).parent / "training_data"
OUTPUT_DIR.mkdir(exist_ok=True)

ALPACA_FILE = OUTPUT_DIR / "training_data_alpaca.jsonl"
SHAREGPT_FILE = OUTPUT_DIR / "training_data_sharegpt.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "5"))
MAX_ENTRIES = int(os.environ.get("MAX_ENTRIES", "0"))  # 0 = all
START_FROM = int(os.environ.get("START_FROM", "0"))

# 生成 prompt
GENERATION_PROMPT = """你是一个数据标注专家。你的任务是从以下视频转录文本中生成高质量的问答训练对。

这些文本来自"产品老曾"——一个资深产品经理的职场分享。请生成 3-5 个问答对，让 AI 学会用他的风格回答问题。

要求：
1. 问题要自然，像真人会问的（"怎么..."、"如果..."、"为什么..."）
2. 回答要保留原文的核心观点和说话风格（直接、案例化、"你想想看"等口头禅）
3. 回答长度 100-300 字，不要太短也不要注水
4. 问题要多样化：有方法论类、场景类、认知类
5. 回答必须基于原文内容，不要编造

输出格式（严格 JSON 数组）：
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]

---

视频分类：{category}
视频标题：{title}

转录原文：
{text}

---

请生成问答对（JSON 数组）："""


def clean_title(filename):
    name = filename
    if name.endswith('.mp4'):
        name = name[:-4]
    if name.endswith('_video'):
        name = name[:-6]
    m = re.match(r'\d{4}-\d{2}-\d{2}\s+\d{2}-\d{2}-\d{2}_', name)
    if m:
        name = name[m.end():]
    idx = name.find('_#')
    if idx > 0:
        name = name[:idx]
    parts = re.split(r'\.{3,}', name)
    name = parts[0] if parts else name
    name = name.strip('_').strip()
    return name if name else filename[:60]


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"processed": 0, "qa_count": 0, "errors": 0}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def generate_qa_pairs(client, item):
    """为单条转录生成 Q&A 对（含 3 次重试 + 指数退避）"""
    title = clean_title(item.get("filename", ""))
    category = item.get("_primary_cat", "未分类")
    text = item.get("text", "")

    # 截断过长文本
    if len(text) > 3000:
        text = text[:3000] + "..."

    prompt = GENERATION_PROMPT.format(
        category=category, title=title, text=text
    )

    # 3 次重试 + 指数退避（2s → 4s → 8s）。最后一次失败原样抛出，由外层计入 errors
    max_retries = 3
    response = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
                timeout=60,  # 60 秒超时，防止 API hang 死
            )
            break  # 成功，退出重试循环
        except Exception as e:
            if attempt < max_retries - 1:
                backoff = 2 ** (attempt + 1)  # 2, 4, 8
                err_name = type(e).__name__
                print(f" [retry {attempt + 1}/{max_retries} in {backoff}s: {err_name}]", end="", flush=True)
                time.sleep(backoff)
            else:
                raise  # 第 3 次仍失败，把异常抛给外层 main()

    content = response.choices[0].message.content.strip()
    content = re.sub(r'^```(?:json)?\s*\n', '', content)
    content = re.sub(r'\n```\s*$', '', content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    start = content.find('[')
    end = content.rfind(']')
    if start >= 0 and end > start:
        try:
            return json.loads(content[start:end+1])
        except json.JSONDecodeError:
            pass
    return None


def to_alpaca_format(question, answer):
    """Alpaca 格式"""
    return {
        "instruction": question,
        "input": "",
        "output": answer,
        "system": "你是产品老曾——互联网大厂资深产品经理，10+年管理经验。说话直接、用案例说话、给可执行建议。",
    }


def to_sharegpt_format(question, answer):
    """ShareGPT 格式（LLaMA Factory 推荐）"""
    return {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ],
        "system": "你是产品老曾——互联网大厂资深产品经理，10+年管理经验。说话直接、用案例说话、给可执行建议。",
    }


def main():
    if not API_KEY:
        print("❌ 请先设置 OPENAI_API_KEY 环境变量")
        print("   export OPENAI_API_KEY='sk-xxx'")
        print("   # 或 DeepSeek: export OPENAI_BASE_URL='https://api.deepseek.com/v1'")
        sys.exit(1)
    
    print("=" * 50)
    print("  产品老曾 LoRA 训练数据生成器")
    print("=" * 50)
    print(f"  Model: {MODEL}")
    print(f"  API: {BASE_URL}")
    print(f"  Batch: {BATCH_SIZE}")
    print()
    
    # 加载数据
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 过滤有效条目（至少 200 字）
    valid_data = [item for item in data if len(item.get("text", "")) >= 200]
    print(f"Total valid entries: {len(valid_data)}")
    
    if MAX_ENTRIES > 0:
        valid_data = valid_data[:MAX_ENTRIES]
        print(f"Limited to: {MAX_ENTRIES}")
    
    # 断点续传
    progress = load_progress()
    start_idx = max(START_FROM, progress["processed"])
    if start_idx > 0:
        print(f"Resuming from entry {start_idx} (already {progress['qa_count']} QA pairs)")
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # 打开输出文件（追加模式）
    alpaca_f = open(ALPACA_FILE, "a", encoding="utf-8")
    sharegpt_f = open(SHAREGPT_FILE, "a", encoding="utf-8")
    
    total_qa = progress["qa_count"]
    errors = progress["errors"]
    
    try:
        for i in range(start_idx, len(valid_data)):
            item = valid_data[i]
            title = clean_title(item.get("filename", ""))[:50]
            
            print(f"[{i+1}/{len(valid_data)}] {title}...", end=" ", flush=True)
            
            try:
                qa_pairs = generate_qa_pairs(client, item)
                
                if qa_pairs and isinstance(qa_pairs, list):
                    for qa in qa_pairs:
                        q = qa.get("question", "").strip()
                        a = qa.get("answer", "").strip()
                        if q and a and len(a) > 50:
                            alpaca_f.write(json.dumps(to_alpaca_format(q, a), ensure_ascii=False) + "\n")
                            sharegpt_f.write(json.dumps(to_sharegpt_format(q, a), ensure_ascii=False) + "\n")
                            total_qa += 1
                    
                    print(f"✓ {len(qa_pairs)} pairs (total: {total_qa})")
                else:
                    print("⚠ no valid output")
                    errors += 1
                    
            except Exception as e:
                print(f"✗ {str(e)[:50]}")
                errors += 1
                if "rate_limit" in str(e).lower():
                    print("  Rate limited, sleeping 30s...")
                    time.sleep(30)
                elif "timeout" in str(e).lower():
                    time.sleep(5)
            
            # 保存进度
            progress = {"processed": i + 1, "qa_count": total_qa, "errors": errors}
            save_progress(progress)
            
            # Rate limiting: 每 batch 后 sleep 2-3 秒 + 随机抖动，避免触发限流
            if (i + 1) % BATCH_SIZE == 0:
                delay = 2 + random.uniform(0, 1)
                time.sleep(delay)
    
    except KeyboardInterrupt:
        print("\n\n⏸ Interrupted! Progress saved.")
    finally:
        alpaca_f.close()
        sharegpt_f.close()
    
    print(f"\n{'='*50}")
    print(f"  完成！")
    print(f"  处理: {progress['processed']}/{len(valid_data)} 条")
    print(f"  生成: {total_qa} 个 QA 对")
    print(f"  错误: {errors}")
    print(f"  输出:")
    print(f"    {ALPACA_FILE}")
    print(f"    {SHAREGPT_FILE}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
