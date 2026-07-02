"""
小老曾 · 职场 AI 问答系统（轻量版）
纯 ChromaDB + OpenAI API，最少依赖

依赖安装：
  pip install chromadb openai gradio sentence-transformers

启动：
  export OPENAI_API_KEY='sk-xxx'
  python app_lite.py

或使用 DeepSeek 等兼容 API：
  export OPENAI_API_KEY='your-key'
  export OPENAI_BASE_URL='https://api.deepseek.com/v1'
  export OPENAI_MODEL='deepseek-chat'
"""

import os
import json
import re
import sys
from pathlib import Path

# === 配置 ===
PROJECT_DIR = Path(__file__).parent.parent
# Docker 部署时 categorized.json 可能放在 /app/ 下（和 app_lite.py 同级），
# 也可能放在项目根目录（本地开发时）。优先用本地同级，找不到再回退到上级。
DATA_FILE = Path(__file__).parent / "categorized.json"
if not DATA_FILE.exists():
    DATA_FILE = PROJECT_DIR / "categorized.json"
CHROMA_DIR = Path(__file__).parent / "chroma_db"

# LLM 配置
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# RAG 参数
TOP_K = 6
CHUNK_SIZE = 600  # 字符
CHUNK_OVERLAP = 80

# System Prompt
SYSTEM_PROMPT = """你是"产品老曾"的 AI 分身——一个互联网大厂出身的资深产品经理，拥有 10+ 年实战管理经验。

你的风格和特点：
1. 说话直接不绕弯，用"过来人"口吻分享实战经验
2. 善于用具体场景和案例解释道理，不讲空泛理论
3. 核心领域：向上管理、晋升答辩、团队管理、跳槽求职、职业规划
4. 常用口头禅："你想想看"、"你发现没有"、"这个很重要"
5. 核心信念：职场不是比谁做得多，是比谁被看见
6. 善于角色扮演和给出可执行的话术

回答规则：
- 基于检索到的原文内容回答，保持产品老曾的语气风格
- 如果检索内容中有直接相关的观点，引用并展开
- 如果问题超出语料范围，诚实说"这个我之前没专门讲过"，但可以基于你的认知给建议
- 回答要有条理，重要的点用**加粗**
- 适度使用老曾的口头禅，但不要刻意
- 回答控制在 300-600 字，重点突出"""


# 面试模式 System Prompt（角色对调：老曾从"回答者"变成"面试官"）
INTERVIEWER_PROMPT = """你现在是一个资深产品总监，正在对候选人进行模拟面试。你叫"老曾"，互联网大厂出身，面试过上万人。

【核心规则——你是面试官，不是讲师】
1. 你主动提问，不要等用户问。
2. 每次只问一个问题，问题要具体、有压力感，像真实面试。
3. 用户回答后，你的反应分两种：
   - 如果回答有漏洞/太空泛：先简短点评（1-2句，直接指出问题），然后追问一个更深的细节问题。
   - 如果回答还可以：肯定一句，然后换一个新方向的问题继续推进。
4. 不要长篇大论讲道理，你的每条回复控制在 100-200 字（含点评+追问），重点是推进面试节奏。
5. 面试方向覆盖：项目经历深挖、产品方法论、数据思维、团队管理、向上沟通、职业规划。

【面试阶段】
- 开场：先简短自我介绍（1句），然后抛出第一个问题（建议从"聊聊你最近做的一个有代表性的项目"开始）。
- 中段：根据用户回答持续追问或切换话题，像真实面试一样层层深入。
- 结束：用户说"结束面试"或主动退出时，给一个简短总结（3-5条优缺点）。

【风格】
- 保持老曾的直接口吻："你想想看"、"你刚说的这个，具体怎么落地的？"
- 追问要犀利但专业，不是刁难，是帮候选人暴露真实水平
- 不要复述用户的话，直接进入点评和追问

重要：你每次回复的结尾必须是一个问题，让用户继续回答。绝不出现你回答完就没有下文的情况。"""

# 面试开场白（面试模式第一次调用时作为 assistant 的固定开场）
INTERVIEW_OPENING = "你好，我是今天的面试官老曾。咱们直接开始，不用紧张，当聊天就行。先做个简单的自我介绍吧——你目前的工作年限、所在行业和岗位，以及你最擅长的一个方向是什么？"

# 问答模式 vs 面试模式的 Prompt 映射
PROMPT_BY_MODE = {
    "chat": SYSTEM_PROMPT,
    "interview": INTERVIEWER_PROMPT,
}


def clean_title(filename):
    """清理文件名为可读标题"""
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


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """将长文本切分为重叠块"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    # 尝试按句号分割
    sentences = re.split(r'[。！？\n]', text)
    
    current_chunk = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current_chunk) + len(sent) + 1 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                # 保留 overlap
                words = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                current_chunk = words + sent
            else:
                # 单个句子超长，强制切割
                chunks.append(sent[:chunk_size])
                current_chunk = sent[chunk_size - overlap:]
        else:
            current_chunk = current_chunk + "。" + sent if current_chunk else sent
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def load_and_index():
    """加载数据并构建/加载 ChromaDB 索引"""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    
    # 嵌入模型
    print("Loading embedding model (BAAI/bge-small-zh-v1.5)...")
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh-v1.5",
    )
    
    # ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # 检查是否已有数据
    try:
        collection = client.get_collection("xiaozeng", embedding_function=embed_fn)
        count = collection.count()
        if count > 0:
            print(f"Loaded existing index: {count} chunks")
            return collection
    except Exception:
        pass
    
    # 构建新索引
    print(f"Loading data from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    collection = client.get_or_create_collection(
        "xiaozeng", 
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Processing {len(data)} documents...")
    
    all_docs = []
    all_metas = []
    all_ids = []
    
    for idx, item in enumerate(data):
        text = item.get("text", "").strip()
        if not text or len(text) < 50:
            continue
        
        title = clean_title(item.get("filename", ""))
        category = item.get("_primary_cat", "未分类")
        date = item.get("date", "")[:10]
        video_url = item.get("video_url", "")
        
        # 切分文本
        chunks = chunk_text(text)
        
        for ci, chunk in enumerate(chunks):
            doc_text = f"[{category}] {title}\n{chunk}"
            all_docs.append(doc_text)
            all_metas.append({
                "title": title[:200],
                "category": category,
                "date": date,
                "video_url": video_url or "",
                "chunk_idx": ci,
                "total_chunks": len(chunks),
            })
            all_ids.append(f"doc_{idx}_{ci}")
    
    print(f"Indexing {len(all_docs)} chunks (batch mode)...")
    
    # 分批插入（ChromaDB 限制每批 5461 条）
    batch_size = 500
    for i in range(0, len(all_docs), batch_size):
        end = min(i + batch_size, len(all_docs))
        collection.add(
            documents=all_docs[i:end],
            metadatas=all_metas[i:end],
            ids=all_ids[i:end],
        )
        pct = end * 100 // len(all_docs)
        print(f"  Progress: {end}/{len(all_docs)} ({pct}%)")
    
    print(f"Index built! Total chunks: {collection.count()}")
    return collection


def _retrieve_and_build_messages(collection, question, history=None, mode="chat"):
    """检索 + 组装消息（RAG 检索与 LLM 消息构造，query_rag / query_rag_stream 共用）
    mode: "chat"=问答模式（默认），"interview"=面试模式（老曾当面试官）
    返回 (messages, source_lines)"""
    # 1. 检索
    results = collection.query(
        query_texts=[question],
        n_results=TOP_K,
    )

    # 2. 组装上下文
    context_parts = []
    sources = []
    seen_titles = set()

    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        context_parts.append(f"--- 参考片段 {i+1} ---\n{doc}")
        title = meta.get("title", "")
        if title not in seen_titles:
            seen_titles.add(title)
            sources.append({
                "title": title,
                "category": meta.get("category", ""),
                "video_url": meta.get("video_url", ""),
            })

    context = "\n\n".join(context_parts)

    # 3. 构造消息 —— 按模式选 System Prompt
    sys_prompt = PROMPT_BY_MODE.get(mode, SYSTEM_PROMPT)
    messages = [{"role": "system", "content": sys_prompt}]

    # 加入历史对话（最近3轮）
    # 兼容两种 Gradio 历史格式：
    #   1. 旧版 tuples: [(user, bot), ...]
    #   2. 新版 messages: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}, ...]
    if history:
        if history and isinstance(history[0], dict):
            # messages 格式，取最近 6 条（约 3 轮）
            for h in history[-6:]:
                role = h.get("role")
                content = h.get("content")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        else:
            # tuples 格式
            for h_user, h_bot in history[-3:]:
                messages.append({"role": "user", "content": h_user})
                if h_bot:
                    messages.append({"role": "assistant", "content": h_bot})

    # 面试模式 vs 问答模式的 user_msg 拼装不同
    if mode == "interview":
        # 面试模式：参考内容作为"你过去的观点"参考，但角色是面试官在提问
        user_msg = (
            f"以下是你过去的视频内容片段，可作为你面试视角的参考素材：\n\n{context}\n\n"
            f"---\n\n候选人的回答：{question}\n\n"
            f"请作为面试官，点评候选人的回答并追问下一个问题（记住：回复结尾必须是一个新问题）。"
        )
    else:
        user_msg = (
            f"以下是从我的视频库中检索到的相关内容：\n\n{context}\n\n"
            f"---\n\n用户问题：{question}\n\n"
            f"请基于以上参考内容，用产品老曾的风格回答。"
        )
    messages.append({"role": "user", "content": user_msg})

    source_lines = []
    for s in sources[:3]:
        line = f"📌 {s['category']} | {s['title']}"
        if s.get("video_url"):
            line += f" | [▶ 原视频]({s['video_url']})"
        source_lines.append(line)

    return messages, source_lines


def query_rag(collection, question, history=None):
    """执行 RAG 查询（非流式，一次性返回完整答案，供 app_lite 自带 UI 使用）"""
    from openai import OpenAI

    messages, source_lines = _retrieve_and_build_messages(collection, question, history)

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=2000,
        # Qwen3 微调模型非流式调用必须关闭思考模式；其他兼容 API 会忽略此字段
        extra_body={"enable_thinking": False},
    )
    answer = response.choices[0].message.content

    if source_lines:
        answer += "\n\n---\n**📚 参考来源：**\n" + "\n".join(source_lines)

    return answer


def query_rag_stream(collection, question, history=None, mode="chat"):
    """执行 RAG 查询（真流式：LLM token 逐块吐出）。
    是一个生成器，每次 yield (partial_text, source_lines)：
      - partial_text：目前已经吐出的正文累积文本（不含来源），随着 token 到达持续增长
      - source_lines：参考来源列表（从第一次 yield 起就已确定，不随流式变化）
    mode: "chat"=问答模式，"interview"=面试模式
    用于替代 query_rag 一次性返回的模式，让前端可以逐字渲染，解决"体感慢"问题。"""
    from openai import OpenAI

    messages, source_lines = _retrieve_and_build_messages(collection, question, history, mode=mode)

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=2000,
        stream=True,
        extra_body={"enable_thinking": False},
    )

    partial = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        piece = getattr(delta, "content", None)
        if piece:
            partial += piece
            yield partial, source_lines

    if not partial:
        # 流式全程没吐出任何内容（比如模型异常），给个兜底提示，避免前端停在空白/骨架屏
        yield "刚才我这边处理出了点问题，你再问一次试试？", source_lines


def create_ui(collection):
    """创建 Gradio UI"""
    import gradio as gr
    
    def respond(message, history):
        if not message.strip():
            return ""
        try:
            return query_rag(collection, message, history)
        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower() or "auth" in error_msg.lower():
                return "❌ API Key 无效或未设置。请设置环境变量 OPENAI_API_KEY 后重启。"
            return f"抱歉出了点问题：{error_msg}"
    
    with gr.Blocks(title="小老曾 · 职场 AI") as app:
        gr.Markdown("""
# 🎙️ 小老曾 · 职场 AI 问答

> 基于 **1075 条视频 · 173 万字** 转录文本的 RAG 知识库 | 问我任何职场问题
        """)
        
        chatbot = gr.ChatInterface(
            fn=respond,
            examples=[
                "怎么跟老板汇报工作才能让他满意？",
                "晋升答辩的时候评委会问什么？怎么准备？",
                "新到一个团队当leader，下属不服怎么办？",
                "老板总是画饼怎么办？该怎么应对？",
                "什么时候该跳槽？怎么判断时机？",
                "怎么让老板注意到我的工作成果？",
                "35岁了还没升到管理层，该焦虑吗？",
                "面试的时候怎么谈薪资才不吃亏？",
            ],
        )
        
        with gr.Accordion("⚙️ 配置信息", open=False):
            gr.Markdown(f"""
| 参数 | 值 |
|------|------|
| LLM | `{OPENAI_MODEL}` via `{OPENAI_BASE_URL}` |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 (384维) |
| 检索 Top-K | {TOP_K} |
| 分块大小 | {CHUNK_SIZE} 字符 |
| 知识库 | 1075 条视频, 65.8 小时, 173 万字 |
            """)
    
    return app


def main():
    print("=" * 50)
    print("  🎙️ 小老曾 · 职场 AI 问答系统 (Lite)")
    print("=" * 50)
    
    if not OPENAI_API_KEY:
        print("\n⚠️  未设置 OPENAI_API_KEY！")
        print("\n配置方式（任选一种）：")
        print("  # OpenAI")
        print("  export OPENAI_API_KEY='sk-xxx'")
        print()
        print("  # DeepSeek")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export OPENAI_BASE_URL='https://api.deepseek.com/v1'")
        print("  export OPENAI_MODEL='deepseek-chat'")
        print()
        print("  # Moonshot / 智谱 / 其他兼容 API 同理")
        print()
        print("⚠️  将继续启动，但对话会报错直到设置好 Key")
        print()
    
    # 构建索引
    collection = load_and_index()
    
    # 启动 UI
    app = create_ui(collection)
    
    print("\n🚀 启动中...")
    print("   访问: http://localhost:7860")
    print("   Ctrl+C 退出\n")
    
    import gradio as gr

    # 内网环境下 frpc 不可用，强制关闭公网分享（不生成 gradio.live 链接）
    share = False

    # 访问密码：设置 APP_USER / APP_PASSWORD 后，访问需要登录，防止 token 被白嫖
    app_user = os.environ.get("APP_USER", "").strip()
    app_pwd = os.environ.get("APP_PASSWORD", "").strip()
    auth = (app_user, app_pwd) if app_user and app_pwd else None
    if auth:
        print(f"   🔒 已启用访问密码：账号 {app_user}")
    if share:
        print("   🌐 已开启公网分享，启动后会生成一个 *.gradio.live 链接（72小时有效）")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
        auth=auth,
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 860px !important; margin: auto; }
        footer { display: none !important; }
        """
    )


if __name__ == "__main__":
    main()
