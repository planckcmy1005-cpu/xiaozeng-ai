# 小老曾 · AI 职场面试官

基于产品老曾 1000+ 条抖音视频转录文本构建的 RAG 知识库，用老曾自己的声音回答职场问题，支持模拟面试模式。

## 功能

- **问答模式**：打字或语音提问，老曾用他自己的风格回答（流式输出 + 语音回复）
- **模拟面试**：老曾当面试官，主动提问 → 你回答 → 老曾点评 + 追问，全程语音
- **语音交互**：录音 → ASR 转写 → 可编辑确认 → 发文字/发原声
- **对话导出**：一键导出整场对话为 Markdown

## 技术架构

```
前端（纯 HTML/CSS/JS）          后端（FastAPI）
┌─────────────────────┐      ┌──────────────────────┐
│  录音 MediaRecorder  │      │  /api/asr  语音识别   │
│  状态机 idle/rec/review │ ←→ │  /api/chat SSE流式回答 │
│  SSE 流式渲染打字机   │      │  /api/tts  语音合成    │
│  Tailwind CSS 样式   │      │  RAG (ChromaDB)       │
└─────────────────────┘      └──────────────────────┘
```

- **RAG 大脑**：ChromaDB + BAAI/bge-small-zh-v1.5 嵌入 + Qwen 微调模型
- **语音识别**：Qwen3-ASR-Flash（比 paraformer 更准）
- **语音合成**：Qwen3-TTS-VC 声音复刻（克隆老曾音色）
- **TTS 缓存**：同一段文字只合成一次，后续秒返回

## 快速开始

### 1. 环境准备

```bash
# Python 3.10+
cd rag_xiaozeng
pip install -r requirements.txt
```

### 2. 配置密钥

```bash
cd ..  # 回到项目根目录
cp .env.example .env
# 编辑 .env，填入你的百炼 API Key
```

在 [阿里云百炼](https://bailian.console.aliyun.com/) 获取 API Key。

### 3. 克隆音色（首次需要）

```bash
cd rag_xiaozeng
# 准备一段老曾的干净语音（10-30秒），放在 voice_assets/ 目录
python clone_voice.py
# 成功后会生成 voice_id.txt
```

### 4. 启动

```bash
./run_server.sh
# 浏览器打开 http://localhost:7863
```

首次启动会构建向量索引（约 1-2 分钟），之后直接加载缓存。

## 知识库数据

`categorized.json` 包含 1075 条视频转录文本（173 万字）。如果你有自己的语料，替换这个文件后重启即可重新构建索引。

## 项目结构

```
.
├── .env.example          # 环境变量模板
├── categorized.json       # 知识库语料（5.5M）
├── rag_xiaozeng/
│   ├── server.py          # FastAPI 后端
│   ├── app_lite.py        # RAG 核心（检索 + LLM 调用）
│   ├── app_voice.py       # ASR/TTS 语音处理
│   ├── clone_voice.py     # 声音克隆工具
│   ├── static/index.html  # 前端页面
│   ├── assets/            # 头像等静态资源
│   ├── requirements.txt
│   └── run_server.sh      # 启动脚本
└── voice_assets/          # 声音克隆参考音频（不随仓库上传）
```

## License

MIT
