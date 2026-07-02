"""
小老曾 · 语音版（说话提问 → 老曾语音回答）· 现代简约聊天界面

链路：麦克风录音 → 本地转 wav → Qwen3-ASR 语音识别 → 转写确认卡（可编辑/可改发文字或原声）
      → 复用 app_lite 的 RAG 大脑（真流式输出）→（发原声时）Qwen 声音复刻异步配音

依赖：
  pip install dashscope gradio chromadb sentence-transformers openai

启动（用 run_voice.sh，里面配好了所有环境变量）：
  export DASHSCOPE_API_KEY='sk-标准百炼Key'        # 语音识别+合成用
  export LAOZENG_VOICE_ID='qwen-tts-vc-laozeng-xxx'  # clone_voice.py 生成的音色ID
  export OPENAI_API_KEY='sk-ws-...'                 # 微调模型用（专属maas Key）
  export OPENAI_BASE_URL='https://ws-...maas.aliyuncs.com/compatible-mode/v1'
  export OPENAI_MODEL='qwen3-8b-d14dd77aa68f'
  python app_voice.py
"""
import os
import re
import sys
import tempfile
from pathlib import Path

# Gradio 是可选依赖：纯 FastAPI 后端（server.py）只需要 transcribe/synthesize/strip_markdown
# 等不依赖 gradio 的函数；只有旧版 app_voice.py 的 Gradio UI 入口才需要它。
try:
    import gradio as gr
except ImportError:
    gr = None
import dashscope

# 复用文字版的 RAG 大脑（同目录）
sys.path.insert(0, str(Path(__file__).parent))
import app_lite

# === 配置 ===
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
# voice_id：优先环境变量，其次读 clone_voice.py 生成的 voice_id.txt
LAOZENG_VOICE_ID = os.environ.get("LAOZENG_VOICE_ID", "").strip()
if not LAOZENG_VOICE_ID:
    _vf = Path(__file__).parent / "voice_id.txt"
    if _vf.exists():
        LAOZENG_VOICE_ID = _vf.read_text(encoding="utf-8").strip()

ASR_MODEL = "qwen3-asr-flash"               # 阿里语音识别（比paraformer准确率更高，对口语/方言更友好）
TTS_MODEL = "qwen3-tts-vc-2026-01-22"       # Qwen 声音复刻，必须和克隆时同一模型

# 头像（聊天气泡左侧）
_ASSETS_DIR = Path(__file__).parent / "assets"
AVATAR_LAOZENG = str(_ASSETS_DIR / "avatar_laozeng.png")
AVATAR_USER = str(_ASSETS_DIR / "avatar_user.png")

dashscope.api_key = DASHSCOPE_API_KEY
# 国内北京区端点（你的 Key 是 cn-beijing）
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

# 全局只建一次索引
_COLLECTION = None


def get_collection():
    global _COLLECTION
    if _COLLECTION is None:
        _COLLECTION = app_lite.load_and_index()
    return _COLLECTION


# ---------- ASR：录音文件 → 文字 ----------
def transcribe(audio_path):
    """用 Qwen3-ASR-Flash 把录音转成文字（比旧的 paraformer-realtime-v2 准确率更高，
    对口语化表达、语速快、轻微方言口音更友好，且原生支持本地wav文件路径）"""
    if not audio_path:
        return ""
    try:
        abs_path = os.path.abspath(audio_path)
        messages = [
            {"role": "user", "content": [{"audio": f"file://{abs_path}"}]}
        ]
        response = dashscope.MultiModalConversation.call(
            model=ASR_MODEL,
            messages=messages,
            result_format="message",
            asr_options={"language": "zh", "enable_itn": True},
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            text = "".join(
                c.get("text", "") for c in content if isinstance(c, dict) and "text" in c
            )
        else:
            print("ASR 调用失败:", response.status_code, getattr(response, "message", ""))
            text = ""
    except Exception as e:
        print("ASR 异常:", repr(e)[:300])
        text = ""
    return text.strip()


# ---------- TTS：文字 → 老曾音色语音文件 ----------
def synthesize(text):
    """用 Qwen 声音复刻(qwen3-tts-vc) + 克隆音色合成语音，返回本地 wav 路径

    实测：qwen3-tts-vc 合成耗时基本随文本长度线性增长——
      100字 ≈ 5.6s，250字 ≈ 12s，500字 ≈ 20-23s。
    这是云端模型本身的真实合成速度（不是网络慢也不是代码卡），
    但老曾的文字回答常常有500+字，念完整版用户要等20秒以上，体感"非常慢"。
    所以语音只念精华摘要（前 200 字），完整内容始终在文字气泡里，
    这样能把等待时间从 20+ 秒压到 8-10 秒左右。
    """
    if not text or not LAOZENG_VOICE_ID:
        return None
    import requests
    speak = text[:200]
    try:
        resp = dashscope.MultiModalConversation.call(
            model=TTS_MODEL,
            text=speak,
            voice=LAOZENG_VOICE_ID,
            stream=False,
        )
        if resp.status_code != 200:
            print("TTS 失败:", resp.status_code, getattr(resp, "message", ""))
            return None
        audio_info = resp.output.get("audio") if isinstance(resp.output, dict) else resp.output["audio"]
        url = audio_info.get("url")
        if not url:
            return None
        data = requests.get(url, timeout=60).content
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out.write(data)
        out.close()
        return out.name
    except Exception as e:
        print("TTS 异常:", repr(e)[:200])
        return None


def strip_markdown(text):
    """去掉 markdown 标记，避免 TTS 念出 ** 和 ---"""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"^---.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"来源[:：].*$", "", text, flags=re.DOTALL)  # 来源链接不念
    return text.strip()


# ---------- 主流程 ----------
def _strip_html(s):
    """把简单的 HTML 标签清掉，只留纯文本。
    用户气泡为了展示效果会包一层 <div class='wx-...'> 标记（比如"语音转文字"标签），
    但这段 HTML 不该原样传给 LLM 当对话历史，否则模型会看到多余标签，影响效果。"""
    return re.sub(r"<[^>]+>", "", s).strip()


def _text_only_history(history):
    """给 RAG 传上下文时，过滤掉音频气泡（gr.Audio对象/dict），只保留纯文字消息，
    并清洗掉展示用的 HTML 标签，否则塞进 OpenAI messages 会因为 content 不是字符串
    直接报错/静默失败，或者把 HTML 标签污染进对话历史。"""
    clean = []
    for h in history or []:
        role = h.get("role")
        content = h.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            clean.append({"role": role, "content": _strip_html(content)})
    return clean


def _format_sources(lines_text):
    """把参考来源整理成一条克制的浅灰小字气泡（去掉 emoji 堆砌、去掉加粗标题）。
    输入：多行文本，每行形如 "📌 分类 | 标题 | [▶原视频](url)"
    输出：<div class='wx-src'>参考&nbsp;· 标题①&nbsp;· 标题② …</div>"""
    lines = [l.strip() for l in lines_text.splitlines() if l.strip().startswith("📌")]
    items = []
    for l in lines:
        l = l.lstrip("📌").strip()
        parts = [p.strip() for p in l.split("|")]
        title = parts[1] if len(parts) >= 2 else (parts[0] if parts else "")
        url = ""
        for p in parts:
            m = re.search(r"\((https?://[^)]+)\)", p)
            if m:
                url = m.group(1)
                break
        if url and title:
            items.append(f"<a href='{url}' target='_blank'>{title}</a>")
        elif title:
            items.append(title)
    if not items:
        return ""
    inner = " · ".join(items)
    return f"<div class='wx-src'>参考 · {inner}</div>"


def _answer_and_speak_stream(question, history, want_audio=True):
    """真流式分条回答：
      ① 立刻弹「老曾正在想…💭」占位（不用等 RAG 跑完）
      ② LLM token 逐块到达 → 正文持续追加更新（真正的流式吐字，不是假打字机）
      ③ 若有参考来源 → 单独追加一条浅灰折叠小气泡
      ④（仅 want_audio=True 时）追加「配音中」占位 → TTS 合成完替换成语音气泡（失败则移除）
    每一条都是独立气泡；want_audio=False（发文字通道）不等 TTS，回答更快落地。"""
    history.append({"role": "assistant", "content": "老曾正在想…💭"})
    yield history

    ctx_history = _text_only_history(history[:-1])
    body = ""
    source_lines = []
    got_any = False
    try:
        for partial, source_lines in app_lite.query_rag_stream(get_collection(), question, ctx_history):
            got_any = True
            body = partial
            history[-1] = {"role": "assistant", "content": body}
            yield history
    except Exception as e:
        print("RAG 流式异常:", repr(e)[:300])

    if not got_any:
        body = "刚才我这边处理出了点问题，你再问一次试试？"
        history[-1] = {"role": "assistant", "content": body}
        yield history

    # ③ 参考来源单独一条浅灰小气泡（有才加）
    src_html = _format_sources("\n".join(source_lines)) if source_lines else ""
    if src_html:
        history.append({"role": "assistant", "content": src_html})
        yield history

    if not want_audio:
        return

    # ④ 「配音中」占位 → 语音气泡（TTS 只念正文，不念来源）
    history.append({"role": "assistant", "content": "🔊 老曾正在配音…"})
    yield history

    audio_out = synthesize(strip_markdown(body))
    if audio_out:
        history[-1] = {"role": "assistant", "content": gr.Audio(value=audio_out)}
    else:
        history.pop()  # 合成失败就去掉占位，只留文字
    yield history


def _reply_stream(question, history, want_audio=False):
    """追加用户文字气泡（立刻显示）+ 流式回答。
    底部常驻文本框打字提问默认不触发语音回复（want_audio=False），回答更快。"""
    history = history or []
    history.append({"role": "user", "content": question})
    yield history
    for h in _answer_and_speak_stream(question, history, want_audio=want_audio):
        yield h


def text_chat(message, history):
    """打字提问（底部常驻输入框）：流式输出，每一步都 yield 给界面"""
    history = history or []
    if not message or not message.strip():
        yield history, ""
        return
    for h in _reply_stream(message.strip(), history, want_audio=False):
        yield h, ""


# ---------- 录音三态：idle → recording(纯前端) → review(转写确认卡，真实后端驱动) ----------
def on_recording_stopped(audio_path):
    """录音停止、本地转好 wav 后触发（mic_audio.change）。
    分两步 yield：① 立刻显示转写确认卡 + 可回听原声（不等 ASR）② ASR 结果落地后显示可编辑文字+两个发送按钮。
    注意：mic_audio 的复位由 .then() 统一执行（见 build_ui()），复位动作会令 value 变为 None
    并再次触发本函数。此时 audio_path=None，我们必须全部 skip（不更新任何输出），
    否则会把刚展示出来的确认卡又打回常态——这就是之前"闪一下就没了"的根因。"""
    if not audio_path:
        yield tuple(gr.skip() for _ in range(10))
        return

    # ① 立刻显示确认卡：可回听原声、可删除重录，同时展示"转写中"
    yield (
        gr.update(visible=False),                          # bar_idle
        gr.update(visible=True),                            # review_group
        "<div class='asr-status'>🎧 转写中…</div>",           # status_html
        gr.update(visible=True),                             # cancel_btn
        gr.update(visible=True, value=audio_path),           # preview_audio
        gr.update(visible=False, value=""),                  # asr_input
        gr.update(visible=False),                             # send_voice_btn
        gr.update(visible=False),                             # send_text_btn
        gr.update(visible=False),                             # fail_retry_btn
        audio_path,                                           # raw_audio_state
    )

    question = transcribe(audio_path)

    if question:
        yield (
            gr.skip(), gr.skip(),
            "",                                               # status_html 清空
            gr.skip(),
            gr.skip(),
            gr.update(visible=True, value=question),           # asr_input 展示可编辑文字
            gr.update(visible=True),                            # send_voice_btn
            gr.update(visible=True),                            # send_text_btn
            gr.update(visible=False),                           # fail_retry_btn
            gr.skip(),
        )
    else:
        yield (
            gr.skip(), gr.skip(),
            "<div class='asr-status asr-status--warn'>⚠️ 没听清，可直接发语音</div>",
            gr.skip(), gr.skip(),
            gr.update(visible=False, value=""),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),                            # fail_retry_btn
            gr.skip(),
        )


def on_cancel_review():
    """删除重录：关闭确认卡，弹回常态输入条（mic_audio 由 change 事件的 .then() 统一负责重置）"""
    return gr.update(visible=True), gr.update(visible=False)


def on_send_text(asr_text, history):
    """发文字：走文字通道——用转写(可编辑)后的文字直接进入 RAG，不等语音合成，最快落地。"""
    history = history or []
    text = (asr_text or "").strip()
    if not text:
        yield history, gr.update(visible=True), gr.update(visible=False)
        return
    history.append({"role": "user", "content": text})
    yield history, gr.update(visible=True), gr.update(visible=False)
    for h in _answer_and_speak_stream(text, history, want_audio=False):
        yield h, gr.skip(), gr.skip()


def on_send_voice(audio_path, asr_text, history):
    """发原声：保留用户原始语音气泡（可回听）+ 走完整语音链路（含老曾语音回复）。"""
    history = history or []
    if not audio_path:
        yield history, gr.update(visible=True), gr.update(visible=False)
        return

    history.append({"role": "user", "content": gr.Audio(value=audio_path)})
    yield history, gr.update(visible=True), gr.update(visible=False)

    question = (asr_text or "").strip()
    if not question:
        question = transcribe(audio_path)  # 兜底：转写失败态下用户仍选择"发语音"时，再试一次

    if not question:
        history.append({"role": "user", "content": "📝 [语音转文字失败，听不清]"})
        history.append({"role": "assistant", "content": "没听清，再说一遍试试？🎤"})
        yield history, gr.skip(), gr.skip()
        return

    history.append({"role": "user", "content": f"<div class='wx-asr'>{question}</div>"})
    yield history, gr.skip(), gr.skip()
    for h in _answer_and_speak_stream(question, history, want_audio=True):
        yield h, gr.skip(), gr.skip()


# ---------- 现代简约风格样式（紫蓝色系，圆角卡片，柔和阴影）----------
WECHAT_CSS = """
/* ===== 全局变量 ===== */
:root, .dark {
    --body-background-fill:      #f4f5fb !important;
    --background-fill-primary:   #ffffff !important;
    --background-fill-secondary: #f7f7fb !important;
    --block-background-fill:     transparent !important;
    --block-border-color:        transparent !important;
    --border-color-primary:      transparent !important;
    --color-accent-soft:         transparent !important;
    --body-text-color:           #1e1e2e !important;
    --body-text-color-subdued:   #8a8aa0 !important;
    --input-placeholder-color:   #aeaec2 !important;
}

.gradio-container,
.gradio-container .fillable {
    background: radial-gradient(circle at 20% 0%, #eef2ff 0%, #f5f6fb 45%, #f4f4f7 100%) !important;
    max-width:  480px !important;
    margin:     0 auto !important;
    padding:    0 !important;
    min-height: 100vh !important;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
#wx-chat, #wx-mic, #wx-text, #wx-inputbar, #bar-idle, #bar-review {
    background:  transparent !important;
    border:      0 !important;
    border-style: none !important;
    box-shadow:  none !important;
    outline:     none !important;
}
#wx-chat .message-wrap,
#wx-chat .message-row,
#wx-chat .flex-wrap,
#wx-chat .bubble-wrap,
#wx-chat .wrapper {
    background:  transparent !important;
    border:      0 !important;
    box-shadow:  none !important;
}

footer, .show-api, .built-with { display: none !important; }

/* ===== 顶部标题栏 ===== */
#wx-header {
    background:    rgba(255,255,255,0.7);
    backdrop-filter: blur(8px);
    text-align:    center;
    padding:       16px 0 12px;
    border-bottom: 1px solid rgba(99,102,241,0.08);
}
#wx-header .wx-title { font-size:17px; font-weight:700; color:#1e1e2e; margin:0; }
#wx-header .wx-sub   { font-size:12px; color:#9a9ab0; margin:4px 0 0; }

/* ===== 聊天区 ===== */
#wx-chat {
    background:    transparent !important;
    height:        calc(100vh - 300px) !important;
    min-height:    320px !important;
    padding:       14px 10px !important;
    overflow-y:    auto !important;
}
#wx-chat .message-wrap,
#wx-chat .wrap, #wx-chat .wrapper,
#wx-chat .bubble-wrap, #wx-chat .messages {
    background: transparent !important;
}
#wx-chat .message-buttons,
#wx-chat .icon-button-wrapper,
#wx-chat .copy-button,
#wx-chat button.icon-button {
    display: none !important;
}

/* 气泡通用：圆角卡片 + 柔和阴影 */
#wx-chat .message,
#wx-chat .bot.message,
#wx-chat .user.message {
    max-width:     78% !important;
    width:         fit-content !important;
    padding:       10px 14px !important;
    border:        0 !important;
    border-radius: 18px !important;
    font-size:     14.5px !important;
    line-height:   1.6 !important;
    box-shadow:    none !important;
    color:         #1e1e2e !important;
}

/* 用户气泡：靛蓝渐变/靠右 */
#wx-chat .user.message,
#wx-chat .user-row .message,
#wx-chat [data-testid="user"] .message {
    background:    linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color:         #ffffff !important;
    margin-left:   auto !important;
    margin-right:  0 !important;
    border-radius: 18px 4px 18px 18px !important;
    box-shadow:    0 6px 16px -6px rgba(79,70,229,0.45) !important;
}

/* 老曾气泡：白底/靠左 */
#wx-chat .bot.message,
#wx-chat .bot-row .message,
#wx-chat [data-testid="bot"] .message {
    background:    #ffffff !important;
    box-shadow:    0 1px 2px rgba(15,23,42,0.06), 0 8px 20px -12px rgba(15,23,42,0.10) !important;
    margin-right:  auto !important;
    margin-left:   0 !important;
    border-radius: 4px 18px 18px 18px !important;
}

/* ===== 头像 ===== */
#wx-chat .avatar-container {
    width:         36px !important;
    height:        36px !important;
    min-width:     36px !important;
    flex-shrink:   0 !important;
    display:       block !important;
    visibility:    visible !important;
    opacity:       1 !important;
    border-radius: 50% !important;
    overflow:      hidden !important;
    margin:        0 8px !important;
    box-shadow:    0 2px 6px rgba(0,0,0,0.12) !important;
}
#wx-chat .avatar-container img,
#wx-chat img.avatar-image {
    width:         36px !important;
    height:        36px !important;
    min-width:     36px !important;
    border-radius: 50% !important;
    object-fit:    cover !important;
    display:       block !important;
    visibility:    visible !important;
    opacity:       1 !important;
}
#wx-chat .message-row,
#wx-chat .bot-row,
#wx-chat .user-row { align-items: flex-start !important; }

/* ===== 底部输入区 ===== */
#wx-inputbar {
    background:  transparent !important;
    padding:     10px 12px 14px !important;
    gap:         8px !important;
}

/* 常态输入条 */
#bar-idle { display: flex !important; align-items: center !important; gap: 8px !important; }
#wx-mic {
    width:         46px !important;
    min-width:     46px !important;
    height:        46px !important;
    background:    #eef0ff !important;
    color:         #6366f1 !important;
    border:        none !important;
    border-radius: 50% !important;
    font-size:     18px !important;
    cursor:        pointer !important;
    display:       flex !important;
    align-items:   center !important;
    justify-content: center !important;
    flex-shrink:   0 !important;
    transition:    background .15s;
}
#wx-mic:hover { background: #e0e3ff !important; }
#wx-text textarea {
    background:    #ffffff !important;
    border:        1px solid #e6e7f5 !important;
    border-radius: 22px !important;
    padding:       11px 16px !important;
    font-size:     14.5px !important;
    color:         #1e1e2e !important;
    resize:        none !important;
    box-shadow:    0 1px 2px rgba(15,23,42,0.03) !important;
}
#wx-send {
    background:    linear-gradient(135deg, #6366f1, #4f46e5) !important;
    border:        0 !important;
    border-radius: 22px !important;
    color:         #ffffff !important;
    min-width:     68px !important;
    font-size:     14px !important;
    font-weight:   600 !important;
    height:        44px !important;
    box-shadow:    0 6px 16px -6px rgba(79,70,229,0.5) !important;
}
#wx-send:hover { opacity: .92 !important; }

/* ===== 转写确认卡 review-card：紫调圆角卡片 ===== */
#bar-review {
    display:       flex !important;
    flex-direction: column !important;
    gap:           10px !important;
    background:    linear-gradient(135deg, #eef0ff, #f2f0ff) !important;
    border-radius: 20px !important;
    padding:       14px 14px 12px !important;
    box-shadow:    0 4px 16px -8px rgba(99,102,241,0.25) !important;
}
#bar-review .asr-status {
    font-size: 12.5px;
    color: #6366f1;
    padding-left: 2px;
}
#bar-review .asr-status--warn { color: #d97706; }

/* 播放器精简：只留一个紧凑播放条 */
#review-audio {
    background: #ffffff !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06) !important;
}
#review-audio .controls { padding: 4px 8px !important; }
#review-audio audio { height: 34px !important; }

#asr-input textarea {
    background:    #ffffff !important;
    border:        1px solid #dcdcf5 !important;
    border-radius: 16px !important;
    padding:       10px 14px !important;
    font-size:     14px !important;
    color:         #1e1e2e !important;
    resize:        none !important;
}

#review-actions { display: flex !important; align-items: center !important; gap: 8px !important; justify-content: flex-end !important; }
#review-cancel {
    width: 40px !important; min-width: 40px !important; height: 40px !important;
    border-radius: 50% !important; background: #ffffff !important; color: #9a9ab0 !important;
    border: none !important; font-size: 16px !important; box-shadow: 0 1px 3px rgba(15,23,42,0.08) !important;
    flex-shrink: 0 !important;
}
#review-cancel:hover { color: #ef4444 !important; }
#btn-send-voice {
    background: #ffffff !important; color: #6366f1 !important; border: none !important;
    border-radius: 20px !important; font-size: 13px !important; font-weight: 600 !important;
    height: 40px !important; box-shadow: 0 1px 3px rgba(15,23,42,0.08) !important;
}
#btn-send-text {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important; color: #ffffff !important;
    border: none !important; border-radius: 20px !important; font-size: 13px !important;
    font-weight: 600 !important; height: 40px !important;
    box-shadow: 0 6px 16px -6px rgba(79,70,229,0.5) !important;
}
#btn-send-fail {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important; color: #ffffff !important;
    border: none !important; border-radius: 20px !important; font-size: 13px !important;
    font-weight: 600 !important; height: 40px !important; width: 100% !important;
}

/* ===== 参考来源：克制的浅灰小字气泡 ===== */
#wx-chat .bot.message .wx-src {
    font-size:   11.5px !important;
    line-height: 1.5 !important;
    color:       #9a9ab0 !important;
}
#wx-chat .bot.message .wx-src a {
    color:           #6366f1 !important;
    text-decoration: none !important;
}
#wx-chat .bot.message .wx-src a:hover { text-decoration: underline !important; }
#wx-chat .bot.message:has(.wx-src) {
    background: #f5f5fa !important;
    box-shadow: none !important;
    padding:    6px 12px !important;
}

#wx-chat .message-row { margin-bottom: 4px !important; }
#wx-chat .message { margin-top: 2px !important; }

#wx-chat .message audio,
#wx-chat .message .audio-container {
    max-width: 220px !important;
}

/* 用户「转文字」小字条（发原声通道下的识别结果展示） */
#wx-chat .user.message:has(.wx-asr) {
    background: rgba(99,102,241,0.14) !important;
    color: #4338ca !important;
    box-shadow: none !important;
    padding:    6px 12px !important;
}
#wx-chat .user.message .wx-asr {
    font-size:   12.5px !important;
    line-height: 1.5 !important;
}
"""


# 关键补丁：用 MediaRecorder API 自建录音 UI（完全绕开 gr.Audio 内部样式）
# - #wx-mic 是一个普通 gr.Button（视觉好看，SVG图标替代emoji）
# - JS 拦截按钮 click：第一次点开始录音（按钮变"停止"），第二次点结束
# - 录音完成后把 webm 用 AudioContext 解码为 PCM，重采样到16kHz，再编码为 wav
# - 喂给隐藏的 gr.Audio(sources=["upload"]) 组件 → 触发 change 事件链 → 后端 on_recording_stopped
# 实际 JS 代码维护在 voice_record_head.html，通过 demo.launch(head_paths=...) 注入 <head>，
# 这是唯一能让 <script> 真正执行的方式（gr.HTML 注入的 <script> 不会执行）。


def build_ui():
    with gr.Blocks(title="小老曾", fill_height=True, fill_width=True) as demo:
        gr.HTML(
            """
            <div id="wx-header">
                <p class="wx-title">产品老曾</p>
                <p class="wx-sub">用老曾自己的声音回答你 · 说话或打字都行</p>
            </div>
            """
        )

        chatbot = gr.Chatbot(
            elem_id="wx-chat",
            show_label=False,
            avatar_images=(AVATAR_USER, AVATAR_LAOZENG),
            container=False,
            value=[{"role": "assistant", "content": "你好，我是产品老曾。有啥职场上的问题，直接问我。👇 点麦克风说话，或者直接打字。"}],
        )

        with gr.Column(elem_id="wx-inputbar"):
            # ===== 常态输入条：麦克风 + 文本框 + 发送 =====
            with gr.Row(elem_id="bar-idle", visible=True) as bar_idle:
                mic_btn = gr.Button("", elem_id="wx-mic", variant="secondary")
                txt = gr.Textbox(
                    placeholder="也可以打字问老曾…",
                    show_label=False,
                    scale=5,
                    elem_id="wx-text",
                    container=False,
                )
                send = gr.Button("发送", scale=1, elem_id="wx-send")

            # 接收录音文件的 Audio 组件（隐藏其自身UI，仅作数据管道）：
            # 关键坑——不能用 visible=False！Gradio 的 Svelte 组件在 visible=False 时会被
            # {#if} 整个从 DOM 卸载，JS 的 querySelector 会找不到 input[type=file]。
            # 改用 visible=True + CSS 挪走，组件才会真实挂载在 DOM 上。
            mic_audio = gr.Audio(
                sources=["upload"],
                type="filepath",
                show_label=False,
                elem_id="wx-mic-audio",
                visible=True,
                container=False,
            )

            # ===== 转写确认卡：回听/删除重录 + 可编辑文字 + 发原声/发文字 =====
            with gr.Column(elem_id="bar-review", visible=False) as review_group:
                status_html = gr.HTML(value="", elem_id="asr-status-wrap")
                with gr.Row():
                    preview_audio = gr.Audio(
                        show_label=False, container=False, visible=False,
                        elem_id="review-audio", interactive=False,
                    )
                    cancel_btn = gr.Button("✕", elem_id="review-cancel", visible=False)
                asr_input = gr.Textbox(
                    show_label=False, container=False, visible=False,
                    elem_id="asr-input", placeholder="识别结果，可编辑纠错…",
                )
                with gr.Row(elem_id="review-actions"):
                    send_voice_btn = gr.Button("发原声", elem_id="btn-send-voice", visible=False)
                    send_text_btn = gr.Button("发文字", elem_id="btn-send-text", visible=False)
                fail_retry_btn = gr.Button("没听清，直接发语音", elem_id="btn-send-fail", visible=False)

        raw_audio_state = gr.State(value=None)

        _review_outputs = [
            bar_idle, review_group, status_html, cancel_btn, preview_audio,
            asr_input, send_voice_btn, send_text_btn, fail_retry_btn,
            raw_audio_state,
        ]
        # 关键：mic_audio 的复位放在 .then() 里，等 on_recording_stopped 这个生成器
        # 完全跑完之后才执行一次，避免在生成器内部改 mic_audio.value 导致
        # change 事件被递归再次触发，把刚显示出来的确认卡瞬间打回常态。
        mic_audio.change(on_recording_stopped, [mic_audio], _review_outputs).then(
            lambda: gr.update(value=None), None, mic_audio
        )

        cancel_btn.click(on_cancel_review, None, [bar_idle, review_group])

        send_text_btn.click(
            on_send_text, [asr_input, chatbot], [chatbot, bar_idle, review_group]
        )
        send_voice_btn.click(
            on_send_voice, [raw_audio_state, asr_input, chatbot], [chatbot, bar_idle, review_group]
        )
        fail_retry_btn.click(
            on_send_voice, [raw_audio_state, asr_input, chatbot], [chatbot, bar_idle, review_group]
        )

        send.click(text_chat, [txt, chatbot], [chatbot, txt])
        txt.submit(text_chat, [txt, chatbot], [chatbot, txt])

    return demo


if __name__ == "__main__":
    if not DASHSCOPE_API_KEY:
        print("⚠️  未设置 DASHSCOPE_API_KEY，语音识别和合成将不可用")
    if not LAOZENG_VOICE_ID:
        print("⚠️  未设置 LAOZENG_VOICE_ID，请先跑 clone_voice.py 生成音色")
    print("启动中，首次会构建向量索引…")
    get_collection()
    demo = build_ui()
    # head_paths 让 Gradio 把 voice_record_head.html 作为 <script> 引入 <head>，
    # 这是唯一可靠让 JS 执行的方式（gr.HTML 注入的 <script> 不执行）
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False,
                theme=gr.themes.Soft(), css=WECHAT_CSS,
                head_paths=[str(Path(__file__).parent / "voice_record_head.html")])
