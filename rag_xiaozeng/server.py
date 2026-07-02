"""
小老曾 · 纯前端 + FastAPI 后端版

架构说明（为什么换掉 Gradio）：
  之前 app_voice.py 用 Gradio，但录音是自建 MediaRecorder+JS 硬桥接到 Gradio 隐藏组件，
  两套事件系统（Gradio 响应式 + 手写 DOM JS）互相不可见地勾连，导致状态经常被意外覆盖
  （比如"确认卡闪一下就没了"），而且样式只能在 Gradio 生成的 DOM 上打 CSS 补丁，改不了结构。

  这一版彻底拆开：
    - 前端（static/index.html）：纯 HTML/CSS/JS，自己管理所有 UI 状态（idle/recording/review），
      样式 100% 还原 ideal_ui_demo.html，不再有任何隐藏组件桥接。
    - 后端（本文件）：只提供 3 个无状态接口，职责单一，出问题只可能是"接口对不对"或
      "前端这行 JS 逻辑对不对"，不会再有黑盒式的框架内部时序问题。

接口：
  GET  /                纯前端页面
  POST /api/asr          上传录音 wav -> 返回识别文字 {"text": "..."}
  POST /api/chat          文字/语音问答 -> SSE 流式返回（打字机 + 参考来源 + 可选语音合成）

依赖：pip install fastapi uvicorn python-multipart
启动：./run_server.sh   （会设置好 DASHSCOPE_API_KEY / OPENAI_API_KEY 等环境变量）
"""
import os
import sys
import json
import hashlib
import base64
import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent))
import app_lite
import app_voice  # 复用其中的 transcribe / synthesize / strip_markdown / _format_sources

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="小老曾")

# 上传录音大小上限（防止恶意超大文件占满内存/磁盘）
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB 足够覆盖几分钟的 16kHz wav 语音

# 访问密码：从环境变量读取，不知道密码的人无法调用 /api/* 接口
# 留空则不启用密码（本地开发时方便）
APP_PASSWORD = os.environ.get("APP_PASSWORD", "").strip()

# TTS 语音缓存目录：同一段文本只合成一次，后续直接返回缓存的 wav
# 面试开场白等固定文本收益最大（首次 ~6s → 后续 <0.1s）
TTS_CACHE_DIR = Path(__file__).parent / "tts_cache"
TTS_CACHE_DIR.mkdir(exist_ok=True)
# 缓存上限（按文件数），超出后清理最旧的
TTS_CACHE_MAX = 200

_COLLECTION = None


def get_collection():
    global _COLLECTION
    if _COLLECTION is None:
        _COLLECTION = app_lite.load_and_index()
    return _COLLECTION


def _tts_cache_key(text):
    """对文本做 hash 作为缓存文件名，避免特殊字符问题"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def synthesize_cached(text):
    """带磁盘缓存的语音合成：
    - 同一段文本只调一次 TTS API，结果存为 wav 文件
    - 后续命中缓存直接返回文件路径（<0.1s）
    - 返回 (wav_path, from_cache: bool)"""
    clean = app_voice.strip_markdown(text)
    if not clean or not app_voice.LAOZENG_VOICE_ID:
        return None, False

    key = _tts_cache_key(clean)
    cached_path = TTS_CACHE_DIR / f"{key}.wav"

    # 命中缓存：直接返回
    if cached_path.exists() and cached_path.stat().st_size > 0:
        return str(cached_path), True

    # 未命中：调 TTS 合成，成功后复制到缓存目录
    tmp_path = app_voice.synthesize(text)
    if not tmp_path:
        return None, False

    try:
        shutil.copy2(tmp_path, cached_path)
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    except Exception as e:
        print("TTS 缓存写入失败:", repr(e)[:200])
        # 缓存失败不影响功能，返回临时文件
        return tmp_path, False

    # 清理最旧缓存（LRU 简化版：按修改时间排序，删最老的）
    _evict_old_cache()

    return str(cached_path), True


def _evict_old_cache():
    """缓存文件数超过上限时，删除最旧的"""
    files = list(TTS_CACHE_DIR.glob("*.wav"))
    if len(files) <= TTS_CACHE_MAX:
        return
    files.sort(key=lambda f: f.stat().st_mtime)
    for f in files[:len(files) - TTS_CACHE_MAX]:
        try:
            f.unlink()
        except Exception:
            pass


def _sse(obj: dict) -> str:
    """组装一条 SSE 消息（data 字段是 JSON 字符串）"""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


def _check_password(request: Request) -> bool:
    """校验请求头里的密码是否正确，没配密码则直接放行"""
    if not APP_PASSWORD:
        return True
    # 优先从 Authorization header 读，兼容 query param（前端 SSE 用 fetch 无法自定义 header）
    auth = request.headers.get("Authorization", "")
    pwd_query = request.query_params.get("pwd", "")
    if auth == f"Bearer {APP_PASSWORD}" or pwd_query == APP_PASSWORD:
        return True
    return False


@app.post("/api/auth")
async def api_auth(request: Request):
    """验证密码是否正确，前端用这个接口做登录校验"""
    if not APP_PASSWORD:
        return {"ok": True, "no_password": True}
    try:
        body = await request.json()
    except Exception:
        body = {}
    pwd = body.get("password", "")
    if pwd == APP_PASSWORD:
        return {"ok": True}
    return JSONResponse({"ok": False, "error": "密码错误"}, status_code=401)


@app.get("/api/tts/{audio_key}")
async def api_tts(audio_key: str):
    """返回缓存的 TTS 音频文件。
    synthesize_cached() 已经把 wav 存在 tts_cache/ 目录，
    前端通过 SSE 收到 audio_key 后直接用 <audio src="/api/tts/xxx"> 播放，
    不用把 800KB 的 base64 塞在 SSE 消息里（大 base64 会被 chunked encoding 切断导致 JSON 解析失败）。"""
    # 安全：只允许十六进制字符的文件名，防止路径穿越
    if not all(c in "0123456789abcdef" for c in audio_key) or len(audio_key) != 32:
        return JSONResponse({"error": "invalid key"}, status_code=400)
    wav_path = TTS_CACHE_DIR / f"{audio_key}.wav"
    if not wav_path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(wav_path), media_type="audio/wav")


@app.post("/api/asr")
async def api_asr(request: Request, file: UploadFile = File(...)):
    """接收前端本地转好的 wav 录音，转写为文字。
    用完即删临时文件，不在磁盘上长期保留用户语音。"""
    if not _check_password(request):
        return JSONResponse({"error": "密码错误"}, status_code=401)
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        return JSONResponse({"text": "", "error": "录音文件过大"}, status_code=400)
    if not content:
        return JSONResponse({"text": "", "error": "空文件"}, status_code=400)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        text = app_voice.transcribe(tmp_path)
    except Exception as e:
        print("ASR 接口异常:", repr(e)[:300])
        return JSONResponse({"text": "", "error": "识别失败"}, status_code=500)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    return {"text": text}


def _clean_history(history):
    """只保留合法的 {role, content} 字符串对，过滤掉前端可能传来的脏数据"""
    clean = []
    if not isinstance(history, list):
        return clean
    for h in history:
        if not isinstance(h, dict):
            continue
        role = h.get("role")
        content = h.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            clean.append({"role": role, "content": content})
    return clean


@app.post("/api/chat")
async def api_chat(request: Request):
    """问答接口：SSE 流式返回。
    请求体 JSON: {
        "message": "问题文字",
        "history": [{"role":...,"content":...}, ...],
        "want_audio": bool,
        "mode": "chat" | "interview"
    }
    响应事件（每行 data: {...}）：
      {"type":"answer","text": 累积文本}          —— 可能多次，流式追加
      {"type":"sources","html": "..."}            —— 参考来源小气泡（可能没有）
      {"type":"tts_start"}                        —— 开始语音合成（仅 want_audio=true）
      {"type":"tts_ready","audio_key":"<md5hash>"} —— 合成成功，前端用 /api/tts/<key> 播放
      {"type":"tts_failed"}                        —— 合成失败
      {"type":"error","message":"..."}             —— 出错提示
      {"type":"done"}                              —— 结束标记
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not _check_password(request):
        # SSE 返回错误事件而不是 JSON，前端能统一处理
        def _auth_error():
            yield _sse({"type": "error", "message": "密码错误"})
            yield _sse({"type": "done"})
        return StreamingResponse(_auth_error(), media_type="text/event-stream")
    message = (body.get("message") or "").strip()
    history = _clean_history(body.get("history"))
    want_audio = bool(body.get("want_audio"))
    mode = body.get("mode") or "chat"
    if mode not in ("chat", "interview"):
        mode = "chat"

    def event_stream():
        # 面试模式特例：开场白不需要走 LLM，直接返回固定话术
        if mode == "interview" and not history:
            opening = app_lite.INTERVIEW_OPENING
            yield _sse({"type": "answer", "text": opening})
            # 开场白也需要语音合成（面试模式老曾每次提问都要说出来）
            if want_audio:
                yield _sse({"type": "tts_start"})
                audio_path = None
                try:
                    audio_path, _cached = synthesize_cached(opening)
                    if audio_path:
                        # 返回文件 key，前端用 /api/tts/<key> 播放，不在 SSE 里塞 base64
                        audio_key = Path(audio_path).stem
                        yield _sse({"type": "tts_ready", "audio_key": audio_key})
                    else:
                        yield _sse({"type": "tts_failed"})
                except Exception as e:
                    print("开场白 TTS 异常:", repr(e)[:300])
                    yield _sse({"type": "tts_failed"})
                finally:
                    if audio_path and "/tmp/" in audio_path:
                        try:
                            os.unlink(audio_path)
                        except Exception:
                            pass
            yield _sse({"type": "done"})
            return

        if not message:
            yield _sse({"type": "done"})
            return

        try:
            collection = get_collection()
        except Exception as e:
            print("加载知识库异常:", repr(e)[:300])
            yield _sse({"type": "error", "message": "知识库加载失败"})
            yield _sse({"type": "done"})
            return

        body_text = ""
        source_lines = []
        got_any = False
        try:
            for partial, source_lines in app_lite.query_rag_stream(collection, message, history, mode=mode):
                got_any = True
                body_text = partial
                yield _sse({"type": "answer", "text": body_text})
        except Exception as e:
            print("RAG 流式异常:", repr(e)[:300])
            yield _sse({"type": "error", "message": "回答生成出错"})

        if not got_any:
            body_text = "刚才我这边处理出了点问题，你再问一次试试？"
            yield _sse({"type": "answer", "text": body_text})

        if source_lines:
            html = app_voice._format_sources("\n".join(source_lines))
            if html:
                yield _sse({"type": "sources", "html": html})

        if want_audio:
            yield _sse({"type": "tts_start"})
            audio_path = None
            try:
                audio_path, _cached = synthesize_cached(body_text)
                if audio_path:
                    audio_key = Path(audio_path).stem
                    yield _sse({"type": "tts_ready", "audio_key": audio_key})
                else:
                    yield _sse({"type": "tts_failed"})
            except Exception as e:
                print("TTS 接口异常:", repr(e)[:300])
                yield _sse({"type": "tts_failed"})
            finally:
                if audio_path and "/tmp/" in audio_path:
                    try:
                        os.unlink(audio_path)
                    except Exception:
                        pass

        yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# 静态资源（图标等，如果以后需要）挂在 /static，不影响上面的 /api/* 和 /
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  未设置 OPENAI_API_KEY，文字问答会报错")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("⚠️  未设置 DASHSCOPE_API_KEY，语音识别/合成将不可用")

    print("启动中，首次会构建向量索引…")
    get_collection()
    port = int(os.environ.get("PORT", "7863"))
    print(f"\n🚀 访问 http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
