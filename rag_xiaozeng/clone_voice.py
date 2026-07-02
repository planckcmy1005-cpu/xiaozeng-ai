"""
小老曾 · 声音克隆（百炼 Qwen 声音复刻 · base64 直传）

【合规说明】音频不落任何存储、不碰任何图床/OSS。
  本地参考音频 → 转 base64 → 直接塞进 HTTP 请求体发给百炼 → 返回 voice_id
和你平时调百炼模型是同一个 Key、同一套域名、同一种动作（就是发个 HTTP 请求）。

用法：
  export DASHSCOPE_API_KEY='sk-你的百炼Key'
  python clone_voice.py                    # 用默认 voice_assets/laozeng_ref_preview.mp3
  python clone_voice.py /path/to/ref.mp3   # 或指定本地音频（mp3/wav 均可）

跑一次即可，打印并保存 voice_id 到 voice_id.txt。
依赖：pip install dashscope requests
"""
import os
import sys
import base64
import pathlib

import requests

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
if not API_KEY:
    print("请先设置 DASHSCOPE_API_KEY 环境变量（百炼 Key，sk- 开头）")
    sys.exit(1)

# 国内北京区端点（你的 Key 是 cn-beijing）
ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
TARGET_MODEL = "qwen3-tts-vc-2026-01-22"   # 合成时必须用同一个模型

# 参考音频：默认用项目里提取好的老曾人声
DEFAULT_REF = pathlib.Path(__file__).parent.parent / "voice_assets" / "laozeng_ref_preview.mp3"
ref_path = pathlib.Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_REF

if not ref_path.exists():
    print(f"❌ 找不到参考音频：{ref_path}")
    sys.exit(1)

# mime 类型：mp3 用 audio/mpeg，wav 用 audio/wav
suffix = ref_path.suffix.lower()
mime = "audio/wav" if suffix == ".wav" else "audio/mpeg"

print(f"参考音频：{ref_path}（{ref_path.stat().st_size} 字节，{mime}）")
print("① 音频转 base64（不落任何存储，直接进请求体）…")
b64 = base64.b64encode(ref_path.read_bytes()).decode()
data_uri = f"data:{mime};base64,{b64}"

print(f"② 调用 Qwen 声音复刻（模型 {TARGET_MODEL}）…")
resp = requests.post(
    ENDPOINT,
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={
        "model": "qwen-voice-enrollment",
        "input": {
            "action": "create",
            "target_model": TARGET_MODEL,
            "preferred_name": "laozeng",
            "audio": {"data": data_uri},
        },
    },
    timeout=120,
)

if resp.status_code != 200:
    print(f"❌ 克隆失败：{resp.status_code}\n{resp.text}")
    sys.exit(1)

voice_id = resp.json()["output"]["voice"]

print("\n" + "=" * 50)
print(f"✅ 克隆成功！voice_id：\n\n    {voice_id}\n")
print("=" * 50)

out = pathlib.Path(__file__).parent / "voice_id.txt"
out.write_text(voice_id, encoding="utf-8")
print(f"已保存到 {out}")
print("下一步：./run_voice.sh 启动语音版（自动读取 voice_id.txt）")
