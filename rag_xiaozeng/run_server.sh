#!/bin/bash
# 小老曾 · 纯前端 + FastAPI 后端版一键启动
# 用法：
#   1. 复制 .env.example 为 .env，填入你的 API Key
#   2. ./run_server.sh

cd "$(dirname "$0")/.."

# 从 .env 加载环境变量（不存在则提示
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo "⚠️  未找到 .env 文件，请先复制 .env.example 为 .env 并填入 API Key"
  exit 1
fi

PY=${PYTHON:-python3}
export PORT=${PORT:-7863}

echo "启动小老曾（纯前端+FastAPI版）… 浏览器打开 http://localhost:${PORT}"
cd rag_xiaozeng
$PY server.py
