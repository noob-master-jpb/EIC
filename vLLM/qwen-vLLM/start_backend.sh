#!/bin/bash
source ../.venv/bin/activate
export $(cat .env | xargs)
# Start vLLM server serving Qwen3.5-0.8B (OpenAI compatible)
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-0.8B --port 8000 --gpu-memory-utilization 0.7 --max-model-len 4096 --disable-log-stats
