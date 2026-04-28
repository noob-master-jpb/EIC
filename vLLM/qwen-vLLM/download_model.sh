#!/bin/bash
source ../.venv/bin/activate

# Load environment variables (like HF_TOKEN)
if [ -f .env ]; then
  export $(cat .env | xargs)
fi

echo "Downloading Qwen/Qwen3.5-0.8B from Hugging Face..."
hf download Qwen/Qwen3.5-0.8B

echo "Download complete! You can now run ./start_backend.sh"
