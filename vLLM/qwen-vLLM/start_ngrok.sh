#!/bin/bash
if [ -f .env ]; then
  export $(cat .env | xargs)
fi

if [ -z "$NGROK_AUTHTOKEN" ]; then
  echo "Error: NGROK_AUTHTOKEN not found in .env"
  echo "Please add 'NGROK_AUTHTOKEN=your_token_here' to Arya-Files/qwen/.env"
  echo "Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken"
  exit 1
fi

ngrok config add-authtoken $NGROK_AUTHTOKEN
ngrok http 8000
