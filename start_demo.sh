#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Starting minillm server..."
./build/minillm-serve \
  --model models/SmolLM2-360M-Instruct \
  --model models/SmolLM2-1.7B-Instruct \
  --quant int8 2>&1 &
SERVER_PID=$!

echo "Waiting for models to load (~20s)..."
until curl -s http://localhost:8080/info > /dev/null 2>&1; do sleep 2; done
echo "Server ready."

echo "Starting ngrok tunnel..."
ngrok http --domain="${NGROK_DOMAIN}" 8080

# cleanup on exit
kill $SERVER_PID 2>/dev/null
