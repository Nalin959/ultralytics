#!/bin/bash
set -e

echo "Setting up everything locally (no sudo required)..."

# 1. Install Node.js locally if not present
NODE_DIR="$HOME/.local/node"
if [ ! -d "$NODE_DIR" ]; then
    echo "Downloading Node.js..."
    mkdir -p "$NODE_DIR"
    wget -qO- https://nodejs.org/dist/v20.15.0/node-v20.15.0-linux-x64.tar.xz | tar -xJ -C "$NODE_DIR" --strip-components=1
fi

export PATH="$NODE_DIR/bin:$PATH"

# 2. Install Claude Code CLI
if ! command -v claude &> /dev/null; then
    echo "Installing Anthropic Claude Code CLI..."
    npm install -g @anthropic-ai/claude-code
fi

# Kill any stuck LiteLLM instances
pkill -f "litellm" || true

# 3. Setup LiteLLM Proxy if not running
if ! pgrep -f "litellm --model nvidia_nim/z-ai/glm-5.2" > /dev/null; then
    export NVIDIA_NIM_API_KEY="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"
    export OPENAI_API_KEY="$NVIDIA_NIM_API_KEY"
    echo "Starting LiteLLM Proxy in the background..."
    nohup litellm --model nvidia_nim/z-ai/glm-5.2 --api_base https://integrate.api.nvidia.com/v1 > litellm.log 2>&1 &
    disown
    sleep 3
fi

# 4. Configure Claude to use the Proxy
export ANTHROPIC_BASE_URL="http://localhost:4000"
export ANTHROPIC_API_KEY="dummy-key"
export ANTHROPIC_MODEL="nvidia_nim/z-ai/glm-5.2"

echo ""
echo "=================================================="
echo "✅ Success! Launching Claude Code with GLM 5.2!"
echo "=================================================="
claude
