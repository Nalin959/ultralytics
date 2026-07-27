#!/bin/bash
export PATH="$HOME/.local/node/bin:$PATH"

# Kill existing LiteLLM proxies to apply new settings
pkill -f "litellm"

export NVIDIA_NIM_API_KEY="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"

# Start proxy with correct NVIDIA_NIM prefix in the background safely!
nohup litellm --detailed_debug --drop_params --model nvidia_nim/z-ai/glm-5.2 > litellm.log 2>&1 &
disown
sleep 3

export ANTHROPIC_BASE_URL="http://localhost:4000"
export ANTHROPIC_API_KEY="dummy-key"
export ANTHROPIC_MODEL="nvidia_nim/z-ai/glm-5.2"

claude
