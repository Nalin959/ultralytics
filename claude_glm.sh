#!/bin/bash

export PATH="$HOME/.local/bin:$HOME/.local/node/bin:$PATH"

systemctl --user start litellm >/dev/null 2>&1

for i in {1..30}; do
    if curl -sf http://127.0.0.1:4000/v1/models >/dev/null; then
        break
    fi
    sleep 1
done

export ANTHROPIC_BASE_URL="http://127.0.0.1:4000"
export ANTHROPIC_API_KEY="dummy-key"
export ANTHROPIC_MODEL="nvidia_nim/z-ai/glm-5.2"

exec /home/nalin/.local/node/lib/node_modules/@anthropic-ai/claude-code/node_modules/@anthropic-ai/claude-code-linux-x64/claude "$@"
