#!/bin/bash
# Refuse to start without a login, then run ComfyUI (local only) behind Caddy basic auth.
set -euo pipefail
export COMFY_USER="${COMFY_USER:-render}"
if [[ -z "${COMFY_PASSWORD:-}" ]]; then
  echo "COMFY_PASSWORD is required; refusing to expose ComfyUI without a login" >&2
  exit 1
fi
COMFY_PASSWORD_HASH="$(caddy hash-password --plaintext "$COMFY_PASSWORD")"
export COMFY_PASSWORD_HASH
unset COMFY_PASSWORD
echo "idle self-terminate: ${IDLE_SECONDS:-1200}s, hard limit: ${MAX_SECONDS:-0}s (0 = off)"

python /opt/ComfyUI/main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch &
caddy run --config /opt/Caddyfile --adapter caddyfile &
wait -n
exit $?
