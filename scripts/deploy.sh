#!/usr/bin/env bash
#
# Part 1 — Model Serving & Deployment.
#
# Initializes the local inference server (Ollama) and pulls the models the
# Agentic Edge Stack needs:
#   - llama3.2:3b       the chat / agent "brain" (tool-calling capable)
#   - nomic-embed-text  the embedding model used by the in-memory RAG (Part 2)
#
# Designed for Ubuntu on WSL2. Idempotent — every step checks current state
# before acting, so it is safe to run repeatedly. After this, run:
#   python scripts/verify.py
#
# Usage:
#   ./scripts/deploy.sh
#
set -euo pipefail

# --- Resolve repo root so the script works from any cwd ----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- Load config from .env (if present); else use defaults -------------------
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
LLM_MODEL="${MODEL_NAME:-llama3.2:3b}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
export OLLAMA_HOST   # the ollama CLI honors this for both `serve` and `pull`

log()  { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[deploy]\033[0m %s\n' "$*"; }

log "Ollama host : $OLLAMA_HOST"
log "Chat model  : $LLM_MODEL"
log "Embed model : $EMBED_MODEL"

# --- 1. Install Ollama if it isn't already present ---------------------------
if command -v ollama >/dev/null 2>&1; then
  log "Ollama already installed: $(ollama --version 2>/dev/null | head -n1)"
else
  log "Installing Ollama..."
  # > Windows-native: install from https://ollama.com/download instead.
  curl -fsSL https://ollama.com/install.sh | sh
fi

# --- 2. Make sure the server is running --------------------------------------
# On systemd-enabled WSL2 the installer registers an 'ollama' service; fall back
# to a background 'ollama serve' otherwise.
is_up() { curl -fsS "$OLLAMA_HOST/api/version" >/dev/null 2>&1; }

start_ollama() {
  if command -v systemctl >/dev/null 2>&1 \
     && systemctl list-unit-files 2>/dev/null | grep -q '^ollama.service'; then
    log "Starting Ollama via systemd..."
    sudo systemctl enable ollama >/dev/null 2>&1 || true
    sudo systemctl start ollama
  else
    warn "systemd service not found; starting 'ollama serve' in the background (/tmp/ollama.log)."
    nohup ollama serve >/tmp/ollama.log 2>&1 &
  fi
}

if is_up; then
  log "Ollama server already responding at $OLLAMA_HOST"
else
  start_ollama
  log "Waiting for Ollama to come up..."
  for _ in $(seq 1 30); do
    if is_up; then break; fi
    sleep 1
  done
  is_up || { warn "Ollama did not become ready in 30s. Check 'systemctl status ollama' or /tmp/ollama.log"; exit 1; }
  log "Ollama is up."
fi

# --- 3. Pull the models (skip the download if already present) ---------------
pull_if_missing() {
  local model="$1"
  if ollama list 2>/dev/null | awk '{print $1}' | grep -qxE "${model}(:latest)?"; then
    log "Model already present: ${model}"
  else
    log "Pulling ${model} ..."
    ollama pull "${model}"
  fi
}

pull_if_missing "$LLM_MODEL"
pull_if_missing "$EMBED_MODEL"

log "Done. Installed models:"
ollama list
echo
echo "Next: python scripts/verify.py"
