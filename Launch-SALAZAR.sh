#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
APP="$ROOT/frontend/release/linux-unpacked/frontend"

if [[ ! -x "$APP" ]]; then
    echo "SALAZAR desktop executable is missing or not executable:"
    echo "$APP"
    echo
    echo "Check that the USB is mounted with execution enabled."
    read -r -p "Press Enter to close..."
    exit 1
fi

if [[ ! -f "$ROOT/models/llm/smollm2-1.7b/smollm2-1.7b-instruct-q4_k_m.gguf" ]]; then
    echo "SALAZAR model files could not be found in:"
    echo "$ROOT/models"
    read -r -p "Press Enter to close..."
    exit 1
fi

export SALAZAR_ROOT="$ROOT"
cd "$ROOT"

exec "$APP" "$@"
