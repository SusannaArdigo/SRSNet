#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SCOPE="lite-paper"
GPU="0"
COMMAND="run"
INHIBIT=1
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope)
      SCOPE="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --dry-run)
      EXTRA+=("--dry-run")
      shift
      ;;
    --force)
      EXTRA+=("--force")
      shift
      ;;
    --keep-going)
      EXTRA+=("--keep-going")
      shift
      ;;
    --max-tasks)
      EXTRA+=("--max-tasks" "$2")
      shift 2
      ;;
    --manifest)
      COMMAND="manifest"
      shift
      ;;
    --collect)
      COMMAND="collect"
      shift
      ;;
    --check-data)
      COMMAND="check-data"
      shift
      ;;
    --check-stale-results)
      COMMAND="check-stale-results"
      shift
      ;;
    --smoke-check)
      COMMAND="smoke-check"
      shift
      ;;
    --smoke-dataset|--smoke-horizon|--smoke-tolerance-mse|--smoke-tolerance-mae)
      EXTRA+=("$1" "$2")
      shift 2
      ;;
    --no-inhibit)
      INHIBIT=0
      shift
      ;;
    --hours)
      # Accepted for compatibility with 24h-style invocations. The runner is
      # resumable; external timeout/systemd timers can stop it safely.
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

PY=(python scripts/repro/paper_repro.py "$COMMAND" --scope "$SCOPE" --gpu "$GPU" "${EXTRA[@]}")

if [[ "$INHIBIT" == "1" ]] && command -v systemd-inhibit >/dev/null 2>&1; then
  exec systemd-inhibit --what=sleep:idle --why="SRSNet paper reproduction" "${PY[@]}"
fi

exec "${PY[@]}"
