#!/usr/bin/env bash
# =============================================================================
# Local launcher for the SRSNet paper repro (RTX 4090 box).
#
# Thin wrapper around paper_repro.py that does three things the Python
# script doesn't do on its own:
#   1. maps friendly Bash flags (--manifest, --collect, --smoke-check, ...)
#      to the right paper_repro.py sub-command
#   2. exports the env vars we need for deterministic CUDA runs
#   3. wraps the call in systemd-inhibit so a long batch doesn't get
#      killed by the box falling asleep
#
# Layout:
#   A. Bootstrap (cd to root, strict mode)
#   B. Defaults
#   C. Flag parser
#   D. Env vars
#   E. Execution
# =============================================================================

# --- A -- Bootstrap -----------------------------------------------------------
# Strict mode + cd to repo root so all relative paths resolve.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# --- B -- Defaults ------------------------------------------------------------
# Bare call = run lite-paper on GPU 0 with inhibit on.
# EXTRA[] collects pass-through flags.
SCOPE="lite-paper"
GPU="0"
COMMAND="run"
INHIBIT=1
EXTRA=()

# --- C -- Flag parser ---------------------------------------------------------
# Each arm either updates a var, switches sub-command, or appends to
# EXTRA[] for pass-through.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope)
      # Pick a different scope (full-paper / psrs-sweep / ...).
      SCOPE="$2"
      shift 2
      ;;
    --gpu)
      # GPU index, used for both CUDA_VISIBLE_DEVICES and --gpus.
      GPU="$2"
      shift 2
      ;;
    --dry-run)
      # Print the planned commands, don't run them.
      EXTRA+=("--dry-run")
      shift
      ;;
    --force)
      # Re-run every task even if it's marked completed.
      EXTRA+=("--force")
      shift
      ;;
    --keep-going)
      # Don't stop on a failed task; record it and move on.
      EXTRA+=("--keep-going")
      shift
      ;;
    --max-tasks)
      # Cap how many tasks to run (debug/smoke).
      EXTRA+=("--max-tasks" "$2")
      shift 2
      ;;
    --parallel)
      # How many tasks in parallel (heavy rows still run alone).
      EXTRA+=("--parallel" "$2")
      shift 2
      ;;
    --manifest)
      # Just write the plan, don't run anything.
      COMMAND="manifest"
      shift
      ;;
    --collect)
      # Aggregate finished results into summary.csv + coverage.md.
      COMMAND="collect"
      shift
      ;;
    --check-data)
      # Check the dataset CSVs are in place.
      COMMAND="check-data"
      shift
      ;;
    --check-stale-results)
      # Print legacy files outside repro/<scope>/ and exit non-zero.
      COMMAND="check-stale-results"
      shift
      ;;
    --smoke-check)
      # Compare best SRSNet (dataset, horizon) against paper Table 2.
      COMMAND="smoke-check"
      shift
      ;;
    --smoke-dataset|--smoke-horizon|--smoke-tolerance-mse|--smoke-tolerance-mae)
      # Smoke-check knobs, passed through.
      EXTRA+=("$1" "$2")
      shift 2
      ;;
    --datasets|--models)
      # Optional task filters (comma-separated), passed through.
      EXTRA+=("$1" "$2")
      shift 2
      ;;
    --no-inhibit)
      # Skip systemd-inhibit (e.g. macOS).
      INHIBIT=0
      shift
      ;;
    --hours)
      # Accepted for compat with 24h-style scripts. We ignore the value --
      # the runner is resumable, external timers can stop us safely.
      shift 2
      ;;
    *)
      # Unknown flag, bail out loudly.
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

# --- D -- Env vars ------------------------------------------------------------
# CUDA_VISIBLE_DEVICES: hide every other GPU from the process.
# PYTHONUNBUFFERED: flush logs in real time so `tail -f` works.
# CUBLAS_WORKSPACE_CONFIG: required for torch deterministic mode on
#   cuBLAS >= 10.2 (NVIDIA-recommended value).
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# --- E -- Execution -----------------------------------------------------------
# Build argv and exec into it (no Bash sub-shell stays alive). If
# systemd-inhibit is on the PATH and INHIBIT=1, wrap the call to keep
# the box awake. exec means signals (SIGTERM/SIGINT) hit Python directly.
PY=(python scripts/repro/paper_repro.py "$COMMAND" --scope "$SCOPE" --gpu "$GPU" "${EXTRA[@]}")

if [[ "$INHIBIT" == "1" ]] && command -v systemd-inhibit >/dev/null 2>&1; then
  exec systemd-inhibit --what=sleep:idle --why="SRSNet paper reproduction" "${PY[@]}"
fi

exec "${PY[@]}"
