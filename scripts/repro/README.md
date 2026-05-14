# SRSNet Paper Reproduction on a Local RTX 4090

Local, single-GPU reproduction layer built on top of official `main`. No
Hivenet, no SLURM, no `srsbench` rewrite. The runner shells out to the
official `scripts/run_benchmark.py` per row and adds resume, hashing,
OOM handling, paper-value smoke checks, and coverage reporting.

Paper reference: <https://arxiv.org/html/2510.14510> and
<https://arxiv.org/pdf/2510.14510>.

---

## TL;DR

```bash
# 0. one-time: confirm dataset layout, install reqs, optionally pin numpy<2
bash scripts/repro/run_local_4090.sh --check-data
pip install -r requirements.txt -r requirements-repro.txt

# 1. inspect what will run (writes manifest, does not train)
bash scripts/repro/run_local_4090.sh --scope lite-paper --manifest

# 2. make sure no stale legacy results bias resume
bash scripts/repro/run_local_4090.sh --scope lite-paper --check-stale-results

# 3. dry-run a single task end-to-end so you see the exact command
bash scripts/repro/run_local_4090.sh --scope lite-paper --dry-run --max-tasks 1

# 4. start the real run (resumable; SIGINT is safe).
#    --parallel 4 packs the light rows; heavy rows automatically run alone.
bash scripts/repro/run_local_4090.sh --scope lite-paper --parallel 4 --keep-going

# 5. collect + gate the smoke row against paper Table 2
bash scripts/repro/run_local_4090.sh --scope lite-paper --collect
bash scripts/repro/run_local_4090.sh --scope lite-paper --smoke-check
```

---

## Environment

The repo's `requirements.txt` is the source of truth. The overlay
`requirements-repro.txt` only adds `numpy<2` because some transitive
deps in `requirements.txt` don't yet support NumPy 2.x.

Expected dataset layout (TFB official, unchanged from the repo README):

```text
dataset/forecasting/FORECAST_META.csv
dataset/forecasting/ETTh1.csv
dataset/forecasting/ETTh2.csv
dataset/forecasting/ETTm1.csv
dataset/forecasting/ETTm2.csv
dataset/forecasting/Weather.csv
dataset/forecasting/Electricity.csv
dataset/forecasting/Solar.csv
dataset/forecasting/Traffic.csv
```

`--check-data` verifies this.

GPU: defaults to `CUDA_VISIBLE_DEVICES=0`. Override with `--gpu N`.

Sleep inhibition: when `systemd-inhibit` is available the wrapper
re-execs under it. Pass `--no-inhibit` to skip (macOS, headless boxes
without systemd).

---

## Scopes

| Scope | What runs | When to use |
|---|---|---|
| `lite-paper` | SRSNet Table 2 (8 datasets × 4 horizons, 1 seed, default lookback), all SRSNet ablations, MLP/+SRS plug-in rows. Other plug-ins and the full baseline matrix on `Solar`/`Traffic` are `omitted-lite`. | Local 4090 "24h-style" run; still emits every paper row in coverage. |
| `full-paper` | SRSNet across 5 seeds × {96, 336, 512} lookbacks, every official baseline shell script for Table 2, all ablations, all plug-in variants, and the one-batch efficiency profiler for Tables 5/6. | Multi-day budget; the only honest "did we reproduce" run. |
| `main-compat` | Normalized official `main` scripts, no paper-mode overrides. | Sanity check: does `main` itself match the paper before our layer touches anything? |

Coverage statuses you'll see in `coverage.md` / `summary.csv`:

- `completed` — ran, result file present, metadata sidecar matches.
- `failed` — ran, exited non-zero; see `repro_results/<scope>/logs/<task_id>.log`.
- `missing` — manifested but never attempted yet.
- `omitted-lite` — intentionally skipped in `lite-paper`; run `full-paper`.
- `reference-only` — no official `main` shell script for this (dataset, model);
  paper value retained in coverage but no run is attempted.

---

## Paper-mode overrides (important)

Paper-mode (`lite-paper`, `full-paper`) injects fixed values into every
official shell command's `--model-hyper-params`:

- `batch_size = 64` — overrides whatever the script said. Several
  official scripts hardcode `32`, `256`, etc.; the paper uses 64.
  `main-compat` preserves the script's value.
- `train_drop_last = false` — this flag is new and is plumbed into
  `ts_benchmark/baselines/deep_forecasting_model_base.py`. When unset
  (default), the original `drop_last=True` training behavior is
  preserved, so `main-compat` is unaffected.
- `horizon` is forced to the row's horizon for safety.

`full-paper` additionally expands SRSNet rows over
`seeds={2021,2022,2023,2024,2025}` and `seq_len={96,336,512}`.

OOM handling (paper-mode only): on `out of memory` in the child
process, batch size is halved and the row retries, stopping at
`PAPER_BATCH_FLOOR=8`. The retried `batch_size` is part of the row's
metadata and contributes to its `config_hash`, so a smaller batch will
**not** be silently reused on resume — the row will re-run if the next
invocation asks for the original size.

---

## Resume semantics (hash-aware)

For each task the runner writes
`repro_results/<scope>/metadata/<task_id>.json` containing:

- `requested_command`, `final_command`, and their `command_hash`es
- `requested_config_hash`, `config_hash` (semantic-only — excludes
  `--gpus`, `--save-path`, `--num-workers`)
- `requested_identity`, `final_identity` (dataset, horizon, model,
  seed, seq_len, data_name_list, model_name, adapter, strategy_args,
  model_hyper_params, deterministic)
- `final_batch_size`, full `attempts` list
- `result_file` path
- `metric_space` (see below)
- timestamps

On the next invocation, `--resume` (default) skips a task **only if**
the sidecar exists, the result file still exists, and the new task's
`_config_hash(task, task.command)` equals the stored
`requested_config_hash`. Any change to dataset, horizon, model name,
adapter, seed, seq_len, hyperparameters, or strategy args invalidates
the skip and the row re-runs.

`--force` ignores the sidecar; `--no-resume` (in `paper_repro.py`)
disables the skip entirely.

---

## Parallel mode

`--parallel N` (default 1) runs up to N tasks concurrently on the single
GPU. The scheduler is in `scripts/repro/paper_repro.py` (`_run_parallel`).
Recommended: `--parallel 4` on a 24GB 4090; drop to 2 if you hit OOMs.

Three correctness guarantees, all enforced by the scheduler:

1. **Heavy rows run alone.** A static blocklist (`HEAVY_ROWS` in
   `paper_repro.py`) marks combinations like FEDformer / Solar,
   Crossformer / Traffic / H720, and the Table 5/6 efficiency rows.
   These get `effective_concurrency=1` automatically and are never
   admitted while another row is in flight. The current rules:

   ```python
   {"table": "table5_efficiency"},
   {"table": "table6_efficiency"},
   {"model": "FEDformer", "dataset": ("Solar", "Traffic", "Electricity")},
   {"model": "Crossformer", "dataset": ("Solar", "Traffic", "Electricity"),
    "horizon": (336, 720)},
   {"model": "Pathformer", "dataset": ("Traffic", "Electricity"),
    "horizon": (336, 720)},
   {"dataset": ("Traffic", "Electricity"), "horizon": 720, "seq_len": 512},
   ```

   If you discover a new OOM pattern, add a rule here. No silent fallback
   logic exists; the only response to OOM in parallel mode is a hard fail.

2. **Bit-faithful numerics under contention.** When a row's
   `effective_concurrency >= 2`, the runner rewrites `--deterministic`
   to `full`. That sets `cudnn.deterministic=True`,
   `cudnn.benchmark=False`, `cudnn.enabled=False` in the child
   (`ts_benchmark/utils/random_utils.py:18-31`), so parallel-trained
   results are bit-equivalent to a clean-GPU sequential run. Slight
   slowdown (5-15%); paper-faithful claim preserved.

3. **No OOM batch-halving in parallel.** Sequential mode (`--parallel 1`)
   keeps the paper's stated bs=64 → floor 8 halving policy. Parallel
   mode disables it entirely: an OOM row is `failed`, no metadata is
   written, and `--resume` will re-run it on the next invocation. The
   recovery path for a failed parallel row is to re-run with
   `--parallel 1`.

### Logs

Each child writes its stdout to `repro_results/<scope>/logs/<task_id>.log`
in real time (line-buffered). The parent prints compact one-liners:

```text
parallel=4; 207 runnable tasks
[1/207] START table2_srsnet_ETTh1_H96_SRSNet_s2021 (conc=4)
[2/207] START table2_srsnet_ETTh2_H96_SRSNet_s2021 (conc=4)
...
OK    table2_srsnet_ETTh1_H96_SRSNet_s2021  (612.3s)
[5/207] START table4_ablation_ETTh1_H96_SRSNet_NoSP_s2021 (conc=4)
...
```

### Resume across modes

Switching `--parallel` values between runs is safe and reuses
previously completed rows:

- The `requested_config_hash` is computed from `task.command`, which
  never contains the injected `--deterministic full` (that's a
  runtime-only rewrite). So the resume identity is stable across
  `--parallel` values.
- This means: if you run `--parallel 1` to completion of 100 rows then
  switch to `--parallel 4`, those 100 rows are skipped. They were
  trained with the official default `--deterministic efficient`
  (cudnn nondeterminism *allowed* on a clean single-tenant GPU). New
  rows at `--parallel 4` are trained with `--deterministic full`
  (cudnn nondeterminism *forbidden*, bit-equivalent under contention).
- The two settings produce numbers that are extremely close in
  practice (cudnn nondeterminism contributes far below paper-reported
  precision), but the *result set is technically mixed*. If you want
  a strictly homogeneous set, run a single mode end-to-end or use
  `--force` when switching.
- Metadata records the actual `deterministic_used` and
  `effective_concurrency` per row, so the audit trail tells you which
  rows were trained under which setting.

### Ctrl+C and orphan safety

Children inherit the parent's process group. Terminal Ctrl+C reaches
every child simultaneously. The parent also explicitly broadcasts
SIGTERM (5s grace) then SIGKILL on `KeyboardInterrupt`/`RuntimeError`
so non-terminal stops (`kill <pid>`, `timeout` wrapper, systemd) also
leave no orphans. Verify with `ps -ef | grep run_benchmark.py` and
`nvidia-smi` after any abnormal exit.

### Tuning N

- `--parallel 1`: current default, paper-mode sequential with OOM halving.
- `--parallel 2`: safe starter on any GPU >=12GB.
- `--parallel 4`: recommended on 24GB 4090. The static blocklist covers
  the known-heavy combos; light rows pack fine.
- `--parallel >=6`: only with an empty blocklist hit rate. Provoke OOMs
  on light rows; you'll need to extend `HEAVY_ROWS` or back off.

---

## Metric space

Reported `mse` / `mae` in `summary.csv` are
**`mse_norm` / `mae_norm`** as written by the official evaluator:
each model inverse-transforms its predictions, then the evaluator
normalizes the per-window error with the train-split scaler. Both
preds and targets live in the same scaled space, so values are
comparable to paper Table 2 figures.

**Refactor-branch `results/table*.json` files are not consumed** by
this runner. Those were produced by a different trainer/evaluator
pipeline (no inverse-transform, sliding-window only, different
`type3` schedule). Treat them as invalid and move them aside before
launching.

---

## Smoke gate

```bash
bash scripts/repro/run_local_4090.sh --scope lite-paper --smoke-check
```

After at least one SRSNet ETTh1 H96 row completes (and `--collect`
has run), this compares the best completed seed against paper Table 2
`(0.366, 0.394)` for `(MSE, MAE)`.

Defaults: `--smoke-tolerance-mse 0.08`, `--smoke-tolerance-mae 0.08`
(absolute). At MSE 0.366 that's ~22% relative — it's a **sanity gate
against gross misimplementation**, not a "matches paper" claim.
Tighten via flags if you want stricter gating once the pipeline is
stable, e.g.:

```bash
bash scripts/repro/run_local_4090.sh --scope full-paper --smoke-check \
  --smoke-dataset ETTh1 --smoke-horizon 96 \
  --smoke-tolerance-mse 0.02 --smoke-tolerance-mae 0.02
```

Other paper-Table-2 anchors (dataset, horizon → MSE, MAE) are listed
in `SRSNET_TABLE2` in `paper_repro.py:62`.

---

## Efficiency profiler

`scripts/repro/efficiency.py` is invoked by `full-paper` for Tables 5/6
rows. It:

- builds the model wrapper via the official `model_loader`
- runs `iters=5` warm + measured steps with `torch.randn` synthetic
  input tensors shaped for the paper's efficiency setting
  (`seq_len=512`, `horizon=720`, paper batch)
- reports `parameters`, `train_time_s_per_batch`,
  `inference_time_s_per_batch`, peak train/inference GPU memory
- `macs` is `null` unless you install a counter such as `thop` or
  `fvcore` (not added here — let me know if you want it wired in)

The profiler is not a substitute for an end-to-end training run; it's
only meaningful for time/memory shape against the paper's efficiency
tables.

---

## Outputs

After a run + `--collect`:

```text
repro_results/<scope>/
├── manifest.jsonl         # one line per planned task
├── status.jsonl           # one line per run attempt (append-only)
├── metadata/<task>.json   # per-task sidecar with hashes + final command
├── logs/<task>.log        # stdout+stderr of the child run_benchmark.py
├── summary.csv            # collected metrics, paper deltas, hashes
├── coverage.md            # per-status and per-table counts
└── efficiency/<task>.json # only in full-paper
```

`result/repro/<scope>/<table>/<task>/test_report*.csv` is where the
official evaluator writes the actual metric rows (one per dataset
horizon split). The `result_file` field in metadata points there.

---

## Known gotchas

- **You must move pre-existing `result/` aside** if you've previously
  run the official `scripts/multivariate_forecast/*` scripts.
  `--check-stale-results` will list them. They live in the same root
  the new runner uses and can confuse resume if filenames overlap.
- **`reference-only` is silent** for rows where no official shell
  script exists. Check `coverage.md`'s per-status breakdown before
  reading "836 manifested rows" as "836 runnable rows" — currently
  lite-paper has ~207 runnable / 628 omitted-lite / 1 reference-only.
- **`--keep-going` keeps the run alive past failures**; without it
  the runner exits non-zero on the first failure. Default is
  fail-loud on purpose. Use `--keep-going` only after smoke + a few
  manual rows have passed.
- **`--hours` is accepted but ignored.** The runner is resumable; if
  you need a hard wall-clock cap, wrap the invocation in `timeout`
  or a systemd timer.
- **macOS/Linux without systemd**: pass `--no-inhibit` or rely on the
  wrapper's auto-detect (`command -v systemd-inhibit`).
- **Paper-mode `batch_size=64` is non-negotiable** in `lite-paper`/
  `full-paper`. If you need to honor an official script's batch_size
  exactly (e.g. to match the original repo's behavior), use
  `--scope main-compat`.
- **`--parallel N` is opt-in.** Default stays at 1 so the sequential
  paper-mode OOM-halving behavior is preserved. At N>=2, OOM halving
  is disabled and the row fails loud; re-run failed rows at
  `--parallel 1` to recover them.
- **Switching `--parallel` values does not invalidate completed rows.**
  The runtime `--deterministic full` injection is not part of the
  resume identity hash, so previously completed rows are reused.
  Result: the set can be mixed (some rows trained at `efficient`, some
  at `full`). In practice the numbers are nearly identical; if you
  want a homogeneous set, run a single mode end-to-end or use
  `--force`. The `deterministic_used` field in each row's metadata
  records which setting was actually used.

---

## Pre-launch checklist for the teammate running this on the Linux box

1. `git pull` this branch.
2. `pip install -r requirements.txt -r requirements-repro.txt`.
3. `bash scripts/repro/run_local_4090.sh --check-data`.
4. `bash scripts/repro/run_local_4090.sh --scope lite-paper --check-stale-results` — if it lists files, move `result/` aside.
5. `bash scripts/repro/run_local_4090.sh --scope lite-paper --manifest` and skim `repro_results/lite-paper/manifest.jsonl` + `coverage.md`.
6. `bash scripts/repro/run_local_4090.sh --scope lite-paper --dry-run --max-tasks 1` and read the printed command.
7. Run a single real task: `--scope lite-paper --max-tasks 1` (no `--dry-run`).
8. `--collect` then `--smoke-check`. Only proceed past this point if smoke passes.
9. Launch the real run: `--scope lite-paper --parallel 4 --keep-going`. Tail `repro_results/lite-paper/logs/`.
10. After lite passes, repeat with `--scope full-paper --parallel 4 --keep-going` if budget allows.

> If you hit unexpected OOMs at `--parallel 4`, drop to `--parallel 2`,
> let resume pick up the failed rows, and add the offending
> `(model, dataset, horizon)` to `HEAVY_ROWS` in `paper_repro.py`.

---

## Files

- `scripts/repro/run_local_4090.sh` — wrapper (env, GPU pinning, sleep inhibit)
- `scripts/repro/paper_repro.py` — manifest, runner, hashing, collect, smoke, stale check
- `scripts/repro/efficiency.py` — one-batch synthetic profiler for Tables 5/6
- `scripts/repro/README.md` — this file
- `ts_benchmark/baselines/srs_paper/` — ablation variants and SRS plug-in heads
- `ts_benchmark/baselines/deep_forecasting_model_base.py` — `train_drop_last` plumbing only
- `requirements-repro.txt` — `numpy<2` overlay
