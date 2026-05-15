# Selectivity Extension Plan for SRSNet Reproduction

Branch: `paper-faithful-repro-ett-extensions`

Status: engineering handoff. This document is the proposed replacement for the
broader extension set (`DDA`, `MSP`, `GAF`, `RandomSRS`) when quality matters
more than quantity and compute time is limited.

## Executive Decision

Focus the extension work on one sharp question:

> How much does SRSNet's performance depend on learned patch selection, compared
> with random or simple patch choices?

Do not present `DDA`, `MSP`, or `GAF` as headline extensions in the near-term
writeup. Keep them as future ideas only. The near-term contribution should be a
selectivity control study with:

1. An inference-time selector-swap diagnostic, if usable SRSNet checkpoints are
   available.
2. Retrained random-selectivity controls with multiple seeds.

This is a stronger story than running many loosely related extensions because it
directly tests the central paper claim that learned selective patching matters.

## Why This Plan

### Why Cut DDA

`DDA` currently changes the alpha initialization using a hand-tuned FFT
seasonality score. The current ETT alpha values are approximately `3.00` to
`3.36`, which gives `sigmoid(alpha) = 0.953` to `0.967`.

In the parent SRS layer, the fusion is:

```python
embedding = weight * E_orig + (1 - weight) * E_rec
weight = sigmoid(alpha)
```

So the current DDA values give only about `3.3%` to `4.7%` reconstruction-view
weight. That makes the SRS contribution very small for many rows, and it makes
the "data-driven" claim hard to defend. Fixing DDA properly would require a
better seasonality metric, a clear sign convention, an alpha sensitivity study,
and probably a sweep baseline. That is too much work for the short extension.

### Why Cut MSP

`MSP` is architecturally broad: it adds three SRS modules, aligns their patch
dimensions, and learns a cross-scale fusion. It changes capacity, cost, and
patching behavior at the same time.

It is also risky in the current implementation because it hardcodes scales
`(8, 24, 48)` and a 24-step reference patch count, while the SRSNet prediction
head is built from the official row's `patch_len` and `stride`. Several ETT rows
use `patch_len=12` or `16`, so MSP can produce a patch count that does not match
the head dimension.

Even after fixing the shape issue, MSP would need efficiency measurements to be
fair because it roughly triples the SRS embedding cost. That does not fit the
time budget.

### Why Cut GAF

`GAF` is interesting, but it is capacity-confounded. It replaces a learned
`[patch_num, d_model]` alpha tensor with a small MLP over `[E_orig, E_rec]`.
If it improves, the result may come from extra capacity rather than a better
fusion mechanism.

A publishable GAF experiment would need a parameter-matched control and a
vanilla-preserving initialization, for example:

```python
gate_logits = alpha + gate_delta(torch.cat([E_orig, E_rec], dim=-1))
gate_delta is initialized to zero
```

That is a good future extension, but it is not the cleanest short-run test.

### Why Keep Random Selectivity

Random patch selection is the smallest meaningful intervention on the SRS
claim. It asks whether the learned scorer is actually doing useful work.

Both outcomes are informative:

- Random selection is close to learned selection: the paper's selectivity claim
  needs qualification because arbitrary patch choices work nearly as well.
- Random selection is worse: the paper gains support from a missing negative
  control.

This control is cheap to implement, easy to explain, and does not require
inventing a new architecture.

## Recommended Experiments

### Experiment 1: Inference-Time Selector Swap

Purpose: test whether the trained learned selector matters without retraining.

Prerequisite: completed SRSNet checkpoints must exist for the target cells. If
`result/repro/...` does not contain loadable checkpoints for the completed
paper-reproduction rows, this experiment is blocked and should be skipped rather
than replaced with a weaker approximation.

Target grid:

```text
Datasets: ETTh1, ETTm2
Horizons: 96, 720
Seeds: existing completed SRSNet seeds
```

Expanded grid, only if checkpoint loading is straightforward:

```text
Datasets: ETTh1, ETTh2, ETTm1, ETTm2
Horizons: 96, 192, 336, 720
Seeds: existing completed SRSNet seeds
```

Selector variants:

| Variant | Meaning |
|---|---|
| `learned` | Original trained SRS `_select` behavior. |
| `random` | Replace selected candidate indices with random candidate indices. |
| `top_variance` | Select candidate windows with largest within-window variance. |
| `periodic` | Select windows aligned to known period offsets, such as 24 for hourly ETT and 96 for 15-minute ETT. |
| `spectral_energy` | Optional: select windows with strongest non-DC FFT magnitude. |

Random selector requirements:

- Use multiple random draws per checkpoint and test cell, for example `5` draws.
- Report mean and standard deviation for the random selector.
- Do not compare one random draw against one learned run as if it were stable.

Output:

```text
report_tables/selectivity_inference_swap.csv
report_tables/selectivity_inference_swap.md
```

Minimum CSV columns:

```text
dataset,horizon,seed,selector,draw,mse_norm,mae_norm,
delta_mse_vs_learned_pct,delta_mae_vs_learned_pct
```

Interpretation:

- If `random` is close to `learned`, the learned scorer may be decorative.
- If simple heuristics are close to `learned`, selection matters but the learned
  scorer may not be necessary.
- If only `learned` works well, the paper's learned-selectivity claim is
  strengthened.

### Experiment 2: Retrained Random-Selectivity Controls

Purpose: test whether models trained with random selection can match models
trained with learned selection.

Primary variants:

| Variant | Meaning |
|---|---|
| `SRSNet` | Original learned select + learned shuffle + adaptive fusion. |
| `SRSNet_RandomSP` | Random candidate selection, learned shuffle retained. |
| `SRSNet_RandomSPRandomShuffle` | Random candidate selection plus non-learned shuffle behavior. |

Optional reference:

| Variant | Meaning |
|---|---|
| `SRSNet_NoSP` | Existing ablation that disables selective patching while keeping dynamic reassembly. Include only if already cheap through existing ablation tasks. |

Use precise naming. Avoid `RandomSRS` unless both selection and shuffling are
randomized. The current "random SRS" idea only randomizes `_select` while still
using learned `_shuffle`, so `RandomSP` is the clearer name.

Small strong grid:

```text
Datasets: ETTh1, ETTm2
Horizons: 96, 720
Seeds: 2021, 2022, 2023, 2024, 2025
Variants: SRSNet, SRSNet_RandomSP, SRSNet_RandomSPRandomShuffle
```

Run count:

```text
2 datasets x 2 horizons x 5 seeds x 3 variants = 60 total rows
```

If matching baseline `SRSNet` rows already exist and pass config-hash checks,
the new training cost is:

```text
2 datasets x 2 horizons x 5 seeds x 2 random variants = 40 new rows
```

Expanded grid, only if the small grid shows a useful pattern:

```text
Datasets: ETTh1, ETTh2, ETTm1, ETTm2
Horizons: 96, 192, 336, 720
Seeds: 2021, 2022, 2023, 2024, 2025
Variants: SRSNet, SRSNet_RandomSP, SRSNet_RandomSPRandomShuffle
```

Run count:

```text
4 datasets x 4 horizons x 5 seeds x 3 variants = 240 total rows
```

If baseline `SRSNet` rows are reused:

```text
4 datasets x 4 horizons x 5 seeds x 2 random variants = 160 new rows
```

Output:

```text
repro_results/selectivity-controls/summary.csv
report_tables/selectivity_controls.csv
report_tables/selectivity_controls.md
```

Minimum CSV columns:

```text
dataset,horizon,seed,model,mse_norm,mae_norm,
baseline_mse_norm,delta_mse_vs_srsnet_pct,
baseline_mae_norm,delta_mae_vs_srsnet_pct
```

Headline metrics:

- Mean delta MSE versus SRSNet across seeds.
- Number of seeds where each random-control variant beats SRSNet.
- Per-cell mean and standard deviation across seeds.
- Aggregate result over the small grid first; full-grid aggregate only if run.

## Implementation Draft

### 1. Add Precise Random-Control Layers

Add two new SRS layer variants under the SRS paper extension module.

`SRSRandomSP`:

- Subclass the parent `SRS`.
- Override `_select` only.
- Sample random candidate indices with the same output shape as learned `_select`.
- Keep `_shuffle`, adaptive fusion, embeddings, dropout, and positional behavior
  unchanged.

Sketch:

```python
class SRSRandomSP(SRS):
    def _select(self, x_rec):
        batch, n_vars, candidate_num, patch_size = x_rec.shape
        idx = torch.randint(
            low=0,
            high=candidate_num,
            size=(batch, n_vars, 1, self.patch_num),
            device=x_rec.device,
        )
        gather_idx = idx.repeat(1, 1, patch_size, 1).permute(0, 1, 3, 2)
        return torch.gather(x_rec, dim=-2, index=gather_idx)
```

`SRSRandomSPRandomShuffle`:

- Subclass `SRSRandomSP`.
- Override `_shuffle`.
- Preferred default: use a random permutation of selected patches per batch and
  variable.
- If random shuffle is too noisy, use identity order and name the class
  `SRSRandomSPNoShuffle` instead. Do not use a misleading name.

Sketch:

```python
class SRSRandomSPRandomShuffle(SRSRandomSP):
    def _shuffle(self, selected_patches):
        batch, n_vars, patch_num, patch_size = selected_patches.shape
        scores = torch.rand(batch, n_vars, patch_num, 1, device=selected_patches.device)
        order = torch.argsort(scores, dim=-2, descending=True)
        gather_idx = order.repeat(1, 1, 1, patch_size)
        return torch.gather(selected_patches, dim=-2, index=gather_idx)
```

Register wrappers:

- `SRSNet_RandomSP`
- `SRSNet_RandomSPRandomShuffle`

These wrappers should follow the existing SRSNet extension wrapper pattern and
report distinct `model_name` values.

### 2. Add a Dedicated Runner Scope

Add a new scope named `selectivity-controls` rather than overloading the existing
`extensions` scope.

Task matrix:

```text
SELECTIVITY_DATASETS_SMALL = ["ETTh1", "ETTm2"]
SELECTIVITY_HORIZONS_SMALL = [96, 720]
SELECTIVITY_SEEDS = [2021, 2022, 2023, 2024, 2025]
SELECTIVITY_VARIANTS = [
    "SRSNet",
    "srs_paper.SRSNet_RandomSP",
    "srs_paper.SRSNet_RandomSPRandomShuffle",
]
```

Behavior:

- Use the official SRSNet shell scripts as the base command source.
- Preserve paper-mode overrides: `batch_size=64`, `train_drop_last=false`, and
  horizon safety override.
- Use hash-aware resume exactly like the existing reproduction scopes.
- Keep baseline SRSNet in the manifest so comparisons are explicit. Reuse only
  if metadata confirms the requested config matches.

Add an optional CLI flag later if needed:

```text
--selectivity-grid small|full
```

Default to `small`.

### 3. Add Report Generation

Add report helpers that read `repro_results/selectivity-controls/summary.csv`
and corresponding `test_report*.csv` files.

Generate:

```text
report_tables/selectivity_controls.csv
report_tables/selectivity_controls.md
```

Report requirements:

- Compare each random-control row to matching `SRSNet` by dataset, horizon, and
  seed.
- Aggregate by dataset and horizon.
- Aggregate over the whole small grid.
- Show seed-level variance before making any qualitative claim.

Verdict labels:

| Condition | Verdict |
|---|---|
| Random control mean delta within `+/-1%` MSE of SRSNet | `matches_learned_selectivity` |
| Random control mean delta worse than `+1%` MSE | `learned_selectivity_helpful` |
| Random control mean delta better than `-1%` MSE | `learned_selectivity_not_supported` |

The verdict threshold is descriptive only. The writeup should still show raw
per-seed numbers.

### 4. Add Inference-Time Swap Script Only If Checkpoints Exist

Before implementing the inference-time script, verify that completed SRSNet
checkpoints exist and can be loaded. If not, mark this experiment blocked in the
document/report and proceed with retrained controls.

Suggested script:

```text
tools/selectivity_inference_swap.py
```

Responsibilities:

- Load a completed SRSNet row and its checkpoint.
- Monkey-patch or configure the SRS selector at evaluation time.
- Evaluate each selector variant on the same test split.
- Emit one CSV row per selector, seed, dataset, horizon, and random draw.

Do not retrain in this script.

## Execution Order

1. Confirm whether baseline checkpoints exist for the target cells.
2. If checkpoints exist, implement and run inference-time selector swap first.
3. Implement `RandomSP` and `RandomSPRandomShuffle` wrappers.
4. Add the `selectivity-controls` scope and manifest generation.
5. Dry-run one task for each random-control variant.
6. Run the small 5-seed grid.
7. Generate `selectivity_controls.csv` and `selectivity_controls.md`.
8. Decide whether the full ETT grid is worth running.

## Suggested Commands

Inspect the new scope:

```bash
bash scripts/repro/run_local_4090.sh --scope selectivity-controls --manifest
```

Dry-run one task:

```bash
bash scripts/repro/run_local_4090.sh --scope selectivity-controls --dry-run --max-tasks 1
```

Run the small grid:

```bash
bash scripts/repro/run_local_4090.sh --scope selectivity-controls --parallel 4 --keep-going
```

Collect:

```bash
bash scripts/repro/run_local_4090.sh --scope selectivity-controls --collect
python tools/generate_paper_comparison.py
```

These commands assume the runner has been updated to accept the new scope.

## Acceptance Criteria

The extension work is ready to report when all of the following are true:

- `RandomSP` and `RandomSPRandomShuffle` are named precisely and documented.
- The small grid has 5 seeds for every completed model/dataset/horizon cell.
- Results compare random-control variants to matching-seed SRSNet baselines.
- The report shows mean and standard deviation across seeds.
- No headline claim depends on a single seed.
- No headline claim uses DDA, MSP, or GAF.
- If inference-time selector swap is skipped, the reason is explicitly stated
  as missing or unusable checkpoints.

## Expected Writeup Framing

Frame the contribution as a missing-control study, not as a new state-of-the-art
model:

> We test whether SRSNet's learned patch selector contributes beyond random or
> simple patch choices. This isolates the paper's central selectivity claim more
> directly than adding higher-capacity architectural variants.

Possible conclusions:

- If random controls match SRSNet, learned selective patching is not strongly
  supported on the tested cells.
- If random controls are consistently worse, learned selective patching is
  supported by a stronger negative control than the original paper provided.
- If simple heuristic selectors match learned selection at inference time,
  selection matters, but a learned scorer may be unnecessary.

Keep conclusions scoped to the tested datasets, horizons, seeds, and hardware.

## Review Notes (added after engineering handoff)

The plan is the right scope. Before implementation, address the following in
order of risk.

### 1. Verify checkpoint availability before scoping Experiment 1

Experiment 1 (inference-time selector swap) is conditional on completed SRSNet
checkpoints, but TFB's default training loop does not reliably persist final
model weights. `result/repro/.../test_report*.csv` contains metrics, not state
dicts. Run a quick check before writing any code for Exp 1:

```bash
find result/ -name '*.pth' -o -name '*.pt' 2>/dev/null | head
```

If only CSVs exist, Exp 1 is either blocked or requires retraining baselines
just to dump checkpoints, which negates the "no extra training compute"
advantage. Either way, decide and document before starting implementation.

### 2. Confirm baseline seed coverage

`lite-paper` scope runs seed 2021 only; only `full-paper` runs all 5 seeds. If
the completed runs are `lite-paper`, the "5-seed SRSNet vs 5-seed RandomSP"
comparison requires 16 additional baseline runs (2 datasets x 2 horizons x 4
missing seeds) on top of the 40 random-control runs. Budget for 56 new
trainings, not 40. Confirm by checking which seeds already appear in
`repro_results/*/summary.csv` before committing.

### 3. Default the second variant to identity shuffle, not random shuffle

Random-shuffle-per-batch introduces a second stochastic source on top of random
selection. This inflates per-seed variance and confounds attribution between
"no learning in _select" and "no learning in _shuffle".

Recommended default: identity shuffle (keep selected patches in candidate
order). It is deterministic, cheaper, and isolates "no learning in _shuffle"
cleanly. Rename to `SRSNet_RandomSP_NoShuffle`. The plan already lists this as
a fallback; promote it to the default.

If a randomized-shuffle variant is still desired, add it as a third optional
variant and clearly mark it as ablating both selection and shuffling
simultaneously.

### 4. Reframe the `periodic` selector as the PatchTST-equivalent baseline

A `periodic` selector with stride equal to `patch_len`, aligned to 24 for
hourly ETT or 96 for 15-minute ETT, is essentially plain PatchTST-style
patching. This is the highest-impact framing in the writeup:

- If `periodic` is close to `learned` at inference time, the takeaway is not
  "selection does not matter" but "a fixed PatchTST stride matches the learned
  scorer", which directly questions the architectural novelty of selective
  patching.
- If `periodic` is worse than `learned`, the paper's selectivity claim gains
  support from a stronger baseline than the original ablations provided.

Make this framing explicit in `report_tables/selectivity_inference_swap.md`,
and label the periodic selector as "PatchTST-equivalent fixed stride" in
plots and tables.

### 5. Tone down the verdict thresholds

The +/-1% MSE bins (`matches_learned_selectivity`,
`learned_selectivity_helpful`, `learned_selectivity_not_supported`) are too
crisp for ETT-MSE seed variance, which is often comparable to or larger than
1% of the mean. Two safer options:

- Use a per-cell `2 * sigma` of the SRSNet baseline across seeds as the
  threshold.
- Drop verdict labels entirely. Report only `mean +/- std` and the number of
  seeds where each variant beats the baseline.

The plan already notes "verdict is descriptive only", but labelled verdicts
will be quoted out of context. Cleaner to not provide them.

### 6. Minor implementation notes

- **RNG control in random selectors.** `torch.randint` inside `_select` is
  called every forward pass and is not tied to the run's `--seed`. To make a
  single seed reproducible end-to-end, route the random draws through a
  `torch.Generator` initialized from the run seed. Otherwise "5 seeds" of
  RandomSP means 5 weight inits + 5 data orders, but the in-loop selection
  randomness is still ambient. Acceptable, but acknowledge in the report.
- **Naming hygiene.** The existing `SRSNet_RandomSRS` wrapper in
  `ts_benchmark/baselines/srs_paper/extensions.py` should be removed or
  aliased to `SRSNet_RandomSP` before the new scope lands. Leaving both will
  confuse downstream summaries and verdict aggregation.
- **Compute estimate.** Small grid: 60 runs x roughly 5-10 min on a 4090 is
  approximately 5-10 GPU-hours. Tractable. Full grid: 240 runs is a weekend
  job. Do not commit to the full grid until the small grid shows a signal
  worth scaling.

### Bottom line

Go ahead with this plan, but gate the work on the checkpoint check (item 1)
before writing any code for Experiment 1, and adopt items 3 (identity shuffle
default), 4 (periodic = PatchTST framing), and 5 (drop or refine verdict bins)
before generating any report tables.
