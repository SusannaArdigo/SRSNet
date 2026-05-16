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

## Future Constructive Extensions (Post-Selectivity Audit)

The selectivity-controls study above is a *negative-control* study: it tests
whether the paper's learned scorer is necessary, but it does not propose a
replacement.  Once the small-grid Random results are in, the report would be
strengthened by pairing the negative result with at least one *constructive*
extension that maps explicitly to a Future Work entry from Sec. 6 of the
SRSNet paper.

The three candidates below were selected because each addresses a distinct
paper Future Work / Limitation pair, and each is small enough to implement
on top of the existing `selectivity-controls` scaffolding (same datasets,
same horizons, 5 seeds, same `paper_repro.py` runner).

A summary table is provided at the end; the per-extension blocks below
expand the rationale and design.

### Extension A: TASP -- Time-Aware Selective Patching

**Paper claims addressed**

* FW#1 -- *"environment-aware mechanism to perceive the patch-wise data
  distributions and patterns more explicitly".*
* L3 -- *"we can only ensure the selected patches are useful for forecasting,
  but not all of them are interpretable".*

**Motivation**

The paper's `_select` is a two-layer MLP over the raw patch values.  Its
output is a vector of scores with no a-priori meaning.  TASP replaces that
MLP with a scorer over a small set of *engineered, interpretable* per-window
statistics:

* dominant FFT magnitude in the seasonal band (hourly: lag 24; 15-min: lag 96)
* lag-1 autocorrelation
* within-window variance
* trend slope (linear fit residual)

The scorer that maps the four-feature vector to a per-candidate score is
itself a tiny MLP (4 -> 16 -> 1).  Total scorer params drop from
`O(patch_len * hidden_size * patch_num)` to `O(80)`, so any improvement
cannot be attributed to extra capacity.

**Design sketch**

```python
class SRSTimeAware(SRS):
    """Scorer over engineered features instead of raw patch values."""

    N_FEATURES = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the learned scorer with a tiny MLP over interpretable features.
        self.scorer_select = nn.Sequential(
            nn.Linear(self.N_FEATURES, 16),
            nn.ReLU(),
            nn.Linear(16, self.patch_num),
        )

    @staticmethod
    def _window_features(x_rec):
        """x_rec: [B, C, candidate_num, patch_size] -> [B, C, candidate_num, 4]."""
        spec = torch.fft.rfft(x_rec, dim=-1).abs()
        dominant = spec[..., 1:].max(dim=-1).values
        mean = x_rec.mean(dim=-1, keepdim=True)
        var = x_rec.var(dim=-1, unbiased=False)
        # Lag-1 autocorrelation via correlation of (x[1:] - mean) and (x[:-1] - mean).
        x_centered = x_rec - mean
        ac1 = (
            (x_centered[..., 1:] * x_centered[..., :-1]).sum(dim=-1)
            / (x_centered.pow(2).sum(dim=-1) + 1e-8)
        )
        # Trend slope: residual of a linear least-squares fit over the window.
        t = torch.arange(x_rec.shape[-1], device=x_rec.device, dtype=x_rec.dtype)
        t_c = t - t.mean()
        denom = (t_c * t_c).sum() + 1e-8
        slope = ((x_rec - mean) * t_c).sum(dim=-1) / denom
        return torch.stack([dominant, ac1, var, slope], dim=-1)

    def _select(self, x_rec):
        feats = self._window_features(x_rec)
        scores = self.scorer_select(feats)  # [B, C, candidates, patch_num]
        # ...remaining gather logic mirrors the parent _select unchanged.
```

**Why this is a contribution beyond `SRSRandomSP`**

* It proposes a *replacement* for the learned scorer, not just an ablation.
* The selected patches are now traceable to a small set of named features.
  A plot of `which feature drove the top score` is a real interpretability
  artifact, not a post-hoc rationalization.
* If TASP matches SRSNet MSE within the seed-level std band, that is a
  stronger statement than the Random result alone: not only does the
  learned MLP not help, but a transparent four-feature scorer suffices.

**Risks / pitfalls**

* The feature set is opinionated.  Reviewers may ask why these four and not
  others.  Honest answer: they are common time-series summary statistics;
  no claim that this is the *best* engineered set.
* If TASP is clearly worse than SRSNet, the writeup must say so.  Do not
  describe TASP as "interpretable so worse is fine".  Either it works or
  it does not, and the result either way is informative.
* RNG-free, but the linear-fit trend slope is sensitive to outliers.
  Mention this as a known limitation.

**Cost estimate**

* Code: roughly 100 LOC for the layer + wrapper + registration.
* Compute: 60 retrained tasks on the small grid
  (2 datasets x 2 horizons x 5 seeds x 3 variants where 2 of the 3 are
  baseline SRSNet and SRSNet_RandomSP from the existing run, so net new
  tasks are 20).  About 1 to 2 GPU-hours on a single 4090.

### Extension B: Hypernet-AF -- Hypernet Adaptive Fusion

**Paper claims addressed**

* FW#3 -- *"more efficient update mechanism for `alpha` deserves exploration".*
* L4 -- *"the initialization of the weights `alpha` seems to be important".*

**Motivation**

SRSNet's adaptive fusion uses a free parameter `alpha` of shape
`[patch_num, d_model]` that is shared across the entire batch and updated
only via the prediction-head gradient.  Two issues stand out:

1. With the script-mode default `alpha = 3.5`, `sigmoid(alpha) = 0.97`, so
   the reconstruction view contributes only about 3% of the embedding.
   This is the regime the DDA experiment lived in -- effectively a no-op.
2. The fusion weight does not depend on the input.  Different batches with
   very different stationarity / seasonality use the same fusion ratio.

Hypernet-AF replaces the free `alpha` parameter with a tiny hypernet that
maps a batch-level context summary to a per-patch, per-feature `alpha`
tensor.  Crucially, the hypernet has very few parameters compared with the
GAF gating MLP (which was capacity-confounded).

**Design sketch**

```python
class SRSHypernetAF(SRS):
    """Replace the learned alpha tensor with a hypernet over batch context."""

    def __init__(self, *args, hyper_hidden=8, **kwargs):
        super().__init__(*args, **kwargs)
        # Strip the free alpha parameter and replace with a 2-layer hypernet
        # over a low-dim batch context.
        del self.alpha
        self.context_proj = nn.Linear(self.value_embedding_org.out_features, hyper_hidden)
        self.hyper = nn.Linear(hyper_hidden, self.patch_num * self.value_embedding_org.out_features)
        # Init the hypernet output bias so that sigmoid(alpha) ~ 0.95 at step 0,
        # matching the paper's default 3.5 initialization and keeping a vanilla-
        # preserving start.
        nn.init.zeros_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)
        nn.init.zeros_(self.hyper.weight)
        nn.init.constant_(self.hyper.bias, 3.5)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        rec = self._rec_view(x)
        orig = self._origin_view(x)
        e_orig = self.value_embedding_org(orig)
        e_rec = self.value_embedding_rec(rec)
        # Context summary: mean over batch+variables+patches of e_orig.
        ctx = e_orig.mean(dim=(0, 1)).mean(dim=0)  # [d_model]
        h = torch.relu(self.context_proj(ctx))
        alpha = self.hyper(h).view(self.patch_num, e_orig.shape[-1])
        weight = torch.sigmoid(alpha)
        embedding = weight * e_orig + (1 - weight) * e_rec
        if self.pos:
            embedding = embedding + self.position_embedding(orig)
        return self.dropout(embedding), n_vars
```

**Why this is a contribution beyond GAF**

* Hypernet-AF is intentionally smaller than the GAF gating MLP that the
  earlier extension set proposed.  The number of new parameters is
  `d_model * hyper_hidden + hyper_hidden * patch_num * d_model`, with
  `hyper_hidden = 8`.  For typical ETT configurations this is on the
  order of 10k parameters, comparable to the free `alpha` tensor it
  replaces.
* The zero-init of the hypernet weights combined with bias = 3.5 means
  the model starts behaving exactly like vanilla SRSNet at step 0.  Any
  improvement during training can only come from the *dynamic*,
  context-dependent behavior of the hypernet, not from extra capacity at
  initialization.  This is the kind of parameter-matched control GAF
  lacked.

**Risks / pitfalls**

* The hypernet may converge to producing a near-constant `alpha`, making
  it indistinguishable from the paper baseline.  This would itself be a
  result: "context does not carry useful information for `alpha`".
* `sigmoid(3.5) = 0.97` saturates the gradient.  If the hypernet does
  not move `alpha` away from 3.5 early in training, the rest of the
  forward pass is effectively the paper's vanilla pipeline.  Track the
  per-cell mean and std of `alpha` during training to confirm whether
  Hypernet-AF actually moves it.
* Add an `alpha` histogram or a per-step mean to the run logs so the
  report can distinguish "the hypernet does not do anything" from
  "the hypernet helps".

**Cost estimate**

* Code: roughly 80 LOC including a tiny logging hook for `alpha`.
* Compute: 20 new tasks on the small grid (only the third variant).
  About 1 to 2 GPU-hours.

### Extension C: PS-SRS -- Pattern-Supervised SRS

**Paper claims addressed**

* FW#4 -- *"design a module to supervise the sample-wise data patterns,
  constructing a more explicit optimization objective between data
  patterns and `alpha`".*
* L3 -- *"the selected patches are useful for forecasting, but not all of
  them are interpretable".*

**Motivation**

In SRSNet the scorer is trained end-to-end with the prediction loss.
Nothing forces it to learn anything that a human can audit.  PS-SRS adds
an auxiliary loss that asks the scorer's intermediate representation to
predict pre-computed pattern descriptors of each candidate window.  If
the scorer can simultaneously do forecasting *and* predict those
descriptors, then "the scorer learned something pattern-related" becomes
a verifiable claim rather than a hand-wave.

This is the most novel of the three because it modifies the *training
objective*, not just the architecture.

**Design sketch**

```python
class SRSPatternSupervised(SRS):
    """Add an auxiliary loss that predicts engineered pattern descriptors."""

    N_DESCRIPTORS = 3  # e.g., dominant FFT magnitude, lag-1 autocorr, variance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # An auxiliary head that pulls features out of the first scorer layer
        # and predicts the N_DESCRIPTORS-dim summary of each window.
        hidden_dim = self.scorer_select[0].out_features
        self.descriptor_head = nn.Linear(hidden_dim, self.N_DESCRIPTORS)

    def _scorer_features(self, x_rec):
        # x_rec: [B, C, candidate_num, patch_size]
        # Reuse the first hidden layer of scorer_select.
        return torch.relu(self.scorer_select[0](x_rec))

    def descriptor_loss(self, x_rec, ground_truth_descriptors):
        feats = self._scorer_features(x_rec)
        pred = self.descriptor_head(feats)
        return F.mse_loss(pred, ground_truth_descriptors)

    # _select is inherited unchanged; the auxiliary head only affects training.
```

The training loop is also modified:

```python
# In SRSNet's _process or training step:
total_loss = mse_loss + lambda_aux * patch_embedding.descriptor_loss(
    x_rec, precomputed_descriptors
)
```

Pre-compute the ground-truth descriptors offline for each (dataset, split)
pair to avoid recomputing them every batch.  `lambda_aux` is a single
scalar hyperparameter.  A small initial value (`1e-3` to `1e-2`) lets us
tune the strength of the auxiliary signal without overwhelming the main
forecasting loss.

**Why this is a contribution beyond TASP**

* TASP makes the scorer *transparent* by changing its inputs.  PS-SRS
  goes further: it forces the scorer's *internal representation* to be
  predictive of named statistics.  The two are not mutually exclusive,
  but PS-SRS is the more aggressive intervention.
* The auxiliary loss adds a meaningful regularizer.  Even if PS-SRS does
  not change end-task MSE, it produces a *probe-ready* scorer: at
  evaluation time, the descriptor head can be queried to verify how
  much of each pattern type the scorer captured per cell.

**Risks / pitfalls**

* `lambda_aux` is now a tunable that interacts with the main loss.  A
  small sweep (`{1e-3, 1e-2, 1e-1}`) on one cell is mandatory before
  running 5 seeds across all cells.
* Pre-computing descriptors requires deciding *which* descriptors to
  use.  If the same descriptors are used for TASP and PS-SRS, the two
  experiments share a methodological choice and should be discussed
  together.
* The probe-ready property is only useful if the auxiliary loss
  converges to a low value.  Log the auxiliary loss alongside MSE and
  report both.

**Cost estimate**

* Code: roughly 150 LOC including the descriptor pre-computation script,
  a small change to the training loop in
  `ts_benchmark/baselines/srsnet/srsnet.py::_process`, and the new layer.
* Compute: 20 new tasks on the small grid, plus a 5-row hyperparameter
  sweep for `lambda_aux`.  About 2 to 3 GPU-hours total.

### Comparison Table

| Extension | Paper FW addressed | Capacity-confounded? | Net new code | Net new tasks (small grid) | Probe-ready output |
|---|---|---|---|---|---|
| TASP        | FW#1 + L3       | No (fewer params than baseline scorer) | ~100 LOC | 20 | engineered features visible per selection |
| Hypernet-AF | FW#3 + L4       | No (parameter-matched, zero-init)      | ~80 LOC  | 20 | per-cell `alpha` distribution over training |
| PS-SRS      | FW#4 + L3       | No (lambda-controlled aux loss)        | ~150 LOC | 20 + 5-row sweep | per-cell descriptor recovery error |

### Recommended Sequencing

The selectivity-controls Random study should remain the headline result of
the report because it directly addresses the paper's central claim.  The
constructive extensions are best presented as a single follow-up section
that demonstrates the gap can be filled, not as separate competing
models.  Suggested ordering:

1. **Random study** (already complete).  Negative-control result.
2. **TASP**.  Lightest constructive extension.  If TASP matches SRSNet
   within the seed band, the narrative becomes "a transparent four-feature
   scorer suffices".
3. **Hypernet-AF**.  Smallest architectural change.  If it does not move
   `alpha` away from 3.5, that itself answers FW#3 with a null result and
   should be reported.
4. **PS-SRS**.  Most novel and most expensive.  Only run if items 2 and
   3 leave a clear opening for "interpretability via training, not just
   architecture".

If compute is constrained, drop PS-SRS first, then Hypernet-AF.  TASP
should remain because it pairs most directly with the Random study: TASP
*is* the constructive counterpart of the negative-control result.

### Acceptance Criteria for Constructive Extensions

* The variant's MSE is reported per (dataset, horizon) cell across all
  five seeds, with mean and standard deviation, exactly like the
  Random study.
* Parameter count is reported alongside MSE so the reader can verify the
  comparison is not silently capacity-confounded.
* For Hypernet-AF and PS-SRS, log at least one diagnostic per run that
  shows whether the new component is actually doing work
  (`alpha` distribution, auxiliary loss curve, descriptor recovery error).
* The writeup framing remains "missing-control plus minimal
  constructive answer", not "new state of the art".  Negative or null
  results are reportable and should be reported.
