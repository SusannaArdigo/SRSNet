# Selectivity-controls study

Retrained random selective-patching controls.  Each (dataset, horizon,
variant) cell is aggregated over the seeds listed in `seeds`.

* Variants:
  * `SRSNet`                       -- learned select + learned shuffle (baseline)
  * `SRSNet_RandomSP`              -- random select, learned shuffle
  * `SRSNet_RandomSPNoShuffle`     -- random select + identity shuffle

No verdict labels are emitted; the writeup should look at the raw mean +/- std
and the seed-level win count.

## MSE summary (mean +/- std across seeds)

| dataset | horizon | seeds | SRSNet (baseline) | SRSNet_RandomSP (delta% vs baseline) | SRSNet_RandomSPNoShuffle (delta% vs baseline) | wins_vs_baseline (per variant) |
|---|---|---|---|---|---|---|
| ETTh1 | 96 | 5 | 0.4360+/-0.0006 | 0.4360+/-0.0004 (+0.02%) | 0.4361+/-0.0002 (+0.03%) | RandomSP=3/5, RandomSPNoShuffle=3/5 |
| ETTh1 | 720 | 5 | 0.6604+/-0.0028 | 0.6545+/-0.0034 (-0.90%) | 0.6534+/-0.0015 (-1.07%) | RandomSP=4/5, RandomSPNoShuffle=5/5 |
| ETTm2 | 96 | 5 | 0.1481+/-0.0011 | 0.1482+/-0.0009 (+0.04%) | 0.1486+/-0.0004 (+0.30%) | RandomSP=3/5, RandomSPNoShuffle=2/5 |
| ETTm2 | 720 | 5 | 0.2708+/-0.0060 | 0.2706+/-0.0014 (-0.08%) | 0.2714+/-0.0033 (+0.23%) | RandomSP=2/5, RandomSPNoShuffle=2/5 |

## Aggregate across all (dataset, horizon, seed) cells

| variant | n_cells | mean_delta_mse_pct | std_delta_mse_pct | n_seeds_beating_baseline |
|---|---|---|---|---|
| SRSNet_RandomSP | 20 | -0.22 | 1.23 | 12/20 |
| SRSNet_RandomSPNoShuffle | 20 | -0.12 | 1.46 | 12/20 |

## Interpretation guidance

- A mean delta MSE close to 0 with random controls means the learned
  scorer is not contributing in this regime.
- A clearly positive mean delta MSE (random worse than SRSNet) supports
  the paper's selectivity claim with a stronger negative control than
  Table 4's NoSP variant.
- A negative mean delta MSE (random better than SRSNet) is a refutation
  but requires checking seed-level variance before reporting.
- Conclusions are scoped to the tested (datasets, horizons, seeds, hardware)
  and do not generalize to other patch-based models.
