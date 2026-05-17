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

| dataset | horizon | seeds | SRSNet (baseline) | SRSNet_RandomSP (delta% vs baseline) | SRSNet_RandomSPNoShuffle (delta% vs baseline) | SRSNet_HypernetAF (delta% vs baseline) | SRSNet_PSRS (delta% vs baseline) | SRSNet_PSRS_HypernetAF (delta% vs baseline) | SRSNet_RandomSP_HypernetAF (delta% vs baseline) | SRSNet_TASP (delta% vs baseline) | SRSNet_TASP_HypernetAF (delta% vs baseline) | wins_vs_baseline (per variant) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | 5 | 0.4360+/-0.0006 | 0.4360+/-0.0004 (+0.02%) | 0.4361+/-0.0002 (+0.03%) | 0.4361+/-0.0004 (+0.04%) | 0.4361+/-0.0006 (+0.04%) | 0.4363+/-0.0005 (+0.08%) | 0.4361+/-0.0001 (+0.04%) | 0.4365+/-0.0005 (+0.13%) | 0.4371+/-0.0006 (+0.26%) | RandomSP=3/5, RandomSPNoShuffle=3/5, HypernetAF=2/5, PSRS=3/5, HypernetAF=2/5, HypernetAF=2/5, TASP=3/5, HypernetAF=1/5 |
| ETTh1 | 720 | 5 | 0.6604+/-0.0028 | 0.6545+/-0.0034 (-0.90%) | 0.6534+/-0.0015 (-1.07%) | 0.6621+/-0.0019 (+0.25%) | 0.6622+/-0.0026 (+0.26%) | 0.6597+/-0.0015 (-0.12%) | 0.6547+/-0.0036 (-0.88%) | 0.6538+/-0.0040 (-1.00%) | 0.6546+/-0.0058 (-0.88%) | RandomSP=4/5, RandomSPNoShuffle=5/5, HypernetAF=2/5, PSRS=3/5, HypernetAF=3/5, HypernetAF=4/5, TASP=5/5, HypernetAF=3/5 |
| ETTm2 | 96 | 5 | 0.1481+/-0.0011 | 0.1482+/-0.0009 (+0.04%) | 0.1486+/-0.0004 (+0.30%) | 0.1477+/-0.0002 (-0.31%) | 0.1482+/-0.0005 (+0.03%) | 0.1477+/-0.0002 (-0.31%) | 0.1481+/-0.0002 (-0.01%) | 0.1491+/-0.0015 (+0.67%) | 0.1483+/-0.0003 (+0.15%) | RandomSP=3/5, RandomSPNoShuffle=2/5, HypernetAF=4/5, PSRS=3/5, HypernetAF=3/5, HypernetAF=3/5, TASP=2/5, HypernetAF=3/5 |
| ETTm2 | 720 | 5 | 0.2708+/-0.0060 | 0.2706+/-0.0014 (-0.08%) | 0.2714+/-0.0033 (+0.23%) | 0.2697+/-0.0025 (-0.40%) | 0.2722+/-0.0071 (+0.52%) | 0.2730+/-0.0074 (+0.83%) | 0.2707+/-0.0034 (-0.03%) | 0.2739+/-0.0034 (+1.16%) | 0.2717+/-0.0014 (+0.35%) | RandomSP=2/5, RandomSPNoShuffle=2/5, HypernetAF=3/5, PSRS=2/5, HypernetAF=2/5, HypernetAF=3/5, TASP=2/5, HypernetAF=2/5 |

## Aggregate across all (dataset, horizon, seed) cells

| variant | n_cells | mean_delta_mse_pct | std_delta_mse_pct | n_seeds_beating_baseline |
|---|---|---|---|---|
| SRSNet_RandomSP | 20 | -0.22 | 1.23 | 12/20 |
| SRSNet_RandomSPNoShuffle | 20 | -0.12 | 1.46 | 12/20 |
| SRSNet_HypernetAF | 20 | -0.09 | 1.44 | 11/20 |
| SRSNet_PSRS | 20 | +0.22 | 1.21 | 11/20 |
| SRSNet_PSRS_HypernetAF | 20 | +0.12 | 1.16 | 10/20 |
| SRSNet_RandomSP_HypernetAF | 20 | -0.20 | 1.65 | 12/20 |
| SRSNet_TASP | 20 | +0.25 | 1.82 | 12/20 |
| SRSNet_TASP_HypernetAF | 20 | -0.02 | 1.23 | 9/20 |

## Pairwise cross-comparison (mean delta MSE row vs column)

Each cell is the mean over all (dataset, horizon, seed) cells of the percentage delta MSE of the row variant relative to the column variant.  Positive values mean the row variant is worse than the column variant on average.

| variant | SRSNet | SRSNet_RandomSP | SRSNet_RandomSPNoShuffle | SRSNet_HypernetAF | SRSNet_PSRS | SRSNet_PSRS_HypernetAF | SRSNet_RandomSP_HypernetAF | SRSNet_TASP | SRSNet_TASP_HypernetAF |
|---|---|---|---|---|---|---|---|---|---|
| SRSNet | - | +0.24 | +0.14 | +0.11 | -0.20 | -0.11 | +0.23 | -0.22 | +0.03 |
| SRSNet_RandomSP | -0.22 | - | -0.10 | -0.12 | -0.43 | -0.34 | -0.01 | -0.46 | -0.20 |
| SRSNet_RandomSPNoShuffle | -0.12 | +0.10 | - | -0.02 | -0.32 | -0.23 | +0.09 | -0.36 | -0.10 |
| SRSNet_HypernetAF | -0.09 | +0.13 | +0.03 | - | -0.30 | -0.21 | +0.12 | -0.33 | -0.07 |
| SRSNet_PSRS | +0.22 | +0.45 | +0.35 | +0.32 | - | +0.11 | +0.44 | -0.01 | +0.25 |
| SRSNet_PSRS_HypernetAF | +0.12 | +0.36 | +0.26 | +0.23 | -0.08 | - | +0.35 | -0.10 | +0.15 |
| SRSNet_RandomSP_HypernetAF | -0.20 | +0.01 | -0.09 | -0.11 | -0.42 | -0.32 | - | -0.45 | -0.19 |
| SRSNet_TASP | +0.25 | +0.47 | +0.37 | +0.35 | +0.04 | +0.14 | +0.46 | - | +0.27 |
| SRSNet_TASP_HypernetAF | -0.02 | +0.20 | +0.10 | +0.08 | -0.23 | -0.14 | +0.20 | -0.26 | - |

## Factorial decomposition: Select x Fusion

Mean MSE across the small grid (lower is better).  Reads the interaction between the selection mechanism (rows) and the fusion mechanism (columns).

| Select \ Fusion | Free alpha | Hypernet alpha |
|---|---|---|
| Learned | 0.3788 | 0.3789 |
| Random | 0.3773 | 0.3774 |
| TASP | 0.3783 | 0.3779 |
| LearnedAux | 0.3797 | 0.3792 |

### Main effects (average effect of switching factor level)

| Factor | Level change | Mean delta MSE (paired across other factor) |
|---|---|---|
| Fusion | FreeAlpha -> Hypernet | -0.10% (std 1.28, n=80) |
| Select | Learned -> Random | -0.17% (std 1.02, n=40) |
| Select | Learned -> TASP | +0.17% (std 1.46, n=40) |
| Select | Learned -> LearnedAux | +0.22% (std 1.33, n=40) |

## Interpretation guidance

- A mean delta MSE close to 0 with random controls means the learned
  scorer is not contributing in this regime.
- A clearly positive mean delta MSE (random worse than SRSNet) supports
  the paper's selectivity claim with a stronger negative control than
  Table 4's NoSP variant.
- A negative mean delta MSE (random better than SRSNet) is a refutation
  but requires checking seed-level variance before reporting.
- The factorial decomposition isolates the Select factor and the Fusion
  factor.  If Fusion main effect is close to zero, Hypernet-AF does not
  meaningfully change the fusion behavior; if Select effects are all
  small, the choice of selector is largely irrelevant.
- Conclusions are scoped to the tested (datasets, horizons, seeds, hardware)
  and do not generalize to other patch-based models.
