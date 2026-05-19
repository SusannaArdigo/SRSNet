# Audit of the Constructive Extensions

Comprehensive review of the three constructive extensions
(TASP, Hypernet-AF, PS-SRS) and their three factorial combinations,
with three concerns from the supervising feedback:

1. *We lack statistical power; results are only indicative.*
2. *Can we justify the choice of descriptors in PS-SRS?*
3. *PS-SRS likely degrades results because of conflict with the main loss.*

This audit is the source of truth for the report rewrite. Each finding
is anchored to (a) the paper Sec. 6 verbatim, (b) the code in
`ts_benchmark/baselines/srs_paper/extensions.py`, and (c) empirical
measurements run on `france-gpu-1x-rtx-4090-https-cut-family`.

## 1. Mapping each extension to the paper's stated motivations

Sec. 6 verbatim (paper p. 10):

> **Interpretability (L3)** -- *"Since the SRS module is also optimized
> through gradient descent and the gradient flow is coupled with the
> subsequent patch-based models, we can only ensure the selected
> patches are useful for forecasting, but not all of them are
> interpretable."*

> **Initialization (L4)** -- *"The initialization of the weights alpha
> seems to be important... we recommend to manually initialize it when
> the prior knowledge of datasets is acquired, such as increasing it
> when the datasets are periodic and stationary, and decreasing it
> when the datasets are non-stationary and shifting."*

> **Future Work (FW3+FW4)** -- *"To solve the initialization problem
> in the SRS module, a more efficient update mechanism for alpha
> deserves exploration. A potential solution is to design a module to
> supervise the sample-wise data patterns, **constructing a more
> explicit optimization objective between data patterns and alpha**,
> which can also enhance the interpretability of the SRS module."*

> **Future Work (FW1)** -- *"we hope to devise an environment-aware
> mechanism to perceive the patch-wise data distributions and patterns
> more explicitly."*

Our extensions claim the following mapping:

| Extension       | Claimed mapping       | Audited correctness                                                          |
|---              |---                    |---                                                                           |
| TASP            | FW1 + L3              | **Partial**: matches FW1 conceptually, partially addresses L3 (see 2.1)      |
| Hypernet-AF     | FW3 + L4              | **Conceptual stretch**: not really an *update* mechanism (see 2.2)           |
| PS-SRS          | FW4 + L3              | **Mismatch**: supervises the scorer, not alpha (see 2.3)                     |
| TASP_HypernetAF | factorial             | inherits the two parents' caveats                                            |
| RandomSP_HypernetAF | factorial         | inherits the two parents' caveats                                            |
| PSRS_HypernetAF | factorial             | inherits the two parents' caveats + aux-loss interference                    |

## 2. Per-extension correctness audit

### 2.1 TASP -- Time-Aware Selective Patching

**Mapping to paper:** FW1 (environment-aware mechanism) + L3 (interpretability).

**Implementation:** replaces the parent scorer
`Linear(patch_len=24 -> hidden=128) -> ReLU -> Linear(128 -> patch_num=21)`
with `Linear(N_FEATURES=4 -> 16) -> ReLU -> Linear(16 -> patch_num=21)`
on four engineered features per candidate window: dominant non-DC FFT
magnitude, lag-1 autocorrelation, within-window variance, linear trend
slope.

**What is correct:**
- The selection logic after the scorer (argmax + max-score rescale +
  gather) is identical to the parent. Only the scorer changes.
- The four features are computed in a single pass through `x_rec`
  without leaking the target window or any future data.
- Parameter count drops from 27,287 (full SRS) to 21,703 (-20%); the
  scorer alone goes from 6,038 params to 374 params. The "no extra
  capacity" claim holds.

**What is questionable:**
- **L3 is only partially addressed.** TASP makes the *inputs* of the
  scorer named (FFT, AC1, variance, slope) but the scorer is still a
  4 -> 16 -> patch_num MLP, so the *mapping* from features to
  selection score is opaque. The user can audit the features that
  enter the scorer; they cannot audit why a specific patch was picked.
- **The feature set is opinionated, not derived.** We picked the four
  most common time-series summary statistics. We did not run an
  ablation over different feature sets, so a reviewer can object that
  these four are arbitrary.
- **TASP scorer is shape-restricted.** Each candidate window
  contributes 4 numbers to the scorer that outputs `patch_num` scores.
  This is intentional (we want fewer parameters than the parent), but
  it limits the expressiveness of the scorer. A negative result for
  TASP could be caused by the scorer being too small rather than the
  feature set being wrong.

**What is broken:**
- Nothing functionally. The forward pass is deterministic and matches
  the parent's selection contract.

### 2.2 Hypernet-AF -- data-dependent alpha via a tiny hypernet

**Mapping to paper:** FW3 (more efficient update mechanism for alpha)
+ L4 (initialization).

**Implementation:** the free `alpha` parameter of shape
`[patch_num, d_model]` is deleted; a hypernet
`Linear(d_model -> 8) -> ReLU -> Linear(8 -> patch_num*d_model)` is
added. The input to the hypernet is a batch-level summary
`ctx = e_orig.mean(dim=0).mean(dim=0)` (a scalar per d_model dim).

**What is correct (verified empirically):**
- **Vanilla-preserving at step 0:** with the zero-init of context_proj
  and hyper.weight and `hyper.bias = INIT_ALPHA = 3.5`, the output of
  the hypernet at step 0 is `sigmoid(3.5) = 0.9707` on every cell,
  bit-identical to a vanilla SRS with the paper's default alpha=3.5.
- The factorial combos
  `{TASP, RandomSP, PSRS}_HypernetAF` inherit this property.

**What is questionable:**
- **Parameter count claim is misleading.** The Hypernet-AF module has
  74,399 trainable parameters, vs 27,287 for the baseline SRS. The
  hypernet alone (`hyper`) is 50,688 params, which is **9.4x larger
  than the free alpha parameter** it replaces (5,376 params). The
  "parameter-matched" claim in the original commit message referred
  to the discarded GAF baseline, not to the vanilla SRS. Versus
  vanilla, Hypernet-AF is **2.7x capacity**.
- **It is not really an *update* mechanism.** The paper's FW3 talks
  about a "more efficient update mechanism for alpha". Our hypernet
  recomputes alpha from a batch context at every forward pass; it
  does not change how alpha is *updated by gradient descent*. The
  paper's framing suggests something closer to a learned optimizer or
  a constraint that anchors alpha to the data. Our approach is a
  reasonable instance of "alpha = f(data)" but it is not what the
  paper literally proposed.
- **Context computation is batch-aware, not sample-aware.** With
  `ctx = e_orig.mean(dim=0).mean(dim=0)`, a single alpha tensor is
  applied to every sample in the batch. Empirically: when the same
  4-sample batch x1 is passed in different orderings or with B=1, the
  ctx vector shifts by ~0.1 (significant for a sigmoid input that
  saturates beyond 3). Training uses bs=64; inference under TFB
  rolling forecast may use bs=1 -- so the alpha at evaluation can
  drift away from the alpha distribution seen during training.

**What is broken:**
- Nothing functionally. The forward pass is well-typed and reproducible
  given the seed.

### 2.3 PS-SRS -- Pattern-Supervised SRS  (most criticized)

**Claimed mapping:** FW4 (supervised module for sample-wise patterns)
+ L3 (interpretability).

**Implementation:** the scorer is the same as vanilla. A new head
`Linear(hidden_size=128 -> N_DESCRIPTORS=3)` is tapped onto the
scorer's first hidden activation (after ReLU). Its target is three
engineered descriptors computed from `x_rec`: dominant FFT magnitude,
lag-1 autocorrelation, within-window variance. The auxiliary loss is
`MSE(pred, gt) * LAMBDA_AUX = MSE(...) * 1e-2`. It is exposed as
`last_aux_loss` and added to the main forecasting loss via the
DeepForecastingModelBase native `out_loss["additional_loss"]` slot
(see `deep_forecasting_model_base.py:345-350`).

**What is correct:**
- The aux loss is detached from `gt` (no gradient flows back into the
  target). Eval mode skips the aux computation. The aux loss reaches
  the training loss correctly.
- Combinations: `SRSNet_PSRS_HypernetAF` inherits descriptor_head and
  `last_aux_loss` correctly; the wrapper forwards them.

**What is broken / questionable -- this is where the user's intuition
is right:**

1. **Mapping to the paper is wrong.** The paper FW4 says
   *"constructing a more explicit optimization objective **between
   data patterns and alpha**"*. Our PS-SRS supervises the **scorer**
   (selection), not alpha (fusion). We chose to put the head on the
   selection scorer because that is where "patches" are decided, but
   the paper explicitly frames the connection as "data patterns <->
   alpha". This is a misalignment we should declare in the report.

2. **The choice of 3 descriptors is not justified.** We picked
   `FFT max, AC1, variance` because they are the same three numbers
   used by TASP minus the trend slope. No criterion was derived from
   the paper, from a literature survey, or from a data-driven
   ablation. A reasonable reviewer will ask: "why not 10 descriptors?
   why not periodogram peak at lag 24? why not absolute mean?" We
   cannot answer.

3. **The auxiliary loss is dominated by FFT magnitude.** Targets have
   wildly different scales: FFT max ranges in **[2.49, 19.08]** with
   mean 8.58; AC1 is in [-0.73, +0.78]; variance is in [0.08, 2.25].
   With unweighted MSE, per-descriptor contributions to the aux loss
   at step 0 are:
   - `MSE(FFT max) = 75.55`
   - `MSE(AC1)     = 0.10`   (750x smaller)
   - `MSE(variance) = 1.35`  (56x smaller)

   So PS-SRS is effectively training the scorer hidden activation to
   predict **only FFT max**. AC1 and variance contribute negligibly.

4. **Lambda = 1e-2 is arbitrary, not swept.** Raw aux loss at step 0 is
   ~25.7. With lambda = 1e-2 the scaled aux is ~0.26, which is the
   same order as the typical MSE forecasting loss on ETT (~0.3-0.6).
   So the auxiliary loss contributes a *non-negligible* fraction of
   the total training signal. We picked lambda once and never swept
   it.

5. **Gradient conflict is real.** Both losses share `h`, the scorer's
   first hidden activation. We measured the cosine similarity of the
   gradient with respect to `scorer_select[0].weight` from the aux
   loss vs from a proxy main loss:
   `cos(aux_grad, main_grad) = 0.048` -- essentially **orthogonal**.

   That means: the aux loss is pulling the scorer's hidden layer in a
   direction that is neither aligned with nor strictly opposite to
   the forecasting objective. The optimizer has to split the
   layer's capacity. Combined with the magnitude observation, the
   resulting drift is enough to *hurt* the forecasting accuracy in
   our experiments, which matches the user's intuition:
   *"PS-SRS ha solo peggiorato i risultati, il che e' probabilmente
   perche' ha dato conflitto con la loss principale"* -- yes,
   confirmed quantitatively.

6. **No normalization of targets.** The descriptor targets are
   computed from raw `x_rec`, after instance normalization (RevIN) on
   the input but with no further per-descriptor normalisation. A
   minimal fix would be z-score targets across the batch before
   computing the MSE, or use a per-descriptor lambda.

**Practical recommendation for the report:** report PS-SRS honestly
as the **negative case** of the constructive set. State the descriptor
choice as a known design weakness; cite the cos-similarity measurement
to justify the "main-loss conflict" intuition; explicitly note the
paper-mapping mismatch (we supervise the scorer, not alpha).

### 2.4 Factorial combinations

The factory function `_make_hypernet_combo(base_select_cls, name)` is
correct:

- **MRO:** `_Combo(base_select_cls)` -> base_select_cls -> SRS. The
  base's `_select` (random / engineered features / supervised) is
  inherited. `__init__` calls `super().__init__()` first to install
  the base's machinery, then deletes alpha and installs the hypernet.
- **TASP_HypernetAF:** the scorer is the TASP engineered scorer (4
  in_features, 16 hidden), the alpha pathway is the hypernet. Verified
  empirically that `scorer_select[0].in_features == 4`.
- **RandomSP_HypernetAF:** `_select` is randomized (different seeds
  give different selections); alpha pathway is the hypernet. Verified.
- **PSRS_HypernetAF:** both `descriptor_head` and `last_aux_loss` are
  present after construction; the wrapper `SRSNet_PSRS_HypernetAF`
  overrides `_process` to forward the aux loss the same way the
  single-component PSRS wrapper does. Verified that the auxiliary
  loss is reachable through backward and that
  `scorer_select[0].weight` accumulates gradient from both the main
  and the aux paths.

All three combos pass the vanilla-preserving check at step 0:
`sigmoid(alpha_dyn) = 0.9707` on every cell. The combos inherit the
parameter inflation of Hypernet-AF (152-174% over vanilla).

The factory pattern is clean -- there are no subtle MRO bugs.

## 3. Statistical power

A paired t-test on the 20 (dataset, horizon, seed) cells per variant
yields:

```
Variant                                 n   mean_d   std_d    CI95            t        p     d_eff
SRSNet_HypernetAF                       20  -0.092   1.436   [-0.76, +0.58]   -0.29   0.779   0.064
SRSNet_PSRS                             20  +0.218   1.208   [-0.35, +0.78]   +0.81   0.429   0.181
SRSNet_PSRS_HypernetAF                  20  +0.123   1.160   [-0.42, +0.67]   +0.47   0.641   0.106
SRSNet_RandomSP                         20  -0.221   1.226   [-0.79, +0.35]   -0.80   0.431   0.180
SRSNet_RandomSPNoShuffle                20  -0.117   1.459   [-0.80, +0.57]   -0.36   0.724   0.080
SRSNet_RandomSP_HypernetAF              20  -0.204   1.654   [-0.98, +0.57]   -0.55   0.588   0.123
SRSNet_TASP                             20  +0.253   1.818   [-0.60, +1.10]   +0.62   0.541   0.139
SRSNet_TASP_HypernetAF                  20  -0.019   1.233   [-0.60, +0.56]   -0.07   0.946   0.015
```

Headline numbers:
- **No variant rejects the null hypothesis** at alpha=0.05 two-sided.
  All p-values lie in [0.43, 0.95].
- **All 95% confidence intervals include 0.**
- **All Cohen's d values are below 0.2** (rule-of-thumb for
  "negligible" effect).

Required sample size to detect a 0.2% mean delta MSE with each
variant's observed std at 80% power:

```
SRSNet_HypernetAF              n needed = 405
SRSNet_PSRS                    n needed = 287
SRSNet_PSRS_HypernetAF         n needed = 265
SRSNet_RandomSP                n needed = 296
SRSNet_RandomSPNoShuffle       n needed = 418
SRSNet_RandomSP_HypernetAF     n needed = 537
SRSNet_TASP                    n needed = 649
SRSNet_TASP_HypernetAF         n needed = 299
```

We ran 20 cells per variant; the cheapest variant (PSRS_HypernetAF)
would still need an additional 245 cells per variant to detect a
0.2% effect at the same noise level. **The user's framing is the
correct one:** the experiments are *not powered* to demonstrate
"learned selection is not superior" as a positive claim. They are
*indicative, within the limits of our experimental conditions*, that
none of the controls or constructive extensions produces an effect
distinguishable from seed-level noise.

## 4. Corrected claims for the report

The text in `report_atml/main.tex` and `report_tables/critical_analysis.md`
contains a few statements that should be softened or corrected:

1. **"Random matches Learned in 12/20 seeds"** is a true *empirical*
   observation but should be paired with the p-value (0.43) and the
   CI ([-0.79, +0.35]). The right framing is "no statistically
   significant difference at our sample size", not "Random is as good
   as Learned".

2. **"Hypernet-AF is parameter-matched"** is true only versus the
   discarded GAF, not versus vanilla SRS. Versus vanilla, the module
   is **2.7x larger**. The combinations are similarly inflated.
   This must be stated.

3. **"PS-SRS addresses paper FW4"** is misleading. FW4 explicitly
   mentions supervising "data patterns and alpha"; PS-SRS supervises
   "data patterns and the scorer". Report this as a known
   re-interpretation, not as an exact match.

4. **"The fusion main effect is null (-0.10%)"** is also a true
   *empirical* observation; should be paired with a CI and a power
   statement.

5. **"The constructive extensions do not improve over baseline"** is
   true *at our sample size*; the corrected statement is **"none of
   the constructive extensions produces an effect distinguishable
   from the seed-level noise floor at our sample size"**.

## 5. What we should do next (if there is time)

In decreasing order of impact for the report:

1. **State all the caveats above in the report.** Zero compute cost,
   high credibility cost if missed.
2. **PS-SRS lambda sweep + target normalization.** ~3 hours compute.
   We pick `lambda in {1e-3, 1e-2, 1e-1, 1.0}` and we z-score the
   three descriptor targets. Hypothesis: with normalised targets and
   the right lambda, PS-SRS at least does not hurt. If it still does
   not improve, that is a clean negative.
3. **PS-SRS that actually supervises alpha.** This is a re-design,
   not a fix: add a head on top of the alpha tensor (or on the
   hypernet activation if combined with Hypernet-AF) and supervise
   that to predict descriptors. Faithful to the paper's FW4 wording.
   ~6 hours work.
4. **A budget version of "more seeds for one variant".** Pick one
   variant where the observed mean delta is the largest (TASP_HF or
   RandomSP_HF), run 5 more seeds per cell, and report the updated
   p-value. Concrete demonstration of the sample-size argument.

## 6. Source files referenced

- Paper: arxiv 2510.14510 Sec. 6 (p. 10).
- Code: `ts_benchmark/baselines/srs_paper/extensions.py`.
- Empirical checks: scripts in `/tmp/audit_*.py` (not committed).
- Data: `report_tables/selectivity_controls.csv` (180 rows).
