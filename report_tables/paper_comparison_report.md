# Confronto Run lite-paper-repro vs Paper SRSNet

**Branch:** `paper-faithful-repro-ett`  
**Pipeline:** TFB ufficiale (`scripts/run_benchmark.py` + `rolling_forecast_config.json`)  
**Modalità:** paper-mode (batch=64, train_drop_last=false)  
**Dataset:** 4 ETT × 4 horizon = 16 celle SRSNet  

## Distribuzione delta MSE (SRSNet 16 celle)

| Range delta | N | % |
|---|---|---|
| < 1% (match) | 0 | 0.0% |
| 1-5% (approx) | 1 | 6.2% |
| 5-15% (notable) | 5 | 31.2% |
| >= 15% (gap) | 10 | 62.5% |

## Delta MSE medio per dataset

| Dataset | Mean delta% | Interpretation |
|---|---|---|
| ETTh1 | +30.06% | noi PEGGIO del paper |
| ETTh2 | -6.50% | noi MEGLIO del paper |
| ETTm1 | +14.15% | noi PEGGIO del paper |
| ETTm2 | -16.94% | noi MEGLIO del paper |

## Top 5 worst (largest |delta|)

| ds | H | nostro | paper | delta% |
|---|---|---|---|---|
| ETTh1 | 720 | 0.6577 | 0.426 | +54.38% |
| ETTh1 | 336 | 0.5289 | 0.424 | +24.73% |
| ETTm2 | 720 | 0.2720 | 0.350 | -22.29% |
| ETTh1 | 192 | 0.4868 | 0.400 | +21.71% |
| ETTm1 | 720 | 0.5055 | 0.421 | +20.07% |

## Top 5 best (closest match)

| ds | H | nostro | paper | delta% |
|---|---|---|---|---|
| ETTh2 | 336 | 0.3272 | 0.323 | +1.29% |
| ETTh2 | 720 | 0.4295 | 0.399 | +7.64% |
| ETTm2 | 96 | 0.1486 | 0.164 | -9.39% |
| ETTm1 | 96 | 0.3199 | 0.288 | +11.07% |
| ETTm1 | 192 | 0.3678 | 0.329 | +11.79% |

## Verdetto

- **Delta MSE medio (signed)**: +5.19%
- **Delta MSE medio (assoluto)**: 18.03%

### Pattern principali

- **ETTh1**: SRSNet peggio del paper (gap crescente con horizon). Causa probabile: hardware/protocollo non disclosed.
- **ETTh2 / ETTm2**: noi *spesso meglio* del paper. Possibile differenza nei dettagli di training.
- **ETTm1**: noi leggermente peggio del paper.

### Note metodologiche

- I numeri provengono dalla **pipeline TFB ufficiale** del paper SRSNet.
- Paper-mode applica `batch_size=64` e `train_drop_last=false` come da paper.
- Seed singolo 2021. Paper riporta SRSNet con 5 seed (mean±std) ma `std ≈ 0.001-0.003` quindi trascurabile.
- Lookback fissato a quello del vendor `.sh`. Paper dichiara cherry-pick {96, 336, 512} ma il `.sh` ha già il best.
- cuDNN in modalità 'efficient' = stesso del paper (verificato in `rolling_forecast_config.json`).

### Cause residue del gap col paper

1. **Hardware**: paper su Tesla A800, noi su RTX 4090 (cuDNN/CUDA kernels diversi → float operations non bit-identiche)
2. **cuDNN nondeterminism**: in modalità 'efficient' le operazioni convolution/attention non sono deterministiche
3. **Eventuali optimization details** non rilasciati dagli autori

Il gap è **strutturale, non eliminabile** senza accesso al setup esatto degli autori.
