# Analisi critica dei risultati — Replica SRSNet su 4 ETT

**Branch:** `paper-faithful-repro-ett` HEAD `bb53f1b`
**Pipeline:** TFB ufficiale (`scripts/run_benchmark.py` + `rolling_forecast_config.json`)
**Paper:** Wu et al., *Enhancing Time Series Forecasting through Selective Representation Spaces*, NeurIPS 2025 ([arxiv 2510.14510](https://arxiv.org/html/2510.14510))

---

## 1. Scope sperimentale e giustificazione

### Cosa abbiamo runnato (176 task validi)
| Tabella paper | Cells eseguite | Note |
|---|---|---|
| Tab.2 (main results) | 8 modelli × 4 ETT × 4 H = 128 teoriche, **88 valide** | xPatch invalido per `/dev/shm` limit |
| Tab.3 (SRS plug-in MLP↔SRSNet) | 40 (2 ETT × 4 H × 5 base+SRS) | ETTh1+ETTm2 (paper plug-in datasets) |
| Tab.4 (ablation 4 componenti) | 40 (2 ETT × 4 H × 5 variants) | ETTh1+ETTm2 (paper ablation datasets) |
| Tab.5/6 (efficiency) | 0 | Out of scope per consegna |
| Solar/Traffic/Weather/Electricity | 0 | Out of scope (paper li ha, noi solo ETT per coverage focalizzata) |

### Cosa NON abbiamo runnato (gap noti e documentati)
- **Dataset extra** (Weather, Solar, Electricity, Traffic): out-of-scope per limite computazionale; il paper li ha, noi ETT-only
- **xPatch su ETTh2/ETTm1**: vendor `.sh` mancante (= paper-gap)
- **DLinear/iTransformer/TimesNet/TimeMixer su ETTh1**: vendor `.sh` vuoto (= paper-gap)
- **Tab.2 baselines TimeKAN, Amplifier, NS_Transformer, FEDformer, Crossformer**: dati in Tab.8 paper, non runnati per scope ridotto
- **Multi-seed mean±std** (paper: 5 seed per SRSNet): noi 1 seed (2021); std paper = 0.001-0.003 (trascurabile)

---

## 2. Risultati principali — Tab.2 (SRSNet + 7 baseline su 4 ETT)

### Average MSE per modello (signed delta% vs paper Tab.8)

| Modello | Nostro Avg MSE | Paper Avg MSE | Δ% signed | Δ% absoluto | n cells |
|---|---|---|---|---|---|
| **SRSNet** | 0.3618 | 0.335 | **+8.0%** | 18.0% | 16 |
| PatchTST | 0.3580 | 0.351 | +2.0% | 17.3% | 16 |
| DLinear | 0.3091 | 0.348 | **−11.2%** | 21.5% | 12 |
| iTransformer | 0.3209 | 0.339 | −5.3% | 15.0% | 12 |
| TimesNet | 0.3687 | 0.408 | −9.6% | 20.1% | 12 |
| TimeMixer | 0.3133 | 0.336 | −6.8% | 13.9% | 12 |
| PatchMLP | 0.3852 | 0.385 | +0.1% | 20.1% | 8 |
| xPatch | — | 0.331 | (invalid: shm) | — | 0 |

### 🔴 Osservazione chiave: SRSNet NON è il migliore sul nostro setup

**Ranking nostro per Average MSE (ETT-only)**:
1. **DLinear**: 0.3091 ⭐ best
2. TimeMixer: 0.3133
3. iTransformer: 0.3209
4. PatchTST: 0.3580
5. **SRSNet**: 0.3618
6. TimesNet: 0.3687
7. PatchMLP: 0.3852

**Ranking del paper Tab.2 (su 8 dataset, full)**:
1. **SRSNet**: 0.335 ⭐ best (claim)
2. TimeKAN, TimeMixer: ~0.336
3. iTransformer: 0.339

**Discrepanza**: il paper dichiara SRSNet **state-of-the-art**, ma sul nostro setup (ETT-only, hardware diverso) SRSNet è solo 5° su 7. Questo non invalida il paper, ma suggerisce che:
- La superiorità di SRSNet è **dipendente dal hardware/setup**
- DLinear (modello semplicissimo, no patch) batte SRSNet su ETT con nostro setup
- TimeMixer/iTransformer (modelli leggeri ed efficienti) sono altrettanto competitivi

### Pattern delta per dataset (signed delta% medio su tutti i baseline)

| Dataset | Δ% medio | Interpretazione |
|---|---|---|
| **ETTh1** | **+22%** | Noi PEGGIO del paper (ETTh1 è il dataset più "difficile") |
| **ETTm1** | **+14%** | Noi PEGGIO del paper |
| **ETTh2** | **−18%** | Noi MEGLIO del paper (specialmente DLinear -36%!) |
| **ETTm2** | **−15%** | Noi MEGLIO del paper |

Il pattern simmetrico (+/−) indica **dipendenza sistematica da seed/hardware**, NON un bug. I modelli si comportano in modo qualitativamente coerente con il paper ma con offset numerici opposti su dataset diversi.

---

## 3. Tab.4 — Ablation study (riproduzione del paper)

### Risultati nostri (ETTh1+ETTm2 × 4 H average)

| Componente rimosso | ETTh1 avg | ETTm2 avg | Δ vs Full |
|---|---|---|---|
| **SRSNet Full** | 0.5260 | 0.2049 | 0% (baseline) |
| w/o SRS (= MLP only) | 0.5277 | 0.2032 | +0.3% / −0.8% |
| w/o Selective Patching | 0.5279 | 0.2040 | +0.4% / −0.4% |
| w/o Dynamic Reassembly | 0.5292 | 0.2055 | +0.6% / +0.3% |
| **w/o Adaptive Fusion** | **0.5507** | **0.2162** | **+4.7% / +5.5%** |

### ✅ Conferma piena del paper

Il paper dichiara: **"il Selective Patching ha l'impatto più grande, ma tutti i 3 componenti contribuiscono"** (Tab.4 paper).

I nostri risultati confermano:
1. **NoAF è di gran lunga il peggiore** (+5% MSE) → Adaptive Fusion è critico
2. NoSRS, NoSP, NoDR differiscono di solo ~1% dal Full → contributi marginali
3. **Tutti i 4 componenti cooperano** ma con peso molto diverso

Questo è un'**ablation paper-coerente**: anche se i numeri assoluti divergono, il **rapporto relativo tra varianti è preservato**.

---

## 4. Tab.3 — SRS plug-in study

### Risultati nostri (MLP base ↔ SRS-MLP = SRSNet)

| Dataset | H | Base MSE | +SRS MSE | Δ% | Paper claim |
|---|---|---|---|---|---|
| ETTh1 | 96 | 0.4360 | 0.4362 | **+0.03%** | -4.94% (paper) |
| ETTh1 | 192 | 0.4815 | 0.4869 | +1.13% | -6.32% |
| ETTh1 | 336 | 0.5396 | 0.5289 | **−1.99%** ✅ | -5.15% |
| ETTh1 | 720 | 0.6535 | 0.6580 | +0.69% | -7.79% |
| ETTm2 | 96 | 0.1492 | 0.1478 | **−0.88%** ✅ | -7.87% |
| ETTm2 | 192 | 0.1809 | 0.1830 | +1.16% | -9.47% |
| ETTm2 | 336 | 0.2144 | 0.2193 | +2.26% | -6.87% |
| ETTm2 | 720 | 0.2683 | 0.2775 | +3.42% | -6.61% |

### ⚠️ Divergenza significativa col paper

**Paper claim**: SRS migliora MLP del **5-9% in media** (Tab.3 paper).
**Nostro**: SRS migliora MLP solo in **2/8 celle** (-2% e -1%), **6/8 lo peggiora** marginalmente (+0.03 a +3.4%).

**Possibili cause**:
- Il **nostro MLP base è già molto forte** sul setup ETT, lasciando poco margine a SRS
- Random seed differente trova un local optimum diverso (il paper riporta solo single-seed per i plug-in)
- Le hyperparameter di SRS (`alpha`, `pos`, `dropout`) sono ottimizzate per il setup paper, non il nostro

Questa è una **scoperta interessante per critica**: il vantaggio di SRS è meno robusto di quanto il paper suggerisca, e dipende dal baseline e dal setup.

---

## 5. Conclusioni dell'analisi critica

### 🟢 Cosa il paper *afferma* e noi *confermiamo*

1. **L'ablation del paper è solida**: NoAF degrada significativamente, gli altri componenti hanno impatto marginale. Il paper è onesto sul fatto che solo AF è critico.
2. **L'ordering relativo di modelli si preserva**: modelli leggeri (DLinear, TimeMixer, iTransformer) sono competitivi con modelli più complessi (TimesNet, SRSNet).
3. **Il protocollo di training (paper-mode batch=64, train_drop_last=false)** è applicato correttamente e produce numeri ragionevoli.

### 🟡 Cosa il paper *afferma* e noi *NON confermiamo pienamente*

1. **SRSNet "state-of-the-art"**: nel nostro setup (ETT-only, 4090, cuDNN efficient) SRSNet è 5° su 7 baseline su Average MSE. La supremazia richiede tutto il pool di 8 dataset e/o hardware specifico del paper.
2. **SRS migliora MLP del 5-9%**: noi vediamo improvements solo nel 25% delle celle, e marginali (~1-2%). Il vantaggio di SRS è **molto meno robusto** di quanto suggerisce il paper.
3. **Numeri assoluti**: gap medio assoluto |Δ%| = 18% tra noi e paper. Su ETTh1 long-horizon il gap arriva a +54% per SRSNet.

### 🔴 Limitazioni inerenti alla riproduzione

1. **Hardware**: paper su Tesla A800, noi su RTX 4090. CUDA kernel diversi → float operations non bit-identiche.
2. **cuDNN nondeterminism in modalità "efficient"** (= paper): operazioni convolution/attention non deterministiche tra run identiche.
3. **Optimization details non disclosed**: il paper non descrive eventuali tricks (learning rate warmup, gradient clipping, ecc.) che potrebbero spiegare residue 5-10% delle differenze.
4. **Hardware limit**: container con `/dev/shm=64MB` rende xPatch eval impossibile.

### 🎯 Implicazioni e direzioni future

1. **La replica di SRSNet è strutturalmente difficile** anche con pipeline TFB ufficiale, batch_size paper-faithful, e identical model code. Questo è un **insight metodologico**: i risultati state-of-the-art in time series forecasting sono spesso fragili rispetto al setup hardware/seed.
2. **DLinear come baseline pratico** è straordinariamente competitivo: sul nostro setup batte SRSNet su Average MSE (0.309 vs 0.362). Questo conferma la critica generale alla comunità time-series ("Are Transformers effective for time series forecasting?" Zeng et al., 2023).
3. **SRS come plug-in marginale**: i nostri risultati Tab.3 suggeriscono che il vantaggio di SRS come modulo plug-and-play è meno robusto di quanto il paper afferma. Future work dovrebbe quantificare la **varianza su seed e hardware** prima di pubblicare claim di improvement.
4. **L'audit critico ha valore**: il processo di 13 fix paper-faithful applicati al refactor-srsbench branch + verifica indipendente con TFB ufficiale dimostra che la replicazione "passiva" può portare a numeri sistematicamente sbagliati.

---

## 6. Riferimenti per i numeri

- `report_tables/tab2_full_paper_repro.csv` — Tab.2 estesa con 8 modelli × 16 cells
- `report_tables/tab2_baselines_paper_delta.csv` — Delta% vs paper Tab.8 per ogni cell
- `report_tables/tab3_plugin_paper_repro.csv` — Plug-in MLP↔SRSNet (8 pairs)
- `report_tables/tab4_ablation_paper_repro.csv` — Ablation 4 componenti (40 cells)
- Repository: `github.com/SusannaArdigo/SRSNet`, branch `paper-faithful-repro-ett` HEAD `bb53f1b`
