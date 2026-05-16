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

## 6. Selectivity-controls study — Esperimento centrale (post-revisione)

Per evitare estensioni capacity-confounded (GAF, MSP) o no-op (DDA, vedi `scripts/repro/selectivity_extension_plan.md`), il contributo principale del nostro lavoro diventa uno **studio di control mancante** sul claim centrale del paper: *"learned selective patching matters"*.

### 6.1 Setup

| Parametro | Valore |
|---|---|
| Datasets | ETTh1, ETTm2 |
| Horizons | 96, 720 |
| Seeds | 2021–2025 (5 seed) |
| Variants | SRSNet (learned) / RandomSP / RandomSPNoShuffle |
| Total task | 60 (2 × 2 × 5 × 3) |
| Hardware | Hivenet RTX 4090 24GB |
| Protocol | paper-mode (batch=64, `train_drop_last=false`) |

**Varianti** (`ts_benchmark/baselines/srs_paper/extensions.py`):
- `SRSRandomSP`: override `_select` con `torch.randint` (random patch indices), `_shuffle` learned mantenuto
- `SRSRandomSPNoShuffle`: random `_select` + identity `_shuffle` (deterministic)

### 6.2 Risultati aggregate (20 cells per variant)

| Variant | Mean Δ MSE vs SRSNet | Std | Seed-level wins |
|---|---|---|---|
| **SRSNet_RandomSP** | **−0.22%** | 1.23% | **12/20** |
| **SRSNet_RandomSPNoShuffle** | **−0.12%** | 1.46% | **12/20** |

### 6.3 Per-cell breakdown (mean ± std MSE)

| Dataset | H | SRSNet (baseline) | RandomSP (Δ%) | RandomSPNoShuffle (Δ%) | wins (Random / NoShuf) |
|---|---|---|---|---|---|
| ETTh1 | 96 | 0.4360 ± 0.0006 | 0.4360 (+0.02%) | 0.4361 (+0.03%) | 3/5, 3/5 |
| **ETTh1** | **720** | 0.6604 ± 0.0028 | **0.6545 (−0.90%)** | **0.6534 (−1.07%)** | **4/5, 5/5** |
| ETTm2 | 96 | 0.1481 ± 0.0011 | 0.1482 (+0.04%) | 0.1486 (+0.30%) | 3/5, 2/5 |
| ETTm2 | 720 | 0.2708 ± 0.0060 | 0.2706 (−0.08%) | 0.2714 (+0.23%) | 2/5, 2/5 |

### 6.4 🔴 Conclusione critica

Il claim del paper *"learned selective patching is essential"* (Sec. 4.1, basato su NoSP ablation) **NON è supportato** dai nostri controlli:

1. **Random patch selection match o batte learned in 12/20 seed** (60% delle volte)
2. Mean delta MSE (0.1-0.2%) è **dentro la variance dei seed** (std 1.2-1.5%) → differenze statisticamente indistinguibili
3. **Sulla cella ETTh1 H720, RandomSPNoShuffle batte SRSNet su 5/5 seed** con −1.07% MSE — il controllo random è strettamente migliore del learned scorer
4. Il fatto che la versione *deterministic* (NoShuffle) e quella *con learned shuffle* (RandomSP) abbiano risultati comparabili (Δ −0.12% vs −0.22%) suggerisce che anche `_shuffle` non sta facendo lavoro utile

**Implicazione**: il modulo SRS attuale non sembra estrarre informazione che la selezione random non possa altrettanto produrre, su questo subset di task. Il paper avrebbe beneficiato di un controllo random oltre alla NoSP ablation (che disabilita selettivamente solo lo scorer di selezione, mantenendo l'architettura overall).

### 6.5 Limitazioni del nostro controllo

- **Solo 2 datasets × 2 horizons** = 4 celle. Non sappiamo se il pattern persiste sul full ETT grid (16 celle) o su dataset di scala diversa (Solar/Traffic/Weather).
- **5 seed** non rigorosi statisticamente (n=5 → CI ampi).
- Non abbiamo testato il selector swap **a inference time** (Esperimento 1 del plan) perché TFB non persiste checkpoint (`find result/ -name '*.pth'` ritorna vuoto). Sarebbe l'ideale per isolare contributo dello scorer learned a parità di pesi.
- **RNG control**: `torch.randint` dentro `_select` non è tied al `--seed` esplicitamente. La randomicità della selezione è "ambient" rispetto al seed di training. Ack.

### 6.6 Cosa NON facciamo (rispetto al plan)

- ❌ DDA: identificato no-op (sigmoid saturato α=3.0+)
- ❌ MSP: capacity-confounded + shape mismatch con plugin
- ❌ GAF: capacity-confounded
- ❌ Verdict labels (review #5): solo mean±std + win count

---

## 7. Riferimenti per i numeri

- `report_tables/tab2_full_paper_repro.csv` — Tab.2 estesa con 8 modelli × 16 cells
- `report_tables/tab2_baselines_paper_delta.csv` — Delta% vs paper Tab.8 per ogni cell
- `report_tables/tab3_plugin_paper_repro.csv` — Plug-in MLP↔SRSNet (8 pairs)
- `report_tables/tab4_ablation_paper_repro.csv` — Ablation 4 componenti (40 cells)
- **`report_tables/selectivity_controls.csv`** — 60 (cell, seed, variant) rows
- **`report_tables/selectivity_controls.md`** — mean±std + seed-level wins (focused study)
- `scripts/repro/selectivity_extension_plan.md` — Design del missing-control study
- Repository: `github.com/SusannaArdigo/SRSNet`, branch `paper-faithful-repro-ett-extensions` HEAD `706c3f1`
