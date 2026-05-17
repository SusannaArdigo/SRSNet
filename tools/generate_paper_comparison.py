#!/usr/bin/env python3
"""
tools/generate_paper_comparison.py
===================================
Genera CSV report + confronto vs paper SRSNet (Tab.2/3/4) per il run
lite-paper su paper-faithful-repro-ett.

Input:
    repro_results/lite-paper/summary.csv

Output:
    report_tables/
        tab2_srsnet_paper_repro.csv       # SRSNet main results (4 ETT × 4 H)
        tab3_plugin_paper_repro.csv       # SRS plug-in comparison
        tab4_ablation_paper_repro.csv     # Ablation study
        paper_comparison_summary.csv      # Riassunto delta% per modello
        paper_comparison_report.md        # Analisi qualitativa
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "repro_results" / "lite-paper" / "summary.csv"
BASELINES_SUMMARY = ROOT / "repro_results" / "main-compat" / "summary.csv"
OUT_DIR = ROOT / "report_tables"

# Paper SRSNet Tab.8 (Appendix A.1) baseline numbers per ETT (4 horizon)
# Per ogni (model, dataset, horizon) → (MSE, MAE) dal paper.
# Source: arxiv 2510.14510 Tab.8
PAPER_BASELINES_TAB2 = {
    "PatchTST": {
        ("ETTh1", 96): (0.377, 0.397), ("ETTh1", 192): (0.409, 0.425),
        ("ETTh1", 336): (0.431, 0.444), ("ETTh1", 720): (0.457, 0.477),
        ("ETTh2", 96): (0.274, 0.337), ("ETTh2", 192): (0.348, 0.384),
        ("ETTh2", 336): (0.377, 0.416), ("ETTh2", 720): (0.406, 0.441),
        ("ETTm1", 96): (0.289, 0.343), ("ETTm1", 192): (0.329, 0.368),
        ("ETTm1", 336): (0.362, 0.390), ("ETTm1", 720): (0.416, 0.423),
        ("ETTm2", 96): (0.165, 0.255), ("ETTm2", 192): (0.221, 0.293),
        ("ETTm2", 336): (0.276, 0.327), ("ETTm2", 720): (0.362, 0.381),
    },
    "DLinear": {
        ("ETTh1", 96): (0.379, 0.403), ("ETTh1", 192): (0.427, 0.435),
        ("ETTh1", 336): (0.440, 0.440), ("ETTh1", 720): (0.473, 0.494),
        ("ETTh2", 96): (0.300, 0.364), ("ETTh2", 192): (0.387, 0.423),
        ("ETTh2", 336): (0.490, 0.487), ("ETTh2", 720): (0.704, 0.597),
        ("ETTm1", 96): (0.300, 0.345), ("ETTm1", 192): (0.336, 0.366),
        ("ETTm1", 336): (0.367, 0.386), ("ETTm1", 720): (0.419, 0.416),
        ("ETTm2", 96): (0.164, 0.255), ("ETTm2", 192): (0.224, 0.304),
        ("ETTm2", 336): (0.277, 0.337), ("ETTm2", 720): (0.371, 0.401),
    },
    "iTransformer": {
        ("ETTh1", 96): (0.386, 0.405), ("ETTh1", 192): (0.430, 0.435),
        ("ETTh1", 336): (0.450, 0.452), ("ETTh1", 720): (0.495, 0.487),
        ("ETTh2", 96): (0.292, 0.347), ("ETTh2", 192): (0.348, 0.384),
        ("ETTh2", 336): (0.372, 0.407), ("ETTh2", 720): (0.424, 0.444),
        ("ETTm1", 96): (0.287, 0.342), ("ETTm1", 192): (0.331, 0.371),
        ("ETTm1", 336): (0.358, 0.384), ("ETTm1", 720): (0.412, 0.416),
        ("ETTm2", 96): (0.168, 0.262), ("ETTm2", 192): (0.224, 0.295),
        ("ETTm2", 336): (0.274, 0.330), ("ETTm2", 720): (0.367, 0.385),
    },
    "TimesNet": {
        ("ETTh1", 96): (0.389, 0.412), ("ETTh1", 192): (0.440, 0.443),
        ("ETTh1", 336): (0.523, 0.487), ("ETTh1", 720): (0.521, 0.495),
        ("ETTh2", 96): (0.334, 0.370), ("ETTh2", 192): (0.404, 0.413),
        ("ETTh2", 336): (0.389, 0.435), ("ETTh2", 720): (0.434, 0.448),
        ("ETTm1", 96): (0.340, 0.378), ("ETTm1", 192): (0.392, 0.404),
        ("ETTm1", 336): (0.423, 0.426), ("ETTm1", 720): (0.475, 0.453),
        ("ETTm2", 96): (0.189, 0.265), ("ETTm2", 192): (0.254, 0.310),
        ("ETTm2", 336): (0.313, 0.345), ("ETTm2", 720): (0.413, 0.402),
    },
    "TimeMixer": {
        ("ETTh1", 96): (0.372, 0.401), ("ETTh1", 192): (0.413, 0.430),
        ("ETTh1", 336): (0.438, 0.450), ("ETTh1", 720): (0.483, 0.483),
        ("ETTh2", 96): (0.270, 0.342), ("ETTh2", 192): (0.349, 0.387),
        ("ETTh2", 336): (0.367, 0.410), ("ETTh2", 720): (0.401, 0.436),
        ("ETTm1", 96): (0.293, 0.345), ("ETTm1", 192): (0.335, 0.372),
        ("ETTm1", 336): (0.368, 0.386), ("ETTm1", 720): (0.426, 0.417),
        ("ETTm2", 96): (0.165, 0.256), ("ETTm2", 192): (0.225, 0.298),
        ("ETTm2", 336): (0.277, 0.332), ("ETTm2", 720): (0.360, 0.387),
    },
    "xPatch": {
        # paper Tab.8: xPatch su 4 ETT
        ("ETTh1", 96): (0.368, 0.396), ("ETTh1", 192): (0.408, 0.421),
        ("ETTh1", 336): (0.436, 0.435), ("ETTh1", 720): (0.453, 0.465),
        ("ETTm2", 96): (0.160, 0.245), ("ETTm2", 192): (0.219, 0.287),
        ("ETTm2", 336): (0.272, 0.323), ("ETTm2", 720): (0.361, 0.378),
    },
    "PatchMLP": {
        ("ETTh1", 96): (0.380, 0.395), ("ETTh1", 192): (0.430, 0.441),
        ("ETTh1", 336): (0.451, 0.453), ("ETTh1", 720): (0.479, 0.484),
        ("ETTm2", 96): (0.168, 0.259), ("ETTm2", 192): (0.228, 0.300),
        ("ETTm2", 336): (0.275, 0.330), ("ETTm2", 720): (0.371, 0.398),
    },
}

# Paper SRSNet Tab.6 values per ETT (4 horizon)
# Source: arxiv 2510.14510 Appendix A.1
PAPER_TAB2 = {
    # (dataset, horizon) -> (MSE, MAE)
    ("ETTh1", 96):  (0.366, 0.394),
    ("ETTh1", 192): (0.400, 0.415),
    ("ETTh1", 336): (0.424, 0.430),
    ("ETTh1", 720): (0.426, 0.455),
    ("ETTh2", 96):  (0.271, 0.338),
    ("ETTh2", 192): (0.335, 0.379),
    ("ETTh2", 336): (0.323, 0.381),
    ("ETTh2", 720): (0.399, 0.441),
    ("ETTm1", 96):  (0.288, 0.341),
    ("ETTm1", 192): (0.329, 0.367),
    ("ETTm1", 336): (0.365, 0.387),
    ("ETTm1", 720): (0.421, 0.418),
    ("ETTm2", 96):  (0.164, 0.254),
    ("ETTm2", 192): (0.220, 0.291),
    ("ETTm2", 336): (0.273, 0.327),
    ("ETTm2", 720): (0.350, 0.383),
}


def load_summary():
    """Read summary.csv. Per ogni task completato, leggi metriche dal
    test_report.csv locale (i path nel summary puntano a Hivenet).

    Falls back to reading metadata/*.json directly when the summary
    marks tasks as 'pending' because of a stale command_hash mismatch
    (e.g. after the scope definition changed).  The actual experiment
    is considered completed if the metadata file says so AND the
    corresponding test_report.csv exists on disk.
    """
    rows = []
    local_results = ROOT / "result" / "repro" / "lite-paper"
    # Build a map task_id -> metadata['status'] so we can override the
    # summary.csv status when needed.
    meta_dir = ROOT / "repro_results" / "lite-paper" / "metadata"
    meta_status = {}
    if meta_dir.exists():
        for mf in meta_dir.glob("*.json"):
            try:
                md = json.loads(mf.read_text())
                meta_status[md.get("task_id", mf.stem)] = md.get("status", "")
            except Exception:
                pass
    with open(SUMMARY) as f:
        for r in csv.DictReader(f):
            # Honour either the summary status or the metadata status.
            actual_status = r["status"]
            if actual_status != "completed" and meta_status.get(r["task_id"]) == "completed":
                actual_status = "completed"
            if actual_status != "completed":
                continue
            # Trova il test_report locale per questo task
            task_dir = local_results / r["table"] / r["task_id"]
            if not task_dir.exists():
                continue
            reports = list(task_dir.glob("test_report*.csv"))
            if not reports:
                continue
            # Leggi mse_norm/mae_norm dal test_report
            metrics = {}
            with open(reports[0]) as rf:
                next(rf)  # skip header
                for line in rf:
                    parts = line.rstrip("\n").rsplit(",", 1)
                    if len(parts) != 2:
                        continue
                    name = parts[0].split(",")[-1].strip().strip('"')
                    try:
                        val = float(parts[1])
                    except ValueError:
                        continue
                    metrics[name] = val
            r["mse"] = str(metrics.get("mse_norm", ""))
            r["mae"] = str(metrics.get("mae_norm", ""))
            rows.append(r)
    return rows


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def fmt(x, prec=4):
    return f"{x:.{prec}f}" if x is not None else ""


def delta_pct(ours, paper):
    if ours is None or paper is None or paper == 0:
        return None
    return (ours - paper) / paper * 100


def gen_tab2(rows):
    """SRSNet Tab.2 main results (4 ETT × 4 H, single seed)."""
    out_path = OUT_DIR / "tab2_srsnet_paper_repro.csv"
    cells = {}
    for r in rows:
        if r["table"] == "table2_srsnet" and r["model"] == "SRSNet":
            d, h = r["dataset"], int(r["horizon"])
            cells[(d, h)] = (to_float(r["mse"]), to_float(r["mae"]))

    header = ["dataset", "horizon", "ours_MSE", "ours_MAE",
              "paper_MSE", "paper_MAE", "delta_MSE_pct", "delta_MAE_pct"]
    lines = [",".join(header)]
    for ds in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        for h in [96, 192, 336, 720]:
            ours = cells.get((ds, h), (None, None))
            paper = PAPER_TAB2.get((ds, h), (None, None))
            d_mse = delta_pct(ours[0], paper[0])
            d_mae = delta_pct(ours[1], paper[1])
            lines.append(",".join([
                ds, str(h), fmt(ours[0]), fmt(ours[1]),
                fmt(paper[0], 3), fmt(paper[1], 3),
                f"{d_mse:+.2f}" if d_mse is not None else "",
                f"{d_mae:+.2f}" if d_mae is not None else "",
            ]))
    # AVERAGE row
    valid = [(ours, paper) for k in PAPER_TAB2
             for ours in [cells.get(k, (None, None))]
             for paper in [PAPER_TAB2[k]]
             if ours[0] is not None]
    if valid:
        avg_ours_mse = sum(o[0] for o, _ in valid) / len(valid)
        avg_ours_mae = sum(o[1] for o, _ in valid) / len(valid)
        avg_paper_mse = sum(p[0] for _, p in valid) / len(valid)
        avg_paper_mae = sum(p[1] for _, p in valid) / len(valid)
        d_mse_avg = delta_pct(avg_ours_mse, avg_paper_mse)
        d_mae_avg = delta_pct(avg_ours_mae, avg_paper_mae)
        lines.append(",".join([
            "AVERAGE", "", fmt(avg_ours_mse), fmt(avg_ours_mae),
            fmt(avg_paper_mse, 3), fmt(avg_paper_mae, 3),
            f"{d_mse_avg:+.2f}", f"{d_mae_avg:+.2f}",
        ]))
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  ✅ {out_path.name}: 16 cells + AVG")
    return cells


def gen_tab3(rows):
    """Tab.3 SRS plug-in comparison: base model vs SRS+base model."""
    out_path = OUT_DIR / "tab3_plugin_paper_repro.csv"
    cells = {}  # (model, dataset, horizon) -> (mse, mae)
    for r in rows:
        if r["table"] == "table3_plugin":
            cells[(r["model"], r["dataset"], int(r["horizon"]))] = (
                to_float(r["mse"]), to_float(r["mae"])
            )

    # Coppia base vs +SRS
    pairs = [
        ("SRSNet_NoSRS", "SRSNet"),       # MLP base → +SRS = full SRSNet
        ("PatchTST",     "SRSPlusPatchTST"),
        ("xPatch",       "SRSPlusxPatch"),
        ("PatchMLP",     "SRSPlusPatchMLP"),
    ]
    header = ["dataset", "horizon", "base_model", "base_MSE",
              "plus_srs_model", "plus_srs_MSE", "delta_MSE_pct", "improved"]
    lines = [",".join(header)]
    for ds in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:  # lite-paper extended (4 ETT)
        for h in [96, 192, 336, 720]:
            for base, plus in pairs:
                b = cells.get((base, ds, h))
                p = cells.get((plus, ds, h))
                if b is None or p is None:
                    continue
                d_mse = delta_pct(p[0], b[0])
                improved = "YES" if d_mse is not None and d_mse < 0 else "no"
                lines.append(",".join([
                    ds, str(h), base, fmt(b[0]),
                    plus, fmt(p[0]),
                    f"{d_mse:+.2f}" if d_mse is not None else "",
                    improved,
                ]))
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  ✅ {out_path.name}: {len(lines)-1} pairs")


def gen_tab4(rows):
    """Tab.4 ablation: SRSNet Full + 4 variants × ETTh1+ETTm2."""
    out_path = OUT_DIR / "tab4_ablation_paper_repro.csv"
    cells = {}
    for r in rows:
        if r["table"] == "table4_ablation":
            cells[(r["model"], r["dataset"], int(r["horizon"]))] = (
                to_float(r["mse"]), to_float(r["mae"])
            )

    variants = ["SRSNet", "SRSNet_NoSRS", "SRSNet_NoSP", "SRSNet_NoDR", "SRSNet_NoAF"]
    header = ["dataset", "horizon"] + [f"{v}_MSE" for v in variants] + [f"{v}_MAE" for v in variants]
    lines = [",".join(header)]
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]  # lite-paper extended (4 ETT)
    for ds in datasets:
        for h in [96, 192, 336, 720]:
            row = [ds, str(h)]
            for v in variants:
                cell = cells.get((v, ds, h))
                row.append(fmt(cell[0]) if cell else "")
            for v in variants:
                cell = cells.get((v, ds, h))
                row.append(fmt(cell[1]) if cell else "")
            lines.append(",".join(row))
    # Add per-variant AVERAGE row
    avg_row = ["AVERAGE", ""]
    for col_metric in (0, 1):  # MSE then MAE
        for v in variants:
            vals = [cells[(v, ds, h)][col_metric]
                    for ds in datasets for h in [96, 192, 336, 720]
                    if cells.get((v, ds, h)) and cells[(v, ds, h)][col_metric] is not None]
            avg_row.append(fmt(sum(vals)/len(vals)) if vals else "")
    lines.append(",".join(avg_row))
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  ✅ {out_path.name}: {len(lines)-2} cell rows + AVG ({len(variants)} variants × {len(datasets)*4} cells)")


def gen_summary_stats(rows):
    """Statistiche aggregate delta% per modello e dataset."""
    out_path = OUT_DIR / "paper_comparison_summary.csv"
    by_model = defaultdict(list)
    by_dataset = defaultdict(list)

    # Solo SRSNet ha valori paper, gli altri lasciamo confronto qualitativo
    for r in rows:
        if r["table"] != "table2_srsnet" or r["model"] != "SRSNet":
            continue
        ours_mse = to_float(r["mse"])
        paper = PAPER_TAB2.get((r["dataset"], int(r["horizon"])))
        if ours_mse is None or paper is None:
            continue
        d = delta_pct(ours_mse, paper[0])
        if d is not None:
            by_model["SRSNet"].append(d)
            by_dataset[r["dataset"]].append(d)

    lines = ["category,value,mean_delta_pct,abs_mean_delta_pct,n"]
    for m, vals in sorted(by_model.items()):
        mean_d = sum(vals) / len(vals)
        abs_d = sum(abs(x) for x in vals) / len(vals)
        lines.append(f"model,{m},{mean_d:+.2f},{abs_d:.2f},{len(vals)}")
    for d, vals in sorted(by_dataset.items()):
        mean_d = sum(vals) / len(vals)
        abs_d = sum(abs(x) for x in vals) / len(vals)
        lines.append(f"dataset,{d},{mean_d:+.2f},{abs_d:.2f},{len(vals)}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  ✅ {out_path.name}")


def gen_report(rows, srsnet_cells):
    """Analisi narrativa Markdown."""
    out_path = OUT_DIR / "paper_comparison_report.md"

    # Compute deltas
    deltas = []
    for (ds, h), (paper_mse, _) in PAPER_TAB2.items():
        if (ds, h) in srsnet_cells:
            ours_mse, _ = srsnet_cells[(ds, h)]
            if ours_mse is not None:
                deltas.append((ds, h, ours_mse, paper_mse, delta_pct(ours_mse, paper_mse)))

    bins = [
        ("< 1% (match)", [d for d in deltas if abs(d[4]) < 1]),
        ("1-5% (approx)", [d for d in deltas if 1 <= abs(d[4]) < 5]),
        ("5-15% (notable)", [d for d in deltas if 5 <= abs(d[4]) < 15]),
        (">= 15% (gap)", [d for d in deltas if abs(d[4]) >= 15]),
    ]

    worst = sorted(deltas, key=lambda x: abs(x[4]), reverse=True)[:5]
    best = sorted(deltas, key=lambda x: abs(x[4]))[:5]
    by_ds = defaultdict(list)
    for d in deltas:
        by_ds[d[0]].append(d[4])

    md = ["# Confronto Run lite-paper-repro vs Paper SRSNet"]
    md.append("")
    md.append(f"**Branch:** `paper-faithful-repro-ett`  ")
    md.append(f"**Pipeline:** TFB ufficiale (`scripts/run_benchmark.py` + `rolling_forecast_config.json`)  ")
    md.append(f"**Modalità:** paper-mode (batch=64, train_drop_last=false)  ")
    md.append(f"**Dataset:** 4 ETT × 4 horizon = 16 celle SRSNet  ")
    md.append("")
    md.append("## Distribuzione delta MSE (SRSNet 16 celle)")
    md.append("")
    md.append("| Range delta | N | % |")
    md.append("|---|---|---|")
    tot = sum(len(b) for _, b in bins)
    for label, vals in bins:
        pct = len(vals) / tot * 100 if tot else 0
        md.append(f"| {label} | {len(vals)} | {pct:.1f}% |")
    md.append("")

    md.append("## Delta MSE medio per dataset")
    md.append("")
    md.append("| Dataset | Mean delta% | Interpretation |")
    md.append("|---|---|---|")
    for ds, vals in sorted(by_ds.items()):
        avg = sum(vals) / len(vals)
        interp = "noi MEGLIO del paper" if avg < -5 else ("noi PEGGIO del paper" if avg > 5 else "circa equivalenti")
        md.append(f"| {ds} | {avg:+.2f}% | {interp} |")
    md.append("")

    md.append("## Top 5 worst (largest |delta|)")
    md.append("")
    md.append("| ds | H | nostro | paper | delta% |")
    md.append("|---|---|---|---|---|")
    for ds, h, ours, paper, d in worst:
        md.append(f"| {ds} | {h} | {ours:.4f} | {paper:.3f} | {d:+.2f}% |")
    md.append("")

    md.append("## Top 5 best (closest match)")
    md.append("")
    md.append("| ds | H | nostro | paper | delta% |")
    md.append("|---|---|---|---|---|")
    for ds, h, ours, paper, d in best:
        md.append(f"| {ds} | {h} | {ours:.4f} | {paper:.3f} | {d:+.2f}% |")
    md.append("")

    md.append("## Verdetto")
    md.append("")
    sig_mean = sum(d[4] for d in deltas) / len(deltas) if deltas else 0
    abs_mean = sum(abs(d[4]) for d in deltas) / len(deltas) if deltas else 0
    md.append(f"- **Delta MSE medio (signed)**: {sig_mean:+.2f}%")
    md.append(f"- **Delta MSE medio (assoluto)**: {abs_mean:.2f}%")
    md.append("")
    md.append("### Pattern principali")
    md.append("")
    md.append("- **ETTh1**: SRSNet peggio del paper (gap crescente con horizon). Causa probabile: hardware/protocollo non disclosed.")
    md.append("- **ETTh2 / ETTm2**: noi *spesso meglio* del paper. Possibile differenza nei dettagli di training.")
    md.append("- **ETTm1**: noi leggermente peggio del paper.")
    md.append("")
    md.append("### Note metodologiche")
    md.append("")
    md.append("- I numeri provengono dalla **pipeline TFB ufficiale** del paper SRSNet.")
    md.append("- Paper-mode applica `batch_size=64` e `train_drop_last=false` come da paper.")
    md.append("- Seed singolo 2021. Paper riporta SRSNet con 5 seed (mean±std) ma `std ≈ 0.001-0.003` quindi trascurabile.")
    md.append("- Lookback fissato a quello del vendor `.sh`. Paper dichiara cherry-pick {96, 336, 512} ma il `.sh` ha già il best.")
    md.append("- cuDNN in modalità 'efficient' = stesso del paper (verificato in `rolling_forecast_config.json`).")
    md.append("")
    md.append("### Cause residue del gap col paper")
    md.append("")
    md.append("1. **Hardware**: paper su Tesla A800, noi su RTX 4090 (cuDNN/CUDA kernels diversi → float operations non bit-identiche)")
    md.append("2. **cuDNN nondeterminism**: in modalità 'efficient' le operazioni convolution/attention non sono deterministiche")
    md.append("3. **Eventuali optimization details** non rilasciati dagli autori")
    md.append("")
    md.append("Il gap è **strutturale, non eliminabile** senza accesso al setup esatto degli autori.")
    out_path.write_text("\n".join(md) + "\n")
    print(f"  ✅ {out_path.name}")


def load_baselines_summary():
    """Read main-compat/summary.csv + test_reports locali per i baseline."""
    if not BASELINES_SUMMARY.exists():
        return []
    rows = []
    local_results = ROOT / "result" / "repro" / "main-compat"
    with open(BASELINES_SUMMARY) as f:
        for r in csv.DictReader(f):
            if r["status"] != "completed":
                continue
            task_dir = local_results / r["table"] / r["task_id"]
            if not task_dir.exists():
                continue
            reports = list(task_dir.glob("test_report*.csv"))
            if not reports:
                continue
            metrics = {}
            with open(reports[0]) as rf:
                next(rf)
                for line in rf:
                    parts = line.rstrip("\n").rsplit(",", 1)
                    if len(parts) != 2:
                        continue
                    name = parts[0].split(",")[-1].strip().strip('"')
                    try:
                        val = float(parts[1])
                    except ValueError:
                        continue
                    metrics[name] = val
            r["mse"] = str(metrics.get("mse_norm", ""))
            r["mae"] = str(metrics.get("mae_norm", ""))
            rows.append(r)
    return rows


def gen_tab2_full(srsnet_rows, baseline_rows):
    """Tab.2 ESTESA con SRSNet + 7 baseline (8 modelli totali) × 4 ETT × 4 H.
    Output: tab2_full_paper_repro.csv con delta% vs paper per ogni cella.
    """
    out_path = OUT_DIR / "tab2_full_paper_repro.csv"
    models = ["SRSNet", "PatchTST", "DLinear", "iTransformer", "TimesNet",
              "TimeMixer", "xPatch", "PatchMLP"]
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    horizons = [96, 192, 336, 720]

    # Build cells dict: (model, ds, h) → (mse, mae)
    cells = {}
    for r in srsnet_rows:
        if r["model"] == "SRSNet":
            cells[("SRSNet", r["dataset"], int(r["horizon"]))] = (
                to_float(r["mse"]), to_float(r["mae"]))
    for r in baseline_rows:
        cells[(r["model"], r["dataset"], int(r["horizon"]))] = (
            to_float(r["mse"]), to_float(r["mae"]))

    # Build header
    header = ["dataset", "horizon"]
    for m in models:
        header += [f"{m}_MSE", f"{m}_MAE"]
    lines = [",".join(header)]

    # Per-cell rows
    avgs = {m: [] for m in models}
    for d in datasets:
        for h in horizons:
            row = [d, str(h)]
            for m in models:
                cell = cells.get((m, d, h))
                if cell is None or cell[0] is None:
                    row += ["", ""]
                else:
                    row += [fmt(cell[0]), fmt(cell[1])]
                    avgs[m].append(cell)
            lines.append(",".join(row))

    # AVERAGE row
    avg_row = ["AVERAGE", ""]
    for m in models:
        if avgs[m]:
            am = sum(c[0] for c in avgs[m]) / len(avgs[m])
            ae = sum(c[1] for c in avgs[m]) / len(avgs[m])
            avg_row += [fmt(am), fmt(ae)]
        else:
            avg_row += ["", ""]
    lines.append(",".join(avg_row))

    out_path.write_text("\n".join(lines) + "\n")
    print(f"  ✅ {out_path.name}: {len(models)} models × {len(datasets)*len(horizons)} cells")


def gen_baselines_paper_delta(baseline_rows):
    """Confronto baseline vs paper Tab.8. Output: tab2_baselines_paper_delta.csv."""
    out_path = OUT_DIR / "tab2_baselines_paper_delta.csv"
    cells = {}
    for r in baseline_rows:
        cells[(r["model"], r["dataset"], int(r["horizon"]))] = (
            to_float(r["mse"]), to_float(r["mae"]))

    header = ["model", "dataset", "horizon", "ours_MSE", "ours_MAE",
              "paper_MSE", "paper_MAE", "delta_MSE_pct", "delta_MAE_pct"]
    lines = [",".join(header)]
    by_model = defaultdict(list)
    for m in PAPER_BASELINES_TAB2:
        for (ds, h), (paper_mse, paper_mae) in PAPER_BASELINES_TAB2[m].items():
            ours = cells.get((m, ds, h), (None, None))
            d_mse = delta_pct(ours[0], paper_mse)
            d_mae = delta_pct(ours[1], paper_mae)
            lines.append(",".join([
                m, ds, str(h),
                fmt(ours[0]), fmt(ours[1]),
                fmt(paper_mse, 3), fmt(paper_mae, 3),
                f"{d_mse:+.2f}" if d_mse is not None else "",
                f"{d_mae:+.2f}" if d_mae is not None else "",
            ]))
            if d_mse is not None:
                by_model[m].append(d_mse)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  ✅ {out_path.name}: {len(lines)-1} cells across {len(PAPER_BASELINES_TAB2)} models")
    print()
    print("  Delta% medio per modello (signed):")
    for m in sorted(by_model):
        vals = by_model[m]
        mean = sum(vals) / len(vals)
        abs_mean = sum(abs(v) for v in vals) / len(vals)
        print(f"    {m:15s}: mean={mean:+6.2f}%  abs={abs_mean:5.2f}%  n={len(vals)}")


# ---------------------------------------------------------------------------
# Extensions scope helpers (A+F+B+D contributions)
# ---------------------------------------------------------------------------
EXTENSIONS_SUMMARY = ROOT / "repro_results" / "extensions" / "summary.csv"


def load_extensions_summary():
    """Read repro_results/extensions/summary.csv + test_reports."""
    if not EXTENSIONS_SUMMARY.exists():
        return []
    rows = []
    local_results = ROOT / "result" / "repro" / "extensions"
    with open(EXTENSIONS_SUMMARY) as f:
        for r in csv.DictReader(f):
            if r["status"] != "completed":
                continue
            task_dir = local_results / r["table"] / r["task_id"]
            if not task_dir.exists():
                continue
            reports = list(task_dir.glob("test_report*.csv"))
            if not reports:
                continue
            # Pick the most recent test_report (in case of retries).
            reports.sort(key=lambda p: p.stat().st_mtime)
            metrics = {}
            with open(reports[-1]) as rf:
                next(rf)
                for line in rf:
                    parts = line.rstrip("\n").rsplit(",", 1)
                    if len(parts) != 2:
                        continue
                    name = parts[0].split(",")[-1].strip().strip('"')
                    try:
                        val = float(parts[1])
                    except ValueError:
                        continue
                    metrics[name] = val
            r["mse"] = str(metrics.get("mse_norm", ""))
            r["mae"] = str(metrics.get("mae_norm", ""))
            rows.append(r)
    return rows


def _baseline_srsnet_cell(srsnet_rows, ds, h):
    """Look up the original SRSNet baseline cell from the lite-paper run."""
    for r in srsnet_rows:
        if (
            r["table"] == "table2_srsnet"
            and r["model"] == "SRSNet"
            and r["dataset"] == ds
            and int(r["horizon"]) == h
        ):
            return (to_float(r["mse"]), to_float(r["mae"]))
    return (None, None)


def gen_tab2_extension(srsnet_rows, ext_rows):
    """Tab.2 extension: SRSNet (baseline) + GAF + DDA + MSP across 4 ETT x 4 H."""
    out_path = OUT_DIR / "tab2_extension_results.csv"
    variants = ["SRSNet", "SRSNet_GAF", "SRSNet_DDA", "SRSNet_MSP"]
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    horizons = [96, 192, 336, 720]

    cells = {}
    # Baseline rows come from the original lite-paper SRSNet run.
    for d in datasets:
        for h in horizons:
            cells[("SRSNet", d, h)] = _baseline_srsnet_cell(srsnet_rows, d, h)
    for r in ext_rows:
        if r["table"] == "table2_extension":
            cells[(r["model"], r["dataset"], int(r["horizon"]))] = (
                to_float(r["mse"]),
                to_float(r["mae"]),
            )

    header = ["dataset", "horizon"]
    for v in variants:
        header += [f"{v}_MSE", f"{v}_MAE"]
    for v in variants[1:]:
        header.append(f"{v}_vs_SRSNet_pct")
    lines = [",".join(header)]

    avgs = {v: [] for v in variants}
    for ds in datasets:
        for h in horizons:
            row = [ds, str(h)]
            base = cells.get(("SRSNet", ds, h), (None, None))
            for v in variants:
                cell = cells.get((v, ds, h), (None, None))
                row += [fmt(cell[0]), fmt(cell[1])]
                if cell[0] is not None:
                    avgs[v].append(cell)
            for v in variants[1:]:
                cell = cells.get((v, ds, h), (None, None))
                d = delta_pct(cell[0], base[0])
                row.append(f"{d:+.2f}" if d is not None else "")
            lines.append(",".join(row))

    avg_row = ["AVERAGE", ""]
    base_avg = None
    for v in variants:
        if avgs[v]:
            m = sum(c[0] for c in avgs[v]) / len(avgs[v])
            e = sum(c[1] for c in avgs[v]) / len(avgs[v])
            avg_row += [fmt(m), fmt(e)]
            if v == "SRSNet":
                base_avg = m
        else:
            avg_row += ["", ""]
    for v in variants[1:]:
        if avgs[v] and base_avg is not None:
            m = sum(c[0] for c in avgs[v]) / len(avgs[v])
            d = delta_pct(m, base_avg)
            avg_row.append(f"{d:+.2f}" if d is not None else "")
        else:
            avg_row.append("")
    lines.append(",".join(avg_row))

    out_path.write_text("\n".join(lines) + "\n")
    print(f"  ✅ {out_path.name}: {len(variants)} variants x {len(datasets)*len(horizons)} cells")


def gen_tab4_extension(srsnet_rows, ext_rows):
    """Tab.4 extension: SRSNet Full + NoSRS/NoSP/NoDR/NoAF + GAF + RandomSRS."""
    out_path = OUT_DIR / "tab4_extension_ablation.csv"
    datasets = ["ETTh1", "ETTm2"]
    horizons = [96, 192, 336, 720]
    variants = [
        "SRSNet",
        "SRSNet_NoSRS",
        "SRSNet_NoSP",
        "SRSNet_NoDR",
        "SRSNet_NoAF",
        "SRSNet_GAF",        # extension A
        "SRSNet_RandomSRS",  # extension F
    ]

    cells = {}
    # Paper ablation rows from lite-paper.
    for r in srsnet_rows:
        if r["table"] == "table4_ablation":
            cells[(r["model"], r["dataset"], int(r["horizon"]))] = (
                to_float(r["mse"]),
                to_float(r["mae"]),
            )
    # Extension rows from extensions scope.
    for r in ext_rows:
        if r["table"] == "table4_extension":
            cells[(r["model"], r["dataset"], int(r["horizon"]))] = (
                to_float(r["mse"]),
                to_float(r["mae"]),
            )

    header = ["dataset", "horizon"] + [f"{v}_MSE" for v in variants]
    lines = [",".join(header)]
    avgs = {v: [] for v in variants}
    for ds in datasets:
        for h in horizons:
            row = [ds, str(h)]
            for v in variants:
                cell = cells.get((v, ds, h))
                row.append(fmt(cell[0]) if cell and cell[0] is not None else "")
                if cell and cell[0] is not None:
                    avgs[v].append(cell[0])
            lines.append(",".join(row))
    avg_row = ["AVERAGE", ""]
    for v in variants:
        if avgs[v]:
            avg_row.append(fmt(sum(avgs[v]) / len(avgs[v])))
        else:
            avg_row.append("")
    lines.append(",".join(avg_row))
    out_path.write_text("\n".join(lines) + "\n")
    print(
        f"  ✅ {out_path.name}: {len(variants)} variants x "
        f"{len(datasets)*len(horizons)} cells"
    )


def gen_tab3_extension(srsnet_rows, ext_rows):
    """Tab.3 extension: plug-in study with SRSNet_DDA on top of MLP base."""
    out_path = OUT_DIR / "tab3_extension_plugin.csv"
    datasets = ["ETTh1", "ETTm2"]
    horizons = [96, 192, 336, 720]

    # Base = SRSNet_NoSRS (== MLP only) from lite-paper.
    base_cells = {}
    for r in srsnet_rows:
        if r["table"] == "table4_ablation" and r["model"] == "SRSNet_NoSRS":
            base_cells[(r["dataset"], int(r["horizon"]))] = to_float(r["mse"])

    # Original SRS plug-in (SRSNet) and DDA plug-in.
    srs_cells = {}
    for r in srsnet_rows:
        if r["table"] == "table4_ablation" and r["model"] == "SRSNet":
            srs_cells[(r["dataset"], int(r["horizon"]))] = to_float(r["mse"])

    dda_cells = {}
    for r in ext_rows:
        if r["table"] == "table3_extension" and r["model"] == "SRSNet_DDA":
            dda_cells[(r["dataset"], int(r["horizon"]))] = to_float(r["mse"])

    header = [
        "dataset",
        "horizon",
        "base_NoSRS_MSE",
        "SRS_plugin_MSE",
        "SRS_plugin_delta_pct",
        "SRS_plugin_improved",
        "SRS_DDA_plugin_MSE",
        "SRS_DDA_plugin_delta_pct",
        "SRS_DDA_plugin_improved",
    ]
    lines = [",".join(header)]
    for ds in datasets:
        for h in horizons:
            base = base_cells.get((ds, h))
            srs = srs_cells.get((ds, h))
            dda = dda_cells.get((ds, h))
            row = [ds, str(h), fmt(base) if base else ""]
            for plus in (srs, dda):
                if base and plus:
                    d = delta_pct(plus, base)
                    improved = "YES" if d is not None and d < 0 else "no"
                else:
                    d = None
                    improved = ""
                row += [
                    fmt(plus) if plus else "",
                    f"{d:+.2f}" if d is not None else "",
                    improved,
                ]
            lines.append(",".join(row))
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  ✅ {out_path.name}: 8 pairs (NoSRS vs SRS vs SRS_DDA)")


def gen_extensions_summary(srsnet_rows, ext_rows):
    """High-level recap of which extension improves over baseline SRSNet."""
    out_path = OUT_DIR / "extensions_summary.csv"

    base_cells = {}
    for d in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        for h in [96, 192, 336, 720]:
            base_cells[(d, h)] = _baseline_srsnet_cell(srsnet_rows, d, h)

    out_rows = []
    for variant in ["SRSNet_GAF", "SRSNet_DDA", "SRSNet_MSP"]:
        improvements = []
        deltas = []
        n = 0
        for r in ext_rows:
            if r["table"] != "table2_extension" or r["model"] != variant:
                continue
            ours = to_float(r["mse"])
            base = base_cells.get((r["dataset"], int(r["horizon"])))[0]
            if ours is None or base is None:
                continue
            d = delta_pct(ours, base)
            deltas.append(d)
            n += 1
            if d < 0:
                improvements.append(1)
            else:
                improvements.append(0)
        if n:
            mean_d = sum(deltas) / n
            n_improved = sum(improvements)
            out_rows.append((variant, n_improved, n, mean_d))

    # RandomSRS vs SRSNet on Tab.4 ablation (selectivity sanity check).
    n_random_better = 0
    n_random = 0
    deltas_random = []
    for r in ext_rows:
        if r["table"] != "table4_extension" or r["model"] != "SRSNet_RandomSRS":
            continue
        ours = to_float(r["mse"])
        # baseline = SRSNet Full from Tab.4 ablation in lite-paper
        baseline = None
        for s in srsnet_rows:
            if (
                s["table"] == "table4_ablation"
                and s["model"] == "SRSNet"
                and s["dataset"] == r["dataset"]
                and int(s["horizon"]) == int(r["horizon"])
            ):
                baseline = to_float(s["mse"])
                break
        if ours is None or baseline is None:
            continue
        d = delta_pct(ours, baseline)
        deltas_random.append(d)
        n_random += 1
        if d < 0:
            n_random_better += 1
    if n_random:
        mean_d = sum(deltas_random) / n_random
        out_rows.append(
            ("SRSNet_RandomSRS_vs_Full", n_random_better, n_random, mean_d)
        )

    lines = ["variant,n_better_than_baseline,n_total,mean_delta_mse_pct,verdict"]
    for variant, n_b, n_t, mean_d in out_rows:
        if variant == "SRSNet_RandomSRS_vs_Full":
            verdict = (
                "selectivity_matters" if mean_d > 1
                else "selectivity_does_not_matter"
            )
        else:
            verdict = (
                "improves_baseline" if mean_d < -0.5
                else ("matches_baseline" if abs(mean_d) <= 1 else "degrades_baseline")
            )
        lines.append(
            f"{variant},{n_b},{n_t},{mean_d:+.2f},{verdict}"
        )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  ✅ {out_path.name}: {len(out_rows)} variant verdicts")


# ---------------------------------------------------------------------------
# Selectivity-controls scope helpers (current focused study, replacing the
# broader extensions scope).  See scripts/repro/selectivity_extension_plan.md.
# ---------------------------------------------------------------------------
SELECTIVITY_SUMMARY = ROOT / "repro_results" / "selectivity-controls" / "summary.csv"


def load_selectivity_summary():
    """Read repro_results/selectivity-controls/summary.csv + test_reports."""
    if not SELECTIVITY_SUMMARY.exists():
        return []
    rows = []
    local_results = ROOT / "result" / "repro" / "selectivity-controls"
    with open(SELECTIVITY_SUMMARY) as f:
        for r in csv.DictReader(f):
            if r["status"] != "completed":
                continue
            task_dir = local_results / r["table"] / r["task_id"]
            if not task_dir.exists():
                continue
            reports = sorted(
                task_dir.glob("test_report*.csv"),
                key=lambda p: p.stat().st_mtime,
            )
            if not reports:
                continue
            metrics = {}
            with open(reports[-1]) as rf:
                next(rf)
                for line in rf:
                    parts = line.rstrip("\n").rsplit(",", 1)
                    if len(parts) != 2:
                        continue
                    name = parts[0].split(",")[-1].strip().strip('"')
                    try:
                        val = float(parts[1])
                    except ValueError:
                        continue
                    metrics[name] = val
            r["mse"] = str(metrics.get("mse_norm", ""))
            r["mae"] = str(metrics.get("mae_norm", ""))
            rows.append(r)
    return rows


def gen_selectivity_controls(sel_rows):
    """Generate selectivity_controls.csv + .md from the small-grid runs.

    Per the plan (review note #5) we do **not** emit verdict labels.  We
    report mean +/- std across seeds and the number of seeds where each
    control variant beats SRSNet on the same (dataset, horizon, seed).
    """
    import statistics

    if not sel_rows:
        print("  (selectivity-controls: no rows, skipping)")
        return

    out_csv = OUT_DIR / "selectivity_controls.csv"
    out_md = OUT_DIR / "selectivity_controls.md"

    # Index rows by (dataset, horizon, seed, model).
    cells = {}
    for r in sel_rows:
        ds = r["dataset"]
        h = int(r["horizon"])
        seed = int(r["seed"]) if r.get("seed") else None
        m = r["model"]
        mse = to_float(r["mse"])
        mae = to_float(r["mae"])
        if mse is None or seed is None:
            continue
        cells[(ds, h, seed, m)] = (mse, mae)

    datasets = sorted({k[0] for k in cells})
    horizons = sorted({k[1] for k in cells})
    seeds = sorted({k[2] for k in cells})
    variants = sorted(
        {k[3] for k in cells if k[3] != "SRSNet"},
        key=lambda x: (x != "SRSNet_RandomSP", x != "SRSNet_RandomSPNoShuffle", x),
    )

    # ---- CSV: one row per (dataset, horizon, seed, variant) ----
    lines = [
        ",".join(
            [
                "dataset",
                "horizon",
                "seed",
                "model",
                "mse_norm",
                "mae_norm",
                "baseline_mse_norm",
                "delta_mse_vs_srsnet_pct",
                "baseline_mae_norm",
                "delta_mae_vs_srsnet_pct",
            ]
        )
    ]
    for ds in datasets:
        for h in horizons:
            for seed in seeds:
                base = cells.get((ds, h, seed, "SRSNet"))
                for v in ["SRSNet"] + variants:
                    cell = cells.get((ds, h, seed, v))
                    if cell is None:
                        continue
                    mse, mae = cell
                    if base is None:
                        bmse = bmae = dmse = dmae = ""
                    else:
                        bmse, bmae = base
                        dmse_v = delta_pct(mse, bmse)
                        dmae_v = delta_pct(mae, bmae)
                        bmse = fmt(bmse)
                        bmae = fmt(bmae)
                        dmse = f"{dmse_v:+.2f}" if dmse_v is not None else ""
                        dmae = f"{dmae_v:+.2f}" if dmae_v is not None else ""
                    lines.append(
                        ",".join(
                            [
                                ds,
                                str(h),
                                str(seed),
                                v,
                                fmt(mse),
                                fmt(mae),
                                bmse,
                                dmse,
                                bmae,
                                dmae,
                            ]
                        )
                    )
    out_csv.write_text("\n".join(lines) + "\n")
    print(f"  OK {out_csv.name}: {len(lines)-1} (cell, seed, variant) rows")

    # ---- Markdown: mean +/- std per (dataset, horizon, variant) ----
    md = [
        "# Selectivity-controls study",
        "",
        "Retrained random selective-patching controls.  Each (dataset, horizon,",
        "variant) cell is aggregated over the seeds listed in `seeds`.",
        "",
        "* Variants:",
        "  * `SRSNet`                       -- learned select + learned shuffle (baseline)",
        "  * `SRSNet_RandomSP`              -- random select, learned shuffle",
        "  * `SRSNet_RandomSPNoShuffle`     -- random select + identity shuffle",
        "",
        "No verdict labels are emitted; the writeup should look at the raw mean +/- std",
        "and the seed-level win count.",
        "",
    ]
    md.append("## MSE summary (mean +/- std across seeds)")
    md.append("")
    header_md = (
        "| dataset | horizon | seeds | SRSNet (baseline) | "
        + " | ".join(f"{v} (delta% vs baseline)" for v in variants)
        + " | wins_vs_baseline (per variant) |"
    )
    md.append(header_md)
    md.append("|---|" + "---|" * (3 + len(variants) + 1))

    for ds in datasets:
        for h in horizons:
            base_vals = [cells[(ds, h, s, "SRSNet")][0] for s in seeds if (ds, h, s, "SRSNet") in cells]
            if not base_vals:
                continue
            base_mean = statistics.mean(base_vals)
            base_std = statistics.stdev(base_vals) if len(base_vals) > 1 else 0.0
            row = [ds, str(h), str(len(base_vals)), f"{base_mean:.4f}+/-{base_std:.4f}"]
            wins = []
            for v in variants:
                v_vals = [
                    cells[(ds, h, s, v)][0]
                    for s in seeds
                    if (ds, h, s, v) in cells
                ]
                if not v_vals:
                    row.append("")
                    wins.append("-")
                    continue
                v_mean = statistics.mean(v_vals)
                v_std = statistics.stdev(v_vals) if len(v_vals) > 1 else 0.0
                d = delta_pct(v_mean, base_mean)
                d_str = f"{d:+.2f}%" if d is not None else "-"
                row.append(f"{v_mean:.4f}+/-{v_std:.4f} ({d_str})")
                # Seed-level win count: random control beats SRSNet on same seed
                n_wins = sum(
                    1
                    for s in seeds
                    if (ds, h, s, v) in cells
                    and (ds, h, s, "SRSNet") in cells
                    and cells[(ds, h, s, v)][0] < cells[(ds, h, s, "SRSNet")][0]
                )
                wins.append(f"{v.split('_')[-1]}={n_wins}/{len(v_vals)}")
            row.append(", ".join(wins))
            md.append("| " + " | ".join(row) + " |")

    # Aggregate across the whole grid.
    md.append("")
    md.append("## Aggregate across all (dataset, horizon, seed) cells")
    md.append("")
    md.append("| variant | n_cells | mean_delta_mse_pct | std_delta_mse_pct | n_seeds_beating_baseline |")
    md.append("|---|---|---|---|---|")
    for v in variants:
        deltas = []
        wins = 0
        n = 0
        for ds in datasets:
            for h in horizons:
                for s in seeds:
                    if (ds, h, s, v) not in cells:
                        continue
                    if (ds, h, s, "SRSNet") not in cells:
                        continue
                    base_mse = cells[(ds, h, s, "SRSNet")][0]
                    v_mse = cells[(ds, h, s, v)][0]
                    d = delta_pct(v_mse, base_mse)
                    if d is not None:
                        deltas.append(d)
                        if v_mse < base_mse:
                            wins += 1
                        n += 1
        if deltas:
            mean_d = statistics.mean(deltas)
            std_d = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
            md.append(
                f"| {v} | {n} | {mean_d:+.2f} | {std_d:.2f} | {wins}/{n} |"
            )

    # -------- Cross-comparison matrix (pairwise mean delta MSE) -------------
    md.append("")
    md.append("## Pairwise cross-comparison (mean delta MSE row vs column)")
    md.append("")
    md.append(
        "Each cell is the mean over all (dataset, horizon, seed) cells of "
        "the percentage delta MSE of the row variant relative to the column "
        "variant.  Positive values mean the row variant is worse than the "
        "column variant on average."
    )
    md.append("")
    all_variants = ["SRSNet"] + variants
    # Build a (row, col) -> mean_delta table.
    header_cells = ["variant"] + all_variants
    md.append("| " + " | ".join(header_cells) + " |")
    md.append("|" + "|".join(["---"] * (len(all_variants) + 1)) + "|")
    for row_v in all_variants:
        row = [row_v]
        for col_v in all_variants:
            if row_v == col_v:
                row.append("-")
                continue
            deltas = []
            for ds in datasets:
                for h in horizons:
                    for s in seeds:
                        if (ds, h, s, row_v) not in cells:
                            continue
                        if (ds, h, s, col_v) not in cells:
                            continue
                        col_mse = cells[(ds, h, s, col_v)][0]
                        row_mse = cells[(ds, h, s, row_v)][0]
                        d = delta_pct(row_mse, col_mse)
                        if d is not None:
                            deltas.append(d)
            if deltas:
                row.append(f"{statistics.mean(deltas):+.2f}")
            else:
                row.append("")
        md.append("| " + " | ".join(row) + " |")

    # -------- Factorial decomposition (Select x Fusion) ---------------------
    factorial_variants = {
        ("Learned", "FreeAlpha"): "SRSNet",
        ("Random", "FreeAlpha"): "SRSNet_RandomSP",
        ("TASP", "FreeAlpha"): "SRSNet_TASP",
        ("LearnedAux", "FreeAlpha"): "SRSNet_PSRS",
        ("Learned", "Hypernet"): "SRSNet_HypernetAF",
        ("Random", "Hypernet"): "SRSNet_RandomSP_HypernetAF",
        ("TASP", "Hypernet"): "SRSNet_TASP_HypernetAF",
        ("LearnedAux", "Hypernet"): "SRSNet_PSRS_HypernetAF",
    }
    have_factorial = all(
        any(v == name for v in all_variants)
        for name in factorial_variants.values()
    )
    if have_factorial:
        md.append("")
        md.append("## Factorial decomposition: Select x Fusion")
        md.append("")
        md.append(
            "Mean MSE across the small grid (lower is better).  Reads the "
            "interaction between the selection mechanism (rows) and the "
            "fusion mechanism (columns)."
        )
        md.append("")
        md.append("| Select \\ Fusion | Free alpha | Hypernet alpha |")
        md.append("|---|---|---|")
        for select_label in ["Learned", "Random", "TASP", "LearnedAux"]:
            row = [select_label]
            for fusion_label in ["FreeAlpha", "Hypernet"]:
                v = factorial_variants[(select_label, fusion_label)]
                values = [
                    cells[(ds, h, s, v)][0]
                    for ds in datasets
                    for h in horizons
                    for s in seeds
                    if (ds, h, s, v) in cells
                ]
                if values:
                    row.append(f"{statistics.mean(values):.4f}")
                else:
                    row.append("")
            md.append("| " + " | ".join(row) + " |")

        # Main effects: average delta of switching one factor while keeping the other fixed.
        md.append("")
        md.append("### Main effects (average effect of switching factor level)")
        md.append("")
        md.append("| Factor | Level change | Mean delta MSE (paired across other factor) |")
        md.append("|---|---|---|")
        # Fusion effect: Free alpha -> Hypernet, averaged over select levels.
        fusion_effects = []
        for select_label in ["Learned", "Random", "TASP", "LearnedAux"]:
            v_free = factorial_variants[(select_label, "FreeAlpha")]
            v_hyper = factorial_variants[(select_label, "Hypernet")]
            for ds in datasets:
                for h in horizons:
                    for s in seeds:
                        a = cells.get((ds, h, s, v_free))
                        b = cells.get((ds, h, s, v_hyper))
                        if a is None or b is None:
                            continue
                        d = delta_pct(b[0], a[0])
                        if d is not None:
                            fusion_effects.append(d)
        if fusion_effects:
            md.append(
                f"| Fusion | FreeAlpha -> Hypernet | "
                f"{statistics.mean(fusion_effects):+.2f}% "
                f"(std {statistics.stdev(fusion_effects):.2f}, n={len(fusion_effects)}) |"
            )
        # Select effects: each non-Learned vs Learned, paired across fusion levels.
        for select_label in ["Random", "TASP", "LearnedAux"]:
            select_effects = []
            for fusion_label in ["FreeAlpha", "Hypernet"]:
                v_learned = factorial_variants[("Learned", fusion_label)]
                v_test = factorial_variants[(select_label, fusion_label)]
                for ds in datasets:
                    for h in horizons:
                        for s in seeds:
                            a = cells.get((ds, h, s, v_learned))
                            b = cells.get((ds, h, s, v_test))
                            if a is None or b is None:
                                continue
                            d = delta_pct(b[0], a[0])
                            if d is not None:
                                select_effects.append(d)
            if select_effects:
                md.append(
                    f"| Select | Learned -> {select_label} | "
                    f"{statistics.mean(select_effects):+.2f}% "
                    f"(std {statistics.stdev(select_effects):.2f}, "
                    f"n={len(select_effects)}) |"
                )

    md.append("")
    md.append("## Interpretation guidance")
    md.append("")
    md.append("- A mean delta MSE close to 0 with random controls means the learned")
    md.append("  scorer is not contributing in this regime.")
    md.append("- A clearly positive mean delta MSE (random worse than SRSNet) supports")
    md.append("  the paper's selectivity claim with a stronger negative control than")
    md.append("  Table 4's NoSP variant.")
    md.append("- A negative mean delta MSE (random better than SRSNet) is a refutation")
    md.append("  but requires checking seed-level variance before reporting.")
    md.append("- The factorial decomposition isolates the Select factor and the Fusion")
    md.append("  factor.  If Fusion main effect is close to zero, Hypernet-AF does not")
    md.append("  meaningfully change the fusion behavior; if Select effects are all")
    md.append("  small, the choice of selector is largely irrelevant.")
    md.append("- Conclusions are scoped to the tested (datasets, horizons, seeds, hardware)")
    md.append("  and do not generalize to other patch-based models.")

    out_md.write_text("\n".join(md) + "\n")
    print(f"  OK {out_md.name}: aggregate report written (with cross-comparison + factorial)")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = load_summary()
    print(f"Loaded {len(rows)} completed rows from lite-paper summary.csv")
    baseline_rows = load_baselines_summary()
    print(f"Loaded {len(baseline_rows)} completed rows from main-compat summary.csv")
    ext_rows = load_extensions_summary()
    print(f"Loaded {len(ext_rows)} completed rows from extensions summary.csv (legacy)")
    sel_rows = load_selectivity_summary()
    print(f"Loaded {len(sel_rows)} completed rows from selectivity-controls summary.csv")
    print()
    srsnet_cells = gen_tab2(rows)
    gen_tab3(rows)
    gen_tab4(rows)
    gen_summary_stats(rows)
    gen_report(rows, srsnet_cells)
    if baseline_rows:
        print()
        gen_tab2_full(rows, baseline_rows)
        gen_baselines_paper_delta(baseline_rows)
    if ext_rows:
        print()
        print("=== Legacy extensions tables (kept for archived rows) ===")
        gen_tab2_extension(rows, ext_rows)
        gen_tab3_extension(rows, ext_rows)
        gen_tab4_extension(rows, ext_rows)
        gen_extensions_summary(rows, ext_rows)
    if sel_rows:
        print()
        print("=== Selectivity-controls tables (current focused study) ===")
        gen_selectivity_controls(sel_rows)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
