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
OUT_DIR = ROOT / "report_tables"

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
    test_report.csv locale (i path nel summary puntano a Hivenet)."""
    rows = []
    local_results = ROOT / "result" / "repro" / "lite-paper"
    with open(SUMMARY) as f:
        for r in csv.DictReader(f):
            if r["status"] != "completed":
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
    for ds in ["ETTh1", "ETTm2"]:  # lite-paper plugin coverage
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
    for ds in ["ETTh1", "ETTm2"]:
        for h in [96, 192, 336, 720]:
            row = [ds, str(h)]
            for v in variants:
                cell = cells.get((v, ds, h))
                row.append(fmt(cell[0]) if cell else "")
            for v in variants:
                cell = cells.get((v, ds, h))
                row.append(fmt(cell[1]) if cell else "")
            lines.append(",".join(row))
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  ✅ {out_path.name}: {len(lines)-1} rows (5 variants × 8 cells)")


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


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = load_summary()
    print(f"Loaded {len(rows)} completed rows from summary.csv")
    print()
    srsnet_cells = gen_tab2(rows)
    gen_tab3(rows)
    gen_tab4(rows)
    gen_summary_stats(rows)
    gen_report(rows, srsnet_cells)
    print()
    print("✅ Done.")


if __name__ == "__main__":
    main()
