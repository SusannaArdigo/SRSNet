#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "result/repro/selectivity-controls-full-simple/selectivity_controls"
OUT_CSV = ROOT / "report_tables/selectivity_controls_full.csv"
OUT_SUMMARY = ROOT / "report_tables/selectivity_controls_full_summary.csv"

TASK_RE = re.compile(
    r"^selectivity_controls_(?P<dataset>ETTh1|ETTh2|ETTm1|ETTm2)"
    r"_H(?P<horizon>\d+)_(?P<variant>SRSNet(?:_[A-Za-z]+)*)_s(?P<seed>\d+)$"
)


def parse_report(path: Path):
    """Return (mse_norm, mae_norm) from a test_report*.csv file."""
    metrics = {}
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 3:
                continue
            _, metric_name, value = row[:3]
            try:
                metrics[metric_name] = float(value)
            except ValueError:
                continue
    return metrics.get("mse_norm"), metrics.get("mae_norm")


def main() -> int:
    rows = []
    missing = []
    for task_dir in sorted(RESULTS_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        m = TASK_RE.match(task_dir.name)
        if not m:
            print(f"skip (no match): {task_dir.name}", file=sys.stderr)
            continue
        ds = m.group("dataset")
        h = int(m.group("horizon"))
        seed = int(m.group("seed"))
        variant = m.group("variant")

        report_files = list(task_dir.glob("test_report.*.csv"))
        if not report_files:
            missing.append(task_dir.name)
            continue
        mse, mae = parse_report(report_files[0])
        if mse is None:
            missing.append(task_dir.name)
            continue
        rows.append({
            "dataset": ds,
            "horizon": h,
            "seed": seed,
            "variant": variant,
            "mse_norm": mse,
            "mae_norm": mae or "",
        })

    if missing:
        print(f"WARNING: {len(missing)} tasks missing test_report:", file=sys.stderr)
        for n in missing[:10]:
            print(f"  - {n}", file=sys.stderr)

    rows.sort(key=lambda r: (r["dataset"], r["horizon"], r["variant"], r["seed"]))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["dataset", "horizon", "seed", "variant", "mse_norm", "mae_norm"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")

    # ---- Summary: per-(dataset, horizon, variant) aggregate over 5 seeds -----
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["dataset"], r["horizon"], r["variant"])].append(r["mse_norm"])

    # paired-delta vs SRSNet baseline (same (dataset, horizon, seed))
    by_baseline = {}
    for r in rows:
        if r["variant"] == "SRSNet":
            by_baseline[(r["dataset"], r["horizon"], r["seed"])] = r["mse_norm"]

    by_variant_delta = defaultdict(list)
    for r in rows:
        if r["variant"] == "SRSNet":
            continue
        base = by_baseline.get((r["dataset"], r["horizon"], r["seed"]))
        if base is None:
            continue
        by_variant_delta[r["variant"]].append((r["mse_norm"] - base) / base * 100.0)

    summary_rows = []
    for (ds, h, variant), msev in sorted(by_cell.items()):
        mean = statistics.mean(msev)
        sd = statistics.stdev(msev) if len(msev) > 1 else 0.0
        summary_rows.append({
            "dataset": ds,
            "horizon": h,
            "variant": variant,
            "mean_mse": round(mean, 6),
            "std_mse": round(sd, 6),
            "n_seeds": len(msev),
        })

    with OUT_SUMMARY.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["dataset", "horizon", "variant", "mean_mse", "std_mse", "n_seeds"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {OUT_SUMMARY} ({len(summary_rows)} rows)")

    # ---- Per-variant overall summary across all 16 cells x 5 seeds ----------
    print()
    print("=" * 78)
    print("Per-variant summary (mean delta% vs SRSNet baseline, paired by seed):")
    print("=" * 78)
    print(f"{'Variant':35s}  {'Mean Δ%':>9s}  {'Std %':>7s}  {'n':>4s}  {'Wins/n':>8s}")
    for variant in sorted(by_variant_delta):
        deltas = by_variant_delta[variant]
        wins = sum(1 for d in deltas if d < 0)
        mean = statistics.mean(deltas)
        sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        print(f"{variant:35s}  {mean:>+8.3f}%  {sd:>6.3f}%  {len(deltas):>4d}  {wins:>3d}/{len(deltas)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
