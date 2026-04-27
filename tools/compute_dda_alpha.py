#!/usr/bin/env python3
"""Compute data-driven alpha (DDA) initialization values per dataset.

The SRSNet paper explicitly recommends manually initializing the
fusion weight ``alpha`` based on prior knowledge of the dataset
("increase it when the datasets are periodic and stationary, and
decrease it when the datasets are non-stationary and shifting" -- Sec.
6, Potential limitations, bullet 4).  This script encodes that
recommendation as a fully automatic procedure that derives ``alpha``
from a FFT-based seasonality strength of the *training split only*.

Output: ``tools/dda_alpha_values.json`` with the schema

    {"ETTh1": 2.45, "ETTh2": 1.85, ...}

Use these values via paper_repro.py's DDA tasks, which override the
``alpha`` hyper-parameter when launching ``SRSNet_DDA`` runs.

Seasonality metric:
    s = max_k |X(k)| / mean_k |X(k)|   for k != 0
where X is the FFT of each univariate series in the training split,
averaged over channels.  We then compute

    alpha_dda = ALPHA_BASE + ALPHA_SCALE * log(s + EPS)

with hard clipping to ALPHA_RANGE so the value stays inside the
sigmoid's sensitive region (sigmoid saturates beyond ~|alpha|=6).

References:
    * Paper Sec. 6, Potential limitations bullet 4 -- alpha init guidance.
    * Paper Sec. 6, Future work -- "more efficient update mechanism for
      alpha deserves exploration" -- DDA is a zero-cost prior used at
      init time only; gradient descent still adapts alpha during
      training.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Default search paths for ETT CSVs.  The first existing one wins.
DEFAULT_DATA_DIRS = [
    Path("/Users/dominikpanzarella/Downloads/forecasting"),
    Path.home() / "Downloads" / "forecasting",
    Path("/home/ubuntu/SRSNet/dataset/forecasting"),
    Path.cwd() / "dataset" / "forecasting",
]

# TFB rolling forecast uses a 60/20/20 split by default for ETT.
TRAIN_FRACTION = 0.6

# Parametri della curva alpha = base + scale * log(seasonality).
# Scale tuned so that ETT seasonalities (~100-500) map to alpha in
# roughly [2.5, 3.5], i.e. nearby but distinguishable values around
# the paper's default alpha=2.0 and the script-mode alpha=3.5.
ALPHA_BASE = 1.8          # below paper default to leave room above
ALPHA_SCALE = 0.25        # gentle log-modulation
ALPHA_MIN = 1.5           # lower bound (non-stationary signals)
ALPHA_MAX = 4.0           # upper bound (highly periodic signals)
EPS = 1e-6


def _locate_csv(dataset: str, override_dir: Path | None) -> Path:
    dirs = [override_dir] if override_dir else []
    dirs.extend(DEFAULT_DATA_DIRS)
    for base in dirs:
        if base is None:
            continue
        candidate = Path(base) / f"{dataset}.csv"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate {dataset}.csv in any of: "
        + ", ".join(str(d) for d in dirs if d is not None)
    )


def _load_wide(csv_path: Path) -> pd.DataFrame:
    """Read a TFB long-format CSV and pivot to a wide DataFrame.

    Expected schema: ``date,data,cols`` with one row per (timestamp,
    channel).  The wide DataFrame has timestamps in the index and
    one numeric column per channel.
    """
    raw = pd.read_csv(csv_path)
    expected = {"date", "data", "cols"}
    if not expected.issubset(raw.columns):
        # Fallback to "already wide": assume first column is the date.
        wide = raw.set_index(raw.columns[0])
        return wide.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    wide = raw.pivot(index="date", columns="cols", values="data")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    return wide.apply(pd.to_numeric, errors="coerce")


def _seasonality_strength(values: np.ndarray) -> float:
    """FFT-based seasonality strength averaged over channels.

    For each univariate column we compute the modulus of the rfft,
    drop the DC component, then take ``max / mean``.  This yields a
    scalar >= 1 (== 1 when the spectrum is flat, e.g. white noise).
    """
    if values.ndim == 1:
        values = values[:, None]
    strengths = []
    for col in range(values.shape[1]):
        series = values[:, col]
        series = series[~np.isnan(series)]
        if len(series) < 16:
            continue
        # Demean to suppress DC and reduce sensitivity to mean shifts.
        series = series - np.mean(series)
        spectrum = np.abs(np.fft.rfft(series))
        if len(spectrum) <= 2:
            continue
        spectrum = spectrum[1:]  # drop DC
        denom = float(np.mean(spectrum))
        if denom <= EPS:
            continue
        strengths.append(float(np.max(spectrum)) / denom)
    if not strengths:
        return 1.0
    return float(np.mean(strengths))


def compute_alpha(values: np.ndarray) -> tuple[float, float]:
    """Return (alpha_dda, seasonality) for a wide value matrix."""
    n_train = max(int(len(values) * TRAIN_FRACTION), 64)
    train = values[:n_train]
    s = _seasonality_strength(train)
    alpha = ALPHA_BASE + ALPHA_SCALE * math.log(s + EPS)
    alpha = max(ALPHA_MIN, min(ALPHA_MAX, alpha))
    return round(alpha, 4), round(s, 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
        help="Datasets to process (matches *.csv filename).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override directory containing the CSVs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "dda_alpha_values.json",
        help="Where to dump the JSON mapping.",
    )
    args = parser.parse_args()

    result = {}
    diagnostics = {}
    for dataset in args.datasets:
        csv_path = _locate_csv(dataset, args.data_dir)
        wide = _load_wide(csv_path)
        values = wide.to_numpy(dtype=np.float64)
        alpha, seasonality = compute_alpha(values)
        result[dataset] = alpha
        diagnostics[dataset] = {
            "csv": str(csv_path),
            "rows": int(len(values)),
            "channels": int(values.shape[1] if values.ndim > 1 else 1),
            "seasonality_strength": seasonality,
            "alpha_dda": alpha,
        }
        print(
            f"{dataset:<8}  rows={len(values):>6}  "
            f"seasonality={seasonality:8.3f}  -> alpha={alpha:.4f}",
            file=sys.stderr,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    diag_path = args.output.with_suffix(".diagnostics.json")
    diag_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.output} and {diag_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
