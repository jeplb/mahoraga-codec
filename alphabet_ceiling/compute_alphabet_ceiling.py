#!/usr/bin/env python3
"""Alphabet-ceiling fraction statistics for the Mahoraga paper.

reads the matched-parity benchmark JSONs under data/bench2/ and computes,
for every (channel, r) cell, what fraction of the quaternary alphabet
ceiling each codec realises. the alphabet ceiling is 2 bits per base pair;
after accounting for stochastic dropout, the maximum density achievable at
physical redundancy r is

    rho_max  = 2 * (1 - exp(-r)) / r * EB_PER_G_PER_BIT_PER_BASE
    fraction = realized_density / rho_max   (30/30 cells only)

the ceiling is channel-independent: it depends only on alphabet size and
the Poisson survival fraction. channel-specific capacity bounds (with
per-base substitution and indel rates) are strictly tighter and would give
fractions closer to 100%. this cross-channel formulation is the only
universal closed-form bound, which is why the paper uses it.

output:
    alphabet_ceiling.csv   (written next to this script)

run:
    python3 alphabet_ceiling/compute_alphabet_ceiling.py
"""

from __future__ import annotations

import glob
import json
import math
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BENCH2_DIR = REPO_ROOT / "data" / "bench2"
OUT_CSV = SCRIPT_DIR / "alphabet_ceiling.csv"

# dsDNA mass → storage-density conversion (derivation in codec/mahoraga_py/pipeline.py).
EB_PER_G_PER_BIT_PER_BASE = 113.7

# alphabet size: 4 nucleotides → log2(4) = 2 bits/base.
BITS_PER_BASE = 2.0

CODECS = ["mahoraga", "dna_aeon", "mgcplus"]
N_TRIALS_FULL = 30  # bench2 runs 30 trials per cell


def alphabet_ceiling_density(r: float) -> float:
    """rho_max at coverage r (EB/g). channel-independent."""
    if r <= 0.0:
        return float("nan")
    coverage_factor = (1.0 - math.exp(-r)) / r
    return BITS_PER_BASE * coverage_factor * EB_PER_G_PER_BIT_PER_BASE


def load_bench2_records(bench2_dir: Path) -> pd.DataFrame:
    """parse bench2 JSONs into one row per trial.

    schema (confirmed against data/bench2/*.json):
      top-level:  codec, channel, r, depth, input_size, n_trials, results[]
      per trial:  trial, seed, success (bool), density_eb_per_g, ...

    density_eb_per_g is present per trial in every file spot-checked; the
    loop falls back to computing it from (input_size, n_seqs, r) only if
    the field is missing AND the trial succeeded. NB: the back-off formula
    assumes the mahoraga 126-nt useful-base convention; it will under-count
    density for codecs with different strand layouts, so warn when it fires.
    """
    rows: List[dict] = []
    fallback_used = 0
    for path in sorted(glob.glob(str(bench2_dir / "*bench2*.json"))):
        with open(path) as f:
            d = json.load(f)
        codec = d["codec"]
        channel = d["channel"]
        r = float(d["r"])
        input_size = d.get("input_size")
        for t in d.get("results", []):
            success = bool(t.get("success", False))
            density = t.get("density_eb_per_g")
            if density is None and success and input_size and t.get("n_seqs", 0) > 0 and r > 0:
                density = input_size * 8.0 / (t["n_seqs"] * 126.0 * r) * EB_PER_G_PER_BIT_PER_BASE
                fallback_used += 1
            rows.append({
                "codec": codec,
                "channel": channel,
                "r": r,
                "trial": t["trial"],
                "success": success,
                "density_eb_per_g": density if density is not None else float("nan"),
            })
    if fallback_used:
        warnings.warn(
            f"density_eb_per_g missing in {fallback_used} trial(s); used "
            "input_size/n_seqs fallback (may under-count for codecs that don't "
            "use the 126-nt useful-base convention).",
            RuntimeWarning, stacklevel=2,
        )
    if not rows:
        raise FileNotFoundError(f"no bench2 JSONs found under {bench2_dir}")
    return pd.DataFrame(rows)


def aggregate_per_cell(df: pd.DataFrame) -> pd.DataFrame:
    """reduce to one row per (codec, channel, r) with n_success and mean density
    over successful trials. non-decoding cells keep NaN density and count 0.
    """
    def _agg(g: pd.DataFrame) -> pd.Series:
        succ = g[g["success"]]
        return pd.Series({
            "n_trials": len(g),
            "n_trials_success": int(succ.shape[0]),
            "density_ebpg": float(succ["density_eb_per_g"].mean()) if len(succ) else float("nan"),
        })
    grouped = df.groupby(["codec", "channel", "r"]).apply(_agg, include_groups=False)
    return grouped.reset_index()


def attach_alphabet_ceiling(df: pd.DataFrame) -> pd.DataFrame:
    """add alphabet_ceiling_ebpg and alphabet_ceiling_pct columns.

    only rows with n_trials_success == N_TRIALS_FULL (full 30/30) get a
    non-NaN alphabet_ceiling_pct; partial-success cells stay NaN so missing
    operating points do not silently contaminate downstream stats.
    """
    out = df.copy()
    out["alphabet_ceiling_ebpg"] = out["r"].apply(alphabet_ceiling_density)
    passes = out["n_trials_success"] == N_TRIALS_FULL
    frac = np.where(
        passes & (out["alphabet_ceiling_ebpg"] > 0),
        out["density_ebpg"] / out["alphabet_ceiling_ebpg"] * 100.0,
        np.nan,
    )
    out["alphabet_ceiling_pct"] = frac
    return out


def sanity_checks(df: pd.DataFrame) -> None:
    """assertions + warnings called out in the spec."""
    # monotonic non-increasing ceiling in r (per channel; ceiling is actually
    # channel-independent so this also checks that invariant).
    for channel in sorted(df["channel"].unique()):
        sub = df[df["channel"] == channel].sort_values("r")
        bounds = sub["alphabet_ceiling_ebpg"].to_numpy()
        diffs = np.diff(bounds)
        assert np.all(diffs <= 1e-9), (
            f"alphabet ceiling must be non-increasing in r for channel={channel}; "
            f"offending diffs: {diffs}"
        )
    # overshoots (fraction > 1.0): flag loudly but do not crash. the alphabet
    # ceiling is a loose bound, so >100% should not happen — if it does, the
    # density computation is off.
    overshoots = df[(df["alphabet_ceiling_pct"].notna()) & (df["alphabet_ceiling_pct"] > 100.0)]
    for _, row in overshoots.iterrows():
        warnings.warn(
            f"alphabet ceiling fraction > 100%: codec={row['codec']} channel={row['channel']} "
            f"r={row['r']} frac={row['alphabet_ceiling_pct']:.2f}% — alphabet ceiling is a "
            "loose bound so >100% indicates a problem in the density computation.",
            RuntimeWarning, stacklevel=2,
        )
    valid = df[df["alphabet_ceiling_pct"].notna()]
    assert (valid["alphabet_ceiling_pct"] >= 0.0).all(), "negative alphabet ceiling fraction"


def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "channel", "r", "codec",
        "density_ebpg", "alphabet_ceiling_ebpg", "alphabet_ceiling_pct",
        "n_trials_success",
    ]
    out = df[cols].sort_values(["channel", "r", "codec"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.4f")
    print(f"wrote {out_path}  ({len(out)} rows)")


def main() -> int:
    df_trials = load_bench2_records(BENCH2_DIR)
    print(f"loaded {len(df_trials)} per-trial records from {BENCH2_DIR}")
    print(f"  codecs:   {sorted(df_trials['codec'].unique())}")
    print(f"  channels: {sorted(df_trials['channel'].unique())}")
    print(f"  r values: {sorted(df_trials['r'].unique())}")
    print()

    agg = aggregate_per_cell(df_trials)
    agg = attach_alphabet_ceiling(agg)
    sanity_checks(agg)

    write_csv(agg, OUT_CSV)

    # summary print
    print()
    print("per-codec 30/30 alphabet-ceiling fractions (mean over cells that decoded):")
    full = agg[agg["n_trials_success"] == N_TRIALS_FULL]
    summary = full.groupby(["codec", "channel"])["alphabet_ceiling_pct"].agg(["mean", "count"])
    print(summary.round(2).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
