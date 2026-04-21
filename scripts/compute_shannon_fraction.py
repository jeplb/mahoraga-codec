#!/usr/bin/env python3
"""Shannon-fraction statistics for the Mahoraga paper.

reads the matched-parity benchmark JSONs under data/bench2/ and computes,
for every (channel, r) cell:

    C           = per-base capacity upper bound of the IDS channel
                  = 2 * (1 - h(p_sub)) - 2 * (p_ins + p_del)
    rho_shannon = C * (1 - exp(-r)) / r * EB_PER_G_PER_BIT_PER_BASE
    fraction    = realized_density / rho_shannon   (per codec, 30/30 cells only)

output:
    outputs/shannon_fraction.csv   one row per (channel, r, codec)

channel error rates are pulled from the idsim source (channel.rs:38-48)
and printed at startup for reproducibility.

run:
    python3 scripts/compute_shannon_fraction.py
"""

from __future__ import annotations

import glob
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BENCH2_DIR = REPO_ROOT / "data" / "bench2"
OUT_CSV = REPO_ROOT / "outputs" / "shannon_fraction.csv"

# dsDNA mass → storage-density conversion (derivation in codec/mahoraga_py/pipeline.py).
EB_PER_G_PER_BIT_PER_BASE = 113.7

# channel error rates (source: idsim/src/channel.rs, methods ChannelParams::hifi/lofi).
# combined IDS probabilities are synth + seq added (independent-event approximation,
# valid in the low-rate regime where both rates << 1).
CHANNEL_RATES = {
    "hifi": {
        "synth_sub": 0.0005, "synth_del": 0.0002, "synth_ins": 0.0001,
        "seq_sub":   0.0008, "seq_del":   0.0001, "seq_ins":   0.00005,
    },
    "lofi": {
        "synth_sub": 0.005,  "synth_del": 0.005,  "synth_ins": 0.001,
        "seq_sub":   0.003,  "seq_del":   0.0005, "seq_ins":   0.0002,
    },
}

CODECS = ["mahoraga", "dna_aeon", "mgcplus"]
N_TRIALS_FULL = 30  # bench2 runs 30 trials per cell


def _h_binary(p: float) -> float:
    """binary entropy, clamped on the endpoints."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def combined_rates(channel: str) -> Dict[str, float]:
    """synth + seq rates, additive (independent-event approximation).

    returns {p_sub, p_del, p_ins}.
    """
    r = CHANNEL_RATES[channel]
    return {
        "p_sub": r["synth_sub"] + r["seq_sub"],
        "p_del": r["synth_del"] + r["seq_del"],
        "p_ins": r["synth_ins"] + r["seq_ins"],
    }


def capacity_per_base(p_sub: float, p_del: float, p_ins: float) -> float:
    """IDS-channel capacity upper bound (bits/base).

    C = 2 * (1 - h(p_sub)) - 2 * (p_ins + p_del)
    """
    return 2.0 * (1.0 - _h_binary(p_sub)) - 2.0 * (p_ins + p_del)


def shannon_bound_density(channel: str, r: float) -> float:
    """rho_shannon at coverage r (EB/g)."""
    cr = combined_rates(channel)
    C = capacity_per_base(cr["p_sub"], cr["p_del"], cr["p_ins"])
    if r <= 0.0:
        return float("nan")
    coverage_factor = (1.0 - math.exp(-r)) / r
    return C * coverage_factor * EB_PER_G_PER_BIT_PER_BASE


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


def attach_shannon(df: pd.DataFrame) -> pd.DataFrame:
    """add shannon_bound_ebpg and shannon_fraction_pct columns.

    only rows with n_trials_success == N_TRIALS_FULL (full 30/30) get a
    non-NaN shannon_fraction_pct; partial-success cells stay NaN so missing
    operating points do not silently contaminate downstream stats.
    """
    out = df.copy()
    out["shannon_bound_ebpg"] = out.apply(
        lambda row: shannon_bound_density(row["channel"], row["r"]), axis=1
    )
    passes = out["n_trials_success"] == N_TRIALS_FULL
    frac = np.where(
        passes & (out["shannon_bound_ebpg"] > 0),
        out["density_ebpg"] / out["shannon_bound_ebpg"] * 100.0,
        np.nan,
    )
    out["shannon_fraction_pct"] = frac
    return out


def sanity_checks(df: pd.DataFrame) -> None:
    """assertions + warnings called out in the spec."""
    # monotonic non-increasing shannon bound in r (per channel)
    for channel in sorted(df["channel"].unique()):
        sub = df[df["channel"] == channel].sort_values("r")
        bounds = sub["shannon_bound_ebpg"].to_numpy()
        diffs = np.diff(bounds)
        assert np.all(diffs <= 1e-9), (
            f"shannon bound must be non-increasing in r for channel={channel}; "
            f"offending diffs: {diffs}"
        )
    # overshoots (fraction > 1.0): flag loudly but do not crash — a loose bound
    # formulation could legitimately produce fractions above 1.
    overshoots = df[(df["shannon_fraction_pct"].notna()) & (df["shannon_fraction_pct"] > 100.0)]
    for _, row in overshoots.iterrows():
        warnings.warn(
            f"shannon fraction > 100%: codec={row['codec']} channel={row['channel']} "
            f"r={row['r']} frac={row['shannon_fraction_pct']:.2f}% — capacity bound "
            "may be loose, or realized density is being overestimated.",
            RuntimeWarning, stacklevel=2,
        )
    # fraction in [0, 100] for rows that have it
    valid = df[df["shannon_fraction_pct"].notna()]
    assert (valid["shannon_fraction_pct"] >= 0.0).all(), "negative shannon fraction"


def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "channel", "r", "codec",
        "density_ebpg", "shannon_bound_ebpg", "shannon_fraction_pct",
        "n_trials_success",
    ]
    out = df[cols].sort_values(["channel", "r", "codec"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.4f")
    print(f"wrote {out_path}  ({len(out)} rows)")


def print_channel_provenance() -> None:
    print("channel error rates (source: idsim/src/channel.rs ChannelParams):")
    for ch in ("hifi", "lofi"):
        raw = CHANNEL_RATES[ch]
        combo = combined_rates(ch)
        C = capacity_per_base(combo["p_sub"], combo["p_del"], combo["p_ins"])
        print(
            f"  {ch}: synth(sub={raw['synth_sub']:.4g}, del={raw['synth_del']:.4g}, "
            f"ins={raw['synth_ins']:.4g})  "
            f"seq(sub={raw['seq_sub']:.4g}, del={raw['seq_del']:.4g}, "
            f"ins={raw['seq_ins']:.4g})  "
            f"combined(sub={combo['p_sub']:.4g}, del={combo['p_del']:.4g}, "
            f"ins={combo['p_ins']:.4g})  C={C:.4f} bits/base"
        )
    print()


def main() -> int:
    print_channel_provenance()

    df_trials = load_bench2_records(BENCH2_DIR)
    print(f"loaded {len(df_trials)} per-trial records from {BENCH2_DIR}")
    print(f"  codecs:   {sorted(df_trials['codec'].unique())}")
    print(f"  channels: {sorted(df_trials['channel'].unique())}")
    print(f"  r values: {sorted(df_trials['r'].unique())}")
    print()

    agg = aggregate_per_cell(df_trials)
    agg = attach_shannon(agg)
    sanity_checks(agg)

    write_csv(agg, OUT_CSV)

    # summary print
    print()
    print("per-codec 30/30 shannon fractions (mean over cells that decoded):")
    full = agg[agg["n_trials_success"] == N_TRIALS_FULL]
    summary = full.groupby(["codec", "channel"])["shannon_fraction_pct"].agg(["mean", "count"])
    print(summary.round(2).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
