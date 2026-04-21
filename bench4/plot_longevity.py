#!/usr/bin/env python3
# density-vs-projected-longevity pareto at 25°C dry storage.
# each codec is a density-vs-years curve across r_initial; upper-right wins.
# points with cliff==r_initial (zero longevity margin) are excluded — see
# the console log for the audit.
#
# data:   ../data/bench4/*.json
# writes: longevity.{pdf,svg} next to this script
#
# the chemistry model lives in the paper Methods and in cliff_to_years()
# below. keep the constants in sync.

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH4_DIR = SCRIPT_DIR.parent / "data" / "bench4"
OUT_PDF = SCRIPT_DIR / "longevity.pdf"
OUT_SVG = SCRIPT_DIR / "longevity.svg"

# shared style (inlined — see bench3/plot_dt4dds_pareto.py)
STYLE = {
    "mahoraga_color": "#1f77b4",
    "prior_edge": "#555555",
    "tick_fontsize": 10,
    "axis_fontsize": 11,
    "panel_title_fontsize": 12,
    "marker_label_fontsize": 10,
    "font_family": ["Helvetica", "Arial", "DejaVu Sans"],
}
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = STYLE["font_family"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False

# chemistry constants (Methods)
K0 = 3.5e-9
E_A = 126e3
R_GAS = 8.314
T_37 = 310.15
T_STORAGE = 298.15
S_DRY = 300
N_PURINES = 63

# density normaliser — must match codec/hmm_rs/src/lib.rs and docker/*/entrypoint.py
EB_PER_G_PER_BIT_PER_BASE = 113.7

# codec metadata: (key, label, face, edge, line_lw, marker_size, edge_lw)
CODECS = [
    ("mahoraga", "Mahoraga", STYLE["mahoraga_color"], "black",   1.8, 120, 1.2),
    ("mgcplus",  "MGC+",     "#888888",               "#555555", 1.2,  80, 0.5),
    ("dna_aeon", "DNA-Aeon", "#bbbbbb",               "#888888", 1.0,  80, 0.5),
]

R_INITIAL_VALUES = (2.0, 5.0, 10.0)

# expected (cliff, density, years) for sanity check
EXPECTED = {
    ("mahoraga",  5.0): (3.25,  34.1, 133.0),
    ("mahoraga", 10.0): (4.0,   17.1, 282.0),
    ("mgcplus",   5.0): (3.5,   19.5, 110.0),
    ("mgcplus",  10.0): (4.0,    9.9, 282.0),
    ("dna_aeon", 10.0): (5.0,   11.0, 214.0),
}


def cliff_to_years(r_initial: float, cliff_r: float) -> float:
    if cliff_r >= r_initial:
        return 0.0
    surviving_fraction = cliff_r / r_initial
    k_storage = K0 * np.exp(E_A / R_GAS * (1.0 / T_37 - 1.0 / T_STORAGE)) / S_DRY
    seconds_to_cliff = -np.log(surviving_fraction) / (N_PURINES * k_storage)
    return seconds_to_cliff / (86400.0 * 365.25)


def load_bench4_all(codec: str):
    """return list of (r_initial, cliff_r_or_None, density) for every r_init
    run found for `codec` in bench4. base and gapfill runs are merged by
    summing trial counts per channel_r.
    """
    runs = defaultdict(lambda: {"meta": None, "cells": defaultdict(lambda: [0, 0])})
    for fp in sorted(BENCH4_DIR.glob("*.json")):
        with open(fp) as f:
            d = json.load(f)
        if d["codec"] != codec:
            continue
        r_init = d["r_initial"]
        if runs[r_init]["meta"] is None:
            runs[r_init]["meta"] = d
        for cell in d["cells"]:
            agg = runs[r_init]["cells"][cell["channel_r"]]
            agg[0] += cell["n_success"]
            agg[1] += cell["n_trials"]

    results = []
    for r_init in sorted(runs):
        meta = runs[r_init]["meta"]
        cands = [cr for cr, (s, _) in runs[r_init]["cells"].items() if s >= 29]
        cliff_r = min(cands) if cands else None
        density = (
            meta["input_size"] * 8
            / (meta["n_seqs"] * meta["seq_len"] * r_init)
            * EB_PER_G_PER_BIT_PER_BASE
        )
        results.append((r_init, cliff_r, density))
    return results


def plot_figure(pareto_by_codec):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    for codec_key, label, face, edge, lw, ms, elw in CODECS:
        pts = pareto_by_codec.get(codec_key, [])
        if not pts:
            continue
        pts_sorted = sorted(pts, key=lambda p: p[0])  # sort by years
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        r_inits = [p[2] for p in pts_sorted]
        if len(pts_sorted) >= 2:
            ax.plot(xs, ys, color=face, linewidth=lw, zorder=3, label=label)
            ax.scatter(xs, ys, s=ms, facecolor=face, edgecolor=edge,
                       linewidth=elw, zorder=4)
        else:
            # single point — still needs a legend entry, so scatter with label
            ax.scatter(xs, ys, s=ms, facecolor=face, edgecolor=edge,
                       linewidth=elw, zorder=4, label=label)
        for x, y, ri in zip(xs, ys, r_inits):
            ax.annotate(
                f"r={ri:g}",
                xy=(x, y), xytext=(8, 0),
                textcoords="offset points",
                fontsize=9, color=edge,
                va="center", ha="left",
                zorder=5,
            )

    ax.set_xlim(50, 350)
    ax.set_ylim(0, 90)
    ax.set_xlabel("Projected storage time at 25°C dry [years]",
                  fontsize=STYLE["axis_fontsize"])
    ax.set_ylabel("Storage density [EB g$^{-1}$]",
                  fontsize=STYLE["axis_fontsize"])
    ax.tick_params(axis="both", labelsize=STYLE["tick_fontsize"])
    ax.set_title("Density-longevity trade-off at 25°C dry storage",
                 fontsize=STYLE["axis_fontsize"], pad=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.legend(loc="upper right", frameon=False, fontsize=9, handletextpad=0.6)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def verify(raw_by_codec, pareto_by_codec):
    print("bench4 audit (all r_initial values):")
    for codec_key, *_ in CODECS:
        for r_init, cliff_r, density in raw_by_codec[codec_key]:
            if cliff_r is None:
                status = "fails (no 29/30 cliff)"
            else:
                y = cliff_to_years(r_init, cliff_r)
                status = f"cliff={cliff_r} years={y:.2f} density={density:.2f}"
                if y == 0.0:
                    status += "  [zero-years, excluded from plot]"
            print(f"  {codec_key:<10} r={r_init:<5}  {status}")

    print("\nexpected-value check:")
    for (codec, r_init), (exp_cliff, exp_d, exp_y) in EXPECTED.items():
        pts = [p for p in pareto_by_codec[codec] if p[2] == r_init]
        if not pts:
            print(f"  MISS: {codec} r={r_init} not in plotted set")
            continue
        y, d, _ = pts[0]
        raw = [r for r in raw_by_codec[codec] if r[0] == r_init][0]
        cliff_r = raw[1]
        ok = (abs(cliff_r - exp_cliff) < 0.01
              and abs(d - exp_d) < 0.5
              and abs(y - exp_y) < 2.0)
        tag = "ok" if ok else "DRIFT"
        print(f"  {tag}: {codec} r={r_init}  cliff={cliff_r} density={d:.2f} "
              f"years={y:.1f}  (expected {exp_cliff}, {exp_d}, {exp_y})")


def main():
    raw_by_codec = {k: load_bench4_all(k) for k, *_ in CODECS}
    # plottable pareto points: drop None cliffs and zero-years entries
    pareto_by_codec = {}
    for codec_key, *_ in CODECS:
        pts = []
        for r_init, cliff_r, density in raw_by_codec[codec_key]:
            if cliff_r is None:
                continue
            years = cliff_to_years(r_init, cliff_r)
            if years <= 0:
                continue
            pts.append((years, density, r_init))
        pareto_by_codec[codec_key] = pts

    verify(raw_by_codec, pareto_by_codec)
    plot_figure(pareto_by_codec)
    print(f"\nwrote {OUT_PDF}")
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
