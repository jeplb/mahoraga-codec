#!/usr/bin/env python3
# density vs. physical redundancy on the high-fidelity channel.
# mahoraga + prior codecs (dna-aeon, mgc+) at bench1 native operating points.
# distinct line styles + marker shapes so the plot reads in black-and-white.
#
# data:   ../paper/stable1_bench1_full.tsv (aggregated from data/bench1/*.json)
# writes: density_vs_r.{pdf,svg} next to this script

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

SCRIPT_DIR = Path(__file__).resolve().parent
SRC = SCRIPT_DIR.parent / "paper" / "stable1_bench1_full.tsv"
OUT_PDF = SCRIPT_DIR / "density_vs_r.pdf"
OUT_SVG = SCRIPT_DIR / "density_vs_r.svg"

STYLE = {
    "axis_fontsize": 11,
    "tick_fontsize": 10,
    "marker_label_fontsize": 10,
    "panel_title_fontsize": 12,
    "font_family": ["Helvetica", "Arial", "DejaVu Sans"],
}
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = STYLE["font_family"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False

# patent figure: black-only line art. codecs are distinguished by line style,
# marker shape, and fill (solid vs. open) rather than any color.
# tuple: (key, label, linestyle, marker, markersize, linewidth, fill, zorder)
CODECS = [
    ("mahoraga", "Mahoraga", "-",  "o", 8, 1.6, "solid", 5),
    ("dna_aeon", "DNA-Aeon", "--", "s", 7, 1.4, "open",  4),
    ("mgcplus",  "MGC+",     ":",  "^", 8, 1.4, "solid", 3),
]

CHANNEL = "hifi"


def load_points():
    """return {codec: [(r, density)]} for cells at 30/30 on the hifi channel."""
    by_codec = defaultdict(list)
    with open(SRC) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["channel"] != CHANNEL:
                continue
            if not row["n_success"] or not row["n_trials"]:
                continue
            if int(row["n_success"]) != int(row["n_trials"]):
                continue
            if not row["mediandensityebperg"]:
                continue
            by_codec[row["codec"]].append(
                (float(row["r"]), float(row["mediandensityebperg"]))
            )
    for k in by_codec:
        by_codec[k].sort()
    return by_codec


def main():
    pts = load_points()
    for codec_key, label, *_ in CODECS:
        rs = [p[0] for p in pts.get(codec_key, [])]
        ds = [p[1] for p in pts.get(codec_key, [])]
        print(f"  {label:<10} n={len(rs):>2}  r range [{min(rs) if rs else '—'}, {max(rs) if rs else '—'}]"
              f"  peak density {max(ds) if ds else '—':.2f}" if ds else
              f"  {label:<10} (no 30/30 points)")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    for codec_key, label, ls, marker, ms, lw, fill, z in CODECS:
        p = pts.get(codec_key, [])
        if not p:
            continue
        rs = [r for r, _ in p]
        ds = [d for _, d in p]
        face = "black" if fill == "solid" else "white"
        ax.plot(
            rs, ds,
            linestyle=ls, linewidth=lw, color="black",
            marker=marker, markersize=ms,
            markerfacecolor=face, markeredgecolor="black",
            markeredgewidth=1.0,
            label=label,
            zorder=z,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.015, 15)
    ax.set_ylim(0, 170)
    ax.set_xlabel("Physical redundancy $r$", fontsize=STYLE["axis_fontsize"])
    ax.set_ylabel("Storage density [EB g$^{-1}$]", fontsize=STYLE["axis_fontsize"])
    ax.tick_params(axis="both", labelsize=STYLE["tick_fontsize"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # legend listing each codec with its line+marker style (B&W-friendly)
    ax.legend(loc="upper right", frameon=False,
              fontsize=STYLE["marker_label_fontsize"],
              handlelength=3.0, handletextpad=0.6)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"\nwrote {OUT_PDF}")
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
