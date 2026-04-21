#!/usr/bin/env python3
"""two-panel density vs coverage plot for benchmark-1.

benchmark-1 measures each codec AT ITS PAPER-DEFAULT OPERATING POINT.
every non-mahoraga codec runs with outer code rate fixed by its
upstream default (MGC+ default RS, DNA-RS n=255 k=170, DNA-Aeon
NOREC4DNA package_redundancy=0.4, DNA Fountain alpha=0.5). mahoraga
re-encodes per-r with RS parity sized for the expected dropout at
that r, which is why it's the only codec without a cliff here.

this is a correct measurement — it shows what a user gets out of the
box — but it is NOT a measurement of each inner code's fundamental
capability. a separate benchmark (benchmark-2) will re-parameterize
each codec's outer rate per-r to isolate inner-code quality.

reads paper/codec_comparison_real.csv (one row per codec, channel, r)
and renders codec_comparison.pdf alongside this script.

panels:
  left  = HiFi channel  (sigma=0.3 lognormal coverage, hifi IDS params)
  right = LoFi channel  (sigma=0.3 lognormal coverage, lofi IDS params)

axes:
  x = physical redundancy r  (coverage multiplier, log scale)
  y = information density    (EB/g, log scale)

marker semantics (benchmark-1 only plots where each codec actually
decodes at >=90% success; decode-failure points are intentionally
omitted so the plot does not mislead about what the inner code could
reach if the outer parity were adequate — that's benchmark-2's job):
  filled circle + solid line = n_success / n_trials >= 90% (decodes cleanly)

conventions enforced:
  - Arial font, sentence case, brackets for units
  - golden ratio width / height
  - Wong palette colorblind-safe
  - pdf.fonttype = 42 (editable in Illustrator)
  - 300 DPI
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["savefig.dpi"] = 300

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE.parent / "data" / "bench1" / "codec_comparison_real.csv"
OUT_PDF = HERE / "codec_comparison.pdf"

# Wong palette, one color per codec. mahoraga gets the strongest blue
# because it's the star of the comparison; the others get the other
# colorblind-safe hues in order of density ranking at r=5 hifi.
CODEC_COLOR = {
    "mahoraga":      "#0072B2",  # blue
    "dna_fountain": "#CC79A7",  # purple (fast but lofi-fragile)
    "dna_aeon":     "#D55E00",  # vermillion
    "dna_rs":       "#009E73",  # green
    "mgcplus":      "#E69F00",  # orange
}
CODEC_LABEL = {
    "mahoraga":      "Mahoraga (this work)",
    "dna_fountain": "DNA Fountain",
    "dna_aeon":     "DNA-Aeon",
    "dna_rs":       "DNA-RS",
    "mgcplus":      "MGC+",
}
# plot order also controls legend order + zorder (mahoraga on top)
CODEC_ORDER = ["mgcplus", "dna_rs", "dna_aeon", "dna_fountain", "mahoraga"]

SUCCESS_THRESHOLD = 0.9  # fraction of trials that must decode to count as "works"


def load_rows() -> list[dict]:
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def by_cell(rows: list[dict]) -> dict:
    out = {}
    for row in rows:
        key = (row["codec"], row["channel"], float(row["r"]))
        out[key] = row
    return out


def plot_panel(ax, rows_by_cell: dict, channel: str, title: str):
    # collect per-codec x/y series — only keep cells where the codec
    # actually decodes at the success threshold. failure cells are
    # intentionally dropped (see module docstring).
    r_values = sorted({k[2] for k in rows_by_cell if k[1] == channel})
    for codec in CODEC_ORDER:
        xs, ys = [], []
        for r in r_values:
            row = rows_by_cell.get((codec, channel, r))
            if row is None:
                continue
            n_trials = int(row["n_trials"])
            n_success = int(row["n_success"])
            if n_trials == 0:
                continue
            dens_str = row["median_density_eb_per_g"]
            if dens_str == "":
                continue
            if n_success / n_trials < SUCCESS_THRESHOLD:
                continue
            xs.append(r)
            ys.append(float(dens_str))

        if not xs:
            # codec never decodes on this channel — no line. still add a
            # legend-only handle so the user can see the codec was tested.
            ax.plot([], [], "-o",
                    color=CODEC_COLOR[codec], label=CODEC_LABEL[codec],
                    markersize=6, linewidth=2.0)
            continue

        ax.plot(
            xs, ys,
            "-o",
            color=CODEC_COLOR[codec], label=CODEC_LABEL[codec],
            markersize=6, markeredgewidth=1.0,
            markerfacecolor=CODEC_COLOR[codec],
            markeredgecolor=CODEC_COLOR[codec],
            linewidth=2.0,
            zorder=10 if codec == "mahoraga" else 5,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("coverage multiplier r")
    ax.set_ylabel("information density [EB/g]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.set_xlim(0.015, 13)


def main():
    rows = load_rows()
    rows_by_cell = by_cell(rows)

    # golden ratio, room for two panels side by side plus legend
    width = 9.5
    height = width / 1.618
    fig, (ax_hifi, ax_lofi) = plt.subplots(1, 2, figsize=(width, height),
                                           sharey=True)

    plot_panel(ax_hifi, rows_by_cell, "hifi", "High-fidelity channel")
    plot_panel(ax_lofi, rows_by_cell, "lofi", "Low-fidelity channel")

    # single shared legend above both panels (avoids duplicating 5 entries)
    handles, labels = ax_hifi.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        fontsize=9,
    )

    # footnote names this as benchmark-1 and points forward to benchmark-2
    fig.text(
        0.5, 0.02,
        "benchmark-1  -  each codec at its paper-default operating point  -  "
        "failure cells omitted",
        ha="center", va="bottom", fontsize=8, color="#444444",
    )

    plt.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
