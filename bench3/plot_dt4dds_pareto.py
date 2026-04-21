#!/usr/bin/env python3
# mahoraga vs. prior codecs on the dt4dds channel.
# two-panel pareto plot (hifi + lofi).
#
# data:    ../data/bench3/mahoraga-bench3-{bestcase-hifi,worstcase-lofi}-*.json
#          + prior codec peaks hardcoded from gimpel 2026 fig 3c
# writes:  dt4dds_pareto.{pdf,svg} next to this script

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH3_DIR = SCRIPT_DIR.parent / "data" / "bench3"
OUT_PDF = SCRIPT_DIR / "dt4dds_pareto.pdf"
OUT_SVG = SCRIPT_DIR / "dt4dds_pareto.svg"

# shared style dict (also used by bench2 + bench4 plot scripts — each inlines
# its own copy; keep them in sync if you restyle)
STYLE = {
    "mahoraga_color": "#1f77b4",
    "mahoraga_edge": "black",
    "prior_face": "#888888",
    "prior_edge": "#555555",
    "caveat_color": "#555555",
    "envelope_size": 30,
    "envelope_alpha": 0.6,
    "envelope_linewidth": 1.2,
    "peak_size": 180,
    "peak_edge_linewidth": 1.5,
    "prior_size": 80,
    "prior_edge_linewidth": 0.5,
    "tick_fontsize": 10,
    "axis_fontsize": 11,
    "panel_title_fontsize": 12,
    "marker_label_fontsize": 10,
    "caveat_fontsize": 9,
    "font_family": ["Helvetica", "Arial", "DejaVu Sans"],
}

# matplotlib global style
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = STYLE["font_family"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False

# prior-codec peak data is the sd=30 cut from fig. 3c of gimpel 2026, sourced
# from the published supplement (41467_2026_70548_MOESM4_ESM.xlsx, sheet
# "Fig. 3c"). the xlsx tabulates (workflow, codec, code_rate, density); the
# physical redundancy r is back-computed from r = 113.7 * code_rate / density
# (the normaliser used in analysis.ipynb cell 11). for each codec the row with
# the highest density was taken. note: the paper text's quoted peaks
# (e.g. dna-aeon 140 EB/g, dna-rs 125 EB/g) describe the full-pareto-curve
# peak in fig. 3a, not the sd=30 cut used in fig. 3c.
PRIOR_HIFI = [
    {"name": "DNA-Aeon", "r": 0.483, "density": 117.81},
    {"name": "DNA-RS", "r": 0.974, "density": 116.73},
    {"name": "HEDGES", "r": 3.224, "density": 37.74},
    {"name": "DNA Fountain", "r": 7.569, "density": 15.02},
    {"name": "Goldman", "r": 17.81, "density": 2.17},
    {"name": "Yin-Yang", "r": 31.93, "density": 6.59},
]

# lofi peaks from the fig. 3c xlsx, same normalisation as hifi. dna-rs peak is
# at code_rate=1.00 (density 6.42), slightly ahead of code_rate=0.50 (6.17),
# so we pick the higher bar. dna fountain, goldman, and yin-yang rows are
# absent from the worstcase sheet — those codecs did not decode.
PRIOR_LOFI = [
    {"name": "DNA-Aeon", "r": 6.716, "density": 16.93},
    {"name": "DNA-RS", "r": 17.71, "density": 6.42},
    {"name": "HEDGES", "r": 8.446, "density": 8.48},
]

LOFI_CAVEAT = (
    "DNA Fountain, Goldman, and Yin-Yang\n"
    "did not decode on the low-fidelity channel"
)


def load_mahoraga_cells(pattern_glob, target_sd):
    """load bench3 jsons matching glob, return list of 30/30 cells at target sd.

    each cell: {"r": physical_redundancy, "sd": sequencing_depth, "density": density_eb_per_g}.
    """
    cells = []
    for path in sorted(BENCH3_DIR.glob(pattern_glob)):
        with open(path) as f:
            d = json.load(f)
        # schema adaptation: fields are physical_redundancy, sequencing_depth, n_success, n_trials, density_eb_per_g
        sd = d["sequencing_depth"]
        if abs(sd - target_sd) > 1e-6:
            continue
        if d["n_success"] != d["n_trials"]:
            continue
        cells.append({
            "r": d["physical_redundancy"],
            "sd": sd,
            "density": d["density_eb_per_g"],
        })
    return cells


def pareto_upper_envelope(cells):
    """compute upper-right pareto front: sort by r ascending, keep only points
    whose density is strictly greater than all points with smaller r.

    here "upper envelope" means the set of points not dominated — lower r with
    equal-or-higher density dominates. we want the monotonically decreasing
    density-vs-r curve (higher density is better, lower r is better).
    returns list of cells on the front, sorted by r ascending.
    """
    if not cells:
        return []
    sorted_cells = sorted(cells, key=lambda c: c["r"])
    front = []
    max_density_so_far = -float("inf")
    # walk from high r to low r, keep points with strictly greater density
    # (pareto optimal: no other point has both lower r and higher density)
    for c in reversed(sorted_cells):
        if c["density"] > max_density_so_far:
            front.append(c)
            max_density_so_far = c["density"]
    front.reverse()
    return front


def plot_panel(ax, cells, priors, title, ylim, mahoraga_sublabel):
    """draw one panel (hifi or lofi)."""
    # pareto envelope: all 30/30 points
    xs = [c["r"] for c in cells]
    ys = [c["density"] for c in cells]
    ax.scatter(
        xs, ys,
        s=STYLE["envelope_size"],
        facecolor=STYLE["mahoraga_color"],
        edgecolor="none",
        alpha=STYLE["envelope_alpha"],
        zorder=3,
    )

    # connect upper pareto front
    front = pareto_upper_envelope(cells)
    if len(front) >= 2:
        fx = [c["r"] for c in front]
        fy = [c["density"] for c in front]
        ax.plot(
            fx, fy,
            color=STYLE["mahoraga_color"],
            linewidth=STYLE["envelope_linewidth"],
            alpha=0.9,
            zorder=4,
        )

    # peak marker: highest-density 30/30 cell
    peak = max(cells, key=lambda c: c["density"])
    ax.scatter(
        [peak["r"]], [peak["density"]],
        s=STYLE["peak_size"],
        marker="o",
        facecolor=STYLE["mahoraga_color"],
        edgecolor=STYLE["mahoraga_edge"],
        linewidth=STYLE["peak_edge_linewidth"],
        zorder=6,
    )
    # label, placed to the right of the marker in log-x offset
    ax.annotate(
        "Mahoraga",
        xy=(peak["r"], peak["density"]),
        xytext=(10, 2),
        textcoords="offset points",
        fontsize=STYLE["marker_label_fontsize"],
        fontweight="bold",
        color=STYLE["mahoraga_color"],
        va="center",
        ha="left",
        zorder=7,
    )
    if mahoraga_sublabel:
        # place sublabel directly below the marker, centered — keeps it clear of DNA-Aeon at r=0.49
        ax.annotate(
            mahoraga_sublabel,
            xy=(peak["r"], peak["density"]),
            xytext=(0, -12),
            textcoords="offset points",
            fontsize=STYLE["marker_label_fontsize"] - 1,
            fontstyle="italic",
            color=STYLE["mahoraga_color"],
            va="top",
            ha="center",
            zorder=7,
        )

    # prior codecs
    for p in priors:
        ax.scatter(
            [p["r"]], [p["density"]],
            s=STYLE["prior_size"],
            facecolor=STYLE["prior_face"],
            edgecolor=STYLE["prior_edge"],
            linewidth=STYLE["prior_edge_linewidth"],
            zorder=5,
        )
        # default label offset
        xoff, yoff = 9, 0
        ha = "left"
        # manual overrides to reduce overlap
        if p["name"] == "DNA-Aeon":
            # sits close to mahoraga peak in hifi; push up-and-right
            xoff, yoff = 9, 8
        elif p["name"] == "DNA-RS":
            xoff, yoff = 9, -2
        elif p["name"] == "DNA Fountain":
            # lift above marker so it doesn't collide with yin-yang label
            xoff, yoff = 0, 10
            ha = "center"
        elif p["name"] == "Yin-Yang":
            xoff, yoff = -9, 0
            ha = "right"
        elif p["name"] == "Goldman":
            # hifi only; sits at (17.65, 2.17). marker is near the x-axis so
            # label goes above-right, into the open band between dna fountain
            # (7.6, 15) and yin-yang (30, 6.6)
            xoff, yoff = 9, 8
            ha = "left"
        ax.annotate(
            p["name"],
            xy=(p["r"], p["density"]),
            xytext=(xoff, yoff),
            textcoords="offset points",
            fontsize=STYLE["marker_label_fontsize"],
            color=STYLE["prior_edge"],
            va="center",
            ha=ha,
            zorder=6,
        )

    # axes
    ax.set_xscale("log")
    ax.set_xlim(0.1, 50)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Physical redundancy r", fontsize=STYLE["axis_fontsize"])
    ax.set_ylabel("Storage density [EB g$^{-1}$]", fontsize=STYLE["axis_fontsize"])
    ax.tick_params(axis="both", labelsize=STYLE["tick_fontsize"])
    ax.grid(False)
    # panel title, top-left above panel
    ax.set_title(title, fontsize=STYLE["panel_title_fontsize"], fontweight="bold", loc="left")


def main():
    # load mahoraga cells
    hifi_cells = load_mahoraga_cells("mahoraga-bench3-bestcase-hifi-*.json", target_sd=15.0)
    lofi_cells = load_mahoraga_cells("mahoraga-bench3-worstcase-lofi-*.json", target_sd=30.0)

    if not hifi_cells:
        raise RuntimeError("no hifi 30/30 cells at sd=15 — check bench3 jsons")
    if not lofi_cells:
        raise RuntimeError("no lofi 30/30 cells at sd=30 — check bench3 jsons")

    # sanity-check expected peaks
    hifi_peak = max(hifi_cells, key=lambda c: c["density"])
    lofi_peak = max(lofi_cells, key=lambda c: c["density"])
    print(f"hifi peak: r={hifi_peak['r']}, density={hifi_peak['density']:.2f} (n cells={len(hifi_cells)})")
    print(f"lofi peak: r={lofi_peak['r']}, density={lofi_peak['density']:.2f} (n cells={len(lofi_cells)})")

    # figure
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.1, 3.35))

    plot_panel(
        ax_a, hifi_cells, PRIOR_HIFI,
        title="A. High-fidelity channel",
        ylim=(0, 170),
        mahoraga_sublabel="sd=15",
    )
    plot_panel(
        ax_b, lofi_cells, PRIOR_LOFI,
        title="B. Low-fidelity channel",
        ylim=(0, 30),
        mahoraga_sublabel=None,
    )

    # lofi caveat annotation inside panel B frame, below data area
    ax_b.text(
        0.5, 0.04, LOFI_CAVEAT,
        transform=ax_b.transAxes,
        fontsize=STYLE["caveat_fontsize"],
        fontstyle="italic",
        color=STYLE["caveat_color"],
        ha="center",
        va="bottom",
    )

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
