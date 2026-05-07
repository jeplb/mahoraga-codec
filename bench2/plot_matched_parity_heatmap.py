#!/usr/bin/env python3
# matched-outer-parity heatmap (codecs × physical-redundancy r), hifi + lofi.
#
# per (codec, channel, r) cell:
#   - mean density across the 30 trials, restricted to successful trials
#   - cell color encodes mean density when 30/30 trials decoded
#   - cells with 0 < n_success < 30 drawn light grey, annotated "k/30"
#   - cells with 0 / 30 trials decoded drawn white, annotated "0/30"
#     (distinguishes "tested and failed" from "not tested")
#
# data:   ../data/bench2/v2/*.json
# writes: matched_parity_heatmap.{pdf,svg} next to this script.

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH2_DIR = SCRIPT_DIR.parent / "data" / "bench2" / "v2"
OUT_PDF = SCRIPT_DIR / "matched_parity_heatmap.pdf"
OUT_SVG = SCRIPT_DIR / "matched_parity_heatmap.svg"

R_GRID = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
CODECS = [
    ("mahoraga",  "Mahoraga"),
    ("dna_aeon",  "DNA-Aeon"),
    ("mgcplus",   "MGC+"),
]

# uniform Arial, editable PDF, single fontsize across the figure.
FONT_SIZE = 10
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["font.size"] = FONT_SIZE
mpl.rcParams["axes.titlesize"] = FONT_SIZE
mpl.rcParams["axes.labelsize"] = FONT_SIZE
mpl.rcParams["xtick.labelsize"] = FONT_SIZE
mpl.rcParams["ytick.labelsize"] = FONT_SIZE
mpl.rcParams["legend.fontsize"] = FONT_SIZE
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False

# colormap: viridis is colorblind-safe and reads density-as-luminance well.
CMAP = mpl.colormaps["viridis"]
PARTIAL_FACE = "#dddddd"
EMPTY_FACE = "#ffffff"


def load_cell(codec, channel, r, tol=1e-6):
    for f in sorted(BENCH2_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        if (d.get("codec") == codec and d.get("channel") == channel
                and abs(float(d.get("r", -999)) - r) < tol):
            results = d.get("results", [])
            n_trials = d.get("n_trials", len(results))
            ok = [x for x in results if x.get("success")]
            n_success = len(ok)
            densities = [x["density_eb_per_g"] for x in ok
                         if x.get("density_eb_per_g") is not None]
            mean_d = statistics.fmean(densities) if densities else None
            return n_success, n_trials, mean_d
    return 0, 0, None


def collect_grid():
    """grid[(codec, channel, r)] = (n_success, n_trials, mean_density)."""
    grid = {}
    for codec_key, _ in CODECS:
        for channel in ("hifi", "lofi"):
            for r in R_GRID:
                grid[(codec_key, channel, r)] = load_cell(codec_key, channel, r)
    return grid


def draw_panel(ax, grid, channel, title, vmax):
    n_rows = len(CODECS)
    n_cols = len(R_GRID)
    for i, (codec_key, _) in enumerate(CODECS):
        for j, r in enumerate(R_GRID):
            ns, nt, md = grid[(codec_key, channel, r)]
            if ns == nt == 30 and md is not None:
                color = CMAP(md / vmax)
                text = f"{md:.1f}"
                lum = sum(color[:3]) / 3
                tcolor = "white" if lum < 0.55 else "#1a1a1a"
            elif ns > 0:
                color = PARTIAL_FACE
                text = f"{ns}/{nt}"
                tcolor = "#555555"
            else:
                color = EMPTY_FACE
                text = f"0/{nt if nt > 0 else 30}"
                tcolor = "#999999"
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   facecolor=color, edgecolor="white",
                                   linewidth=1.0, zorder=2))
            ax.text(j, i, text, ha="center", va="center",
                    color=tcolor, zorder=3)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)  # invert y so the first codec is on top
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(r) for r in R_GRID])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([label for _, label in CODECS])
    ax.set_xlabel("Physical redundancy $r$")
    ax.set_title(title)
    # auto aspect lets cells stretch with the panel width so numbers fit.
    ax.set_aspect("auto")
    ax.tick_params(axis="both", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def plot(grid):
    densities = [md for (_, _, _), (ns, nt, md) in grid.items()
                 if md is not None and ns == nt == 30]
    vmax = max(densities) if densities else 1.0

    # vertically stacked panels (hifi on top, lofi on bottom). wider cells
    # give every numeric annotation horizontal room. shared colorbar to the
    # right spans both panels.
    fig = plt.figure(figsize=(8.0, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[40, 0.8],
                          hspace=0.55, wspace=0.04)
    ax_hifi = fig.add_subplot(gs[0, 0])
    ax_lofi = fig.add_subplot(gs[1, 0])
    ax_cbar = fig.add_subplot(gs[:, 1])

    draw_panel(ax_hifi, grid, "hifi", "High-fidelity channel", vmax)
    draw_panel(ax_lofi, grid, "lofi", "Low-fidelity channel", vmax)

    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, cax=ax_cbar)
    cb.set_label("Storage density [EB per g of dsDNA]")
    cb.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def verify(grid):
    spot = [
        ("mahoraga", "hifi", 0.02, 153.2),
        ("mahoraga", "lofi", 0.5, 92.3),
    ]
    print("verification:")
    for codec, ch, r, exp in spot:
        _, _, md = grid[(codec, ch, r)]
        ok = md is not None and abs(md - exp) < 1.0
        print(f"  {codec} {ch} r={r}: got {md}, expected ~{exp} "
              f"{'OK' if ok else 'DRIFT'}")


def main():
    grid = collect_grid()
    verify(grid)
    plot(grid)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
