#!/usr/bin/env python3
# matched-outer-parity comparison on the in-silico channel simulator.
# bench2 — each codec's peak 30/30 density with outer parity sized to match
# mahoraga's auto-sized parity per cell.
#
# data:   ../data/bench2/reference_baseline.csv
# writes: matched_parity.{pdf,svg} next to this script
#
# the csv only has dna_aeon, mahoraga, mgcplus rows — dna fountain and dna-rs
# never reached 30/30 under matched parity, so they're annotated in the figure
# rather than plotted.

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / "data" / "bench2" / "reference_baseline.csv"
OUT_PDF = SCRIPT_DIR / "matched_parity.pdf"
OUT_SVG = SCRIPT_DIR / "matched_parity.svg"

# shared style (inlined from bench3/plot_dt4dds_pareto.py — if you restyle
# there, mirror it here so the figures stay consistent)
STYLE = {
    "mahoraga_color": "#1f77b4",
    "mahoraga_edge": "black",
    "prior_face": "#888888",
    "prior_edge": "#555555",
    "caveat_color": "#555555",
    "tick_fontsize": 10,
    "axis_fontsize": 11,
    "panel_title_fontsize": 12,
    "marker_label_fontsize": 10,
    "caveat_fontsize": 9,
    "font_family": ["Helvetica", "Arial", "DejaVu Sans"],
}
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = STYLE["font_family"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False

# codec display metadata — order controls bar order within each group
CODECS = [
    ("mahoraga", "Mahoraga", STYLE["mahoraga_color"], "black", 1.2, True),
    ("dna_aeon", "DNA-Aeon", "#888888", STYLE["prior_edge"], 0.5, False),
    ("mgcplus",  "MGC+",     "#bbbbbb", STYLE["prior_edge"], 0.5, False),
]

# three explicit operating-point groups (channel, r) spanning both the
# low-redundancy regime where MGC+ fails and the redundancy range where all
# three codecs decode. at (hifi, r=0.02) MGC+ does not reach 30/30.
GROUPS = [
    ("hifi", 0.02, "HiFi, $r=0.02$"),
    ("hifi", 1.00, "HiFi, $r=1.0$"),
    ("lofi", 0.50, "LoFi, $r=0.5$"),
]

# expected densities from bench2 at the above operating points
EXPECTED = {
    ("mahoraga", "hifi", 0.02): 153.25,
    ("dna_aeon", "hifi", 0.02): 107.66,
    ("mgcplus",  "hifi", 0.02): None,     # did not decode
    ("mahoraga", "hifi", 1.00):  97.78,
    ("dna_aeon", "hifi", 1.00):  66.33,
    ("mgcplus",  "hifi", 1.00):  59.86,
    ("mahoraga", "lofi", 0.50):  92.25,
    ("dna_aeon", "lofi", 0.50):  64.88,
    ("mgcplus",  "lofi", 0.50):  59.09,
}

CAVEAT = (
    "DNA Fountain and DNA-RS did not reach 30 of 30 recovery at "
    "any redundancy under matched parity."
)


def load_bench2_at(csv_path, codec, channel, r, tol=1e-6):
    """return (density, succeeded) for the bench2 cell at (codec, channel, r).
    succeeded means n_success == n_trials. density is None on miss."""
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if (row["codec"] == codec
                    and row["channel"] == channel
                    and abs(float(row["r"]) - r) < tol):
                ns = int(row["n_success"] or 0)
                nt = int(row["n_trials"] or 30)
                dens = row["median_density_eb_per_g"]
                return (float(dens) if dens else None), (ns == nt)
    return None, False


def verify_against_expected(values, tol=0.5):
    """warn loudly if any loaded density differs from EXPECTED by more than tol."""
    for (codec, ch, r), exp in EXPECTED.items():
        got = values.get((codec, ch, r))
        if exp is None and got is None:
            print(f"  ok:   ({codec}, {ch}, r={r}) did not decode [as expected]")
        elif got is None:
            print(f"  WARN: ({codec}, {ch}, r={r}) missing, expected {exp}")
        elif exp is None:
            print(f"  WARN: ({codec}, {ch}, r={r}) expected non-decode, got {got:.2f}")
        elif abs(got - exp) > tol:
            print(f"  DRIFT: ({codec}, {ch}, r={r}) expected {exp:.2f}, got {got:.2f}")
        else:
            print(f"  ok:   ({codec}, {ch}, r={r}) density {got:.2f} (expected ~{exp:.2f})")


def plot_figure(values):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    bar_width = 0.25
    group_gap = 0.35
    n_codecs = len(CODECS)
    offsets = [(i - (n_codecs - 1) / 2) * bar_width for i in range(n_codecs)]
    group_x = [i * (n_codecs * bar_width + group_gap) for i in range(len(GROUPS))]

    top_by_group_codec = {}
    for g_i, (ch_key, r_val, _group_label) in enumerate(GROUPS):
        for c_i, (codec_key, label, face, edge, lw, _) in enumerate(CODECS):
            density = values.get((codec_key, ch_key, r_val))
            x = group_x[g_i] + offsets[c_i]
            if density is None:
                # no bar — render a hollow placeholder and the "n/d" annotation
                ax.text(x, 2, "n/d", ha="center", va="bottom",
                        fontsize=8, fontstyle="italic",
                        color=STYLE["prior_edge"], zorder=3)
                top_by_group_codec[(g_i, codec_key)] = (x, None)
                continue
            ax.bar(x, density, width=bar_width,
                   facecolor=face, edgecolor=edge, linewidth=lw, zorder=3)
            is_m = codec_key == "mahoraga"
            ax.text(x, density + 2, f"{math.floor(density * 10 + 0.5) / 10:.1f}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if is_m else "normal",
                    color=face if is_m else STYLE["prior_edge"], zorder=4)
            top_by_group_codec[(g_i, codec_key)] = (x, density)

    # ratio bracket: Mahoraga vs DNA-Aeon in each group (both always decode)
    for g_i in range(len(GROUPS)):
        x_m, y_m = top_by_group_codec[(g_i, "mahoraga")]
        x_a, y_a = top_by_group_codec[(g_i, "dna_aeon")]
        if y_m is None or y_a is None:
            continue
        ratio = y_m / y_a
        y_line = y_m + 15
        ax.plot([x_m, x_a], [y_line, y_line], color="#555555", linewidth=0.8, zorder=2)
        for xx, yy in [(x_m, y_m + 2), (x_a, y_a + 2)]:
            ax.plot([xx, xx], [yy, y_line], color="#555555", linewidth=0.8, zorder=2)
        ax.text((x_m + x_a) / 2, y_line + 2, f"{ratio:.2f}×",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#333333", zorder=4)

    ax.set_xticks(group_x)
    ax.set_xticklabels([g[2] for g in GROUPS], fontsize=11)
    ax.tick_params(axis="x", length=0)

    # y-axis
    ax.set_ylim(0, 170)
    ax.set_ylabel("Storage density [EB g$^{-1}$]", fontsize=STYLE["axis_fontsize"])
    ax.tick_params(axis="y", labelsize=STYLE["tick_fontsize"])

    # strip top/right spines for a cleaner look
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # legend (top-right, small)
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=face, edgecolor=edge, linewidth=lw, label=label)
        for _, label, face, edge, lw, _ in CODECS
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        fontsize=9,
        handlelength=1.3,
        handleheight=1.0,
    )

    # caveat annotation below x-axis, inside figure frame
    fig.text(
        0.5, 0.02, CAVEAT,
        ha="center", va="bottom",
        fontsize=9, fontstyle="italic",
        color=STYLE["caveat_color"],
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    values = {}
    for codec_key, *_ in CODECS:
        for ch_key, r_val, _ in GROUPS:
            dens, ok = load_bench2_at(CSV_PATH, codec_key, ch_key, r_val)
            values[(codec_key, ch_key, r_val)] = dens if ok else None
    print("bench2 cells loaded:")
    verify_against_expected(values)

    plot_figure(values)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
