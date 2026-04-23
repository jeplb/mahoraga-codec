#!/usr/bin/env python3
"""Shannon-fraction statistics for the Mahoraga paper.

reads the matched-parity benchmark JSONs under data/bench2/ and computes,
for every (codec, channel, r) cell, the codec's storage density as a
fraction of three successively tighter capacity ceilings.

    1. alphabet ceiling  C_alpha = 2 bits/base (channel-independent)
    2. Shomorony-Heckel  C_SH    = 2*(1-h(p_sub)) - 2*(p_ins+p_del)
    3. Lenz et al        C_L     = 2 * (sum_d P_c(d)*C_d - beta*(1-exp(-c)))

all three are converted to EB/g by the same Poisson-survival scaling,
    rho_max_* = C_*  *  (1 - exp(-r)) / r  *  113.7
so the *relative* ordering is purely a function of the per-base capacity
factor. the alphabet ceiling is the only universal closed-form bound;
the Shomorony-Heckel and Lenz bounds inject the actual DT4DDS error
rates (and for Lenz also the per-cell codebook size).

caveats on the Lenz bound:
  (a) Shomorony-Heckel and Lenz are not universally ordered. Shomorony
      charges for indels additively (which Lenz does not model); on the
      lofi channel at high coverage Shomorony is the tighter of the two.
  (b) the Lenz Poisson-reads-per-reference channel model underestimates
      dropout relative to the DT4DDS compound model (Poisson(r) molecular
      copies each yielding ~sd reads) at very low physical redundancy.
      at r=0.02 (P(dropout)=0.98 under DT4DDS, 0.55 under Poisson(c=0.59)),
      the Lenz bound ceases to be a valid upper bound and codec fractions
      overshoot 100%. we warn on overshoot rather than suppress, so the
      user sees exactly which cells are affected.

output: shannon_fraction.csv + shannon_fraction_table.tex (next to script)
usage:  python3 shannon_fraction/compute_shannon_fraction.py
"""

from __future__ import annotations

import glob
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binom, poisson

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BENCH2_DIR = REPO_ROOT / "data" / "bench2"
OUT_CSV = SCRIPT_DIR / "shannon_fraction.csv"
OUT_TEX = SCRIPT_DIR / "shannon_fraction_table.tex"

# dsDNA mass → storage-density conversion (derivation in codec/mahoraga_py/pipeline.py).
EB_PER_G_PER_BIT_PER_BASE = 113.7

# alphabet size: 4 nucleotides → log2(4) = 2 bits/base.
BITS_PER_BASE = 2.0

# strand length used by Mahoraga encoding (DNA nucleotides).
L_NT = 126

# sequencing depth from the idsim channel config (ChannelParams::seq_depth = 30).
SEQ_DEPTH = 30.0

CODECS = ["mahoraga", "dna_aeon", "mgcplus"]
N_TRIALS_FULL = 30  # bench2 runs 30 trials per cell

# combined synth+seq error rates per channel. values are the sum of the
# ChannelParams::hifi/lofi components in hamming/idsim/src/channel.rs
# (synth_sub+seq_sub, synth_ins+seq_ins, synth_del+seq_del).
CHANNEL_RATES: Dict[str, Dict[str, float]] = {
    "hifi": {"p_sub": 1.3e-3, "p_ins": 1.5e-4, "p_del": 3.0e-4},
    "lofi": {"p_sub": 8.0e-3, "p_ins": 1.2e-3, "p_del": 5.5e-3},
}

# small-probability guard for log(0) in Lenz's C_d sum.
EPS = 1e-15


# ---------------------------------------------------------------------------
# capacity functions
# ---------------------------------------------------------------------------


def h_binary(p: float) -> float:
    """binary entropy in bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def shomorony_capacity(p_sub: float, p_ins: float, p_del: float) -> float:
    """Shomorony-Heckel additive bound: bits per DNA base."""
    return 2.0 * (1.0 - h_binary(p_sub)) - 2.0 * (p_ins + p_del)


def _lenz_c_d(d: int, p: float, cache: Dict[Tuple[int, float], float]) -> float:
    """C_d from Lenz Lemma 3: BSC mutual information at d reads, crossover p.

    C_d = 1 + sum_{k=0}^{d} B(d,p,k) * log2( B(d,p,k) / (B(d,p,k) + B(d,p,d-k)) )

    guards: skip terms where both B(d,p,k) and B(d,p,d-k) fall below EPS
    (denominator would be ill-defined); skip terms where B(d,p,k) alone
    falls below EPS (weight-zero contribution).
    """
    key = (d, p)
    if key in cache:
        return cache[key]
    if d == 0:
        cache[key] = 0.0
        return 0.0
    ks = np.arange(d + 1)
    b_k = binom.pmf(ks, d, p)
    b_dk = binom.pmf(d - ks, d, p)
    denom = b_k + b_dk
    mask = (b_k >= EPS) & (denom >= EPS)
    ratio = np.where(mask, b_k / np.where(denom > 0, denom, 1.0), 1.0)
    # log2(1) = 0 for masked-out positions → zero contribution.
    terms = np.where(mask, b_k * np.log2(ratio), 0.0)
    c_d = 1.0 + float(terms.sum())
    cache[key] = c_d
    return c_d


def lenz_capacity_binary(
    c: float,
    p: float,
    beta: float,
    c_d_cache: Dict[Tuple[int, float], float],
    d_max: int | None = None,
) -> float:
    """Lenz capacity per binary input symbol.

    returns NaN if outside validity region:
        0 < p < 1/8  and  0 < beta < (1 - H(4p)) / 2
    """
    if not (0.0 < p < 0.125):
        return float("nan")
    beta_max = (1.0 - h_binary(4.0 * p)) / 2.0
    if not (0.0 < beta < beta_max):
        return float("nan")
    if c <= 0.0:
        return 0.0
    if d_max is None:
        d_max = max(50, int(math.ceil(c + 10.0 * math.sqrt(c))))

    ds = np.arange(d_max + 1)
    pois = poisson.pmf(ds, c)
    c_d_vals = np.array([_lenz_c_d(int(d), p, c_d_cache) for d in ds])
    expected_c_d = float(np.sum(pois * c_d_vals))
    return expected_c_d - beta * (1.0 - math.exp(-c))


def capacity_to_density(c_bits_per_base: float, r: float) -> float:
    """convert per-base capacity (bits/base) to EB/g at physical redundancy r."""
    if not math.isfinite(c_bits_per_base) or c_bits_per_base <= 0.0 or r <= 0.0:
        return float("nan")
    return c_bits_per_base * (1.0 - math.exp(-r)) / r * EB_PER_G_PER_BIT_PER_BASE


# ---------------------------------------------------------------------------
# bench2 ingest (ported from compute_alphabet_ceiling.py, extended with n_seqs)
# ---------------------------------------------------------------------------


def load_bench2_records(bench2_dir: Path) -> pd.DataFrame:
    """parse bench2 JSONs into one row per trial.

    extracts density_eb_per_g and n_seqs per trial. n_seqs is the codec's
    encoded reference-strand count for that input size (deterministic per
    codec+r in bench2), needed for the Lenz bound's beta parameter.
    """
    rows: List[dict] = []
    fallback_used = 0
    for path in sorted(glob.glob(str(bench2_dir / "*bench2*.json"))):
        with open(path) as f:
            d = json.load(f)
        codec = d["codec"]
        channel = d["channel"]
        r = float(d["r"])
        depth = float(d.get("depth", SEQ_DEPTH))
        input_size = d.get("input_size")
        for t in d.get("results", []):
            success = bool(t.get("success", False))
            density = t.get("density_eb_per_g")
            n_seqs = t.get("n_seqs") or 0
            if density is None and success and input_size and n_seqs and r > 0:
                density = (
                    input_size * 8.0 / (n_seqs * L_NT * r) * EB_PER_G_PER_BIT_PER_BASE
                )
                fallback_used += 1
            rows.append({
                "codec": codec,
                "channel": channel,
                "r": r,
                "depth": depth,
                "trial": t["trial"],
                "success": success,
                "density_eb_per_g": density if density is not None else float("nan"),
                "n_seqs": int(n_seqs),
            })
    if fallback_used:
        warnings.warn(
            f"density_eb_per_g missing in {fallback_used} trial(s); used "
            "input_size/n_seqs fallback.",
            RuntimeWarning, stacklevel=2,
        )
    if not rows:
        raise FileNotFoundError(f"no bench2 JSONs found under {bench2_dir}")
    return pd.DataFrame(rows)


def aggregate_per_cell(df: pd.DataFrame) -> pd.DataFrame:
    """reduce to one row per (codec, channel, r) with success-cell means."""
    def _agg(g: pd.DataFrame) -> pd.Series:
        succ = g[g["success"]]
        # n_seqs is deterministic per codec-cell; use any trial (the first).
        n_seqs = int(g["n_seqs"].iloc[0]) if len(g) else 0
        depth = float(g["depth"].iloc[0]) if len(g) else SEQ_DEPTH
        return pd.Series({
            "n_trials": len(g),
            "n_trials_success": int(succ.shape[0]),
            "density_ebpg": float(succ["density_eb_per_g"].mean()) if len(succ) else float("nan"),
            "n_seqs": n_seqs,
            "depth": depth,
        })
    grouped = df.groupby(["codec", "channel", "r"]).apply(_agg, include_groups=False)
    return grouped.reset_index()


# ---------------------------------------------------------------------------
# bound attachment + sanity checks
# ---------------------------------------------------------------------------


def attach_all_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """add rho_max_* and fraction_*_pct columns for all three bounds.

    only rows with n_trials_success == N_TRIALS_FULL get non-NaN fractions;
    partial-decoding cells keep NaN so they do not contaminate downstream
    summaries.
    """
    out = df.copy()
    c_d_cache: Dict[Tuple[int, float], float] = {}

    rho_alpha = []
    rho_shom = []
    rho_lenz = []
    lenz_valid = []
    for _, row in out.iterrows():
        r = float(row["r"])
        channel = row["channel"]
        rates = CHANNEL_RATES[channel]
        p_sub, p_ins, p_del = rates["p_sub"], rates["p_ins"], rates["p_del"]

        # alphabet ceiling: C = 2 bits/base.
        rho_alpha.append(capacity_to_density(BITS_PER_BASE, r))

        # Shomorony-Heckel: channel-specific per-base capacity.
        c_sh = shomorony_capacity(p_sub, p_ins, p_del)
        rho_shom.append(capacity_to_density(c_sh, r))

        # Lenz: per-cell (depends on codec M via beta, and on c via depth).
        depth = float(row["depth"])
        c_cov = depth * (1.0 - math.exp(-r))
        n_seqs = int(row["n_seqs"])
        if n_seqs <= 1:
            rho_lenz.append(float("nan"))
            lenz_valid.append(False)
            continue
        # Lenz models each DNA base as 2 binary channel uses. strand length
        # in binary symbols is 2*L_NT; beta is the log2(M) rate per binary
        # symbol. using beta = log2(M)/L_NT double-counts and yields Lenz
        # fractions > 100% at low r (verified empirically), which is
        # physically impossible for a capacity bound.
        beta = math.log2(n_seqs) / (2.0 * L_NT)
        c_lenz_bin = lenz_capacity_binary(c_cov, p_sub, beta, c_d_cache)
        if math.isnan(c_lenz_bin):
            rho_lenz.append(float("nan"))
            lenz_valid.append(False)
            continue
        # per-binary-symbol → per-DNA-base: multiply by 2.
        rho_lenz.append(capacity_to_density(2.0 * c_lenz_bin, r))
        lenz_valid.append(True)

    out["rho_max_alphabet"] = rho_alpha
    out["rho_max_shomorony"] = rho_shom
    out["rho_max_lenz"] = rho_lenz
    out["_lenz_valid"] = lenz_valid

    passes = out["n_trials_success"] == N_TRIALS_FULL
    for label in ("alphabet", "shomorony", "lenz"):
        rho = out[f"rho_max_{label}"]
        frac = np.where(
            passes & rho.notna() & (rho > 0),
            out["density_ebpg"] / rho * 100.0,
            np.nan,
        )
        out[f"fraction_{label}_pct"] = frac
    return out


def sanity_checks(df: pd.DataFrame) -> None:
    """spec-mandated assertions + warnings."""
    # 1. bound ordering.
    #
    # alphabet > Shomorony-Heckel always holds: Shomorony charges for both
    # substitutions and indels against the 2-bits/base alphabet, and both
    # p_sub and (p_ins+p_del) are strictly positive on both channels.
    #
    # Shomorony vs Lenz is *not* universally ordered. the two bounds model
    # different aspects of the channel: Shomorony is a loose additive bound
    # that counts indels against capacity; Lenz is tighter on the BSC
    # sub-channel but does not model indels at all. on indel-heavy channels
    # like lofi, Shomorony can therefore be tighter than Lenz at high
    # coverage, where Lenz saturates at 2(1-beta) and ignores the ~1.3%
    # combined indel rate that Shomorony does charge for. we report both
    # and flag the ordering outcome rather than asserting one way.
    for channel, rates in CHANNEL_RATES.items():
        c_sh = shomorony_capacity(rates["p_sub"], rates["p_ins"], rates["p_del"])
        assert c_sh < BITS_PER_BASE, (
            f"Shomorony C={c_sh:.4f} must be < alphabet {BITS_PER_BASE} on {channel}"
        )
    sub = df[df["_lenz_valid"] & df["rho_max_lenz"].notna()].copy()
    violates = sub[sub["rho_max_shomorony"] > sub["rho_max_alphabet"] + 1e-9]
    assert violates.empty, (
        f"Shomorony bound must be tighter than alphabet: {len(violates)} cell(s) violate"
    )
    n_lenz_looser = int(
        (sub["rho_max_lenz"] > sub["rho_max_shomorony"] + 1e-9).sum()
    )
    n_lenz_tighter = int(
        (sub["rho_max_lenz"] < sub["rho_max_shomorony"] - 1e-9).sum()
    )
    if n_lenz_looser or n_lenz_tighter:
        print(
            f"note: Lenz vs Shomorony ordering varies with (c, p, beta) — "
            f"{n_lenz_tighter} cells tighter, {n_lenz_looser} cells looser "
            f"than Shomorony. see docstring for why."
        )

    # 2. monotonic non-increasing rho_max in r per (channel, codec). only
    # applies to the alphabet and Shomorony-Heckel bounds, which have a
    # fixed per-base capacity C — so rho_max = C*(1-exp(-r))/r*113.7
    # inherits the (1-exp(-r))/r envelope's monotone decay. the Lenz
    # bound's C_L itself grows with c = sd*(1-exp(-r)): at low r few
    # reads starve the BSC majority-vote subchannel, and C_L rises
    # faster than (1-exp(-r))/r decays. so rho_max_lenz is non-monotone
    # in r by construction and cannot be subject to this assertion.
    for (channel, codec), grp in df.groupby(["channel", "codec"]):
        for label in ("alphabet", "shomorony"):
            col = f"rho_max_{label}"
            gs = grp.sort_values("r")
            vals = gs[col].dropna().to_numpy()
            if len(vals) < 2:
                continue
            diffs = np.diff(vals)
            assert np.all(diffs <= 1e-6), (
                f"rho_max_{label} must be non-increasing in r for "
                f"channel={channel} codec={codec}; offending diffs: {diffs}"
            )

    # 3. overshoots (fraction > 100%) — warn but don't crash.
    for label in ("alphabet", "shomorony", "lenz"):
        col = f"fraction_{label}_pct"
        overshoots = df[(df[col].notna()) & (df[col] > 100.0)]
        for _, row in overshoots.iterrows():
            warnings.warn(
                f"{label} fraction > 100%: codec={row['codec']} "
                f"channel={row['channel']} r={row['r']} frac={row[col]:.2f}%",
                RuntimeWarning, stacklevel=2,
            )
        valid = df[df[col].notna()]
        assert (valid[col] >= 0.0).all(), f"negative {label} fraction"


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "channel", "r", "codec",
        "density_ebpg",
        "rho_max_alphabet", "fraction_alphabet_pct",
        "rho_max_shomorony", "fraction_shomorony_pct",
        "rho_max_lenz", "fraction_lenz_pct",
        "n_trials_success",
    ]
    out = df[cols].sort_values(["channel", "r", "codec"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.4f")
    print(f"wrote {out_path}  ({len(out)} rows)")


def _fmt_pct(v: float) -> str:
    if not math.isfinite(v):
        return "---"
    return f"{v:.1f}\\%"


def _fmt_rho(v: float) -> str:
    if not math.isfinite(v):
        return "---"
    return f"{v:.1f}"


def write_latex_table(df: pd.DataFrame, out_path: Path) -> None:
    """supplementary table: one row per (channel, r, codec) with all three
    fractions. channels grouped by midrule, r ascending within each channel,
    codecs in CODECS order within each r.
    """
    pretty_channel = {"hifi": "High-fidelity", "lofi": "Low-fidelity"}
    pretty_codec = {"mahoraga": "Mahoraga", "dna_aeon": "DNA-Aeon", "mgcplus": "MGC+"}

    lines: List[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Codec densities as a fraction of three capacity ceilings "
        r"at matched-parity operating points. The alphabet ceiling "
        r"$C_{\alpha}=2$~bits per base is channel-independent; the "
        r"Shomorony-Heckel bound $C_{\mathrm{SH}}=2(1-h(p_{\mathrm{sub}}))"
        r"-2(p_{\mathrm{ins}}+p_{\mathrm{del}})$ injects the combined "
        r"synthesis-plus-sequencing error rates on each channel; the Lenz "
        r"bound $C_{\mathrm{L}}$ adds per-cell codebook size $M$ via "
        r"$\beta=\log_2 M / (2L)$ (with $L=126$~nt strands, i.e.\ $2L=252$ "
        r"binary channel uses) and coverage $c=s_d(1-e^{-r})$. Each entry "
        r"is the codec's density as a percentage of $\rho_{\max}$ for the "
        r"given bound. Dashes denote cells that did not reach 30/30 "
        r"decoding. Values above 100\% occur at $r=0.02$ because the Lenz "
        r"Poisson-reads-per-reference channel model systematically "
        r"underestimates the heavier molecular-copy dropout of the DT4DDS "
        r"channel at very low physical redundancy, so the bound ceases to "
        r"be a valid upper bound in that regime.}"
    )
    lines.append(r"\label{app:shannon_fraction_full}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Channel & $r$ & Codec & Alphabet & Shomorony-Heckel & Lenz \\"
    )
    lines.append(r"\midrule")

    channels_order = ["hifi", "lofi"]
    for i, ch in enumerate(channels_order):
        sub = df[df["channel"] == ch].copy()
        rs = sorted(sub["r"].unique())
        for r in rs:
            cell = sub[sub["r"] == r]
            for codec in CODECS:
                row = cell[cell["codec"] == codec]
                if row.empty:
                    continue
                row = row.iloc[0]
                line = (
                    f"{pretty_channel[ch]} & {r:g} & {pretty_codec[codec]} & "
                    f"{_fmt_pct(row['fraction_alphabet_pct'])} & "
                    f"{_fmt_pct(row['fraction_shomorony_pct'])} & "
                    f"{_fmt_pct(row['fraction_lenz_pct'])} \\\\"
                )
                lines.append(line)
        if i < len(channels_order) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _print_capacities() -> None:
    """self-documenting dump of the three per-base capacities at channel rates."""
    print("per-base capacity values:")
    print(f"  alphabet ceiling:        {BITS_PER_BASE:.4f} bits/base")
    for ch, rates in CHANNEL_RATES.items():
        c_sh = shomorony_capacity(rates["p_sub"], rates["p_ins"], rates["p_del"])
        print(
            f"  Shomorony-Heckel [{ch}]: {c_sh:.4f} bits/base  "
            f"(p_sub={rates['p_sub']}, p_ins={rates['p_ins']}, p_del={rates['p_del']})"
        )
    # Lenz is cell-specific. print two reference points per channel:
    # r=1.0 (high coverage — C_d saturates, channels converge to 1-beta),
    # r=0.02 (low coverage — BSC majority-vote subchannel dominates, so
    # channel-specific p shows through).
    beta_ref = math.log2(1134) / (2.0 * L_NT)
    for r_ref in (0.02, 1.0):
        c_cov = SEQ_DEPTH * (1.0 - math.exp(-r_ref))
        for ch, rates in CHANNEL_RATES.items():
            cache: Dict[Tuple[int, float], float] = {}
            c_l = lenz_capacity_binary(c_cov, rates["p_sub"], beta_ref, cache)
            if math.isfinite(c_l):
                print(
                    f"  Lenz [{ch}, r={r_ref}, M=1134, c={c_cov:.2f}]: "
                    f"{2.0 * c_l:.4f} bits/base  (beta={beta_ref:.4f})"
                )
    print()


def main() -> int:
    _print_capacities()

    df_trials = load_bench2_records(BENCH2_DIR)
    print(f"loaded {len(df_trials)} per-trial records from {BENCH2_DIR}")
    print(f"  codecs:   {sorted(df_trials['codec'].unique())}")
    print(f"  channels: {sorted(df_trials['channel'].unique())}")
    print(f"  r values: {sorted(df_trials['r'].unique())}")
    print()

    agg = aggregate_per_cell(df_trials)
    agg = attach_all_bounds(agg)
    sanity_checks(agg)

    write_csv(agg, OUT_CSV)
    write_latex_table(agg, OUT_TEX)

    print()
    print("per-codec 30/30 fractions (mean over cells that decoded):")
    full = agg[agg["n_trials_success"] == N_TRIALS_FULL]
    for label in ("alphabet", "shomorony", "lenz"):
        col = f"fraction_{label}_pct"
        summary = full.groupby(["codec", "channel"])[col].agg(["mean", "count"])
        print(f"\n  {label}:")
        print(summary.round(2).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
