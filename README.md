# mahoraga-codec

A DNA data-storage codec in pure python. Short strands (126 nt by default), soft-decision inner decode (LDPC + OSD with CRC gating), outer Reed-Solomon over GF(2^16) for the file layer, optional turbo feedback on the noisy channel. Everything here is readable, everything is reproducible, and every number in the accompanying paper ties back to a JSON file under `data/`.

## Performance note

This is a reference implementation written for clarity rather than speed. Expect seconds to minutes per oligo for HMM forward-backward and OSD decoding on a single CPU core.  The benchmarks in `data/` were produced with an optimised implementation. The Python code here reproduces the same output on the same inputs but runs approximately slower per decode.

Performance-sensitive applications should expect the decoder to be re-implemented in a compiled language. The HMM forward-backward, OSD enumeration, and GF(2^16) arithmetic all have straightforward vectorised and parallelised implementations. The algorithms and data structures in this codebase transfer directly.

## Layout

```
codec/
  mahoraga_py/   the codec (10 modules, ~2,700 lines)
  tests/         pytest suite — roundtrips + RS/GF unit tests
bench1/          plot scripts for codec comparison (density_vs_r, codec_comparison)
bench2/          plot script for matched-parity comparison
bench3/          plot scripts for the DT4DDS pipeline (pareto, gimpel_style)
bench4/          plot script for longevity projection
data/            benchmark JSON, organised to match the paper
  bench1/          codec comparison (supp. table s1)
    v2/              same cells re-run on the v2 channel (see "Channel versions")
  bench2/          matched-parity comparison (supp. table s2, fig 1 c/d)
    v2/
  bench3/          DT4DDS pipeline replication (supp. table s3, fig 3)
  bench4/          longevity (supp. table s4, fig 4)
    v2/
  bench5/          strand length sweep (supp. fig)
    v2/
alphabet_ceiling/  capacity-ceiling analysis (script + CSV)
```

Filenames encode the experimental cell: `<codec>-bench<n>-<channel>-<params>.json`. The per-trial JSONs carry seed, git SHA, codec/channel parameters, and timings.

Each `benchN/` directory co-locates its plot script(s) next to its input data.

## Install

```
cd codec
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[test]'
```

## Smoke test

```python
from mahoraga_py import pipeline, viterbi

data = b"hello " * 50
inner = pipeline.InnerCode.new("hifi")
seqs, _ = pipeline.encode_to_dna(data, inner, physical_redundancy=1.28)

# treat the encoded seqs as perfect reads
hmm = viterbi.HmmParams.default_ids()
rec, stats = pipeline.decode_from_reads(seqs, seqs, inner, hmm, len(data))
assert rec == data
```

## Tests

```
cd codec && pytest -q
```

29 tests: noiseless roundtrips on hifi and lofi presets, GF(2^8) and GF(2^16) RS encode/erasure/BM, CRC-16 and CRC-32 known-answer vectors.

## Modules

| module | purpose |
|---|---|
| `mahoraga_py.viterbi`     | banded Viterbi / Forward / Forward-Backward on the profile HMM |
| `mahoraga_py.kmer_index`  | 2-bit k-mer index for read-to-reference prefilter |
| `mahoraga_py.identify`    | batch read assignment (kmer prefilter → HMM forward scoring) |
| `mahoraga_py.llr_bridge`  | posterior → LLR, xorshift64 scrambler, DNA↔bit mapping |
| `mahoraga_py.rs_erasure`  | RS over GF(2^8), CRC-16 CCITT, CRC-32 MPEG-2 |
| `mahoraga_py.rs16`        | RS over GF(2^16): encode, erasure decode, Berlekamp-Massey, Forney |
| `mahoraga_py.peg`         | progressive edge growth LDPC construction |
| `mahoraga_py.ldpc`        | LDPC BP decoder, GF(2) row reduction, systematic encode |
| `mahoraga_py.osd`         | ordered-statistics decoder (OSD-0/1/2/3) with CRC gating |
| `mahoraga_py.pipeline`    | top-level: InnerCode, encode_to_dna, decode_from_reads, turbo RS |

## Channel versions (v1, v2)

The benchmarks ship two idsim channel models. Each `data/benchN/` directory carries v1 results at the top level; v2 results live in a `v2/` subdirectory alongside.

**v1** (`coverage_sigma=0.3`). Three-stage idsim: synthesis IDS errors → lognormal-weighted Poisson coverage → per-read sequencing IDS errors with uniform per-base rates. This is what produced the original paper numbers; JSONs at the top level of each `data/benchN/`.

**v2** (`coverage_sigma=0.5` + Q5 PCR + iSeq position-dependent errors + iSeq-100 NGmerge). Adds three optional stages to match Gimpel 2026 / DT4DDS behavior more faithfully. When all three additive stages are off, v2 reduces to v1 modulo the `coverage_sigma` default.

| v2 stage | model |
|---|---|
| coverage variance | lognormal `sigma=0.5` (DT4DDS-style; v1 uses 0.3) |
| PCR amplification | Q5 high-fidelity, 15+25 cycles. Per-cycle efficiency drawn `Normal(0.95, 0.0051)` and clamped to `[0,1]`; the per-template amplification weight `(1+eff)^cycles` multiplies the lognormal coverage weight, then is renormalized so mean coverage still equals `physical_redundancy`. Cumulative polymerase substitution rate `40 cycles × 5e-7 / cycle = 2e-5` per base. |
| sequencer errors | position-dependent substitution: `e(pos) = e_min + (e_max − e_min) × (pos/L)^power` with `e_min = 1e-4`, `e_max = 1e-2`, `power = 2`. Replaces the uniform `seq_sub` rate; del / ins stay uniform. Mean sub rate over `[0, L]` is `e_min + (e_max − e_min) / (power + 1) ≈ 3.4e-3`. Per-base Phred quality `Q = -10·log10(e(pos))` is returned alongside each read. |
| paired-end / merge | iSeq-100 paired-end with `read_len = 150`, NGmerge with `overlap_min = 20`. Reads from a strand of length `L` are dropped when `L > 2·read_len − overlap_min = 280 nt` — this matches the iSeq-100 merge limit. For `L ≤ 280` the merge always succeeds. |

Densities at v2 land roughly 6–9% below v1 at matched cells; the gap comes mostly from the higher dropout variance pushing outer-RS utilization closer to capacity.

The v2 dataset:

| path | contents |
|---|---|
| `data/bench1/v2/` | codec comparison cells (5 codecs × 10 `r` × 2 channels), v2 channel |
| `data/bench2/v2/` | matched-outer-parity comparison cells, v2 channel |
| `data/bench4/v2/` | longevity sweep cells (3 codecs × 4 `r_initial`). The four `r_initial_{1,2,5,10}.0.json` files (mahoraga) carry per-trial decoder-stage telemetry — HMM rejection, posterior confidence, OSD order histogram, CRC pass rate, RS utilization, Berlekamp-Massey error count. The `mgcplus-…` and `dna_aeon-…` files are pass-rate only (those codecs are not instrumented). |
| `data/bench5/v2/bench5_v2_telemetry.json` | strand-length sweep at v2 with per-trial telemetry for every cell |

The v2 telemetry surfaces which decoder stage absorbs the v2-extra errors. In bench5 (varying `L`) the inner-code OSD-fail rate is the only field that moves — it scales 3.2× from `L=126` to `L=300`. In bench4 (fixed encoding, sweeping `channel_r` downward) the per-strand decode is invariant and only RS utilization moves with dropout.

## Reproducing the paper

Every cell in the paper has a corresponding JSON under `data/`. The supplementary tables are produced on the fly from the JSONs by grouping on `(codec, channel, r)` and counting `n_success / n_trials`. No intermediate aggregate files are checked in.

Replot any figure from its data:

```
pip install numpy matplotlib  # plot-only dependencies
python3 bench1/plot_density_vs_r.py       # density vs. redundancy (hifi)
python3 bench1/plot_codec_comparison.py   # full codec comparison panel
python3 bench2/plot_matched_parity.py     # matched-outer-parity bars
python3 bench3/plot_dt4dds_pareto.py      # DT4DDS pareto, hifi + lofi
python3 bench3/plot_bench3_gimpel_style.py  # gimpel 2026-style panels
python3 bench4/plot_longevity.py          # density vs years at 25°C
```

Each script reads from `data/benchN/` and writes its PDF + SVG next to itself. Regenerating the figures should take a couple of seconds. If you want to re-run the DT4DDS channel itself (rather than just replot the existing JSONs) you need the DT4DDS package; that pipeline is not in this repo.

The `data/bench2/python_*.json` files record per-trial results from the mahoraga-py codec at 2 KB (60 trials) and 15,360 B (20 trials): `ok_py` / `md5_py` are the python port's outcome, `ok_rs` / `md5_rs` columns record a reference decoder's outcome on the same channel output for cross-validation, `byte_mismatch` flags any per-trial divergence.

## Alphabet-ceiling fraction analysis

How close does each codec get to the quaternary alphabet ceiling at each operating point? `alphabet_ceiling/compute_alphabet_ceiling.py` answers that from the bench2 trial JSONs. The alphabet ceiling is 2 bits per base pair. After accounting for stochastic dropout, the maximum density achievable at physical redundancy `r` is

```
rho_max  = 2 * (1 - exp(-r)) / r * 113.7   [EB/g]
fraction = realized_density / rho_max      (30/30 cells only)
```

and writes `alphabet_ceiling/alphabet_ceiling.csv` with one row per `(channel, r, codec)`. Dependencies are `numpy` and `pandas`. The script asserts the ceiling is monotone non-increasing in `r` and flags any fraction > 100% as a `RuntimeWarning`.

The alphabet ceiling is channel-independent because it depends only on alphabet size (2 bits per base pair) and the Poisson survival fraction `(1 - exp(-r)) / r`. Channel-specific capacity bounds that incorporate per-base substitution and indel rates are strictly tighter than this ceiling and would give fractions closer to 100%. The ceiling is used here because it is the only universal closed-form bound and therefore allows cross-channel comparison without committing to a specific indel-substitution capacity formula.

## License

PolyForm Noncommercial 1.0.0. See `LICENSE`.

Noncommercial use, namely research, teaching, peer review, reproducing paper results, is permitted. Commercial use (including internal use by a for-profit entity for its own products or services) requires a separate license from the copyright holder, James L. Banal.
