# mahoraga-codec

A DNA data-storage codec in pure python. Short strands (126 nt by default), soft-decision inner decode (LDPC + OSD with CRC gating), outer Reed-Solomon over GF(2^16) for the file layer, optional turbo feedback on the noisy channel. Everything here is readable, everything is reproducible, and every number in the accompanying paper ties back to a JSON file under `data/`.

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
  bench2/          matched-parity comparison (supp. table s2, fig 1 c/d)
  bench3/          DT4DDS pipeline replication (supp. table s3, fig 3)
  bench4/          longevity (supp. table s4, fig 4)
  bench5/          strand length sweep (supp. fig)
paper/           stable1..stable4 aggregated tables
```

Filenames encode the experimental cell: `<codec>-bench<n>-<channel>-<params>.json`. The per-trial JSONs carry seed, git SHA, codec/channel parameters, and timings.

Each `benchN/` directory co-locates its plot script(s) next to its input data. Running a script writes its PDF/SVG in the same folder. Drop the PDF into LaTeX and move on.

## Install

```
cd codec
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[test]'
```

That's it. Only dependency is `numpy`.

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

## Reproducing the paper

Every cell in the paper has a corresponding JSON under `data/`. The aggregated `paper/stable*.tsv` files are what the supplementary tables show; they were produced from the JSONs by grouping on `(codec, channel, r)` and counting `n_success / n_trials`.

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

Each script reads from `data/benchN/` or `paper/stable*.tsv` and writes its PDF + SVG next to itself. Regenerating the figures should take a couple of seconds. If you want to re-run the DT4DDS channel itself (rather than just replot the existing JSONs) you need the DT4DDS package; that pipeline is not in this repo.

The `data/bench2/python_*.json` files record per-trial results from the mahoraga-py codec at 2 KB (60 trials) and 15,360 B (20 trials): `ok_py` / `md5_py` are the python port's outcome, `ok_rs` / `md5_rs` columns record a reference decoder's outcome on the same channel output for cross-validation, `byte_mismatch` flags any per-trial divergence. Pilot: 60/60 byte-identical. Full size: 20/20 byte-identical.

## Caveats

- Density constant 113.7 EB/g/bit/base lives in `codec/mahoraga_py/pipeline.py`. If you change it, the paper numbers stop being comparable.
- Channel parameters in the benchmark harness are matched to Gimpel 2026 electrochemical synthesis + iSeq100 sequencing. Different instruments need a re-fit.
- The pure-python codec prioritises readability over speed. A file larger than ~20 KB at low physical redundancy takes hours to decode; it's fine for audit, teaching, and reproducing the paper at the published sizes, but it is not a production codec.
- Some `data/bench3/` cells have `sd=1` entries which are a sanity grid, not a paper claim. The paper quotes sd ∈ {15, 30}.

## License

PolyForm Noncommercial 1.0.0. See `LICENSE`.

Noncommercial use, namely research, teaching, peer review, reproducing paper results, is permitted. Commercial use (including internal use by a for-profit entity for its own products or services) requires a separate license from the copyright holder, James L. Banal.
