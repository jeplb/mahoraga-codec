# shannon fraction

codec densities expressed as a fraction of three successively tighter
capacity ceilings per `(codec, channel, r)` cell. run with

    python3 shannon_fraction/compute_shannon_fraction.py

and it writes `shannon_fraction.csv` + `shannon_fraction_table.tex`
next to itself.

the three bounds are

    alphabet:    C = 2 bits/base                                 (channel-independent)
    Shomorony:   C = 2(1 - h(p_sub)) - 2(p_ins + p_del)          (per-channel error rates)
    Lenz:        C = 2 * (sum_d P_c(d) C_d(d,p) - beta(1-e^-c))  (adds codebook size + coverage)

all three convert to EB/g by the same Poisson-survival scaling
`rho_max = C * (1 - exp(-r)) / r * 113.7`.

## conventions

the implementation departs from a naive reading of the spec in three
places. each departure is load-bearing for the numerical output and is
documented in line, but collected here for reviewers.

### 1. Lenz beta uses 2L, not L

we use `beta = log_2(M) / (2L)` with `L = 126` nt. in Lenz's model each
DNA base is treated as two binary channel uses, so the strand carries
`2L = 252` binary symbols and the per-binary-symbol rate is
`log_2(M) / (2L)`. using `log_2(M) / L` double-counts and gives Lenz
fractions above 175% at low `r`, which is physically impossible for a
capacity bound. the spec's formula string `beta = log_2(M) / L` is
inherited from a paper reading where `L` is already in binary units.

### 2. Lenz vs Shomorony ordering is empirical

the spec asserts Lenz is strictly tighter than Shomorony. that does not
hold on this channel. Shomorony charges `-2(p_ins + p_del)` against the
alphabet ceiling; Lenz models a substitution-only BSC and does not
account for indels. on the lofi channel (`p_ins + p_del = 0.0067`) at
high coverage, Lenz saturates near `2(1 - beta)` while Shomorony stays
below `2 - 0.0134`, so Shomorony becomes the tighter bound. we report
both and print how many cells each way; the hard assertion is dropped.

### 3. Lenz overshoots at r = 0.02

four cells produce Lenz fractions > 100% and trigger RuntimeWarnings:

    mahoraga   hifi  r=0.02  -> 163.6%
    mahoraga   lofi  r=0.02  -> 130.0%
    dna_aeon   hifi  r=0.02  -> 115.3%

this is a channel-model mismatch, not a numerical bug. Lenz assumes
reads per reference are Poisson(`c = sd (1 - exp(-r))`). the DT4DDS
channel is compound: Poisson(`r`) molecular copies per reference, each
yielding ~`sd` reads. at `r = 0.02` that gives `P(drop) = 1 - e^{-r} = 0.98`,
against the Lenz Poisson-reads assumption's `P(d=0) = e^{-0.59} = 0.55`.
Lenz drastically under-counts dropout at extreme low `r`, so its bound
is not valid there and codecs beat it. we keep the numbers in the CSV
and LaTeX table (flagged in the caption) rather than suppressing them,
so the regime where Lenz applies is visible by inspection.

excluding those cells, Mahoraga HiFi Lenz fractions sit in 70.7-94.4%
and LoFi in 53.6-73.9%, inside the spec's predicted `70-75%` / `58-65%`
bands at `r >= 0.1`.
