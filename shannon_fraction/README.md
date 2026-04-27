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
