"""mahoraga DNA storage codec (pure python).

ten modules compose the full encode/decode pipeline: HMM (viterbi),
read assignment (kmer_index, identify), soft bits (llr_bridge), reed-
solomon over GF(2^8) and GF(2^16) (rs_erasure, rs16), LDPC construction
(peg) and decoding (ldpc), ordered-statistics decoder (osd), and the
top-level orchestration (pipeline).

import what you need directly: ``from mahoraga_py import viterbi``.
"""

__all__ = [
    "llr_bridge",
    "viterbi",
    "kmer_index",
    "identify",
    "rs_erasure",
    "rs16",
    "peg",
    "ldpc",
    "osd",
    "pipeline",
]
