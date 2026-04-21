"""batch read assignment: k-mer prefilter → HMM banded_forward scoring.

for each read:
  1. k-mer index prefilter gives up to n_candidates reference ids
  2. banded_forward is run against each candidate
  3. the highest-likelihood ref_id is picked
  4. reads with best_ll < -0.5 * read_len are rejected (unassigned)


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from . import viterbi
from .kmer_index import KmerIndex


@dataclass
class ReadAssignment:
    ref_id: int
    log_likelihood: float


def batch_identify(
    reads: Sequence[bytes],
    reference_seqs: Sequence[bytes],
    params: viterbi.HmmParams,
    band_width: int = 10,
    k: int = 8,
    n_candidates: int = 15,
) -> List[Optional[ReadAssignment]]:
    index = KmerIndex.build(reference_seqs, k)
    out: List[Optional[ReadAssignment]] = []

    for read in reads:
        candidates = index.prefilter(read, n_candidates)
        if not candidates:
            out.append(None)
            continue

        best_id = candidates[0]
        best_ll = float("-inf")
        for cand_id in candidates:
            ll = viterbi.banded_forward(reference_seqs[cand_id], read, params, band_width)
            if ll > best_ll:
                best_ll = ll
                best_id = cand_id

        threshold = -0.5 * len(read)
        if best_ll < threshold:
            out.append(None)
        else:
            out.append(ReadAssignment(ref_id=best_id, log_likelihood=best_ll))

    return out


def group_reads_by_ref(
    reads: Sequence[bytes],
    assignments: Sequence[Optional[ReadAssignment]],
    n_refs: int,
) -> List[List[int]]:
    groups: List[List[int]] = [[] for _ in range(n_refs)]
    for read_idx, a in enumerate(assignments):
        if a is not None:
            groups[a.ref_id].append(read_idx)
    return groups
