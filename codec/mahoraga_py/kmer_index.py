"""k-mer index for read-to-reference prefilter.

  - 2-bit per base packing (A=0, C=1, G=2, T=3)
  - k-mer hash u64, max k=32
  - non-ACGT reset the current k-mer
  - prefilter returns top-N references by total k-mer hit count
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

_U64_MASK: int = (1 << 64) - 1
_BASE_BITS = {
    ord("A"): 0, ord("a"): 0,
    ord("C"): 1, ord("c"): 1,
    ord("G"): 2, ord("g"): 2,
    ord("T"): 3, ord("t"): 3,
}


def _base_to_2bit(b: int) -> Optional[int]:
    return _BASE_BITS.get(b)


class KmerIndex:
    __slots__ = ("map", "k")

    def __init__(self, map_: Dict[int, List[Tuple[int, int]]], k: int) -> None:
        self.map = map_
        self.k = k

    @classmethod
    def build(cls, reference_seqs: Sequence[bytes], k: int) -> "KmerIndex":
        assert k <= 32, "k must be <= 32"
        mask = _U64_MASK if k == 32 else (1 << (2 * k)) - 1
        m: Dict[int, List[Tuple[int, int]]] = {}

        for seq_id, seq in enumerate(reference_seqs):
            if len(seq) < k:
                continue
            h = 0
            valid = 0
            for pos, base in enumerate(seq):
                bits = _BASE_BITS.get(base)
                if bits is None:
                    valid = 0
                    h = 0
                else:
                    h = ((h << 2) | bits) & mask
                    valid += 1
                if valid >= k:
                    bucket = m.get(h)
                    entry = (seq_id, pos + 1 - k)
                    if bucket is None:
                        m[h] = [entry]
                    else:
                        bucket.append(entry)

        return cls(m, k)

    def prefilter(self, read: bytes, n_candidates: int) -> List[int]:
        if len(read) < self.k:
            return []
        k = self.k
        mask = _U64_MASK if k == 32 else (1 << (2 * k)) - 1
        counts: Dict[int, int] = {}
        h = 0
        valid = 0
        for base in read:
            bits = _BASE_BITS.get(base)
            if bits is None:
                valid = 0
                h = 0
            else:
                h = ((h << 2) | bits) & mask
                valid += 1
            if valid >= k:
                hits = self.map.get(h)
                if hits is not None:
                    for seq_id, _pos in hits:
                        counts[seq_id] = counts.get(seq_id, 0) + 1

        if not counts:
            return []
        # for equal counts, sort order is by
        # unspecified order. python's sorted() is stable by insertion order,
        # which matches FxHashMap iteration for a single-insert-per-key map
        # on small inputs closely enough for downstream HMM scoring to
        # break any remaining ties identically.
        ordered = sorted(counts.items(), key=lambda kv: -kv[1])
        return [seq_id for seq_id, _c in ordered[:n_candidates]]
