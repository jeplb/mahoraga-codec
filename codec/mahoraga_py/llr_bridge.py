"""posterior-to-LLR conversion, xorshift64 scrambler, DNA/bit mapping.

implements:
  - bit mapping 00=A, 01=T, 10=C, 11=G
  - bit1 LLR = ln((P(A)+P(T)) / (P(C)+P(G)))
  - bit2 LLR = ln((P(A)+P(C)) / (P(T)+P(G)))
  - scrambler: xorshift64 seeded with chunk_index+1 (never 0)
  - output LLRs clamped to ±20
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

LLR_CLAMP: float = 20.0
_EPS: float = 1e-10
_U64_MASK: int = (1 << 64) - 1


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def posteriors_to_llrs(
    posteriors: Sequence[Sequence[float]],
    scrambler_bits: Optional[Sequence[int]] = None,
) -> List[float]:
    """[P(A), P(C), P(G), P(T)] per position → 2*n_pos LLRs.

    scrambler_bits: optional descrambler stream (2 bits per position). when
    a scrambler bit is 1, the corresponding LLR's sign is flipped — which
    is the descrambling operation (scramble is its own inverse over XOR).
    """
    n = len(posteriors)
    llrs: List[float] = []
    llrs_extend = llrs.extend

    for i in range(n):
        post = posteriors[i]
        p_a = post[0] if post[0] > _EPS else _EPS
        p_c = post[1] if post[1] > _EPS else _EPS
        p_g = post[2] if post[2] > _EPS else _EPS
        p_t = post[3] if post[3] > _EPS else _EPS

        llr_bit1 = _clamp(math.log((p_a + p_t) / (p_c + p_g)), -LLR_CLAMP, LLR_CLAMP)
        llr_bit2 = _clamp(math.log((p_a + p_c) / (p_t + p_g)), -LLR_CLAMP, LLR_CLAMP)

        if scrambler_bits is not None:
            s1 = -1.0 if scrambler_bits[2 * i] != 0 else 1.0
            s2 = -1.0 if scrambler_bits[2 * i + 1] != 0 else 1.0
        else:
            s1 = 1.0
            s2 = 1.0

        llrs_extend((llr_bit1 * s1, llr_bit2 * s2))

    return llrs


def _xorshift64(state: int) -> int:
    # 64-bit wrapping xorshift, seed ≠ 0
    state ^= (state << 13) & _U64_MASK
    state ^= (state >> 7) & _U64_MASK
    state ^= (state << 17) & _U64_MASK
    return state & _U64_MASK


def scrambler_bits(chunk_index: int, n_bits: int) -> List[int]:
    """xorshift64 bit stream seeded with chunk_index+1 (seed-0 guard)."""
    state = (chunk_index + 1) & _U64_MASK
    bits: List[int] = [0] * n_bits
    for i in range(n_bits):
        state = _xorshift64(state)
        bits[i] = state & 1
    return bits


def scramble(data_bits: Sequence[int], chunk_index: int) -> List[int]:
    mask = scrambler_bits(chunk_index, len(data_bits))
    return [d ^ m for d, m in zip(data_bits, mask)]


_BITS_TO_DNA = {
    (0, 0): ord("A"),
    (0, 1): ord("T"),
    (1, 0): ord("C"),
    (1, 1): ord("G"),
}


def bits_to_dna(bits: Sequence[int]) -> bytes:
    assert len(bits) % 2 == 0, "bits length must be even"
    out = bytearray(len(bits) // 2)
    for i in range(0, len(bits), 2):
        out[i // 2] = _BITS_TO_DNA[(bits[i], bits[i + 1])]
    return bytes(out)


def dna_to_bits(dna: bytes) -> List[int]:
    # non-ACGT bases map to (0,0).
    bits: List[int] = [0] * (len(dna) * 2)
    for i, base in enumerate(dna):
        if base in (ord("A"), ord("a")):
            b1, b2 = 0, 0
        elif base in (ord("T"), ord("t")):
            b1, b2 = 0, 1
        elif base in (ord("C"), ord("c")):
            b1, b2 = 1, 0
        elif base in (ord("G"), ord("g")):
            b1, b2 = 1, 1
        else:
            b1, b2 = 0, 0
        bits[2 * i] = b1
        bits[2 * i + 1] = b2
    return bits
