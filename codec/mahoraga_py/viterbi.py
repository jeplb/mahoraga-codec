"""profile-HMM banded Viterbi / Forward / Forward-Backward.

semantics:

states per reference position:
  0 = match  — emits observed base with sub probability
  1 = insert — emits any base uniformly (p=0.25)
  2 = delete — silent, skips a reference position

all arithmetic in log-space (float64). banded by ``band_width``: only
reference positions within ``±band_width`` of the current read index
are populated.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

NEG_INF: float = float("-inf")
LOG_QUARTER: float = math.log(0.25)


class HmmParams:
    __slots__ = (
        "p_match_to_match",
        "p_match_to_insert",
        "p_match_to_delete",
        "p_insert_to_match",
        "p_insert_to_insert",
        "p_sub",
        "log_mm",
        "log_mi",
        "log_md",
        "log_im",
        "log_ii",
        "log_emit_match",
        "log_emit_sub",
    )

    def __init__(
        self,
        p_match_to_match: float = 0.994,
        p_match_to_insert: float = 0.001,
        p_match_to_delete: float = 0.005,
        p_insert_to_match: float = 0.5,
        p_insert_to_insert: float = 0.5,
        p_sub: float = 0.005,
    ) -> None:
        self.p_match_to_match = p_match_to_match
        self.p_match_to_insert = p_match_to_insert
        self.p_match_to_delete = p_match_to_delete
        self.p_insert_to_match = p_insert_to_match
        self.p_insert_to_insert = p_insert_to_insert
        self.p_sub = p_sub
        self.log_mm = math.log(p_match_to_match)
        self.log_mi = math.log(p_match_to_insert)
        self.log_md = math.log(p_match_to_delete)
        self.log_im = math.log(p_insert_to_match)
        self.log_ii = math.log(p_insert_to_insert)
        self.log_emit_match = math.log(1.0 - p_sub)
        self.log_emit_sub = math.log(p_sub / 3.0)

    @classmethod
    def default_ids(cls) -> "HmmParams":
        return cls(0.994, 0.001, 0.005, 0.5, 0.5, 0.005)

    @classmethod
    def lofi_ids(cls) -> "HmmParams":
        # combined synth+seq rates: sub~0.008, del~0.0055, ins~0.0012
        return cls(0.987, 0.002, 0.011, 0.5, 0.5, 0.01)

    def log_emit_match_obs(self, ref_base: int, obs_base: int) -> float:
        return self.log_emit_match if ref_base == obs_base else self.log_emit_sub


# log-sum-exp helpers with NEG_INF short-circuit semantics

def _lse2(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _lse3(a: float, b: float, c: float) -> float:
    return _lse2(_lse2(a, b), c)


def _band_range(r: int, ref_len: int, band_width: int) -> Tuple[int, int]:
    lo = r - band_width if r >= band_width else 0
    hi = r + band_width + 1
    if hi > ref_len + 1:
        hi = ref_len + 1
    return lo, hi


def _as_bytes(seq) -> bytes:
    if isinstance(seq, (bytes, bytearray)):
        return bytes(seq)
    if isinstance(seq, str):
        return seq.encode("ascii")
    return bytes(seq)


def banded_viterbi(
    ref_seq,
    read,
    params: HmmParams,
    band_width: int = 10,
) -> Tuple[float, List[int]]:
    """banded Viterbi. returns (log_likelihood, decoded_sequence as list of int bases)."""
    ref_seq = _as_bytes(ref_seq)
    read = _as_bytes(read)
    ref_len = len(ref_seq)
    read_len = len(read)
    if read_len == 0 or ref_len == 0:
        return NEG_INF, []

    sl = ref_len + 1  # stride for l
    size = (read_len + 1) * sl * 3
    dp = [NEG_INF] * size
    tb = [(0, 0, 0)] * size

    def idx(r: int, l: int, s: int) -> int:
        return (r * sl + l) * 3 + s

    # init
    dp[idx(0, 0, 0)] = 0.0

    # initial deletions along l at r=0
    lo0, hi0 = _band_range(0, ref_len, band_width)
    for l in range(1, ref_len + 1):
        if l < lo0 or l >= hi0:
            continue
        from_m = dp[idx(0, l - 1, 0)] + params.log_md
        from_d = dp[idx(0, l - 1, 2)] + params.log_md
        if from_m >= from_d:
            dp[idx(0, l, 2)] = from_m
            tb[idx(0, l, 2)] = (0, l - 1, 0)
        else:
            dp[idx(0, l, 2)] = from_d
            tb[idx(0, l, 2)] = (0, l - 1, 2)

    for r in range(1, read_len + 1):
        lo, hi = _band_range(r, ref_len, band_width)
        obs = read[r - 1]

        for l in range(lo, hi):
            # match
            if l >= 1:
                emit = params.log_emit_match if ref_seq[l - 1] == obs else params.log_emit_sub
                from_m = dp[idx(r - 1, l - 1, 0)] + params.log_mm + emit
                from_i = dp[idx(r - 1, l - 1, 1)] + params.log_im + emit
                from_d = dp[idx(r - 1, l - 1, 2)] + params.log_mm + emit
                best = from_m
                best_tb = (r - 1, l - 1, 0)
                if from_i > best:
                    best = from_i
                    best_tb = (r - 1, l - 1, 1)
                if from_d > best:
                    best = from_d
                    best_tb = (r - 1, l - 1, 2)
                dp[idx(r, l, 0)] = best
                tb[idx(r, l, 0)] = best_tb

            # insert
            emit = LOG_QUARTER
            from_m = dp[idx(r - 1, l, 0)] + params.log_mi + emit
            from_i = dp[idx(r - 1, l, 1)] + params.log_ii + emit
            best = from_m
            best_tb = (r - 1, l, 0)
            if from_i > best:
                best = from_i
                best_tb = (r - 1, l, 1)
            dp[idx(r, l, 1)] = best
            tb[idx(r, l, 1)] = best_tb

            # delete
            if l >= 1:
                from_m = dp[idx(r, l - 1, 0)] + params.log_md
                from_d = dp[idx(r, l - 1, 2)] + params.log_md
                best = from_m
                best_tb = (r, l - 1, 0)
                if from_d > best:
                    best = from_d
                    best_tb = (r, l - 1, 2)
                dp[idx(r, l, 2)] = best
                tb[idx(r, l, 2)] = best_tb

    # best ending state
    m_score = dp[idx(read_len, ref_len, 0)]
    i_score = dp[idx(read_len, ref_len, 1)]
    d_score = dp[idx(read_len, ref_len, 2)]
    if m_score >= i_score and m_score >= d_score:
        best_score, cur_state = m_score, 0
    elif i_score >= d_score:
        best_score, cur_state = i_score, 1
    else:
        best_score, cur_state = d_score, 2

    decoded: List[int] = []
    cur_r = read_len
    cur_l = ref_len
    while cur_r > 0 or cur_l > 0:
        pr, pl, ps = tb[idx(cur_r, cur_l, cur_state)]
        if cur_state == 0:
            decoded.append(read[cur_r - 1])
        elif cur_state == 2:
            decoded.append(ref_seq[cur_l - 1])
        # cur_state == 1: insert, skip
        cur_r = pr
        cur_l = pl
        cur_state = ps

    decoded.reverse()
    return best_score, decoded


def banded_forward(
    ref_seq,
    read,
    params: HmmParams,
    band_width: int = 10,
) -> float:
    """banded forward. returns log P(read | ref)."""
    ref_seq = _as_bytes(ref_seq)
    read = _as_bytes(read)
    ref_len = len(ref_seq)
    read_len = len(read)
    if read_len == 0 or ref_len == 0:
        return NEG_INF

    sl = ref_len + 1
    row_size = sl * 3
    prev = [NEG_INF] * row_size
    curr = [NEG_INF] * row_size

    def ri(l: int, s: int) -> int:
        return l * 3 + s

    prev[ri(0, 0)] = 0.0
    lo0, hi0 = _band_range(0, ref_len, band_width)
    for l in range(1, ref_len + 1):
        if l < lo0 or l >= hi0:
            continue
        prev[ri(l, 2)] = _lse2(
            prev[ri(l - 1, 0)] + params.log_md,
            prev[ri(l - 1, 2)] + params.log_md,
        )

    for r in range(1, read_len + 1):
        for i in range(row_size):
            curr[i] = NEG_INF
        lo, hi = _band_range(r, ref_len, band_width)
        obs = read[r - 1]
        for l in range(lo, hi):
            if l >= 1:
                emit = params.log_emit_match if ref_seq[l - 1] == obs else params.log_emit_sub
                curr[ri(l, 0)] = _lse3(
                    prev[ri(l - 1, 0)] + params.log_mm + emit,
                    prev[ri(l - 1, 1)] + params.log_im + emit,
                    prev[ri(l - 1, 2)] + params.log_mm + emit,
                )
            emit = LOG_QUARTER
            curr[ri(l, 1)] = _lse2(
                prev[ri(l, 0)] + params.log_mi + emit,
                prev[ri(l, 1)] + params.log_ii + emit,
            )
            if l >= 1:
                curr[ri(l, 2)] = _lse2(
                    curr[ri(l - 1, 0)] + params.log_md,
                    curr[ri(l - 1, 2)] + params.log_md,
                )
        prev, curr = curr, prev

    return _lse3(prev[ri(ref_len, 0)], prev[ri(ref_len, 1)], prev[ri(ref_len, 2)])


def forward_backward_posteriors(
    ref_seq,
    read,
    params: HmmParams,
    band_width: int = 10,
) -> List[List[float]]:
    """forward-backward posteriors. returns list of [P(A), P(C), P(G), P(T)]
    for each reference position (0-indexed, length ref_len)."""
    ref_seq = _as_bytes(ref_seq)
    read = _as_bytes(read)
    ref_len = len(ref_seq)
    read_len = len(read)
    if read_len == 0 or ref_len == 0:
        return [[0.25, 0.25, 0.25, 0.25] for _ in range(ref_len)]

    row_size = (ref_len + 1) * 3

    def ri(l: int, s: int) -> int:
        return l * 3 + s

    # forward pass: store all rows
    fwd = [[NEG_INF] * row_size for _ in range(read_len + 1)]

    fwd[0][ri(0, 0)] = 0.0
    lo0, hi0 = _band_range(0, ref_len, band_width)
    for l in range(1, ref_len + 1):
        if l < lo0 or l >= hi0:
            continue
        fwd[0][ri(l, 2)] = _lse2(
            fwd[0][ri(l - 1, 0)] + params.log_md,
            fwd[0][ri(l - 1, 2)] + params.log_md,
        )

    for r in range(1, read_len + 1):
        lo, hi = _band_range(r, ref_len, band_width)
        obs = read[r - 1]
        fwd_r = fwd[r]
        fwd_rm1 = fwd[r - 1]
        for l in range(lo, hi):
            if l >= 1:
                emit = params.log_emit_match if ref_seq[l - 1] == obs else params.log_emit_sub
                fwd_r[ri(l, 0)] = _lse3(
                    fwd_rm1[ri(l - 1, 0)] + params.log_mm + emit,
                    fwd_rm1[ri(l - 1, 1)] + params.log_im + emit,
                    fwd_rm1[ri(l - 1, 2)] + params.log_mm + emit,
                )
            emit = LOG_QUARTER
            fwd_r[ri(l, 1)] = _lse2(
                fwd_rm1[ri(l, 0)] + params.log_mi + emit,
                fwd_rm1[ri(l, 1)] + params.log_ii + emit,
            )
            if l >= 1:
                fwd_r[ri(l, 2)] = _lse2(
                    fwd_r[ri(l - 1, 0)] + params.log_md,
                    fwd_r[ri(l - 1, 2)] + params.log_md,
                )

    # backward pass: store all rows
    bwd = [[NEG_INF] * row_size for _ in range(read_len + 1)]
    bwd[read_len][ri(ref_len, 0)] = 0.0
    bwd[read_len][ri(ref_len, 1)] = 0.0
    bwd[read_len][ri(ref_len, 2)] = 0.0

    # backward deletions at r=read_len
    lo_rL, hi_rL = _band_range(read_len, ref_len, band_width)
    for l in range(ref_len - 1, -1, -1):
        if l < lo_rL or l >= hi_rL:
            continue
        bwd[read_len][ri(l, 0)] = _lse2(
            bwd[read_len][ri(l, 0)],
            params.log_md + bwd[read_len][ri(l + 1, 2)],
        )
        bwd[read_len][ri(l, 2)] = _lse2(
            bwd[read_len][ri(l, 2)],
            params.log_md + bwd[read_len][ri(l + 1, 2)],
        )

    for r in range(read_len - 1, -1, -1):
        lo, hi = _band_range(r, ref_len, band_width)
        next_obs = read[r]  # read[r] consumed at step r+1
        bwd_r = bwd[r]
        bwd_rp1 = bwd[r + 1]

        for l in range(lo, hi):
            # transitions to match at (r+1, l+1)
            if l < ref_len:
                emit = params.log_emit_match if ref_seq[l] == next_obs else params.log_emit_sub
                match_next = bwd_rp1[ri(l + 1, 0)]
                if match_next > NEG_INF:
                    bwd_r[ri(l, 0)] = _lse2(
                        bwd_r[ri(l, 0)], params.log_mm + emit + match_next,
                    )
                    bwd_r[ri(l, 1)] = _lse2(
                        bwd_r[ri(l, 1)], params.log_im + emit + match_next,
                    )
                    bwd_r[ri(l, 2)] = _lse2(
                        bwd_r[ri(l, 2)], params.log_mm + emit + match_next,
                    )
            # transitions to insert at (r+1, l)
            emit = LOG_QUARTER
            ins_next = bwd_rp1[ri(l, 1)]
            if ins_next > NEG_INF:
                bwd_r[ri(l, 0)] = _lse2(
                    bwd_r[ri(l, 0)], params.log_mi + emit + ins_next,
                )
                bwd_r[ri(l, 1)] = _lse2(
                    bwd_r[ri(l, 1)], params.log_ii + emit + ins_next,
                )

        # delete transitions within row r (right-to-left)
        # iterate l from hi-2 down to lo
        hi_minus1 = hi - 1 if hi >= 1 else 0
        for l in range(hi_minus1 - 1, lo - 1, -1):
            if l < ref_len:
                del_next = bwd_r[ri(l + 1, 2)]
                if del_next > NEG_INF:
                    bwd_r[ri(l, 0)] = _lse2(
                        bwd_r[ri(l, 0)], params.log_md + del_next,
                    )
                    bwd_r[ri(l, 2)] = _lse2(
                        bwd_r[ri(l, 2)], params.log_md + del_next,
                    )

    # compute posteriors
    posteriors = [[0.0, 0.0, 0.0, 0.0] for _ in range(ref_len)]
    A, C, G, T = ord("A"), ord("C"), ord("G"), ord("T")
    bases = (A, C, G, T)

    for l in range(1, ref_len + 1):
        log_probs = [NEG_INF, NEG_INF, NEG_INF, NEG_INF]
        ref_base = ref_seq[l - 1]

        for r in range(1, read_len + 1):
            lo, hi = _band_range(r, ref_len, band_width)
            if l < lo or l >= hi:
                continue
            fwd_match = fwd[r][ri(l, 0)]
            bwd_match = bwd[r][ri(l, 0)]
            if fwd_match == NEG_INF or bwd_match == NEG_INF:
                continue

            obs = read[r - 1]
            current_emit = params.log_emit_match if ref_base == obs else params.log_emit_sub
            path_no_emit = fwd_match + bwd_match - current_emit

            for b in range(4):
                obs_given_b = params.log_emit_match if obs == bases[b] else params.log_emit_sub
                log_probs[b] = _lse2(log_probs[b], path_no_emit + obs_given_b)

        max_lp = max(log_probs)
        if max_lp == NEG_INF:
            posteriors[l - 1] = [0.25, 0.25, 0.25, 0.25]
        else:
            row = [math.exp(lp - max_lp) for lp in log_probs]
            s = sum(row)
            posteriors[l - 1] = [v / s for v in row]

    return posteriors
