"""reed-solomon over GF(2^16).

  - GF(2^16) with primitive polynomial 0x1002D
  - barycentric Lagrange encode + erasure decode
  - Berlekamp-Massey + Forney syndromes + Chien search error locator
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

_GF_ORDER: int = 65536
_GF_POLY: int = 0x1_002D  # x^16 + x^5 + x^3 + x^2 + 1
_ORD: int = _GF_ORDER - 1  # 65535


def _build_gf16_tables() -> Tuple[List[int], List[int]]:
    exp_ = [0] * (2 * _ORD)
    log_ = [0] * _GF_ORDER
    x = 1
    for i in range(_ORD):
        exp_[i] = x
        exp_[i + _ORD] = x
        log_[x] = i
        x <<= 1
        if x & 0x10000:
            x ^= _GF_POLY
    return exp_, log_


_EXP, _LOG = _build_gf16_tables()


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[(_LOG[a] + _LOG[b]) % _ORD]


def _gf_inv(a: int) -> int:
    # caller must ensure a != 0.
    return _EXP[_ORD - _LOG[a]]


def _eval_point(j: int) -> int:
    return 1 if j == 0 else _EXP[j % _ORD]


def rs16_encode(data: Sequence[int], n_parity: int) -> List[int]:
    """systematic GRS encode via barycentric Lagrange over alpha^j nodes."""
    k = len(data)
    n = k + n_parity
    assert n <= _ORD, f"n={n} exceeds GF(2^16) max codeword length"

    cw = list(data) + [0] * n_parity

    xs = [_eval_point(i) for i in range(k)]
    # v_i = 1 / prod_{m != i} (x_i - x_m)
    v = [0] * k
    for i in range(k):
        prod = 1
        for m in range(k):
            if m != i:
                prod = _gf_mul(prod, xs[i] ^ xs[m])
        v[i] = _gf_inv(prod)

    dv = [_gf_mul(data[i], v[i]) for i in range(k)]

    for j in range(k, n):
        xj = _eval_point(j)
        n_prod = 1
        for m in range(k):
            n_prod = _gf_mul(n_prod, xj ^ xs[m])
        accum = 0
        for i in range(k):
            diff = xj ^ xs[i]
            accum ^= _gf_mul(dv[i], _gf_inv(diff))
        cw[j] = _gf_mul(n_prod, accum)

    return cw


def rs16_erasure_decode(
    received: Sequence[int],
    erasure_positions: Sequence[int],
    k: int,
    n_parity: int,
) -> Optional[List[int]]:
    """erasure-only decode: return the original k info symbols, or None if too many erasures."""
    n = k + n_parity
    assert len(received) == n
    ne = len(erasure_positions)
    if ne > n_parity:
        return None

    erased = set(erasure_positions)
    data = [0] * k
    info_erased: List[int] = []
    for j in range(k):
        if j in erased:
            info_erased.append(j)
        else:
            data[j] = received[j]
    if not info_erased:
        return data

    # pick first k surviving positions as interpolation nodes
    nodes: List[int] = []
    for j in range(n):
        if j not in erased:
            nodes.append(j)
            if len(nodes) == k:
                break
    if len(nodes) < k:
        return None

    xs = [_eval_point(j) for j in nodes]
    y = [received[j] for j in nodes]

    v = [0] * k
    for i in range(k):
        prod = 1
        for m in range(k):
            if m != i:
                prod = _gf_mul(prod, xs[i] ^ xs[m])
        v[i] = _gf_inv(prod)

    yv = [_gf_mul(y[i], v[i]) for i in range(k)]

    for j in info_erased:
        eval_ = _eval_point(j)

        # pass-through if eval equals an interpolation node
        match_i = -1
        for i in range(k):
            if xs[i] == eval_:
                match_i = i
                break
        if match_i >= 0:
            data[j] = y[match_i]
            continue

        n_prod = 1
        for m in range(k):
            n_prod = _gf_mul(n_prod, eval_ ^ xs[m])

        accum = 0
        for i in range(k):
            diff = eval_ ^ xs[i]
            accum ^= _gf_mul(yv[i], _gf_inv(diff))

        data[j] = _gf_mul(n_prod, accum)

    return data


class GrsPrecomputed:
    """Precomputed eval points and column multipliers for n positions."""
    __slots__ = ("eval_pts", "col_mult")

    def __init__(self, eval_pts: List[int], col_mult: List[int]) -> None:
        self.eval_pts = eval_pts
        self.col_mult = col_mult


def rs16_precompute(n: int) -> GrsPrecomputed:
    eval_pts = [1 if j == 0 else _EXP[j % _ORD] for j in range(n)]
    col_mult = [0] * n
    for j in range(n):
        prod = 1
        for i in range(n):
            if i != j:
                prod = _gf_mul(prod, eval_pts[j] ^ eval_pts[i])
        col_mult[j] = _gf_inv(prod)
    return GrsPrecomputed(eval_pts, col_mult)


def _berlekamp_massey_impl(syndromes: Sequence[int]) -> List[int]:
    nn = len(syndromes)
    if nn == 0 or all(v == 0 for v in syndromes):
        return [1]
    c = [1]
    b = [1]
    l = 0
    m = 1
    delta_b = 1

    for i in range(nn):
        d = syndromes[i]
        for j in range(1, len(c)):
            if i >= j:
                d ^= _gf_mul(c[j], syndromes[i - j])
        if d == 0:
            m += 1
        elif 2 * l <= i:
            t = list(c)
            f = _gf_mul(d, _gf_inv(delta_b))
            new_len = max(len(c), len(b) + m)
            if new_len > len(c):
                c = c + [0] * (new_len - len(c))
            for j in range(len(b)):
                c[j + m] ^= _gf_mul(f, b[j])
            l = i + 1 - l
            b = t
            delta_b = d
            m = 1
        else:
            f = _gf_mul(d, _gf_inv(delta_b))
            new_len = max(len(c), len(b) + m)
            if new_len > len(c):
                c = c + [0] * (new_len - len(c))
            for j in range(len(b)):
                c[j + m] ^= _gf_mul(f, b[j])
            m += 1
    return c


def rs16_find_errors_precomputed(
    received: Sequence[int],
    erasure_positions: Sequence[int],
    k: int,
    n_parity: int,
    pre: GrsPrecomputed,
) -> Optional[List[int]]:
    n = k + n_parity
    assert len(received) == n
    s = len(erasure_positions)
    if s > n_parity:
        return None

    # syndromes S[i] = sum_j r[j]*v[j]*x_j^i
    syndromes = [0] * n_parity
    for i in range(n_parity):
        for j in range(n):
            rv = _gf_mul(received[j], pre.col_mult[j])
            if rv != 0:
                xji = 1 if i == 0 else _EXP[(j * i) % _ORD]
                syndromes[i] ^= _gf_mul(rv, xji)

    if all(v == 0 for v in syndromes):
        return []

    # Gamma(x) = prod (1 - x*X_l): ng[idx] ^= v; ng[idx+1] ^= v*xl
    # so gamma[d] = coefficient of x^d starting with gamma[0]=1
    gamma = [1]
    for pos in erasure_positions:
        xl = pre.eval_pts[pos]
        ng = [0] * (len(gamma) + 1)
        for idx, v in enumerate(gamma):
            ng[idx] ^= v
            ng[idx + 1] ^= _gf_mul(v, xl)
        gamma = ng

    # Forney syndromes: T = (S * gamma) mod x^n_parity
    conv = [0] * n_parity
    for i in range(n_parity):
        for j in range(min(len(gamma), i + 1)):
            conv[i] ^= _gf_mul(syndromes[i - j], gamma[j])

    bm_len = max(n_parity - s, 0)
    if bm_len == 0:
        return []
    bm_input = conv[s : s + bm_len]

    sigma = _berlekamp_massey_impl(bm_input)
    n_errors = len(sigma) - 1
    if 2 * n_errors + s > n_parity:
        return None
    if n_errors == 0:
        return []

    erased_set = set(erasure_positions)
    error_positions: List[int] = []
    for j in range(n):
        if j in erased_set:
            continue
        inv_xj = _gf_inv(pre.eval_pts[j])
        val = 0
        pw = 1
        for c in sigma:
            val ^= _gf_mul(c, pw)
            pw = _gf_mul(pw, inv_xj)
        if val == 0:
            error_positions.append(j)

    if len(error_positions) != n_errors:
        return None
    return error_positions


def rs16_find_errors(
    received: Sequence[int],
    erasure_positions: Sequence[int],
    k: int,
    n_parity: int,
) -> Optional[List[int]]:
    pre = rs16_precompute(k + n_parity)
    return rs16_find_errors_precomputed(received, erasure_positions, k, n_parity, pre)


def bytes_to_symbols(data: bytes) -> List[int]:
    syms: List[int] = []
    for i in range(0, len(data), 2):
        hi = data[i]
        lo = data[i + 1] if i + 1 < len(data) else 0
        syms.append((hi << 8) | lo)
    return syms


def symbols_to_bytes(syms: Sequence[int], n_bytes: int) -> bytes:
    out = bytearray()
    for s in syms:
        out.append((s >> 8) & 0xFF)
        out.append(s & 0xFF)
    return bytes(out[:n_bytes])
