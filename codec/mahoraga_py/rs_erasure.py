"""reed-solomon over GF(2^8) + CRC-16/32.

implements:
  - GF(2^8) with primitive polynomial 0x11D
  - Vandermonde parity check H[i][j] = alpha^((i+1)*j)
  - systematic encode + erasure decode via gaussian elimination
  - CRC-16 CCITT (polynomial 0x1021, init 0xFFFF)
  - CRC-32 (polynomial 0x04C11DB7, init 0xFFFFFFFF, MSB-first, no reflect)
"""

from __future__ import annotations

from typing import List, Sequence

_GF_POLY: int = 0x11D  # x^8 + x^4 + x^3 + x^2 + 1


def _build_gf_tables():
    exp_ = [0] * 512
    log_ = [0] * 256
    x = 1
    for i in range(255):
        exp_[i] = x
        exp_[i + 255] = x
        log_[x] = i
        x <<= 1
        if x & 0x100:
            x ^= _GF_POLY
    return exp_, log_


_EXP, _LOG = _build_gf_tables()


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[(_LOG[a] + _LOG[b]) % 255]


def _gf_inv(a: int) -> int:
    assert a != 0
    return _EXP[255 - _LOG[a]]


def _h_entry(i: int, j: int) -> int:
    if j == 0:
        return 1
    return _EXP[((i + 1) * j) % 255]


def rs_encode(data: Sequence[int], n_parity: int) -> List[int]:
    """systematic GF(2^8) RS encode: k bytes → k + n_parity bytes."""
    k = len(data)

    # syndromes s[i] = sum_j H[i][j] * data[j]
    syndromes = [0] * n_parity
    for i in range(n_parity):
        acc = 0
        for j in range(k):
            acc ^= _gf_mul(_h_entry(i, j), data[j])
        syndromes[i] = acc

    # H_parity[i][p] = H[i][k+p]; solve n_parity × n_parity system
    mat = [[0] * (n_parity + 1) for _ in range(n_parity)]
    for i in range(n_parity):
        for p in range(n_parity):
            mat[i][p] = _h_entry(i, k + p)
        mat[i][n_parity] = syndromes[i]

    _gf_gauss(mat, n_parity)

    cw = list(data)
    for p in range(n_parity):
        cw.append(mat[p][n_parity])
    return cw


def rs_erasure_decode(
    received: List[int],
    erasure_positions: Sequence[int],
    k: int,
    n_parity: int,
) -> bool:
    """in-place erasure decode. returns True on success."""
    n = k + n_parity
    assert len(received) == n
    ne = len(erasure_positions)
    if ne == 0:
        return True
    if ne > n_parity:
        return False

    erased = set(erasure_positions)
    syndromes = [0] * n_parity
    for i in range(n_parity):
        acc = 0
        for j in range(n):
            if j not in erased:
                acc ^= _gf_mul(_h_entry(i, j), received[j])
        syndromes[i] = acc

    mat = [[0] * (ne + 1) for _ in range(ne)]
    for i in range(ne):
        for t in range(ne):
            mat[i][t] = _h_entry(i, erasure_positions[t])
        mat[i][ne] = syndromes[i]

    if not _gf_gauss(mat, ne):
        return False

    for t in range(ne):
        received[erasure_positions[t]] = mat[t][ne]
    return True


def _gf_gauss(mat: List[List[int]], n: int) -> bool:
    for col in range(n):
        pivot = n
        for row in range(col, n):
            if mat[row][col] != 0:
                pivot = row
                break
        if pivot == n:
            return False
        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]

        inv = _gf_inv(mat[col][col])
        ncols = len(mat[col])
        for j in range(col, ncols):
            mat[col][j] = _gf_mul(mat[col][j], inv)

        for row in range(n):
            if row != col and mat[row][col] != 0:
                f = mat[row][col]
                for j in range(col, ncols):
                    mat[row][j] ^= _gf_mul(f, mat[col][j])
    return True


def crc16(data: bytes) -> int:
    """CRC-16 CCITT, init 0xFFFF, polynomial 0x1021, MSB-first, no reflect/xorout."""
    crc = 0xFFFF
    for byte in data:
        crc ^= (byte & 0xFF) << 8
        crc &= 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc32(data: bytes) -> int:
    """CRC-32 init 0xFFFFFFFF, polynomial 0x04C11DB7, MSB-first, no reflect/xorout.

    note: this is NOT the common "CRC-32/ISO-HDLC" used by zlib — it is the
    straight MSB-first polynomial without reflection (MPEG-2-style).
    """
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= (byte & 0xFF) << 24
        crc &= 0xFFFFFFFF
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
    return crc
