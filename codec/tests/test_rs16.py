"""tests for rs16: GF(2^16) RS encode, erasure decode, BM error correction."""

from __future__ import annotations

import pytest

from mahoraga_py import rs16


def test_gf16_mul_inv():
    for x in (1, 2, 255, 1000, 30000, 65535):
        assert rs16._gf_mul(x, rs16._gf_inv(x)) == 1, f"inv failed for {x}"


def test_rs16_roundtrip():
    data = list(range(1, 11))
    cw = rs16.rs16_encode(data, 5)
    assert len(cw) == 15
    recovered = rs16.rs16_erasure_decode(cw, [], 10, 5)
    assert recovered == data


def test_rs16_erasure_small():
    data = list(range(100, 120))
    cw = rs16.rs16_encode(data, 8)
    erasures = [0, 3, 7, 10, 15, 20, 25, 27]
    recovered = rs16.rs16_erasure_decode(cw, erasures, 20, 8)
    assert recovered == data


def test_rs16_80pct_erasure():
    k = 200
    n_parity = 800
    data = [((i * 37 + 13) & 0xFFFF) for i in range(k)]
    cw = rs16.rs16_encode(data, n_parity)
    erasures = list(range(800))
    recovered = rs16.rs16_erasure_decode(cw, erasures, k, n_parity)
    assert recovered == data


def test_bytes_roundtrip():
    data = b"hello world!"
    syms = rs16.bytes_to_symbols(data)
    recovered = rs16.symbols_to_bytes(syms, len(data))
    assert recovered == data


def test_bm_no_errors():
    data = list(range(1, 11))
    cw = rs16.rs16_encode(data, 6)
    errs = rs16.rs16_find_errors(cw, [], 10, 6)
    assert errs == []


def test_bm_one_error():
    data = list(range(1, 11))
    n_parity = 6
    cw = rs16.rs16_encode(data, n_parity)
    cw[3] ^= 0x1234
    errs = rs16.rs16_find_errors(cw, [], 10, n_parity)
    assert errs == [3]


def test_bm_two_errors():
    data = list(range(1, 21))
    n_parity = 10
    cw = rs16.rs16_encode(data, n_parity)
    cw[5] ^= 0xABCD
    cw[17] ^= 0x5678
    errs = rs16.rs16_find_errors(cw, [], 20, n_parity)
    assert errs is not None
    assert len(errs) == 2
    assert 5 in errs and 17 in errs


def test_bm_errors_plus_erasures():
    data = list(range(1, 21))
    n_parity = 10
    cw = rs16.rs16_encode(data, n_parity)
    cw[2] ^= 0x1111
    cw[15] ^= 0x2222
    erasures = [0, 7, 12, 25]
    errs = rs16.rs16_find_errors(cw, erasures, 20, n_parity)
    assert errs is not None
    assert len(errs) == 2
    assert 2 in errs and 15 in errs


def test_bm_full_decode():
    data = list(range(100, 150))
    n_parity = 20
    cw = rs16.rs16_encode(data, n_parity)
    # 3 errors
    cw[0] ^= 0x0001
    cw[30] ^= 0xFFFF
    cw[60] ^= 0x8000
    erasures = [10, 20, 40, 50, 65]

    errs = rs16.rs16_find_errors(cw, erasures, 50, n_parity)
    assert errs is not None
    assert len(errs) == 3

    all_erasures = sorted(set(list(erasures) + list(errs)))
    recovered = rs16.rs16_erasure_decode(cw, all_erasures, 50, n_parity)
    assert recovered == data


def test_bm_max_errors():
    data = list(range(1, 31))
    n_parity = 10
    cw = rs16.rs16_encode(data, n_parity)
    cw[0] ^= 1
    cw[5] ^= 2
    cw[10] ^= 3
    cw[20] ^= 4
    cw[35] ^= 5
    errs = rs16.rs16_find_errors(cw, [], 30, n_parity)
    assert errs is not None
    assert len(errs) == 5

    recovered = rs16.rs16_erasure_decode(cw, errs, 30, n_parity)
    assert recovered == data
