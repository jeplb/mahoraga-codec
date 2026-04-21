"""tests for rs_erasure: roundtrips on GF(2^8) RS + CRC known-answer checks."""

from __future__ import annotations

import pytest

from mahoraga_py import rs_erasure as rse


def test_rs_roundtrip_noiseless():
    data = [1, 2, 3, 4, 5]
    cw = rse.rs_encode(data, 4)
    assert len(cw) == 9
    assert cw[:5] == data


def test_rs_erasure_2():
    data = [10, 20, 30, 40, 50]
    cw = rse.rs_encode(data, 4)
    rx = list(cw)
    rx[1] = 0
    rx[3] = 0
    assert rse.rs_erasure_decode(rx, [1, 3], 5, 4)
    assert rx[:5] == data


def test_rs_max_erasure():
    data = list(range(10))
    cw = rse.rs_encode(data, 6)
    rx = list(cw)
    erasures = list(range(6))
    for e in erasures:
        rx[e] = 0
    assert rse.rs_erasure_decode(rx, erasures, 10, 6)
    assert rx[:10] == data


def test_rs_parity_erasure():
    data = list(range(100, 110))
    cw = rse.rs_encode(data, 5)
    rx = list(cw)
    erasures = [10, 11, 12]
    for e in erasures:
        rx[e] = 0
    assert rse.rs_erasure_decode(rx, erasures, 10, 5)
    assert rx[:10] == data


def test_rs_large():
    data = [(i * 37 + 13) & 0xFF for i in range(200)]
    n_parity = 50
    cw = rse.rs_encode(data, n_parity)
    rx = list(cw)
    erasures = list(range(0, len(cw), 5))[:n_parity]
    for e in erasures:
        rx[e] = 0
    assert rse.rs_erasure_decode(rx, erasures, 200, n_parity)
    assert rx[:200] == data


def test_rs_decode_refuses_too_many():
    data = [1, 2, 3, 4, 5]
    cw = rse.rs_encode(data, 3)
    rx = list(cw)
    # 4 erasures > 3 parity — should fail
    assert not rse.rs_erasure_decode(rx, [0, 1, 2, 3], 5, 3)


# --- CRC known-answer tests ---

def test_crc16_ccitt_known_values():
    # CRC-16/CCITT-FALSE (init=0xFFFF, poly=0x1021, MSB-first, no reflect):
    # "123456789" = 0x29B1 — standard vector
    assert rse.crc16(b"123456789") == 0x29B1
    assert rse.crc16(b"") == 0xFFFF


def test_crc16_stability():
    # same input → same output; 1-bit flip → different
    assert rse.crc16(b"hello") == rse.crc16(b"hello")
    assert rse.crc16(b"hello") != rse.crc16(b"hellp")


def test_crc32_mpeg2_known_values():
    # CRC-32/MPEG-2 (init=0xFFFFFFFF, poly=0x04C11DB7, no reflect, no xorout)
    # reference vector: "123456789" -> 0x0376E6E7
    assert rse.crc32(b"123456789") == 0x0376E6E7
    assert rse.crc32(b"") == 0xFFFFFFFF


def test_crc32_stability():
    assert rse.crc32(b"hello") == rse.crc32(b"hello")
    assert rse.crc32(b"hello") != rse.crc32(b"hellp")
