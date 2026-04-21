"""pure-python encode→decode roundtrip. no external dependencies.

this is the closed-loop check that the codec is functional at the package
level: encode some bytes into DNA, treat those DNA seqs as perfect reads,
decode them, assert the output matches the input.
"""

from __future__ import annotations

import random

import pytest

from mahoraga_py import pipeline, viterbi


@pytest.mark.parametrize("channel", ["hifi", "lofi"])
@pytest.mark.parametrize("n_bytes", [100, 500, 1000])
def test_noiseless_roundtrip(channel: str, n_bytes: int):
    rng = random.Random(n_bytes * 7 + (1 if channel == "hifi" else 2))
    data = bytes(rng.randint(0, 255) for _ in range(n_bytes))

    inner = pipeline.InnerCode.new(channel)
    dna_seqs, _ = pipeline.encode_to_dna(data, inner, physical_redundancy=1.28)

    hmm = viterbi.HmmParams.default_ids() if channel == "hifi" else viterbi.HmmParams.lofi_ids()
    rec, stats = pipeline.decode_from_reads(
        list(dna_seqs), dna_seqs, inner, hmm, len(data)
    )
    assert rec == data, f"roundtrip failed: {stats}"


def test_crc32_roundtrip():
    data = b"the quick brown fox jumps over the lazy dog" * 10
    inner = pipeline.InnerCode.new_crc32("hifi")
    dna_seqs, _ = pipeline.encode_to_dna(data, inner, physical_redundancy=1.28)
    hmm = viterbi.HmmParams.default_ids()
    rec, _ = pipeline.decode_from_reads(list(dna_seqs), dna_seqs, inner, hmm, len(data))
    assert rec == data


def test_pipeline_encode_shape():
    # quick sanity: pipeline_encode returns str list with correct oligo length
    data = b"x" * 200
    seqs = pipeline.pipeline_encode(data, channel_type="hifi", physical_redundancy=1.28)
    assert all(isinstance(s, str) and len(s) == 126 for s in seqs)
    assert all(set(s) <= set("ACGT") for s in seqs)
