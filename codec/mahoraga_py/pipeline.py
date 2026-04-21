"""top-level encode/decode pipeline.

implements the paths actually exercised by the paper benchmarks:

  - ``InnerCode`` with presets: new, new_no_crc, new_crc32, new_crc32_dc,
    new_crc32_dc_len
  - ``encode_to_dna`` / ``encode_to_dna_with_margin`` / ``_with_layout``
  - outer RS block planner (``plan_rs_layout``, ``layout_from_counts``)
  - ``rs_encode_payloads_layout`` and ``rs_decode_payloads_layout``
  - ``decode_from_reads_inner`` with adaptive depth + early termination
  - turbo RS feedback (LoFi only) — left as an opt-in flag

symbol-OSD is omitted (never called from the decode path).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from . import identify as _identify
from . import ldpc as _ldpc
from . import llr_bridge as _llr_bridge
from . import osd as _osd
from . import peg as _peg
from . import rs16 as _rs16
from . import rs_erasure as _rs_erasure
from . import viterbi as _viterbi

# defaults for 126-nt strand length (overridden per call when oligo_length is set)
SEQ_LEN: int = 126
N_CODE: int = 252
CRC_BITS: int = 16

ADAPTIVE_INITIAL_DEPTH_HIFI: int = 20
ADAPTIVE_INITIAL_DEPTH_LOFI: int = 30
CONVERGENCE_THRESHOLD: float = 0.999

MAX_K_PER_BLOCK: int = 2048


def _dc_for(channel: str, seq_len: int) -> int:
    base_dc = 84 if channel == "hifi" else (21 if channel == "lofi" else 84)
    n = seq_len * 2
    dv = 3
    target_dc = base_dc * (seq_len / 126.0)
    target_m = max(1, round((n * dv) / target_dc))
    for delta in range(n * dv):
        for cand in (target_m + delta, max(0, target_m - delta)):
            if cand == 0:
                continue
            if (n * dv) % cand == 0:
                return (n * dv) // cand
    return base_dc


@dataclass
class InnerCode:
    n: int
    k: int
    m: int
    info_cols: List[int]
    parity_cols: List[int]
    parity_deps: List[List[int]]
    useful_bytes: int
    use_crc: bool = True
    crc_bits: int = 16
    channel: str = "hifi"

    def seq_len(self) -> int:
        return self.n // 2

    @classmethod
    def _build_with_dc_len(cls, channel: str, dc_override: int, seq_len: int) -> "InnerCode":
        dv = 3
        dc = dc_override if dc_override > 0 else _dc_for(channel, seq_len)
        n = seq_len * 2
        c2v = _peg.peg_ldpc(n, dv, dc, 42)
        indptr, indices, m = _peg.adj_to_csr(c2v, n)
        mat = _ldpc.Gf2Matrix.from_csr(m, n, indptr, indices)
        pivot_cols, free_cols = mat.row_reduce()
        info_set = set(free_cols)
        info_col_to_idx = {c: i for i, c in enumerate(free_cols)}
        parity_deps: List[List[int]] = []
        for r in range(len(pivot_cols)):
            nz = mat.row_nonzeros(r)
            deps = [info_col_to_idx[c] for c in nz if c != pivot_cols[r] and c in info_set]
            parity_deps.append(deps)
        k = len(free_cols)
        useful_bytes = ((k - CRC_BITS) // 8) & ~1
        return cls(
            n=n,
            k=k,
            m=len(pivot_cols),
            info_cols=list(free_cols),
            parity_cols=list(pivot_cols),
            parity_deps=parity_deps,
            useful_bytes=useful_bytes,
            use_crc=True,
            crc_bits=16,
            channel=channel,
        )

    @classmethod
    def new(cls, channel: str) -> "InnerCode":
        return cls._build_with_dc_len(channel, 0, SEQ_LEN)

    @classmethod
    def new_with_dc(cls, channel: str, dc_override: int) -> "InnerCode":
        return cls._build_with_dc_len(channel, dc_override, SEQ_LEN)

    @classmethod
    def new_with_dc_len(cls, channel: str, dc_override: int, seq_len: int) -> "InnerCode":
        return cls._build_with_dc_len(channel, dc_override, seq_len)

    @classmethod
    def new_no_crc(cls, channel: str) -> "InnerCode":
        code = cls.new(channel)
        code.useful_bytes = (code.k // 8) & ~1
        code.use_crc = False
        code.crc_bits = 0
        return code

    @classmethod
    def new_crc32(cls, channel: str) -> "InnerCode":
        return cls.new_crc32_dc(channel, 0)

    @classmethod
    def new_crc32_dc(cls, channel: str, dc: int) -> "InnerCode":
        return cls.new_crc32_dc_len(channel, dc, SEQ_LEN)

    @classmethod
    def new_crc32_dc_len(cls, channel: str, dc: int, seq_len: int) -> "InnerCode":
        code = cls._build_with_dc_len(channel, dc, seq_len)
        code.useful_bytes = ((code.k - 32) // 8) & ~1
        code.crc_bits = 32
        return code

    def encode(self, info_bits: Sequence[int]) -> List[int]:
        assert len(info_bits) == self.k
        cw = [0] * self.n
        for i, col in enumerate(self.info_cols):
            cw[col] = info_bits[i]
        for p, deps in enumerate(self.parity_deps):
            v = 0
            for j in deps:
                v ^= info_bits[j]
            cw[self.parity_cols[p]] = v
        return cw


# ---------------------------------------------------------------------------
# bit-byte conversion helpers
# ---------------------------------------------------------------------------


def _bits_to_bytes_msb_first(bits: Sequence[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(min(8, len(bits) - i)):
            byte |= bits[i + j] << (7 - j)
        out.append(byte)
    return bytes(out)


def _payload_to_info(payload: Sequence[int], k: int) -> List[int]:
    payload_bits = k - CRC_BITS
    info = [0] * payload_bits
    for i, byte in enumerate(payload):
        for b in range(8):
            idx = i * 8 + b
            if idx < payload_bits:
                info[idx] = (byte >> (7 - b)) & 1
    payload_packed = _bits_to_bytes_msb_first(info)
    crc = _rs_erasure.crc16(payload_packed)
    for b in range(16):
        info.append((crc >> (15 - b)) & 1)
    return info


def _payload_to_info_crc32(payload: Sequence[int], k: int) -> List[int]:
    payload_bits = k - 32
    info = [0] * payload_bits
    for i, byte in enumerate(payload):
        for b in range(8):
            idx = i * 8 + b
            if idx < payload_bits:
                info[idx] = (byte >> (7 - b)) & 1
    payload_packed = _bits_to_bytes_msb_first(info)
    crc = _rs_erasure.crc32(payload_packed)
    for b in range(32):
        info.append((crc >> (31 - b)) & 1)
    return info


def _payload_to_info_no_crc(payload: Sequence[int], k: int) -> List[int]:
    info = [0] * k
    for i, byte in enumerate(payload):
        for b in range(8):
            idx = i * 8 + b
            if idx < k:
                info[idx] = (byte >> (7 - b)) & 1
    return info


# ---------------------------------------------------------------------------
# RS outer layout
# ---------------------------------------------------------------------------


@dataclass
class RsBlock:
    k: int
    n: int
    seq_start: int
    byte_start: int


@dataclass
class RsLayout:
    blocks: List[RsBlock] = field(default_factory=list)
    total_seqs: int = 0
    useful_bytes: int = 0


def inner_pass_rate_for(channel: str) -> float:
    return 0.75 if channel == "lofi" else 0.99


def plan_total_seqs(
    data_len: int,
    useful_bytes: int,
    physical_redundancy: float,
    margin: float,
    inner_pass_rate: float,
) -> Tuple[int, int]:
    k_rs_total = (data_len + useful_bytes - 1) // useful_bytes
    dropout_rate = math.exp(-physical_redundancy)
    effective = max(0.001, (1.0 - dropout_rate) * inner_pass_rate)
    ratio = margin / effective
    cap_by_field = int(65000.0 / ratio)  # floor
    max_k_per_block = max(16, min(cap_by_field, MAX_K_PER_BLOCK))
    n_blocks = max(1, (k_rs_total + max_k_per_block - 1) // max_k_per_block)
    base_k = k_rs_total // n_blocks
    remainder = k_rs_total % n_blocks
    total = 0
    for b in range(n_blocks):
        k_b = base_k + (1 if b < remainder else 0)
        raw_n = math.ceil(k_b * ratio)
        n_parity_b = max(2, k_b // 10, raw_n - k_b)
        total += k_b + n_parity_b
    return k_rs_total, total


def plan_rs_layout(
    data_len: int,
    useful_bytes: int,
    physical_redundancy: float,
    margin: float,
    inner_pass_rate: float,
) -> RsLayout:
    k_rs_total, total_seqs = plan_total_seqs(
        data_len, useful_bytes, physical_redundancy, margin, inner_pass_rate
    )
    return layout_from_counts(k_rs_total, total_seqs, useful_bytes)


def layout_from_counts(k_rs_total: int, n_seqs: int, useful_bytes: int) -> RsLayout:
    if k_rs_total == 0:
        return RsLayout([], 0, useful_bytes)
    ratio = n_seqs / k_rs_total
    cap_by_field = int(65000.0 / ratio)
    max_k_per_block = max(16, min(cap_by_field, MAX_K_PER_BLOCK))
    n_blocks = max(1, (k_rs_total + max_k_per_block - 1) // max_k_per_block)
    base_k = k_rs_total // n_blocks
    remainder = k_rs_total % n_blocks

    blocks: List[RsBlock] = []
    seq_cursor = 0
    byte_cursor = 0
    for b in range(n_blocks):
        k_b = base_k + (1 if b < remainder else 0)
        raw_n = math.ceil(k_b * ratio)
        n_parity_b = max(2, k_b // 10, raw_n - k_b)
        n_b = k_b + n_parity_b
        blocks.append(RsBlock(k=k_b, n=n_b, seq_start=seq_cursor, byte_start=byte_cursor))
        seq_cursor += n_b
        byte_cursor += k_b * useful_bytes

    if seq_cursor != n_seqs and blocks:
        delta = n_seqs - seq_cursor
        last = blocks[-1]
        new_n = max(last.k, last.n + delta)
        last.n = new_n
        seq_cursor = n_seqs

    return RsLayout(blocks=blocks, total_seqs=seq_cursor, useful_bytes=useful_bytes)


def rs_encode_payloads_layout(data: bytes, layout: RsLayout) -> List[bytearray]:
    useful_bytes = layout.useful_bytes
    syms_per_seq = (useful_bytes + 1) // 2
    all_payloads: List[bytearray] = [bytearray(useful_bytes) for _ in range(layout.total_seqs)]

    for block in layout.blocks:
        k_b = block.k
        n_b = block.n
        n_parity_b = n_b - k_b

        # gather info bytes per chunk (pad with zeros)
        file_chunks: List[bytearray] = []
        for chunk_id in range(k_b):
            start = block.byte_start + chunk_id * useful_bytes
            chunk = bytearray(useful_bytes)
            for j in range(useful_bytes):
                idx = start + j
                if idx < len(data):
                    chunk[j] = data[idx]
            file_chunks.append(chunk)

        for sym_idx in range(syms_per_seq):
            byte_lo = sym_idx * 2
            byte_hi = byte_lo + 1

            data_syms: List[int] = []
            for c in file_chunks:
                hi = c[byte_lo] if byte_lo < len(c) else 0
                lo = c[byte_hi] if byte_hi < len(c) else 0
                data_syms.append((hi << 8) | lo)

            rs_cw = _rs16.rs16_encode(data_syms, n_parity_b)

            for seq_off in range(n_b):
                sym = rs_cw[seq_off]
                slot = all_payloads[block.seq_start + seq_off]
                if byte_lo < useful_bytes:
                    slot[byte_lo] = (sym >> 8) & 0xFF
                if byte_hi < useful_bytes:
                    slot[byte_hi] = sym & 0xFF

    return all_payloads


def rs_encode_payloads(
    data: bytes,
    useful_bytes: int,
    physical_redundancy: float,
    margin: float,
    inner_pass_rate: float,
) -> Tuple[List[bytearray], int, RsLayout]:
    layout = plan_rs_layout(len(data), useful_bytes, physical_redundancy, margin, inner_pass_rate)
    k_rs_total = sum(b.k for b in layout.blocks)
    payloads = rs_encode_payloads_layout(data, layout)
    return payloads, k_rs_total, layout


def rs_decode_payloads_layout(
    payloads: Sequence[Optional[bytes]],
    layout: RsLayout,
    data_len: int,
) -> Optional[bytes]:
    useful_bytes = layout.useful_bytes
    syms_per_seq = (useful_bytes + 1) // 2
    assert len(payloads) == layout.total_seqs

    block_results: List[Optional[bytearray]] = [None] * len(layout.blocks)

    for bi, block in enumerate(layout.blocks):
        k_b = block.k
        n_b = block.n
        n_parity_b = n_b - k_b
        block_payloads = payloads[block.seq_start : block.seq_start + n_b]

        erasure_positions = [i for i, p in enumerate(block_payloads) if p is None]
        if len(erasure_positions) > n_parity_b:
            return None

        pre = _rs16.rs16_precompute(n_b)

        block_data = bytearray(k_b * useful_bytes)
        failed = False

        for sym_idx in range(syms_per_seq):
            byte_lo = sym_idx * 2
            byte_hi = byte_lo + 1
            received = [0] * n_b
            for sid, payload in enumerate(block_payloads):
                if payload is not None:
                    hi = payload[byte_lo] if byte_lo < len(payload) else 0
                    lo = payload[byte_hi] if byte_hi < len(payload) else 0
                    received[sid] = (hi << 8) | lo

            error_positions = _rs16.rs16_find_errors_precomputed(
                received, erasure_positions, k_b, n_parity_b, pre
            )
            if error_positions is None:
                failed = True
                break

            all_erasures = sorted(set(list(erasure_positions) + list(error_positions)))
            data_syms = _rs16.rs16_erasure_decode(received, all_erasures, k_b, n_parity_b)
            if data_syms is None:
                failed = True
                break

            for chunk_id, sym in enumerate(data_syms):
                base = chunk_id * useful_bytes
                if byte_lo < useful_bytes:
                    block_data[base + byte_lo] = (sym >> 8) & 0xFF
                if byte_hi < useful_bytes:
                    block_data[base + byte_hi] = sym & 0xFF

        if failed:
            return None
        block_results[bi] = block_data

    out = bytearray()
    for block, result in zip(layout.blocks, block_results):
        if result is None:
            return None
        want = min(block.k * useful_bytes, max(0, data_len - block.byte_start))
        out.extend(result[:want])
    return bytes(out[:data_len])


def rs_decode_payloads(
    payloads: Sequence[Optional[bytes]],
    k_rs: int,
    n_parity: int,
    data_len: int,
    useful_bytes: int,
) -> Optional[bytes]:
    layout = RsLayout(
        blocks=[RsBlock(k=k_rs, n=k_rs + n_parity, seq_start=0, byte_start=0)],
        total_seqs=k_rs + n_parity,
        useful_bytes=useful_bytes,
    )
    return rs_decode_payloads_layout(payloads, layout, data_len)


# ---------------------------------------------------------------------------
# encode to DNA
# ---------------------------------------------------------------------------


def encode_to_dna_with_layout(
    data: bytes,
    inner: InnerCode,
    physical_redundancy: float,
    margin: float,
) -> Tuple[List[bytes], List[List[int]], RsLayout]:
    ipr = inner_pass_rate_for(inner.channel)
    all_payloads, _k, layout = rs_encode_payloads(
        data, inner.useful_bytes, physical_redundancy, margin, ipr
    )
    dna_seqs: List[bytes] = []
    all_infos: List[List[int]] = []
    for i, payload in enumerate(all_payloads):
        if inner.crc_bits == 32:
            info = _payload_to_info_crc32(payload, inner.k)
        elif inner.crc_bits == 0:
            info = _payload_to_info_no_crc(payload, inner.k)
        else:
            info = _payload_to_info(payload, inner.k)
        cw = inner.encode(info)
        scrambled = _llr_bridge.scramble(cw, i)
        dna = _llr_bridge.bits_to_dna(scrambled)
        dna_seqs.append(dna)
        all_infos.append(info)
    return dna_seqs, all_infos, layout


def encode_to_dna_with_margin(
    data: bytes,
    inner: InnerCode,
    physical_redundancy: float,
    margin: float,
) -> Tuple[List[bytes], List[List[int]]]:
    dna, infos, _layout = encode_to_dna_with_layout(data, inner, physical_redundancy, margin)
    return dna, infos


def encode_to_dna(
    data: bytes,
    inner: InnerCode,
    physical_redundancy: float,
) -> Tuple[List[bytes], List[List[int]]]:
    return encode_to_dna_with_margin(data, inner, physical_redundancy, 1.20)


def pipeline_encode(
    data: bytes,
    channel_type: str = "hifi",
    physical_redundancy: float = 1.28,
    margin: float = 1.08,
    oligo_length: int = 126,
) -> List[str]:
    inner = InnerCode.new_crc32_dc_len(channel_type, 0, oligo_length)
    dna_seqs, _ = encode_to_dna_with_margin(data, inner, physical_redundancy, margin)
    return [s.decode("ascii") for s in dna_seqs]


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


@dataclass
class DecodeStats:
    n_reads: int = 0
    n_seqs: int = 0
    n_dropped: int = 0
    n_surviving: int = 0
    n_inner_pass: int = 0
    n_erasures: int = 0
    n_rs_capacity: int = 0
    total_reads_used: int = 0
    converged: bool = False
    n_turbo_recovered: int = 0
    read_counts: List[int] = field(default_factory=list)


@dataclass
class _SeqDecodeResult:
    decoded_bits: List[int]
    info_bits: List[int]
    crc_ok: bool
    n_reads_used: int
    order_used: int
    llrs: List[float]


def decode_from_reads_core(
    reads: Sequence[bytes],
    ref_seqs: Sequence[bytes],
    inner: InnerCode,
    hmm_params: _viterbi.HmmParams,
    data_len: int,
    is_lofi: bool,
    groups: Sequence[Sequence[int]],
    use_crc: bool = True,
) -> Tuple[Optional[bytes], DecodeStats]:
    n_seqs = len(ref_seqs)
    payload_bits = inner.k - inner.crc_bits
    osd_max_order = 0 if not use_crc else (3 if is_lofi else 2)

    adaptive_depth = ADAPTIVE_INITIAL_DEPTH_LOFI if is_lofi else ADAPTIVE_INITIAL_DEPTH_HIFI
    seq_len = inner.seq_len()

    decoder = _osd.OsdDecoder.from_encoding_crc(
        inner.n, inner.info_cols, inner.parity_cols, inner.parity_deps, inner.crc_bits
    )

    seq_results: List[Optional[_SeqDecodeResult]] = [None] * n_seqs

    for seq_id in range(n_seqs):
        read_indices = groups[seq_id]
        if len(read_indices) == 0:
            seq_results[seq_id] = None
            continue

        initial_count = min(len(read_indices), adaptive_depth)
        log_p = [[0.0] * 4 for _ in range(seq_len)]
        reads_used = 0

        # initial batch
        for ri in read_indices[:initial_count]:
            post = _viterbi.forward_backward_posteriors(
                ref_seqs[seq_id], reads[ri], hmm_params, 10
            )
            lim = min(seq_len, len(post))
            for pos in range(lim):
                for b in range(4):
                    v = post[pos][b]
                    log_p[pos][b] += math.log(v if v > 1e-10 else 1e-10)
            reads_used += 1

        # convergence check after initial batch
        def _check_converged() -> bool:
            for pos in range(seq_len):
                mx = max(log_p[pos])
                s = 0.0
                mv = 0.0
                for b in range(4):
                    v = math.exp(log_p[pos][b] - mx)
                    s += v
                    if v > mv:
                        mv = v
                if mv / s < CONVERGENCE_THRESHOLD:
                    return False
            return True

        converged = _check_converged()

        # additional reads with periodic check every 5
        if not converged and len(read_indices) > initial_count:
            for ri in read_indices[initial_count:]:
                post = _viterbi.forward_backward_posteriors(
                    ref_seqs[seq_id], reads[ri], hmm_params, 10
                )
                lim = min(seq_len, len(post))
                for pos in range(lim):
                    for b in range(4):
                        v = post[pos][b]
                        log_p[pos][b] += math.log(v if v > 1e-10 else 1e-10)
                reads_used += 1
                if reads_used % 5 == 0 and _check_converged():
                    break

        # normalize
        posteriors: List[List[float]] = [[0.0] * 4 for _ in range(seq_len)]
        for pos in range(seq_len):
            mx = max(log_p[pos])
            row = [math.exp(log_p[pos][b] - mx) for b in range(4)]
            s = sum(row)
            posteriors[pos] = [v / s for v in row]

        # LLR + descramble
        scr = _llr_bridge.scrambler_bits(seq_id, N_CODE)
        llrs = _llr_bridge.posteriors_to_llrs(posteriors, scr)
        llrs_trimmed = llrs[: inner.n]

        result = decoder.decode(llrs_trimmed, 2, inner.k)
        if not result.crc_ok and osd_max_order >= 3:
            result = decoder.decode(llrs_trimmed, 3, 50)

        seq_results[seq_id] = _SeqDecodeResult(
            decoded_bits=list(result.decoded),
            info_bits=list(result.info_bits),
            crc_ok=result.crc_ok,
            n_reads_used=reads_used,
            order_used=result.order_used,
            llrs=list(llrs_trimmed),
        )

    # assemble payloads (CRC gate if enabled)
    k_rs = (data_len + inner.useful_bytes - 1) // inner.useful_bytes
    n_parity_rs = n_seqs - k_rs
    rs_layout = layout_from_counts(k_rs, n_seqs, inner.useful_bytes)

    n_inner_pass = 0
    n_surviving = 0
    total_reads_used = 0
    decoded_payloads: List[Optional[bytes]] = []

    for r in seq_results:
        if r is None:
            decoded_payloads.append(None)
            continue
        accept = r.crc_ok if use_crc else True
        n_surviving += 1
        total_reads_used += r.n_reads_used
        if accept:
            payload = bytearray(inner.useful_bytes)
            for byte_idx in range(inner.useful_bytes):
                bv = 0
                for b in range(8):
                    bit_idx = byte_idx * 8 + b
                    if bit_idx < payload_bits:
                        bv |= r.info_bits[bit_idx] << (7 - b)
                payload[byte_idx] = bv
            decoded_payloads.append(bytes(payload))
            n_inner_pass += 1
        else:
            decoded_payloads.append(None)

    n_dropped = sum(1 for r in seq_results if r is None)
    read_counts = [0 if r is None else r.n_reads_used for r in seq_results]

    stats = DecodeStats(
        n_reads=len(reads),
        n_seqs=n_seqs,
        n_dropped=n_dropped,
        n_surviving=n_surviving,
        n_inner_pass=n_inner_pass,
        n_erasures=n_seqs - n_inner_pass,
        n_rs_capacity=n_parity_rs,
        total_reads_used=total_reads_used,
        converged=n_inner_pass >= k_rs,
        n_turbo_recovered=0,
        read_counts=read_counts,
    )

    recovered_data = rs_decode_payloads_layout(decoded_payloads, rs_layout, data_len)

    # turbo RS feedback (lofi+crc only)
    if recovered_data is not None and use_crc and is_lofi:
        all_payloads = rs_encode_payloads_layout(recovered_data, rs_layout)
        n_turbo = 0
        for sid, r in enumerate(seq_results):
            if r is None or r.crc_ok:
                continue
            if sid >= len(all_payloads):
                continue
            true_payload = all_payloads[sid]
            if inner.crc_bits == 32:
                true_info = _payload_to_info_crc32(true_payload, inner.k)
            else:
                true_info = _payload_to_info(true_payload, inner.k)
            true_cw = inner.encode(true_info)
            hd = sum(1 for a, b in zip(true_cw, r.decoded_bits) if a != b)
            if hd > 5:
                continue
            fixed_llrs = list(r.llrs)
            lim = min(inner.n, len(true_cw), len(fixed_llrs))
            for i in range(lim):
                if true_cw[i] != r.decoded_bits[i]:
                    fixed_llrs[i] = 20.0 if true_cw[i] == 0 else -20.0
            retry = decoder.decode(fixed_llrs, 2, inner.k)
            if retry.crc_ok:
                payload = bytearray(inner.useful_bytes)
                for byte_idx in range(inner.useful_bytes):
                    bv = 0
                    for b in range(8):
                        bit_idx = byte_idx * 8 + b
                        if bit_idx < payload_bits:
                            bv |= retry.info_bits[bit_idx] << (7 - b)
                    payload[byte_idx] = bv
                decoded_payloads[sid] = bytes(payload)
                n_turbo += 1
        if n_turbo > 0:
            recovered_turbo = rs_decode_payloads_layout(decoded_payloads, rs_layout, data_len)
            n_inner_pass_turbo = sum(1 for p in decoded_payloads if p is not None)
            stats_turbo = DecodeStats(
                n_reads=stats.n_reads,
                n_seqs=n_seqs,
                n_dropped=n_dropped,
                n_surviving=n_surviving,
                n_inner_pass=n_inner_pass_turbo,
                n_erasures=n_seqs - n_inner_pass_turbo,
                n_rs_capacity=n_parity_rs,
                total_reads_used=total_reads_used,
                converged=n_inner_pass_turbo >= k_rs,
                n_turbo_recovered=n_turbo,
                read_counts=stats.read_counts,
            )
            if recovered_turbo is not None:
                return recovered_turbo, stats_turbo
            if recovered_data is not None:
                return recovered_data, stats_turbo
            return None, stats_turbo

    return recovered_data, stats


def decode_from_reads_inner_opts(
    reads: Sequence[bytes],
    ref_seqs: Sequence[bytes],
    inner: InnerCode,
    hmm_params: _viterbi.HmmParams,
    data_len: int,
    is_lofi: bool,
    use_crc: bool = True,
) -> Tuple[Optional[bytes], DecodeStats]:
    assignments = _identify.batch_identify(reads, ref_seqs, hmm_params, 10, 8, 15)
    groups = _identify.group_reads_by_ref(reads, assignments, len(ref_seqs))
    return decode_from_reads_core(
        reads, ref_seqs, inner, hmm_params, data_len, is_lofi, groups, use_crc
    )


def decode_from_reads_inner(
    reads: Sequence[bytes],
    ref_seqs: Sequence[bytes],
    inner: InnerCode,
    hmm_params: _viterbi.HmmParams,
    data_len: int,
    is_lofi: bool,
) -> Tuple[Optional[bytes], DecodeStats]:
    return decode_from_reads_inner_opts(
        reads, ref_seqs, inner, hmm_params, data_len, is_lofi, use_crc=True
    )


def decode_from_reads_with_groups(
    reads: Sequence[bytes],
    ref_seqs: Sequence[bytes],
    inner: InnerCode,
    hmm_params: _viterbi.HmmParams,
    data_len: int,
    is_lofi: bool,
    groups: Sequence[Sequence[int]],
) -> Tuple[Optional[bytes], DecodeStats]:
    return decode_from_reads_core(
        reads, ref_seqs, inner, hmm_params, data_len, is_lofi, groups, use_crc=True
    )


def decode_from_reads(
    reads: Sequence[bytes],
    ref_seqs: Sequence[bytes],
    inner: InnerCode,
    hmm_params: _viterbi.HmmParams,
    data_len: int,
) -> Tuple[Optional[bytes], DecodeStats]:
    return decode_from_reads_inner(reads, ref_seqs, inner, hmm_params, data_len, False)


def pipeline_decode_from_reads(
    reads: Sequence[str],
    ref_seqs: Sequence[str],
    channel_type: str = "hifi",
    data_len: int = 0,
    oligo_length: int = 126,
) -> Optional[bytes]:
    inner = InnerCode.new_crc32_dc_len(channel_type, 0, oligo_length)
    is_lofi = channel_type == "lofi"
    hmm = _viterbi.HmmParams.lofi_ids() if is_lofi else _viterbi.HmmParams.default_ids()
    reads_b = [r.encode("ascii") if isinstance(r, str) else bytes(r) for r in reads]
    refs_b = [r.encode("ascii") if isinstance(r, str) else bytes(r) for r in ref_seqs]
    recovered, _stats = decode_from_reads_inner(reads_b, refs_b, inner, hmm, data_len, is_lofi)
    return recovered
