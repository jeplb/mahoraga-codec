"""ordered statistics decoder (OSD) for short LDPC codes.

including the exact candidate enumeration order, tie-break rule, CRC verification, and ``order_used`` bookkeeping.

we use a clever shortcut: with a precomputed systematic form
(info_cols / parity_cols / parity_deps), OSD never actually re-reduces the
matrix — it just decides which subset of the info bits to flip. our port
follows that exact shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from . import rs_erasure


@dataclass
class OsdResult:
    decoded: List[int] = field(default_factory=list)  # n-bit codeword (original column order)
    info_bits: List[int] = field(default_factory=list)  # k info bits (info-index order)
    crc_ok: bool = False
    metric: float = 0.0
    order_used: int = 0  # 0, 1, 2, or 3


class OsdDecoder:
    """OSD decoder with pre-specified systematic form.

    holds (info_cols, parity_cols, parity_deps) plus a CRC width (0, 16, or 32).
    """

    __slots__ = ("n", "k", "m", "parity_deps", "info_cols", "parity_cols", "crc_bits")

    def __init__(
        self,
        n: int,
        info_cols: Sequence[int],
        parity_cols: Sequence[int],
        parity_deps: Sequence[Sequence[int]],
        crc_bits: int = 16,
    ) -> None:
        self.n = n
        self.k = len(info_cols)
        self.m = len(parity_cols)
        self.info_cols = list(info_cols)
        self.parity_cols = list(parity_cols)
        self.parity_deps = [list(deps) for deps in parity_deps]
        self.crc_bits = crc_bits

    @classmethod
    def from_encoding(
        cls,
        n: int,
        info_cols: Sequence[int],
        parity_cols: Sequence[int],
        parity_deps: Sequence[Sequence[int]],
    ) -> "OsdDecoder":
        return cls(n, info_cols, parity_cols, parity_deps, crc_bits=16)

    @classmethod
    def from_encoding_crc(
        cls,
        n: int,
        info_cols: Sequence[int],
        parity_cols: Sequence[int],
        parity_deps: Sequence[Sequence[int]],
        crc_bits: int,
    ) -> "OsdDecoder":
        return cls(n, info_cols, parity_cols, parity_deps, crc_bits=crc_bits)

    def _encode_from_info(self, info: Sequence[int]) -> List[int]:
        cw = [0] * self.n
        for i, c in enumerate(self.info_cols):
            cw[c] = info[i]
        for p, deps in enumerate(self.parity_deps):
            v = 0
            for d in deps:
                v ^= info[d]
            cw[self.parity_cols[p]] = v
        return cw

    def _metric(self, llrs: Sequence[float], cw: Sequence[int]) -> float:
        m = 0.0
        for i in range(self.n):
            sign = 1.0 - 2.0 * cw[i]
            m += llrs[i] * sign
        return m

    def _check_crc(self, info: Sequence[int]) -> bool:
        if self.crc_bits == 0:
            return False
        if self.k < self.crc_bits + 16:
            return False
        payload_bits = self.k - self.crc_bits
        payload_bytes = bytearray()
        for chunk_start in range(0, payload_bits, 8):
            byte = 0
            for b in range(8):
                if chunk_start + b < payload_bits:
                    byte = (byte << 1) | info[chunk_start + b]
            payload_bytes.append(byte)

        if self.crc_bits == 32:
            expected = rs_erasure.crc32(bytes(payload_bytes))
            actual = 0
            for b in range(32):
                actual = (actual << 1) | info[payload_bits + b]
            return expected == (actual & 0xFFFFFFFF)
        else:
            expected = rs_erasure.crc16(bytes(payload_bytes))
            actual = 0
            for b in range(16):
                actual = (actual << 1) | info[payload_bits + b]
            return expected == (actual & 0xFFFF)

    def decode(self, llrs: Sequence[float], max_order: int = 2, w: int = 30) -> OsdResult:
        assert len(llrs) == self.n

        # order-0: hard-decide each info bit from llrs[info_col]
        info_hard = [0 if llrs[c] >= 0.0 else 1 for c in self.info_cols]

        cw0 = self._encode_from_info(info_hard)
        metric0 = self._metric(llrs, cw0)
        crc0 = self._check_crc(info_hard)

        if crc0:
            return OsdResult(
                decoded=cw0,
                info_bits=list(info_hard),
                crc_ok=True,
                metric=metric0,
                order_used=0,
            )

        if max_order == 0:
            return OsdResult(
                decoded=cw0,
                info_bits=list(info_hard),
                crc_ok=False,
                metric=metric0,
                order_used=0,
            )

        # reliability order for info positions (lowest |LLR| = least reliable first)
        info_rel: List[Tuple[int, float]] = [
            (i, abs(llrs[c])) for i, c in enumerate(self.info_cols)
        ]
        info_rel.sort(key=lambda t: t[1])
        all_info_positions = [i for i, _ in info_rel]
        search_positions_o2 = [i for i, _ in info_rel[: min(w, self.k)]]

        best_metric = metric0
        best_info = list(info_hard)
        best_cw = cw0
        best_crc = False
        best_order = 0

        # order-1: flip each info bit (k candidates)
        for pos in all_info_positions:
            info_hard[pos] ^= 1
            cw = self._encode_from_info(info_hard)
            metric = self._metric(llrs, cw)
            crc = self._check_crc(info_hard)
            if crc and (not best_crc or metric > best_metric):
                best_metric = metric
                best_info = list(info_hard)
                best_cw = cw
                best_crc = True
                best_order = 1
            info_hard[pos] ^= 1

        if best_crc and max_order <= 1:
            return OsdResult(
                decoded=best_cw,
                info_bits=best_info,
                crc_ok=True,
                metric=best_metric,
                order_used=best_order,
            )

        # order-2: each pair among search_positions_o2
        if max_order >= 2:
            w_actual = len(search_positions_o2)
            for i in range(w_actual):
                for j in range(i + 1, w_actual):
                    p1 = search_positions_o2[i]
                    p2 = search_positions_o2[j]
                    info_hard[p1] ^= 1
                    info_hard[p2] ^= 1

                    cw = self._encode_from_info(info_hard)
                    metric = self._metric(llrs, cw)
                    crc = self._check_crc(info_hard)
                    if crc and (not best_crc or metric > best_metric):
                        best_metric = metric
                        best_info = list(info_hard)
                        best_cw = cw
                        best_crc = True
                        best_order = 2

                    info_hard[p1] ^= 1
                    info_hard[p2] ^= 1

        if best_crc and max_order <= 2:
            return OsdResult(
                decoded=best_cw,
                info_bits=best_info,
                crc_ok=True,
                metric=best_metric,
                order_used=best_order,
            )

        # order-3: each triple among search_positions_o2
        if max_order >= 3:
            w_actual = len(search_positions_o2)
            for i in range(w_actual):
                for j in range(i + 1, w_actual):
                    for l in range(j + 1, w_actual):
                        p1 = search_positions_o2[i]
                        p2 = search_positions_o2[j]
                        p3 = search_positions_o2[l]
                        info_hard[p1] ^= 1
                        info_hard[p2] ^= 1
                        info_hard[p3] ^= 1

                        cw = self._encode_from_info(info_hard)
                        metric = self._metric(llrs, cw)
                        crc = self._check_crc(info_hard)
                        if crc and (not best_crc or metric > best_metric):
                            best_metric = metric
                            best_info = list(info_hard)
                            best_cw = cw
                            best_crc = True
                            best_order = 3

                        info_hard[p1] ^= 1
                        info_hard[p2] ^= 1
                        info_hard[p3] ^= 1

        return OsdResult(
            decoded=best_cw,
            info_bits=best_info,
            crc_ok=best_crc,
            metric=best_metric,
            order_used=best_order,
        )
