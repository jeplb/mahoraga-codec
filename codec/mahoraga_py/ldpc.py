"""LDPC belief propagation decoder + GF(2) elimination utilities.

includes the subset of LDPC utilities needed by the mahoraga pipeline:

  - `LdpcCode` with edge-index adjacency tables (from_csr / from_adj)
  - `bp_decode_parallel`: flooding BP, sum-product via phi(x) = -ln(tanh(|x|/2))
  - `Gf2Matrix` (bit-packed) with row_reduce and row_nonzeros
  - `ldpc_encode`: systematic from info_bits + parity_columns

windowed BP and SC-LDPC construction are intentionally omitted — the
mahoraga pipeline uses flat BP+OSD, and paper benchmarks never exercised
the SC path.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

_LLR_CLAMP: float = 30.0


def _phi(x: float) -> float:
    ax = abs(x)
    if ax < 1e-10:
        return 30.0
    if ax > 19.0:
        # exp(-|x|) for large |x|
        return math.exp(-ax)
    return -math.log(math.tanh(ax * 0.5))


def _clamp(x: float, lo: float = -_LLR_CLAMP, hi: float = _LLR_CLAMP) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


class LdpcCode:
    """edge-indexed parity check matrix with precomputed adjacency tables."""

    __slots__ = ("n", "m", "check_edges", "var_edges", "n_edges")

    def __init__(
        self,
        n: int,
        m: int,
        check_edges: List[List[Tuple[int, int]]],
        var_edges: List[List[Tuple[int, int]]],
        n_edges: int,
    ) -> None:
        self.n = n
        self.m = m
        self.check_edges = check_edges
        self.var_edges = var_edges
        self.n_edges = n_edges

    @classmethod
    def from_csr(
        cls,
        n: int,
        m: int,
        indptr: Sequence[int],
        indices: Sequence[int],
    ) -> "LdpcCode":
        check_edges: List[List[Tuple[int, int]]] = [[] for _ in range(m)]
        var_edges: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
        edge_id = 0
        for c in range(m):
            start = indptr[c]
            end = indptr[c + 1]
            for k in range(start, end):
                v = indices[k]
                check_edges[c].append((v, edge_id))
                var_edges[v].append((c, edge_id))
                edge_id += 1
        return cls(n, m, check_edges, var_edges, edge_id)

    @classmethod
    def from_adj(
        cls,
        n: int,
        m: int,
        check_to_var: Sequence[Sequence[int]],
    ) -> "LdpcCode":
        check_edges: List[List[Tuple[int, int]]] = [[] for _ in range(m)]
        var_edges: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
        edge_id = 0
        for c, vars_ in enumerate(check_to_var):
            for v in vars_:
                check_edges[c].append((v, edge_id))
                var_edges[v].append((c, edge_id))
                edge_id += 1
        return cls(n, m, check_edges, var_edges, edge_id)


def bp_decode_parallel(
    channel_llrs: Sequence[float],
    code: LdpcCode,
    max_iters: int = 100,
) -> Tuple[List[int], bool]:
    """flooding BP decoder. returns (hard_bits, converged)."""
    n = code.n
    ne = code.n_edges

    v2c: List[float] = [0.0] * ne
    c2v: List[float] = [0.0] * ne

    # init v2c from channel LLRs
    for v in range(n):
        for (_c, eid) in code.var_edges[v]:
            v2c[eid] = channel_llrs[v]

    hard_bits: List[int] = [0] * n

    for _it in range(max_iters):
        # check-to-variable update
        for edges in code.check_edges:
            dc = len(edges)
            if dc == 0:
                continue
            sign_product = 1
            phi_sum = 0.0
            signs: List[int] = [0] * dc
            phi_vals: List[float] = [0.0] * dc
            for j, (_v, eid) in enumerate(edges):
                msg = v2c[eid]
                s = 1 if msg >= 0.0 else -1
                signs[j] = s
                sign_product *= s
                pv = _phi(msg)
                phi_vals[j] = pv
                phi_sum += pv

            for j, (_v, eid) in enumerate(edges):
                es = sign_product * signs[j]
                ep = phi_sum - phi_vals[j]
                mag = _phi(ep)
                c2v[eid] = _clamp(es * mag)

        # variable-to-check update + hard decision
        for v in range(n):
            edges = code.var_edges[v]
            total = channel_llrs[v]
            for (_c, eid) in edges:
                total += c2v[eid]
            hard_bits[v] = 0 if total >= 0.0 else 1
            for (_c, eid) in edges:
                v2c[eid] = _clamp(total - c2v[eid])

        # syndrome check
        all_ok = True
        for edges in code.check_edges:
            parity = 0
            for (v, _eid) in edges:
                parity ^= hard_bits[v]
            if parity != 0:
                all_ok = False
                break
        if all_ok:
            return hard_bits, True

    return hard_bits, False


def ldpc_encode(
    info_bits: Sequence[int],
    parity_columns: Sequence[Sequence[int]],
    n: int,
) -> List[int]:
    """legacy systematic encoder: codeword = [info | parity].

    parity_columns[p] lists info indices contributing to parity bit p.
    """
    k = len(info_bits)
    n_parity = n - k
    assert len(parity_columns) == n_parity
    codeword: List[int] = list(info_bits)
    for col in parity_columns:
        bit = 0
        for idx in col:
            bit ^= info_bits[idx]
        codeword.append(bit)
    return codeword


# ---------------------------------------------------------------------------
# GF(2) elimination (bit-packed rows)
# ---------------------------------------------------------------------------


class Gf2Matrix:
    """bit-packed GF(2) matrix with row_reduce and row_nonzeros.

    stores each row as a python int (arbitrary-precision bitmask). row_reduce
    produces a row-echelon form and returns (pivot_cols, free_cols):
      - pivot_cols[r] = column where row r pivoted (in row order, not sorted)
      - free_cols     = complementary columns, sorted ascending

    pivot selection order: for each pivot_row (increasing),
    find the **highest-index** unclaimed column set in the scanning row,
    then swap that row into pivot_row position.
    """

    __slots__ = ("rows", "cols", "data")

    def __init__(self, rows: int, cols: int, data: Optional[List[int]] = None) -> None:
        self.rows = rows
        self.cols = cols
        self.data = data if data is not None else [0] * rows

    @classmethod
    def from_csr(
        cls,
        rows: int,
        cols: int,
        indptr: Sequence[int],
        indices: Sequence[int],
    ) -> "Gf2Matrix":
        mat = cls(rows, cols)
        for r in range(rows):
            acc = 0
            for k in range(indptr[r], indptr[r + 1]):
                acc |= 1 << indices[k]
            mat.data[r] = acc
        return mat

    def get(self, r: int, c: int) -> bool:
        return (self.data[r] >> c) & 1 == 1

    def set(self, r: int, c: int) -> None:
        self.data[r] |= 1 << c

    def row_reduce(self) -> Tuple[List[int], List[int]]:
        pivot_cols: List[int] = []
        col_used = [False] * self.cols
        pivot_row = 0

        for row_scan in range(self.rows):
            if pivot_row >= self.rows:
                break

            row_val = self.data[row_scan]
            if row_val == 0:
                continue

            # find highest unclaimed set bit (scan words in reverse
            # and within each word scans bits 63..0 — which is exactly
            # "highest unclaimed bit" via a bitmask strip of already-used
            # columns.
            unused_mask = row_val
            # clear bits at columns already claimed as pivots
            for pc in pivot_cols:
                if (unused_mask >> pc) & 1:
                    unused_mask &= ~(1 << pc)
            if unused_mask == 0:
                continue
            # highest set bit
            fc = unused_mask.bit_length() - 1

            if row_scan != pivot_row:
                self.data[row_scan], self.data[pivot_row] = (
                    self.data[pivot_row],
                    self.data[row_scan],
                )

            col_used[fc] = True
            pivot_cols.append(fc)

            pivot_data = self.data[pivot_row]
            pivot_mask = 1 << fc
            for r in range(self.rows):
                if r != pivot_row and (self.data[r] & pivot_mask):
                    self.data[r] ^= pivot_data

            pivot_row += 1

        pivot_set = set(pivot_cols)
        free_cols = [c for c in range(self.cols) if c not in pivot_set]
        return pivot_cols, free_cols

    def row_nonzeros(self, r: int) -> List[int]:
        out: List[int] = []
        val = self.data[r]
        while val:
            lsb = val & -val
            col = lsb.bit_length() - 1
            if col < self.cols:
                out.append(col)
            val ^= lsb
        return out


def gf2_extract_encoding(
    h_indptr: Sequence[int],
    h_indices: Sequence[int],
    m: int,
    n: int,
) -> Tuple[List[int], List[int], List[List[int]]]:
    """row-reduce H, extract encoding dependencies per pivot row.

    for each pivot row r, deps = (nonzeros of row r in reduced form) minus the
    pivot column. this is the set of cols that XOR with the pivot to close
    the check — used by `InnerCode.encode` and the OSD decoder.

    returns (pivot_cols, free_cols, enc_deps).
    """
    mat = Gf2Matrix.from_csr(m, n, h_indptr, h_indices)
    pivot_cols, free_cols = mat.row_reduce()
    enc_deps: List[List[int]] = []
    for r in range(len(pivot_cols)):
        pc = pivot_cols[r]
        nz = mat.row_nonzeros(r)
        deps = [c for c in nz if c != pc]
        enc_deps.append(deps)
    return pivot_cols, free_cols, enc_deps
