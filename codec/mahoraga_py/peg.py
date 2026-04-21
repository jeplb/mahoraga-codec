"""progressive edge growth (PEG) LDPC construction.

deterministic for fixed (n, dv, dc, seed): the returned check-to-var adjacency is fully determined by the seed.

subtle determinism points:
  - the PRNG state is initialized as `(seed + 1) | 1` and the xorshift64
    closure is never actually consumed by the BFS (the `_rng` arg is
    it, but the initialization is kept for potential future tie-breaking.
  - BFS safety cap at dist > 100.
  - candidate tie-break: sort by (check_fill, index) ascending.
"""

from __future__ import annotations

from typing import List, Tuple

_U64_MASK: int = (1 << 64) - 1


def peg_ldpc(n: int, dv: int, dc: int, seed: int) -> List[List[int]]:
    """build a PEG LDPC code. returns check-to-var adjacency (m lists of var ids)."""
    m = n * dv // dc
    assert n * dv == m * dc, "n*dv must equal m*dc"

    var_to_check: List[List[int]] = [[] for _ in range(n)]
    check_to_var: List[List[int]] = [[] for _ in range(m)]
    check_fill: List[int] = [0] * m

    # rng_state = (seed + 1) | 1; the BFS function does not consume
    # but never calls it. We mirror the initialization for determinism.
    _rng_state = ((seed + 1) & _U64_MASK) | 1
    _ = _rng_state  # unused, kept for future tie-breaking

    for v in range(n):
        for _edge in range(dv):
            excluded = set(var_to_check[v])
            best_check = _bfs_furthest_check(
                v, var_to_check, check_to_var, check_fill, dc, m, excluded,
            )
            var_to_check[v].append(best_check)
            check_to_var[best_check].append(v)
            check_fill[best_check] += 1

    return check_to_var


def _bfs_furthest_check(
    v: int,
    var_to_check: List[List[int]],
    check_to_var: List[List[int]],
    check_fill: List[int],
    check_capacity_uniform: int,
    m: int,
    excluded: set,
) -> int:
    n_vars = len(var_to_check)
    var_visited = [False] * n_vars
    check_dist = [-1] * m

    var_visited[v] = True
    current_vars: List[int] = [v]
    dist = 0

    best_candidates: List[int] = []
    best_dist = -1

    while current_vars:
        # vars → checks
        next_checks: List[int] = []
        for vv in current_vars:
            for c in var_to_check[vv]:
                if check_dist[c] < 0:
                    check_dist[c] = dist
                    next_checks.append(c)

        # newly reached, available
        for c in next_checks:
            if c not in excluded and check_fill[c] < check_capacity_uniform:
                if dist > best_dist:
                    best_dist = dist
                    best_candidates = []
                if dist == best_dist:
                    best_candidates.append(c)

        # checks → vars
        next_vars: List[int] = []
        for c in next_checks:
            for vv in check_to_var[c]:
                if not var_visited[vv]:
                    var_visited[vv] = True
                    next_vars.append(vv)

        current_vars = next_vars
        dist += 1
        if dist > 100:
            break

    if not best_candidates:
        # pick from unreached, non-excluded checks with room
        for c in range(m):
            if check_dist[c] < 0 and c not in excluded and check_fill[c] < check_capacity_uniform:
                best_candidates.append(c)

    if not best_candidates:
        # fallback: any non-excluded check with minimum fill
        min_fill = None
        for c in range(m):
            if c not in excluded:
                if min_fill is None or check_fill[c] < min_fill:
                    min_fill = check_fill[c]
        if min_fill is None:
            min_fill = 0
        for c in range(m):
            if c not in excluded and check_fill[c] == min_fill:
                best_candidates.append(c)

    best_candidates.sort(key=lambda c: (check_fill[c], c))
    return best_candidates[0]


def adj_to_csr(check_to_var: List[List[int]], _n: int) -> Tuple[List[int], List[int], int]:
    """convert check-to-var adjacency to scipy-style CSR: (indptr, indices, m)."""
    m = len(check_to_var)
    indptr: List[int] = [0] * (m + 1)
    indices: List[int] = []
    for c, vars_ in enumerate(check_to_var):
        indices.extend(vars_)
        indptr[c + 1] = len(indices)
    return indptr, indices, m
