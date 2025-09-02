from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np
from sage.combinat.posets.posets import Poset

try:
    from pulp import (
        PULP_CBC_CMD,
        LpBinary,
        LpMinimize,
        LpProblem,
        LpVariable,
        lpSum,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "PuLP is required for ILP functionality. Install via  pip install pulp"
    ) from exc


Edge = tuple[int, int]
EdgeList = list[Edge]


# ----------------------------------------------------------------------
# 1.   Potential poset from weighted adjacency matrix
# ----------------------------------------------------------------------
def potential_poset_from_adjacency(
    A: np.ndarray | Sequence[Sequence[float]],
) -> EdgeList:
    """
    Derive directed cover relations from the *skew-symmetric* component
    of a weighted adjacency matrix.

    For every unordered pair (i, j):

        •  If  R_ij > 0  ⇒  i ≺ j
        •  If  R_ij < 0  ⇒  j ≺ i

    Parameters
    ----------
    A : (n, n) array-like (will be coerced to float)
        Weighted adjacency matrix.

    Returns
    -------
    covers : list[(i, j)]
        Minimal set of covers implied by the sign pattern.
    """
    A = np.asarray(A, dtype=float).copy()
    np.fill_diagonal(A, 0.0)  # force A_ii = 0

    R = (A - A.T) / 2  # skew-symmetric part
    n = A.shape[0]

    covers: EdgeList = []
    for i in range(n):
        for j in range(i + 1, n):
            val = R[i, j]
            if val > 0:
                covers.append((i, j))
            elif val < 0:
                covers.append((j, i))
    return covers


# ----------------------------------------------------------------------
# 2.   Cycle detection (DFS with recursion stack)
# ----------------------------------------------------------------------
def has_cycle(edges: Iterable[Edge], n: int) -> bool:
    """
    Depth-first search for a directed cycle.

    Parameters
    ----------
    edges : iterable[(u, v)]
        Directed edges.
    n     : int
        Number of vertices, assumed labelled 0 … n-1.

    Returns
    -------
    bool   – True iff a cycle exists.
    """
    G: dict[int, list[int]] = defaultdict(list)
    for u, v in edges:
        G[u].append(v)

    visited = [False] * n
    in_stack = [False] * n

    def dfs(v: int) -> bool:
        visited[v] = True
        in_stack[v] = True
        for w in G[v]:
            if not visited[w]:
                if dfs(w):
                    return True
            elif in_stack[w]:  # back-edge ⇒ cycle
                return True
        in_stack[v] = False
        return False

    return any(dfs(v) for v in range(n) if not visited[v])


# ----------------------------------------------------------------------
# 3.   Exact minimum-edge removal (ILP)
# ----------------------------------------------------------------------
def break_cycles_exact(
    covers: EdgeList,
    n: int,
    solver=None,
) -> tuple[EdgeList, EdgeList]:
    """
    Remove the *smallest* number of edges so the graph is acyclic.

    Uses a standard MILP formulation of the minimum feedback-arc set.

    Returns
    -------
    kept_edges, dropped_edges : (list, list)
        Partition of `covers` such that `kept_edges` is acyclic and
        `dropped_edges` is the proven minimum set removed.
    """
    prob = LpProblem("min_feedback_arc_set", LpMinimize)

    e = {(u, v): LpVariable(f"keep_{u}_{v}", cat=LpBinary) for u, v in covers}

    # topological order variables   0 ≤ ord_i ≤ n−1
    order = {i: LpVariable(f"ord_{i}", lowBound=0, upBound=n - 1) for i in range(n)}

    M = n
    for (u, v), keep in e.items():
        # if keep = 1  ⇒  order[u] + 1 ≤ order[v]
        prob += order[u] - order[v] + M * keep <= M - 1

    # minimise edges removed = Σ(1 - keep)
    prob += lpSum(1 - keep for keep in e.values())

    prob.solve(solver or PULP_CBC_CMD(msg=False))

    kept = [edge for edge, var in e.items() if var.value() > 0.5]
    dropped = [edge for edge, var in e.items() if var.value() < 0.5]
    return kept, dropped


# ----------------------------------------------------------------------
# 4.   Linear-extension check via SageMath
# ----------------------------------------------------------------------
def is_linear_extension(covers: EdgeList, ordering: Sequence[int]) -> bool:
    """
    True iff `ordering` is a linear extension of the poset defined by `covers`.
    """

    poset = Poset((range(len(ordering)), covers), cover_relations=True)
    return cast(bool, poset.is_linear_extension(ordering))


# ----------------------------------------------------------------------
# 5.   Quick demo when run as a script
# ----------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    A = np.array(
        [[0, 3, 0], [1, 0, 2], [0, 4, 0]],
        dtype=float,
    )

    covers = potential_poset_from_adjacency(A)
    print("Covers inferred:", covers)

    print("Has cycle?      ", has_cycle(covers, n=A.shape[0]))

    kept, dropped = break_cycles_exact(covers, n=A.shape[0])
    print("Kept edges:     ", kept)
    print("Dropped edges:  ", dropped)
    print("Acyclic now?    ", not has_cycle(kept, n=A.shape[0]))

    total_order = [0, 1, 2]
    print("Linear extension?", is_linear_extension(kept, total_order))
