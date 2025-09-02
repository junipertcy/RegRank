import numpy as np
import pytest

from regrank.core import SpringRankLegacy
from regrank.solvers import cp, legacy_cvx
from regrank.utils.datasets import random_graph, small_graph


def compute(g, alpha):
    # sg = SmallGraph()
    v_cvx = legacy_cvx(g, alpha=alpha)
    primal_s = cp.Variable((g.num_vertices(), 1))
    problem = cp.Problem(cp.Minimize(v_cvx.objective_fn_primal(primal_s)))  # for legacy
    problem.solve(verbose=False)

    v_cvx_output = primal_s.value.reshape(-1, 1)

    sr = SpringRankLegacy(alpha=alpha)

    result = sr.fit(g)
    bicgstab_output = result["rank"]
    return v_cvx_output, bicgstab_output


def test_small_graph():
    alpha = np.random.rand()
    with pytest.warns(
        DeprecationWarning,
        match=(
            "The 'SpringRankLegacy' class is deprecated and will be removed "
            "in a future version. Please use 'regrank.SpringRank' with "
            "regularizer=legacy instead."
        ),
    ):
        v_cvx_output, bicgstab_output = compute(small_graph(), alpha)

    print(v_cvx_output, bicgstab_output)
    assert np.isclose(v_cvx_output, bicgstab_output, atol=1e-3).all()


def test_random_graph_10_times():
    for _ in range(10):
        alpha = np.random.rand()
        with pytest.warns(
            DeprecationWarning,
            match=(
                "The 'SpringRankLegacy' class is deprecated and will be removed "
                "in a future version. Please use 'regrank.SpringRank' with "
                "regularizer=legacy instead."
            ),
        ):
            v_cvx_output, bicgstab_output = compute(random_graph(), alpha)
        assert np.isclose(v_cvx_output, bicgstab_output, atol=1e-3).all()
