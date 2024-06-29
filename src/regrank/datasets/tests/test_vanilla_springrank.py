import numpy as np

import regrank as rr
from regrank.optimize.cvx import cp, vanilla_cvx
from regrank.optimize.models import SpringRank


def compute(g, alpha):
    # sg = SmallGraph()
    v_cvx = vanilla_cvx(g, alpha=alpha)
    primal_s = cp.Variable((g.num_vertices(), 1))
    problem = cp.Problem(
        cp.Minimize(v_cvx.objective_fn_primal(primal_s))
    )  # for vanilla
    problem.solve(verbose=False)

    v_cvx_output = primal_s.value.reshape(-1, 1)

    sr = SpringRank(alpha=alpha)

    result = sr.fit(g)
    bicgstab_output = result["rank"]
    return v_cvx_output, bicgstab_output


def test_small_graph():
    alpha = np.random.rand()
    v_cvx_output, bicgstab_output = compute(rr.small_graph(), alpha)
    print(v_cvx_output, bicgstab_output)
    assert np.isclose(v_cvx_output, bicgstab_output, atol=1e-3).all()


def test_random_graph_10_times():
    for _ in range(10):
        alpha = np.random.rand()
        v_cvx_output, bicgstab_output = compute(rr.random_graph(), alpha)
        assert np.isclose(v_cvx_output, bicgstab_output, atol=1e-3).all()
