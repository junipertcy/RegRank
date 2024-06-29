import numpy as np

from rSpringRank.optimize.cvx import cp, vanilla_cvx
from rSpringRank.optimize.models import SpringRank
from rSpringRank.stats.experiments import RandomGraph, SmallGraph


def compute(obj, alpha):
    # sg = SmallGraph()
    g = obj.get_data()
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
    v_cvx_output, bicgstab_output = compute(SmallGraph(), alpha)
    print(v_cvx_output, bicgstab_output)
    assert np.isclose(v_cvx_output, bicgstab_output, atol=1e-3).all()


def test_random_graph_10_times():
    for _ in range(10):
        alpha = np.random.rand()
        v_cvx_output, bicgstab_output = compute(RandomGraph(), alpha)
        assert np.isclose(v_cvx_output, bicgstab_output, atol=1e-3).all()
