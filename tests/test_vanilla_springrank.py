# from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from math import comb
from collections import defaultdict

from numpy.random import default_rng

from scipy.sparse.linalg import inv, LinearOperator, aslinearoperator, lsqr

from reg_spr.utils import *
from reg_spr.losses import *
from reg_spr.regularizers import *
from reg_spr.experiments import *
from reg_spr.firstOrderMethods import (
    createTestProblem,
    gradientDescent,
    lassoSolver,
    runAllTestProblems,
)

from reg_spr.cvx import *
import reg_spr


def compute(obj, alpha):
    # sg = SmallGraph()
    g = obj.get_data()
    v_cvx = vanilla_cvx(g, alpha=alpha)
    primal_s = cp.Variable((g.num_vertices(), 1))
    problem = cp.Problem(cp.Minimize(v_cvx.objective_fn_primal(primal_s)))  # for vanilla
    problem.solve(verbose=False)
    
    v_cvx_output = primal_s.value.reshape(-1,)

    sr = reg_spr.SpringRank(alpha=alpha)

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
