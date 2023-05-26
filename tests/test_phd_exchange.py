# from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from math import comb
from collections import defaultdict

from numpy.random import default_rng

from scipy.sparse.linalg import inv, LinearOperator, aslinearoperator, lsqr

from reg_sr.utils import *
from reg_sr.losses import *
from reg_sr.regularizers import *
from reg_sr.experiments import *
from reg_sr.firstOrderMethods import (
    createTestProblem,
    gradientDescent,
    lassoSolver,
    runAllTestProblems,
)

from reg_sr.cvx import *

# import gurobipy as gp
# HOW TO SUPPRESS GUROBI OUTPUT (Set parameter Username)?
# unknown.......
#
# with gp.Env(empty=True) as env:
#     env.setParam("OutputFlag", 0)
#     env.start()
#     m = gp.Model()
#     m.Params.LogToConsole = 0

# g = pde.get_data(goi="sector")
# g = pde.get_data(goi="stabbr")

def compute(goi):
    pde = PhDExchange()
    g = pde.get_data(goi=goi)
    L = compute_ell(g)
    sm_cvx = same_mean_cvx(g, L)

    num_classes = len(set(np.array(list(g.vp["goi"]))))
    num_pairs_classes = comb(num_classes, 2)


    ### Our method; DUAL ###

    sslc = sum_squared_loss_conj()
    sslc.setup(g, alpha=1)
    f = lambda x: sslc.evaluate(x)
    grad = lambda x: sslc.prox(x)
    prox = lambda x, t: same_mean_reg(lambd=1).prox(x, t)
    prox_fcn = lambda x: same_mean_reg(lambd=1).evaluate(x)

    x0 = np.random.rand(num_pairs_classes, 1)

    # errFcn = lambda x: norm(x - xTrue) / norm(xTrue)
    Lip_c = sslc.find_Lipschitz_constant()
    xNew, data = gradientDescent(
        f,
        grad,
        x0,
        prox=prox,
        prox_obj=prox_fcn,
        stepsize=Lip_c ** -1,
        printEvery=5000,
        maxIters=1e5,
        tol=1e-14,  # orig 1e-14
        # errorFunction=errFcn,
        saveHistory=True,
        linesearch=False,
        acceleration=False,
        restart=50
    )

    ### CVXPY; PRIMAL ###
    primal_s = cp.Variable((g.num_vertices(), 1))
    problem = cp.Problem(cp.Minimize(sm_cvx.objective_fn_primal(primal_s, lambd=1)))
    problem.solve(solver=cp.GUROBI, verbose=False, reltol=1e-14, abstol=1e-14, max_iters=1e5)

    ### CVXPY; DUAL ###
    n = (pde.num_dual_vars, 1)
    tau = 1
    dual_v = cp.Variable(n)
    constraints = [ cp.norm( dual_v, np.inf ) <= tau ]
    problem = cp.Problem(cp.Minimize(sm_cvx.objective_fn(dual_v)), constraints )
    problem.solve(solver=cp.GUROBI, verbose=False, reltol=1e-14, abstol=1e-14, max_iters=1e5)


    ### CODE FOR BENCHMARKING ###
    ssl = sum_squared_loss()
    ssl.setup(g, alpha=1)

    tau = 1
    f_all_primal = lambda x : ssl.evaluate(x) + tau * np.linalg.norm(ssl.ell @ x, 1)
    
    xNew = np.array(xNew).reshape(-1, 1)
    dual_v = np.array(dual_v.value).reshape(-1, 1)
    
    our_dual = f_all_primal(sslc.dual2primal(xNew))
    cvx_dual = f_all_primal(sslc.dual2primal(dual_v))
    cvx_prim = f_all_primal(primal_s.value.reshape(1, -1).T)
    return our_dual, cvx_dual, cvx_prim

def test_c18basic():
    our_dual, cvx_dual, cvx_prim = compute("c18basic")
    print("### begin:: c18basic ###")
    print("Our dual: ", our_dual)
    print("CVX dual: ", cvx_dual)
    print("CVX primal: ", cvx_prim)
    print("### end:: c18basic ###")
    assert np.isclose(our_dual, cvx_dual, atol=1e-3)
    assert np.isclose(our_dual, cvx_prim, atol=1e-3)
    assert np.isclose(cvx_dual, cvx_prim, atol=1e-3)

def test_sector():
    our_dual, cvx_dual, cvx_prim = compute("sector")
    print("### begin:: sector ###")
    print("Our dual: ", our_dual)
    print("CVX dual: ", cvx_dual)
    print("CVX primal: ", cvx_prim)
    print("### end:: sector ###")
    assert np.isclose(our_dual, cvx_dual, atol=1e-3)
    assert np.isclose(our_dual, cvx_prim, atol=1e-3)
    assert np.isclose(cvx_dual, cvx_prim, atol=1e-3)

def test_stabbr():
    our_dual, cvx_dual, cvx_prim = compute("stabbr")
    print("### begin:: stabbr ###")
    print("Our dual: ", our_dual)
    print("CVX dual: ", cvx_dual)
    print("CVX primal: ", cvx_prim)
    print("### end:: stabbr ###")
    assert np.isclose(our_dual, cvx_dual, atol=1e-1)
    assert np.isclose(our_dual, cvx_prim, atol=1e-1)
    assert np.isclose(cvx_dual, cvx_prim, atol=1e-1)




if __name__ == "__main__":
    pass
