from math import comb

import numpy as np

from rSpringRank.io.graph2mat import compute_ell
from rSpringRank.optimize.cvx import cp, same_mean_cvx
from rSpringRank.optimize.firstOrderMethods import gradientDescent
from rSpringRank.optimize.losses import sum_squared_loss, sum_squared_loss_conj
from rSpringRank.optimize.regularizers import same_mean_reg
from rSpringRank.stats.experiments import PhDExchange

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
    L = compute_ell(g, key="goi")
    sm_cvx = same_mean_cvx(g, L, goi="goi")

    num_classes = len(set(np.array(list(g.vp["goi"]))))
    num_pairs_classes = comb(num_classes, 2)

    ### Our method; DUAL ###
    sslc = sum_squared_loss_conj()
    sslc.setup(g, alpha=1, goi="goi")

    def f(x):
        return sslc.evaluate(x)

    def grad(x):
        return sslc.prox(x)

    def prox(x, t):
        return same_mean_reg(lambd=1).prox(x, t)

    def prox_fcn(x):
        return same_mean_reg(lambd=1).evaluate(x)

    x0 = np.random.rand(num_pairs_classes, 1).astype(np.float64)

    # errFcn = lambda x: norm(x - xTrue) / norm(xTrue)
    Lip_c = sslc.find_Lipschitz_constant()
    xNew, data = gradientDescent(
        f,
        grad,
        x0,
        prox=prox,
        prox_obj=prox_fcn,
        stepsize=Lip_c**-1,
        printEvery=5000,
        maxIters=1e5,
        tol=1e-14,  # orig 1e-14
        # errorFunction=errFcn,
        saveHistory=True,
        linesearch=False,
        acceleration=False,
        restart=50,
    )

    ### CVXPY; PRIMAL ###
    primal_s = cp.Variable((g.num_vertices(), 1))
    problem = cp.Problem(cp.Minimize(sm_cvx.objective_fn_primal(primal_s, lambd=1)))
    problem.solve(solver=cp.GUROBI, verbose=False)

    ### CVXPY; DUAL ###
    n = (pde.num_dual_vars, 1)
    tau = 1
    dual_v = cp.Variable(n)
    constraints = [cp.norm(dual_v, np.inf) <= tau]
    problem = cp.Problem(cp.Minimize(sm_cvx.objective_fn(dual_v)), constraints)
    problem.solve(solver=cp.GUROBI, verbose=False)

    ### CODE FOR BENCHMARKING ###
    ssl = sum_squared_loss()
    ssl.setup(g, alpha=1, goi="goi")

    tau = 1

    def f_all_primal(x):
        return ssl.evaluate(x) + tau * np.linalg.norm(ssl.ell @ x, 1)

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
    assert np.isclose(our_dual, cvx_dual, atol=1e-3)
    assert np.isclose(our_dual, cvx_prim, atol=1e-3)
    assert np.isclose(cvx_dual, cvx_prim, atol=1e-3)


if __name__ == "__main__":
    pass
