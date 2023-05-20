#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Regularized-SpringRank -- regularized methods for efficient ranking in networks
#
# Copyright (C) 2023 Tzu-Chi Yen <tzuchi.yen@colorado.edu>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from math import comb
from collections import defaultdict


from numpy.random import default_rng

from scipy.sparse.linalg import inv, LinearOperator, aslinearoperator, lsqr

from utils import *
from losses import *
from regularizers import *
from experiments import *
from firstOrderMethods import (
    createTestProblem,
    gradientDescent,
    lassoSolver,
    runAllTestProblems,
)

from cvx import *

# import gurobipy as gp
# HOW TO SUPPRESS GUROBI OUTPUT (Set parameter Username)?
# unknown.......
#
# with gp.Env(empty=True) as env:
#     env.setParam("OutputFlag", 0)
#     env.start()
#     m = gp.Model()
#     m.Params.LogToConsole = 0


pde = PhDExchange()
g = pde.get_data(annot="c18basic")
# g = pde.get_data(annot="sector")
# g = pde.get_data(annot="stabbr")

L = compute_ell(g)
sm_cvx = same_mean_cvx(g, L)

num_classes = len(set(np.array(list(g.vp["class"]))))
num_pairs_classes = comb(num_classes, 2)


### Our method; DUAL ###

sslc = sum_squared_loss_conj()
sslc.setup(g, alpha=1)
f = lambda x: sslc.evaluate(x)
grad = lambda x: sslc.prox(x)
prox = lambda x, t: same_mean_reg(tau=1).prox(x, t)
prox_fcn = lambda x: same_mean_reg(tau=1).evaluate(x)

x0 = np.random.rand(num_pairs_classes, 1)

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
problem.solve(
    solver=cp.GUROBI, verbose=False, reltol=1e-14, abstol=1e-14, max_iters=1e5
)

### CVXPY; DUAL ###
n = (pde.num_dual_vars, 1)
tau = 1
dual_v = cp.Variable(n)
constraints = [cp.norm(dual_v, np.inf) <= tau]
problem = cp.Problem(cp.Minimize(sm_cvx.objective_fn(dual_v)), constraints)
problem.solve(
    solver=cp.GUROBI, verbose=False, reltol=1e-14, abstol=1e-14, max_iters=1e5
)


### CODE FOR BENCHMARKING ###
ssl = sum_squared_loss()
ssl.setup(g, alpha=1)

tau = 1
f_all_primal = lambda x: ssl.evaluate(x) + tau * np.linalg.norm(ssl.ell @ x, 1)

our_dual = f_all_primal(pde.dual2primal(xNew))
cvx_dual = f_all_primal(pde.dual2primal(dual_v))
cvx_prim = f_all_primal(primal_s.value.reshape(1, -1).T)

print("Our dual: ", our_dual)
print("CVX dual: ", cvx_dual)
print("CVX primal: ", cvx_prim)


if __name__ == "__main__":
    pass
