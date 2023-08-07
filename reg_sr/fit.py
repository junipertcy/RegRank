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

from scipy.sparse.linalg import lsqr
import numpy as np

from reg_sr.cvx import *
from reg_sr.utils import *
from reg_sr.losses import *
from reg_sr.regularizers import *
from reg_sr.firstOrderMethods import gradientDescent


class rSpringRank(object):
    def __init__(
        self,
        method="vanilla",
    ):
        self.alpha = 0
        self.lambd = 0
        self.method = method
        self.result = dict()
        self.sslc = None
        self.fo_setup = dict()
        self.result["primal"] = None
        self.result["dual"] = None
        self.result["timewise"] = None
        pass

    # *args stand for other regularization parameters
    # **kwargs stand for other parameters (required by solver, for filtering data, etc)
    def fit(self, data, alpha=1, **kwargs):
        self.alpha = alpha
        self.lambd = kwargs.get("lambd", 1)
        self.cvxpy = kwargs.get("cvxpy", False)
        if self.method == "vanilla":
            if self.cvxpy:
                v_cvx = vanilla_cvx(data, alpha=self.alpha)
                primal_s = cp.Variable((data.num_vertices(), 1))
                problem = cp.Problem(
                    cp.Minimize(v_cvx.objective_fn_primal(primal_s))
                )  # for vanilla
                problem.solve(
                    verbose=False,
                )
                primal = primal_s.value.reshape(
                    -1,
                )
                self.result["primal"] = primal
                self.result["f_primal"] = problem.value
            else:
                B, b = cast2sum_squares_form(
                    data,
                    alpha=self.alpha
                )
                self.result["primal"] = lsqr(B, b.toarray())[:1][0].reshape(-1,)
                # compute primal functional value
                f_all_primal = lambda x: 0.5 * norm(B @ x - b) ** 2
                self.result["f_primal"] = f_all_primal(self.result["primal"].reshape(-1, 1))

        elif self.method == "annotated":
            # In this case, we use the dual-based proximal gradient descent algorithm
            # to solve the problem.
            if self.cvxpy:
                raise NotImplementedError("CVXPY not implemented for annotated.")
            else:
                self.sslc = sum_squared_loss_conj()
                self.sslc.setup(data, alpha=self.alpha)
                self.fo_setup["f"] = lambda x: self.sslc.evaluate(x)
                self.fo_setup["grad"] = lambda x: self.sslc.prox(x)
                self.fo_setup["prox"] = lambda x, t: same_mean_reg(lambd=self.lambd).prox(
                    x, t
                )
                self.fo_setup["prox_fcn"] = lambda x: same_mean_reg(
                    lambd=self.lambd
                ).evaluate(x)

                # first order kwargs
                self.fo_setup["printEvery"] = kwargs.get("printEvery", 5000)
                self.fo_setup["ArmijoLinesearch"] = kwargs.get("ArmijoLinesearch", True)
                self.fo_setup["linesearch"] = kwargs.get(
                    "linesearch", False
                )  # do not use True, still buggy
                self.fo_setup["acceleration"] = kwargs.get("acceleration", False)
                self.fo_setup["x0"] = kwargs.get("x0", np.random.rand(self.sslc.ell.shape[0], 1)).reshape(-1, 1)
                self.fo_setup["Lip_c"] = kwargs.get("Lip_c", self.sslc.find_Lipschitz_constant())
                self.fo_setup["maxIters"] = kwargs.get("maxIters", 1e6)
                self.fo_setup["tol"] = kwargs.get("tol", 1e-14)
                dual, _ = gradientDescent(
                    self.fo_setup["f"],
                    self.fo_setup["grad"],
                    self.fo_setup["x0"],
                    prox=self.fo_setup["prox"],
                    prox_obj=self.fo_setup["prox_fcn"],
                    stepsize=self.fo_setup["Lip_c"] ** -1,
                    printEvery=self.fo_setup["printEvery"],
                    maxIters=self.fo_setup["maxIters"],
                    tol=self.fo_setup["tol"],  # orig 1e-14
                    # errorFunction=errFcn,
                    saveHistory=True,
                    linesearch=self.fo_setup["linesearch"],
                    ArmijoLinesearch=self.fo_setup["ArmijoLinesearch"],
                    acceleration=self.fo_setup["acceleration"],
                    restart=50,
                )
                self.result["dual"] = np.array(dual).reshape(1, -1)[0]
                self.result["primal"] = self.sslc.dual2primal(dual).reshape(1, -1)[0]
                self.result["fo_output"] = _

                # compute primal functional value
                f_all_primal = lambda x: 0.5 * norm(self.sslc.B @ x - self.sslc.b) ** 2 + self.lambd * np.linalg.norm(self.sslc.ell @ x, 1)
                self.result["f_primal"] = f_all_primal(self.result["primal"].reshape(-1, 1))
                self.result["f_dual"] = self.sslc.evaluate(self.result["dual"].reshape(-1, 1))
        elif self.method == "time::l1":
            # In this case, we cast to sum-of-squares form
            # and use the dual-based proximal gradient descent algorithm
            # to solve the problem.
            if self.cvxpy:
                raise NotImplementedError("CVXPY not implemented for time::l1.")
            else:
                from_year = kwargs.get("from_year", 1960)
                to_year = kwargs.get("to_year", 2001)
                top_n = kwargs.get("top_n", 70)

                self.sslc = sum_squared_loss_conj()
                self.sslc.setup(
                    data,
                    alpha=self.alpha,
                    lambd=self.lambd,
                    from_year=from_year,
                    to_year=to_year,
                    top_n=top_n,
                    method="time::l1",
                )

                self.fo_setup["f"] = lambda x: self.sslc.evaluate(x)
                self.fo_setup["grad"] = lambda x: self.sslc.prox(x)
                # Do not change the lambd value here.
                self.fo_setup["prox"] = lambda x, t: same_mean_reg(lambd=1).prox(x, t)
                self.fo_setup["prox_fcn"] = lambda x: same_mean_reg(lambd=1).evaluate(x)
                # first order kwargs
                self.fo_setup["printEvery"] = kwargs.get("printEvery", 5000)
                self.fo_setup["ArmijoLinesearch"] = kwargs.get("ArmijoLinesearch", True)
                self.fo_setup["linesearch"] = kwargs.get(
                    "linesearch", False
                )  # do not use True, still buggy
                self.fo_setup["acceleration"] = kwargs.get("acceleration", False)
                self.fo_setup["x0"] = kwargs.get("x0", np.random.rand(self.sslc.ell.shape[0], 1)).reshape(-1, 1)
                self.fo_setup["Lip_c"] = kwargs.get("Lip_c", self.sslc.find_Lipschitz_constant())
                self.fo_setup["maxIters"] = kwargs.get("maxIters", 1e5)
                dual_time, _ = gradientDescent(
                    self.fo_setup["f"],
                    self.fo_setup["grad"],
                    self.fo_setup["x0"],
                    prox=self.fo_setup["prox"],
                    prox_obj=self.fo_setup["prox_fcn"],
                    stepsize=self.fo_setup["Lip_c"] ** -1,
                    printEvery=self.fo_setup["printEvery"],
                    maxIters=self.fo_setup["maxIters"],
                    tol=1e-14,  # orig 1e-14
                    # errorFunction=errFcn,
                    saveHistory=True,
                    linesearch=self.fo_setup["linesearch"],
                    ArmijoLinesearch=self.fo_setup["ArmijoLinesearch"],
                    acceleration=self.fo_setup["acceleration"],
                    restart=50,
                )
                primal_time = self.sslc.dual2primal(dual_time)
                self.result["timewise"] = primal_time.reshape(-1, top_n)
                self.result["fo_output"] = _

        elif self.method == "time::l2":
            # In this case, we cast to sum-of-squares form
            # and use LSQR to solve the problem.
            if self.cvxpy:
                raise NotImplementedError("CVXPY not implemented for time::l2.")
            else:
                from_year = kwargs.get("from_year", 1960)
                to_year = kwargs.get("to_year", 2001)
                top_n = kwargs.get("top_n", 70)

                B, b, _ = cast2sum_squares_form_t(
                    data,
                    alpha=self.alpha,
                    lambd=self.lambd,
                    from_year=from_year,
                    to_year=to_year,
                    top_n=top_n,
                )
                primal_time = lsqr(B, b.toarray())[:1][0]
                self.result["timewise"] = primal_time.reshape(-1, top_n)

        elif self.method == "huber":
            # In this case we use CVXPY to solve the problem.
            if self.cvxpy:
                self.M = kwargs.get("M", 1)
                self.incl_reg = kwargs.get("incl_reg", True)
                h_cvx = huber_cvx(data, alpha=self.alpha, M=self.M, incl_reg=self.incl_reg)
                primal_s = cp.Variable((data.num_vertices(), 1))
                problem = cp.Problem(cp.Minimize(h_cvx.objective_fn_primal(primal_s)))  # for huber
                try:
                    problem.solve(verbose=False)
                except cp.SolverError:
                    problem.solve(
                        solver=cp.GUROBI,
                        verbose=False,
                        reltol=1e-13,
                        abstol=1e-13,
                        max_iters=1e5,
                    )
                primal = primal_s.value.reshape(-1, )
                self.result["primal"] = primal
                self.result["f_primal"] = problem.value
            else:
                raise NotImplementedError("First-order solver for Huber norm has not been not implemented. " +
                                          "Please set explicitly that cvxpy=True.")
        else:
            raise NotImplementedError("Method not implemented.")

        return self.result
