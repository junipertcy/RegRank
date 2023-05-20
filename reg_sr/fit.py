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

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


from scipy.sparse.linalg import inv, LinearOperator, lsqr
import numpy as np
from scipy.sparse import csc_matrix

from reg_sr.cvx import *
from reg_sr.utils import *
from reg_sr.losses import *
from reg_sr.regularizers import *
from reg_sr.experiments import *
from reg_sr.firstOrderMethods import gradientDescent

import reg_sr


class rSpringRank(object):
    def __init__(
        self,
        method="vanilla",
    ):
        self.method = method
        self.fo_setup = dict()
        self.result = dict()
        self.result["primal"] = None
        self.result["dual"] = None
        self.result["timewise"] = None
        pass

    # *args stand for other regularization parameters
    # **kwargs stand for other parameters (required by solver, for filtering data, etc)
    def fit(self, data, alpha=1, **kwargs):
        if self.method == "vanilla":
            v_cvx = vanilla_cvx(data, alpha=alpha)
            primal_s = cp.Variable((data.num_vertices(), 1))
            problem = cp.Problem(
                cp.Minimize(v_cvx.objective_fn_primal(primal_s))
            )  # for vanilla
            problem.solve(
                solver=cp.GUROBI,
                verbose=False,
                reltol=1e-13,
                abstol=1e-13,
                max_iters=100000,
            )
            primal = primal_s.value.reshape(
                -1,
            )
            self.result["primal"] = primal

        elif self.method == "annotated":
            # In this case, we use the dual-based proximal gradient descent algorithm
            # to solve the problem.
            sslc = sum_squared_loss_conj()
            sslc.setup(data, alpha=alpha)
            self.fo_setup["f"] = lambda x: sslc.evaluate(x)
            self.fo_setup["grad"] = lambda x: sslc.prox(x)
            self.fo_setup["prox"] = lambda x, t: same_mean_reg(tau=1).prox(x, t)
            self.fo_setup["prox_fcn"] = lambda x: same_mean_reg(tau=1).evaluate(x)

            x0 = np.random.rand(sslc.ell.shape[0], 1)

            Lip_c = sslc.find_Lipschitz_constant()
            dual, _ = gradientDescent(
                self.fo_setup["f"],
                self.fo_setup["grad"],
                x0,
                prox=self.fo_setup["prox"],
                prox_obj=self.fo_setup["prox_fcn"],
                stepsize=Lip_c**-1,
                printEvery=5000,
                maxIters=1e5,
                tol=1e-16,  # orig 1e-14
                # errorFunction=errFcn,
                saveHistory=True,
                linesearch=False,
                acceleration=False,
                restart=50,
            )
            self.result["dual"] = np.array(dual).reshape(1, -1)[0]
            self.result["primal"] = sslc.dual2primal(dual).reshape(1, -1)[0]
        elif self.method == "time::l1":
            # In this case, we cast to sum-of-squares form
            # and use the dual-based proximal gradient descent algorithm
            # to solve the problem.
            from_year = kwargs.get("from_year", 1960)
            to_year = kwargs.get("to_year", 2001)
            top_n = kwargs.get("top_n", 70)
            lambd = kwargs.get("lambd", 1)

            sslc = sum_squared_loss_conj()
            sslc.setup(
                data,
                alpha=alpha,
                lambd=lambd,
                from_year=from_year,
                to_year=to_year,
                top_n=top_n,
                method="time::l1",
            )

            self.fo_setup["f"] = lambda x: sslc.evaluate(x)
            self.fo_setup["grad"] = lambda x: sslc.prox(x)
            self.fo_setup["prox"] = lambda x, t: same_mean_reg(tau=1).prox(x, t)
            self.fo_setup["prox_fcn"] = lambda x: same_mean_reg(tau=1).evaluate(x)

            x0 = np.random.rand(sslc.ell.shape[0], 1)

            Lip_c = sslc.find_Lipschitz_constant()
            dual_time, _ = gradientDescent(
                self.fo_setup["f"],
                self.fo_setup["grad"],
                x0,
                prox=self.fo_setup["prox"],
                prox_obj=self.fo_setup["prox_fcn"],
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
            primal_time = sslc.dual2primal(dual_time)
            self.result["timewise"] = primal_time.reshape(-1, top_n)
        elif self.method == "time::l2":
            # In this case, we cast to sum-of-squares form
            # and use LSQR to solve the problem.
            from_year = kwargs.get("from_year", 1960)
            to_year = kwargs.get("to_year", 2001)
            top_n = kwargs.get("top_n", 70)
            lambd = kwargs.get("lambd", 1)
            B, b, _ = cast2sum_squares_form_t(
                data,
                alpha=alpha,
                lambd=lambd,
                from_year=from_year,
                to_year=to_year,
                top_n=top_n,
            )
            primal_time = lsqr(B, b.toarray())[:1][0]
            self.result["timewise"] = primal_time.reshape(-1, top_n)

        elif self.method == "huber":
            # In this case we use CVXPY to solve the problem.
            pass
        else:
            raise NotImplementedError("Method not implemented.")

        return self.result
