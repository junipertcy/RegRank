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

try:
    import graph_tool.all as gt
except ModuleNotFoundError:
    print("graph_tool not found. Please install graph_tool.")

import warnings

import cvxpy as cp
import numpy as np
import scipy.sparse.linalg
from numpy.linalg import norm
from scipy.optimize import brentq
from scipy.sparse import SparseEfficiencyWarning, csr_matrix, spdiags
from scipy.sparse.linalg import lsmr, lsqr

from ..io import cast2sum_squares_form, cast2sum_squares_form_t
from .cvx import huber_cvx, vanilla_cvx
from .firstOrderMethods import gradientDescent
from .losses import sum_squared_loss_conj
from .regularizers import same_mean_reg, zero_reg

warnings.simplefilter("ignore", SparseEfficiencyWarning)


class BaseModel:
    def __init__(self, loss, reg=zero_reg()):
        self.loss = loss
        self.local_reg = reg


class SpringRank:
    def __init__(self, alpha=0):
        self.alpha = alpha
        # pass
        # self.change_base_model(BaseModel)

    def fit_scaled(self, data, scale=0.75):
        if type(data) == gt.Graph:
            adj = gt.adjacency(data)
        else:
            raise NotImplementedError
        # from Hunter's code
        ranks = self.get_ranks(adj)
        inverse_temperature = self.get_inverse_temperature(adj, ranks)
        scaling_factor = 1 / (np.log(scale / (1 - scale)) / (2 * inverse_temperature))
        scaled_ranks = self.scale_ranks(ranks, scaling_factor)

        info = {"rank": scaled_ranks}
        return info

    def fit(self, data):
        if type(data) == gt.Graph:
            adj = gt.adjacency(data)
        else:
            raise NotImplementedError
        # print(f"bicgstab: adj = {adj.toarray()[:5,:5]}")
        ranks = self.get_ranks(adj)

        info = {"rank": ranks.reshape(-1, 1)}
        return info

    # below came from Hunter's code
    def get_ranks(self, A):
        """
        params:
        - A: a (square) np.ndarray

        returns:
        - ranks, np.array

        TODO:
        - support passing in other formats (eg a sparse matrix)
        """
        return self.compute_sr(A, self.alpha)

    def get_inverse_temperature(self, A, ranks):
        """given an adjacency matrix and the ranks for that matrix, calculates the
        temperature of those ranks"""
        betahat = brentq(self.eqs39, 0.01, 20, args=(ranks, A))
        return betahat

    @staticmethod
    def scale_ranks(ranks, scaling_factor):
        return ranks * scaling_factor

    @staticmethod
    def csr_SpringRank(A):
        """
        Main routine to calculate SpringRank by solving linear system
        Default parameters are initialized as in the standard SpringRank model

        Arguments:
            A: Directed network (np.ndarray, scipy.sparse.csr.csr_matrix)

        Output:
            rank: N-dim array, indeces represent the nodes' indices used in ordering the matrix A
        """

        N = A.shape[0]
        k_in = np.array(A.sum(axis=0))
        k_out = np.array(A.sum(axis=1).transpose())

        # form the graph laplacian
        operator = csr_matrix(spdiags(k_out + k_in, 0, N, N) - A - A.transpose())

        # form the operator A (from Ax=b notation)
        # note that this is the operator in the paper, but augmented
        # to solve a Lagrange multiplier problem that provides the constraint
        operator.resize(N + 1, N + 1)
        operator[N, 0] = 1
        operator[0, N] = 1

        # form the solution vector b (from Ax=b notation)
        solution_vector = np.append((k_out - k_in), np.array([0])).transpose()

        # perform the computations
        ranks = scipy.sparse.linalg.bicgstab(
            scipy.sparse.csr_matrix(operator), solution_vector, atol=1e-8
        )[0]

        return ranks[:-1]

    def compute_sr(self, A, alpha=0):
        """
        Solve the SpringRank system.
        If alpha = 0, solves a Lagrange multiplier problem.
        Otherwise, performs L2 regularization to make full rank.

        Arguments:
            A: Directed network (np.ndarray, scipy.sparse.csr.csr_matrix)
            alpha: regularization term. Defaults to 0.

        Output:
            ranks: Solution to SpringRank
        """

        if alpha == 0:
            rank = self.csr_SpringRank(A)
        else:
            if type(A) == np.ndarray:
                A = scipy.sparse.csr_matrix(A)
            # print("Running bicgstab to solve Ax=b ...")
            # print("adj matrix A:\n", A.toarray())
            N = A.shape[0]
            k_in = scipy.sparse.csr_matrix.sum(A, 0)
            k_out = scipy.sparse.csr_matrix.sum(A, 1).T

            k_in = scipy.sparse.diags(np.array(k_in)[0])
            k_out = scipy.sparse.diags(np.array(k_out)[0])

            C = A + A.T
            D1 = k_in + k_out

            B = k_out - k_in
            B = B @ np.ones([N, 1])

            A = alpha * scipy.sparse.eye(N) + D1 - C

            rank = scipy.sparse.linalg.bicgstab(A, B, atol=1e-8)[0]

        return np.transpose(rank)

    # @jit(nopython=True)
    def eqs39(self, beta, s, A):
        N = A.shape[0]
        x = 0
        for i in range(N):
            for j in range(N):
                if A[i, j] == 0:
                    continue
                else:
                    x += (s[i] - s[j]) * (
                        A[i, j]
                        - (A[i, j] + A[j, i]) / (1 + np.exp(-2 * beta * (s[i] - s[j])))
                    )
        return x


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
        self.bicgstab = kwargs.get("bicgstab", True)
        if np.sum([self.cvxpy, self.bicgstab]) > 1:
            raise ValueError("Only one of cvxpy and bicgstab can be True.")

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
            elif self.bicgstab:
                self.result["primal"] = SpringRank(alpha=self.alpha).fit(data)["rank"]
            else:
                B, b = cast2sum_squares_form(data, alpha=self.alpha)
                b_array = b.toarray(order="C")
                _lsmr = lsmr(B, b_array)[:1][0]
                self.result["primal"] = _lsmr.reshape(-1, 1)

                # compute primal functional value
                def f_all_primal(x):
                    return 0.5 * norm(B @ x - b_array) ** 2

                self.result["f_primal"] = f_all_primal(self.result["primal"])

        elif self.method == "annotated":
            # In this case, we use the dual-based proximal gradient descent algorithm
            # to solve the problem.
            if self.cvxpy:
                raise NotImplementedError("Not implemented for method='annotated'.")
            else:
                goi = kwargs.get("goi", None)
                self.sslc = sum_squared_loss_conj()
                self.sslc.setup(data, alpha=self.alpha, goi=goi)
                self.fo_setup["f"] = lambda x: self.sslc.evaluate(x)
                self.fo_setup["grad"] = lambda x: self.sslc.prox(x)
                self.fo_setup["prox"] = lambda x, t: same_mean_reg(
                    lambd=self.lambd
                ).prox(x, t)
                self.fo_setup["prox_fcn"] = lambda x: same_mean_reg(
                    lambd=self.lambd
                ).evaluate(x)

                # first order kwargs
                self.fo_setup["printEvery"] = kwargs.get("printEvery", 5000)
                self.fo_setup["ArmijoLinesearch"] = kwargs.get(
                    "ArmijoLinesearch", False
                )
                self.fo_setup["linesearch"] = kwargs.get(
                    "linesearch", True
                )  # do not use True, still buggy
                self.fo_setup["acceleration"] = kwargs.get("acceleration", True)
                self.fo_setup["x0"] = kwargs.get(
                    "x0", np.random.rand(self.sslc.ell.shape[0], 1).astype(np.float64)
                ).reshape(-1, 1)
                # TODO: Using view() can change dtype in place. Not sure if it's better.
                self.fo_setup["x0"] = self.fo_setup["x0"].astype(np.float64)

                # You may replace 1. with self.sslc.find_Lipschitz_constant()
                # But when linesearch is True and acceleration is True, using 1 is faster.
                self.fo_setup["Lip_c"] = kwargs.get("Lip_c", 1.0)
                self.fo_setup["maxIters"] = kwargs.get("maxIters", 1e6)
                self.fo_setup["tol"] = kwargs.get("tol", 1e-12)

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
                def f_all_primal(x):
                    return 0.5 * norm(
                        self.sslc.B @ x - self.sslc.b
                    ) ** 2 + self.lambd * np.linalg.norm(self.sslc.ell @ x, 1)

                self.result["f_primal"] = f_all_primal(
                    self.result["primal"].reshape(-1, 1)
                )
                self.result["f_dual"] = self.sslc.evaluate(
                    self.result["dual"].reshape(-1, 1)
                )
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
                self.fo_setup["ArmijoLinesearch"] = kwargs.get(
                    "ArmijoLinesearch", False
                )
                self.fo_setup["linesearch"] = kwargs.get(
                    "linesearch", True
                )  # do not use True, still buggy
                self.fo_setup["acceleration"] = kwargs.get("acceleration", True)
                self.fo_setup["x0"] = kwargs.get(
                    "x0", np.random.rand(self.sslc.ell.shape[0], 1).astype(np.float64)
                ).reshape(-1, 1)
                # TODO: Using view() can change dtype in place. Not sure if it's better.
                self.fo_setup["x0"] = self.fo_setup["x0"].astype(np.float64)
                # You may replace 1. with self.sslc.find_Lipschitz_constant()
                # But when linesearch is True and acceleration is True, using 1 is faster.
                self.fo_setup["Lip_c"] = kwargs.get("Lip_c", 1.0)
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
                primal_time = lsqr(B, b.toarray(order="C"))[:1][0]
                self.result["timewise"] = primal_time.reshape(-1, top_n)

        elif self.method == "huber":
            # In this case we use CVXPY to solve the problem.
            if self.cvxpy:
                self.M = kwargs.get("M", 1)
                self.incl_reg = kwargs.get("incl_reg", True)
                h_cvx = huber_cvx(
                    data, alpha=self.alpha, M=self.M, incl_reg=self.incl_reg
                )
                primal_s = cp.Variable((data.num_vertices(), 1))
                problem = cp.Problem(
                    cp.Minimize(h_cvx.objective_fn_primal(primal_s))
                )  # for huber
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
                primal = primal_s.value.reshape(
                    -1,
                )
                self.result["primal"] = primal
                self.result["f_primal"] = problem.value
            else:
                raise NotImplementedError(
                    "First-order solver for Huber norm has not been not implemented. "
                    + "Please set explicitly that cvxpy=True."
                )
        else:
            raise NotImplementedError("Method not implemented.")

        return self.result
