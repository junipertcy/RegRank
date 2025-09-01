#!/usr/bin/env python3
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
from collections import defaultdict

import cvxpy as cp
import numpy as np
import scipy.sparse.linalg
from numpy.linalg import norm
from scipy.optimize import brentq
from scipy.sparse import SparseEfficiencyWarning, csr_matrix, spdiags
from scipy.sparse.linalg import lsmr, lsqr
from sklearn.linear_model import Lasso
from sklearn.neighbors import NearestNeighbors

from ..io import cast2sum_squares_form, cast2sum_squares_form_t
from .cvx import huber_cvx, legacy_cvx
from .firstOrderMethods import gradientDescent
from .losses import sum_squared_loss_conj
from .regularizers import same_mean_reg, zero_reg

warnings.simplefilter("ignore", SparseEfficiencyWarning)


def determine_optimal_epsilon(arr, min_samples=2):
    """
    Determines the optimal epsilon value for clustering using the k-distance graph method.

    Parameters:
    arr (np.ndarray): The input array of float values.
    min_samples (int): The number of nearest neighbors to consider.

    Returns:
    float: The optimal epsilon value.
    """
    # Reshape the array for NearestNeighbors
    arr = arr.reshape(-1, 1)

    # Compute the nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(arr)
    distances, indices = neighbors_fit.kneighbors(arr)

    # Sort the distances to the nearest neighbors
    distances = np.sort(distances[:, 1], axis=0)

    # Find the point of maximum curvature (elbow)
    diff = np.diff(distances)
    optimal_index = np.argmax(diff)
    optimal_epsilon = distances[optimal_index]

    return optimal_epsilon


def cluster_1d_array(arr, min_samples=2):
    """
    Clusters a 1D array of float values such that adjacent values are grouped together.

    Parameters:
    arr (list or np.ndarray): The input array of float values.
    min_samples (int): The number of nearest neighbors to consider for determining epsilon.

    Returns:
    tuple: A tuple containing a list of clusters and a dictionary mapping each index of the input array to the index of the identified cluster.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    # Determine the optimal epsilon value
    eps = determine_optimal_epsilon(arr, min_samples)

    # Sort the array and keep track of original indices
    sorted_indices = np.argsort(arr)
    sorted_arr = arr[sorted_indices]

    # Initialize clusters and index mapping
    clusters = []
    index_mapping = {}
    current_cluster = [sorted_arr[0]]
    current_cluster_index = 0

    # Iterate through the sorted array and form clusters
    for i in range(1, len(sorted_arr)):
        if sorted_arr[i] - sorted_arr[i - 1] <= eps:
            current_cluster.append(sorted_arr[i])
        else:
            clusters.append(current_cluster)
            for idx in sorted_indices[i - len(current_cluster) : i]:
                index_mapping[idx] = current_cluster_index
            current_cluster = [sorted_arr[i]]
            current_cluster_index += 1

    # Append the last cluster
    clusters.append(current_cluster)
    for idx in sorted_indices[len(sorted_arr) - len(current_cluster) :]:
        index_mapping[idx] = current_cluster_index

    # Sort clusters by their mean values in descending order
    cluster_means = [np.mean(cluster) for cluster in clusters]
    sorted_cluster_indices = np.argsort(cluster_means)[::-1]
    sorted_clusters = [clusters[i] for i in sorted_cluster_indices]

    # Update index_mapping according to the new cluster order
    new_index_mapping = {}
    for new_cluster_index, original_cluster_index in enumerate(sorted_cluster_indices):
        for idx in index_mapping:
            if index_mapping[idx] == original_cluster_index:
                new_index_mapping[idx] = new_cluster_index

    return sorted_clusters, new_index_mapping


class BaseModel:
    def __init__(self, loss, reg=None):
        if reg is None:
            reg = zero_reg()
        self.loss = loss
        self.local_reg = reg

    @staticmethod
    def compute_summary(g, goi, sslc=None, dual_v=None, primal_s=None):
        if dual_v is not None and primal_s is not None:
            raise AttributeError("Only use either dual_v or primal_s.")
        elif dual_v is None and primal_s is None:
            raise AttributeError("You need some input data.")
        elif dual_v is not None:
            # We take firstOrderMethods.py output directly
            dual_v = np.array(dual_v).reshape(-1, 1)
            output = sslc.dual2primal(dual_v)
        else:
            output = primal_s
        node_metadata = np.array(list(g.vp[goi]))
        data_goi = defaultdict(list)
        for idx, _c in enumerate(node_metadata):
            data_goi[_c].append(output[idx])

        summary = {}
        keys = []
        diff_avgs = []
        for idx, key in enumerate(data_goi):
            keys.append(data_goi[key])
            diff_avgs.append(np.mean(data_goi[key]))
            summary[idx] = (key, len(data_goi[key]))

        summary["avg_clusters"], summary["keyid2clusterid"] = cluster_1d_array(
            diff_avgs
        )
        summary["goi"] = data_goi
        summary["rankings"] = output
        return summary


class SpringRankLegacy:
    def __init__(self, alpha=0):
        self.alpha = alpha
        # pass
        # self.change_base_model(BaseModel)

    def fit_from_adjacency(self, adj):
        """Fit SpringRank directly from adjacency matrix."""
        ranks = self.compute_sr(adj, self.alpha)
        return {"rank": ranks.reshape(-1, 1)}

    def fit_scaled(self, data, scale=0.75):
        if type(data) is gt.Graph:
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
        if type(data) in [gt.Graph, gt.GraphView]:
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
            if type(A) is np.ndarray:
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


class SpringRank(BaseModel):
    def __init__(self, method="legacy", **kwargs):
        self.alpha = 0
        self.lambd = 0
        self.method = method
        if method == "annotated":
            self.goi = kwargs.get(
                "goi", None
            )  # Will check for this parameter when fit is called
        self.result = {}
        self.sslc = None
        self.fo_setup = {}
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

        if self.method == "legacy":
            if self.cvxpy:
                v_cvx = legacy_cvx(data, alpha=self.alpha)
                primal_s = cp.Variable((data.num_vertices(), 1))
                problem = cp.Problem(
                    cp.Minimize(v_cvx.objective_fn_primal(primal_s))
                )  # for legacy
                problem.solve(
                    verbose=False,
                )
                primal = primal_s.value.reshape(
                    -1,
                )
                self.result["primal"] = primal
                self.result["f_primal"] = problem.value
            elif self.bicgstab:
                self.result["primal"] = (
                    SpringRankLegacy(alpha=self.alpha).fit(data)["rank"].reshape(-1)
                )
            else:
                B, b = cast2sum_squares_form(data, alpha=self.alpha)
                b_array = b.toarray(order="C")
                _lsmr = lsmr(B, b_array)[:1][0]
                self.result["primal"] = _lsmr.reshape(-1)

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
                if goi is None:
                    raise ValueError(
                        "The 'goi' parameter is required for method='annotated'."
                    )

                self.sslc = sum_squared_loss_conj()
                try:
                    self.sslc.setup(data, alpha=self.alpha, goi=goi)
                except Exception as e:
                    print(f"Error during setup: {e}")
                    raise

                # Verify sslc is properly initialized before defining lambda functions
                if self.sslc is None or not hasattr(self.sslc, "evaluate"):
                    raise AttributeError(
                        "self.sslc.setup did not properly initialize the 'evaluate' method"
                    )

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
                if primal_s.value is None:
                    print("Warning: CVXPY solver did not return a solution.")
                    primal = np.zeros(data.num_vertices())
                else:
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
        elif self.method == "dictionary":
            # Model parameters
            n_components = kwargs.get("n_components", 10)  # K
            gamma = kwargs.get("gamma", 10.0)  # Reconstruction weight
            lambda_sparse = kwargs.get(
                "lambda_sparse", 0.1
            )  # Sparsity weight for codes

            # Solver parameters
            max_iters_outer = kwargs.get("max_iters_outer", 20)
            tol_outer = kwargs.get("tol_outer", 1e-8)
            N = data.num_vertices()

            # Initialize ranks `s` using standard SpringRank.
            s = (
                SpringRankLegacy(alpha=self.alpha)
                .fit(data)["rank"]
                .astype(np.float64)
                .reshape(-1, 1)
            )

            # Initialize Dictionary D with a FIXED DC component as the first atom.
            D = np.random.randn(N, n_components)
            D[:, 0] = 1.0 / np.sqrt(N)  # The fixed DC atom (normalized)

            # Initialize the learnable part of the dictionary
            D_learnable = D[:, 1:]
            D_learnable -= D_learnable.mean(axis=0, keepdims=True)
            D_learnable /= np.linalg.norm(D_learnable, axis=0, keepdims=True)
            D[:, 1:] = D_learnable

            # Initialize sparse codes A (alpha)
            A = np.zeros((n_components, 1))

            # Get SpringRank quadratic form
            B_sr, b_sr_dense = cast2sum_squares_form(data, alpha=self.alpha)
            b_sr = b_sr_dense.toarray(order="C")

            # Pre-compute for s-update
            AtA = B_sr.T @ B_sr
            Atb = B_sr.T @ b_sr

            print(
                f"Starting Fixed DC-Atom Dictionary Learning SpringRank with {max_iters_outer} iterations..."
            )
            for i in range(max_iters_outer):
                s_old = s.copy()

                # --- STEP A: Update Sparse Codes A (alpha) ---
                # Objective: min_A (gamma/2)*||s - DA||_F^2 + lambda_sparse*||A||_1
                lasso = Lasso(
                    alpha=lambda_sparse / gamma, fit_intercept=False, max_iter=100
                )
                lasso.fit(D, s)
                A = lasso.coef_.reshape(-1, 1)

                # --- STEP B: Update Dictionary D ---
                # [MODIFIED] Use the K-SVD-like helper to update D while keeping d_0 fixed.
                D, A = self._update_dict_with_fixed_dc(s, D, A)

                # --- STEP C: Update Ranks s ---
                # Objective: min_s 1/2*||B@s - b||^2 + gamma/2*||s - DA||^2
                reconstruction = D @ A
                system_matrix = AtA + gamma * scipy.sparse.eye(N, format="csr")
                system_vector = Atb + gamma * reconstruction

                s_new, exit_code = scipy.sparse.linalg.bicgstab(
                    system_matrix, system_vector, x0=s.flatten()
                )
                if exit_code != 0:
                    print(
                        f"Warning: bicgstab exited with code {exit_code} at iteration {i}"
                    )

                s = s_new.reshape(-1, 1)

                change = np.linalg.norm(s - s_old) / (np.linalg.norm(s_old) + 1e-9)
                if i > 0 and change < tol_outer:
                    print(
                        f"DLSR converged at iteration {i + 1} with relative change: {change:.6f}"
                    )
                    break

            self.result["primal"] = s.flatten()
            self.result["dictionary"] = D
            self.result["codes"] = A.flatten()

            # The global mean is captured by the first code coefficient and the DC atom.
            self.result["global_mean_reconstructed"] = (
                D[:, 0] * A[0, 0]
            ).mean() * np.sqrt(N)

            self.result["reconstruction_error"] = np.linalg.norm(s - (D @ A))
        elif self.method == "branch_constrained":
            # Extract parameters
            branch_constraints = kwargs.get("branch_constraints", [])
            if not branch_constraints:
                raise ValueError(
                    "branch_constraints parameter is required for method='branch_constrained'."
                )

            # Optimization parameters with better defaults
            max_iter = kwargs.get("max_iter", 5000)
            tol = kwargs.get("tol", 1e-8)
            initial_step_size = kwargs.get("step_size", 1e-3)
            use_armijo = kwargs.get("use_armijo", True)

            # Get adjacency matrix
            if type(data) in [gt.Graph, gt.GraphView]:
                adj = gt.adjacency(data).toarray()
            else:
                adj = data

            N = adj.shape[0]

            # Initialize with standard SpringRank solution (better starting point)
            r = SpringRankLegacy(alpha=self.alpha).compute_sr(adj, self.alpha)

            # Precompute SpringRank system matrices
            k_in = adj.sum(axis=0)
            k_out = adj.sum(axis=1)
            L = np.diag(k_in + k_out) - (adj + adj.T)
            L_reg = L + self.alpha * np.eye(N)
            b_sr = k_out - k_in

            # Validate constraints
            for i_m, j_m, d_m in branch_constraints:
                if type(i_m) is not int or type(j_m) is not int:
                    raise ValueError(
                        f"i_m or j_m must be integers, got i_m={i_m}, j_m={j_m}."
                    )
                if not (0 <= i_m < N and 0 <= j_m < N):
                    raise ValueError(
                        f"Invalid node indices in constraint ({i_m}, {j_m}, {d_m})"
                    )
                if d_m not in [-1, 1]:
                    raise ValueError(f"Decision value d_m must be +1 or -1, got {d_m}")

            print(
                f"Starting branch-constrained SpringRank with {len(branch_constraints)} constraints..."
            )

            step_size = initial_step_size

            # Proximal gradient descent with Armijo line search
            for iteration in range(max_iter):
                r_old = r.copy()

                # Compute full gradient
                grad = self._compute_branch_gradient(
                    r, L_reg, b_sr, branch_constraints, self.lambd
                )

                # Check for numerical issues
                if not np.all(np.isfinite(grad)):
                    print(
                        f"Gradient contains non-finite values at iteration {iteration}"
                    )
                    step_size *= 0.5
                    continue

                # Armijo line search for adaptive step size
                if use_armijo:
                    step_size = self._armijo_line_search(
                        r, grad, L_reg, b_sr, branch_constraints, self.lambd, step_size
                    )

                # Gradient descent update
                r_new = r - step_size * grad

                # Numerical stability: clip extreme values
                r_new = np.clip(r_new, -1e3, 1e3)

                # Check convergence
                change = np.linalg.norm(r_new - r_old) / (np.linalg.norm(r_old) + 1e-9)

                if iteration % 500 == 0:
                    obj_val = self._evaluate_branch_objective(
                        r_new, L_reg, b_sr, branch_constraints, self.lambd
                    )
                    if np.isfinite(obj_val):
                        print(
                            f"Iteration {iteration}: objective = {obj_val:.6f}, change = {change:.6e}"
                        )
                    else:
                        print(f"Iteration {iteration}: numerical instability detected")
                        step_size *= 0.1  # Reduce step size drastically

                r = r_new

                if change < tol:
                    print(
                        f"Converged at iteration {iteration} with relative change: {change:.6e}"
                    )
                    break

            # Store results
            self.result["primal"] = r
            self.result["branch_constraints"] = branch_constraints
            self.result["final_objective"] = self._evaluate_branch_objective(
                r, L_reg, b_sr, branch_constraints, self.lambd
            )

            # Analyze constraint satisfaction
            constraint_analysis = []
            for i_m, j_m, d_m in branch_constraints:
                diff = r[i_m] - r[j_m]
                violation = max(0, -d_m * diff)
                satisfied = d_m * diff >= 0
                constraint_analysis.append({
                    "constraint": (i_m, j_m, d_m),
                    "difference": diff,
                    "violation_magnitude": violation,
                    "satisfied": satisfied,
                })

            self.result["constraint_analysis"] = constraint_analysis
        else:
            raise NotImplementedError("Method not implemented.")

        return self.result

    def _compute_branch_gradient(self, r, L_reg, b_sr, branch_constraints, lambd):
        """Compute gradient with numerical stability checks."""
        # SpringRank gradient
        grad = L_reg @ r - b_sr

        # Add branch constraint penalty gradients
        for i_m, j_m, d_m in branch_constraints:
            diff = r[i_m] - r[j_m]
            violation = max(0, -d_m * diff)

            if violation > 0:
                # Gradient of max(0, -d_m * (r_i - r_j))²
                grad_penalty = -2 * lambd * d_m * violation
                # Apply with numerical stability
                if abs(grad_penalty) < 1e6:  # Prevent extremely large gradients
                    grad[i_m] += grad_penalty
                    grad[j_m] -= grad_penalty

        return grad

    def _armijo_line_search(
        self, r, grad, L_reg, b_sr, branch_constraints, lambd, initial_step
    ):
        """Armijo line search for adaptive step sizing."""
        c1 = 1e-4  # Armijo constant
        beta = 0.5  # Step reduction factor
        max_backtracks = 20

        current_obj = self._evaluate_branch_objective(
            r, L_reg, b_sr, branch_constraints, lambd
        )
        grad_norm_sq = np.dot(grad, grad)

        step_size = initial_step

        for _ in range(max_backtracks):
            r_new = r - step_size * grad
            r_new = np.clip(r_new, -1e3, 1e3)  # Numerical stability

            new_obj = self._evaluate_branch_objective(
                r_new, L_reg, b_sr, branch_constraints, lambd
            )

            # Armijo condition
            if new_obj <= current_obj - c1 * step_size * grad_norm_sq:
                return step_size

            step_size *= beta

        return step_size

    def _evaluate_branch_objective(self, r, L_reg, b_sr, branch_constraints, lambd):
        """Evaluate objective with overflow protection."""
        try:
            # SpringRank term
            residual = L_reg @ r - b_sr
            springrank_term = 0.5 * np.dot(residual, residual)

            # Branch constraint penalty terms
            penalty_term = 0.0
            for i_m, j_m, d_m in branch_constraints:
                diff = r[i_m] - r[j_m]
                violation = max(0, -d_m * diff)
                penalty_term += violation**2

            total_obj = springrank_term + lambd * penalty_term

            # Return inf if numerical issues
            if not np.isfinite(total_obj):
                return np.inf

            return total_obj

        except (OverflowError, FloatingPointError):
            return np.inf

    @staticmethod
    def _update_dict_with_fixed_dc(S, D_in, A):
        """
        Updates the dictionary D using a K-SVD-like process, keeping the first
        atom (DC component) fixed.

        Args:
            S (np.ndarray): The rank matrix (N, num_signals).
            D_in (np.ndarray): The current dictionary (N, K).
            A (np.ndarray): The sparse code matrix (K, num_signals).
            gamma (float): Reconstruction weight.

        Returns:
            np.ndarray: The updated dictionary D (N, K).
        """
        D = D_in.copy()
        # Iterate through each atom, skipping the first (DC) atom
        for k in range(1, D.shape[1]):
            # Find the signals that use this atom
            omega_k = np.where(A[k, :] != 0)[0]
            if len(omega_k) == 0:
                # Atom is not used, no need to update
                continue

            # --- K-SVD Update Step ---
            # 1. Calculate the residual error if this atom were removed.
            # E_k = S - sum_{j!=k} d_j * a_j
            D_without_k = np.delete(D, k, axis=1)
            A_without_k = np.delete(A, k, axis=0)
            E_k = S - D_without_k @ A_without_k

            # Restrict the error matrix to only the signals that use atom k
            E_k_restricted = E_k[:, omega_k]

            # 2. Use SVD to find the best replacement for d_k and a_k.
            # E_k_restricted ≈ d_k * a_k_restricted
            try:
                # Perform Singular Value Decomposition
                u, s_val, vt = np.linalg.svd(E_k_restricted, full_matrices=False)

                # The best rank-1 approximation is the first singular vector/value pair.
                # Update the dictionary atom d_k with the first left singular vector.
                d_k_new = u[:, 0]

                # Update the corresponding sparse codes for this atom.
                a_k_new_restricted = s_val[0] * vt[0, :]

                # Update the dictionary and the sparse code matrix
                D[:, k] = d_k_new
                A[k, omega_k] = a_k_new_restricted

            except np.linalg.LinAlgError:
                # SVD can fail if the residual matrix is zero or ill-conditioned.
                # In this case, we just skip the update for this atom.
                continue

        return D, A
