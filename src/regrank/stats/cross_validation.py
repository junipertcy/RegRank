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
#
#
# This code is translated to Python from MATLAB code by ChatGPT.
# The MATLAB code was originally written by Daniel Larremore, at:
# [https://github.com/cdebacco/SpringRank/blob/master/matlab/crossValidation.m](https://github.com/cdebacco/SpringRank/blob/master/matlab/crossValidation.m)


import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import csr_matrix

try:
    import graph_tool.all as gt
except ModuleNotFoundError:
    print("graph_tool not found. Please install graph_tool.")
import warnings
from collections import defaultdict
from typing import Any

from numba import njit  # type: ignore
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=RuntimeWarning)


@njit(parallel=True, cache=True)
def negacc(A: np.ndarray, ranking: np.ndarray, beta: float) -> float:
    """Calculates the negative accuracy for beta optimization."""
    total_weight = np.sum(A)
    num_nodes = len(ranking)
    total_deviation = 0.0
    for i in range(num_nodes):
        for j in range(num_nodes):
            rank_diff = ranking[i] - ranking[j]
            p_ij = 1.0 / (1.0 + np.exp(-2 * beta * rank_diff))
            total_deviation += np.abs(A[i, j] - (A[i, j] + A[j, i]) * p_ij)

    if total_weight == 0:
        return 0.0
    return float(total_deviation / total_weight - 1.0)


@njit(cache=True)
def f_objective(A: np.ndarray, ranking: np.ndarray, beta: float) -> float:
    """Objective function for global beta optimization."""
    num_nodes = len(ranking)
    y = 0.0
    for i in range(num_nodes):
        for j in range(num_nodes):
            rank_diff = ranking[i] - ranking[j]
            p_ij = 1.0 / (1.0 + np.exp(-2 * beta * rank_diff))
            y += rank_diff * (A[i, j] - (A[i, j] + A[j, i]) * p_ij)
    return float(y)  # FIX: Explicitly cast to float to satisfy mypy


@njit(cache=True)
def compute_accuracy(
    A: np.ndarray, ranking: np.ndarray, beta_local: float, beta_global: float
) -> tuple[float, float]:
    """Computes local and global accuracy metrics."""
    total_weight = np.sum(A)
    if total_weight == 0:
        return 0.5, 0.0

    num_nodes = len(ranking)
    y_local = 0.0
    log_likelihood_global = 0.0

    for i in range(num_nodes):
        for j in range(num_nodes):
            rank_diff = ranking[i] - ranking[j]
            p_local = 1.0 / (1.0 + np.exp(-2 * beta_local * rank_diff))
            p_global = 1.0 / (1.0 + np.exp(-2 * beta_global * rank_diff))

            y_local += abs(A[i, j] - (A[i, j] + A[j, i]) * p_local)

            if p_global > 1e-9 and p_global < 1 - 1e-9:
                if A[i, j] > 0:
                    log_likelihood_global += A[i, j] * np.log(p_global)
                if A[j, i] > 0:
                    log_likelihood_global += A[j, i] * np.log(1 - p_global)

    accuracy_local = 1.0 - 0.5 * y_local / total_weight
    accuracy_global = log_likelihood_global / total_weight

    return accuracy_local, accuracy_global


def betaLocal(adj: csr_matrix, ranking: np.ndarray) -> float:
    """Finds the optimal local beta value."""
    adj_dense = adj.toarray()

    def objective(beta):
        return negacc(adj_dense, ranking, beta)

    result = minimize_scalar(objective, bounds=(1e-6, 1000), method="bounded")
    return float(result.x)


def betaGlobal(adj: csr_matrix, ranking: np.ndarray) -> float:
    """Finds the optimal global beta value."""
    adj_dense = adj.toarray()

    def objective(beta):
        return f_objective(adj_dense, ranking, beta) ** 2

    result = minimize_scalar(objective, bounds=(1e-6, 1000), method="bounded")
    return float(result.x)


class CrossValidation:
    def __init__(
        self,
        g: gt.Graph,
        n_folds: int = 5,
        n_subfolds: int = 4,
        n_reps: int = 3,
        seed: int = 42,
    ):
        self.g = g
        self.n_folds = n_folds
        self.n_subfolds = n_subfolds
        self.n_reps = n_reps
        self.seed = seed
        self.model: Any | None = None

        self.main_cv_splits: dict[int, gt.EdgePropertyMap] = self.get_cv_realization(
            g, n_folds, seed=seed
        )
        self.sub_cv_splits: dict[int, dict[int, dict[int, gt.EdgePropertyMap]]] = (
            defaultdict(lambda: defaultdict(dict))
        )
        self.cv_results: dict[str, dict[str, dict[int, Any]]] = defaultdict(
            lambda: defaultdict(dict)
        )

    @staticmethod
    def get_cv_realization(
        graph: gt.Graph, n_splits: int, seed: int | None = None
    ) -> dict[int, gt.EdgePropertyMap]:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        all_edges = list(graph.edges())
        edge_filter_dict: dict[int, gt.EdgePropertyMap] = {}

        for idx, (_train_indices, test_indices) in enumerate(kf.split(all_edges)):
            edge_filter = graph.new_edge_property("bool", val=True)
            for edge_idx in test_indices:
                edge_filter[all_edges[edge_idx]] = False
            edge_filter_dict[idx] = edge_filter

        return edge_filter_dict

    def gen_all_train_validate_splits(self):
        print("Generating all cross-validation splits...")
        rng = np.random.default_rng(self.seed)
        for fold_id in range(self.n_folds):
            main_train_view = gt.GraphView(self.g, efilt=self.main_cv_splits[fold_id])
            for rep in range(self.n_reps):
                sub_seed = int(rng.integers(0, 1e6))
                self.sub_cv_splits[fold_id][rep] = self.get_cv_realization(
                    main_train_view, self.n_subfolds, seed=sub_seed
                )
        print("Done.")

    def _run_single_validation(
        self, g_train: gt.GraphView, g_validate: gt.GraphView, params: dict[str, float]
    ) -> tuple[float, float]:
        if self.model is None:
            raise RuntimeError("Model has not been set.")

        adj_train = gt.adjacency(g_train)
        adj_validate = gt.adjacency(g_validate)

        result = self.model.fit(g_train, printEvery=0, **params)
        ranking = result.get("primal", np.array([]))

        if np.std(ranking) < 1e-9:
            return 0.5, -np.inf

        b_local = betaLocal(adj_train, ranking)
        b_global = betaGlobal(adj_train, ranking)

        acc_local, acc_global = compute_accuracy(
            adj_validate.toarray(), ranking, b_local, b_global
        )
        return float(acc_local), float(acc_global)

    def train_and_validate(
        self,
        model: Any,
        fold_id: int,
        params_grid: dict[str, list[float]],
        interp_grid: dict[str, list[float]] | None = None,
    ):
        self.model = model
        main_train_view = gt.GraphView(self.g, efilt=self.main_cv_splits[fold_id])

        results_a: dict[tuple, list[float]] = defaultdict(list)
        results_L: dict[tuple, list[float]] = defaultdict(list)

        param_names = list(params_grid.keys())
        from itertools import product

        param_combinations = list(product(*params_grid.values()))

        for param_values in param_combinations:
            current_params = dict(zip(param_names, param_values, strict=False))
            for rep in range(self.n_reps):
                for subfold in range(self.n_subfolds):
                    efilter = self.sub_cv_splits[fold_id][rep][subfold]
                    sub_train_view = gt.GraphView(main_train_view, efilt=efilter)
                    validate_filter_map = main_train_view.new_edge_property("bool")
                    validate_filter_map.a = np.logical_not(efilter.a)
                    sub_validate_view = gt.GraphView(
                        main_train_view, efilt=validate_filter_map
                    )
                    sig_a, sig_L = self._run_single_validation(
                        sub_train_view, sub_validate_view, current_params
                    )
                    results_a[param_values].append(sig_a)
                    results_L[param_values].append(sig_L)

        ScoreDict1D = dict[tuple[float], float]
        ScoreDict2D = dict[tuple[float, ...], float]

        mean_neg_a: ScoreDict2D = {k: float(-np.mean(v)) for k, v in results_a.items()}
        mean_neg_L: ScoreDict2D = {k: float(-np.mean(v)) for k, v in results_L.items()}

        if not hasattr(self.model, "method"):
            return

        model_method = self.model.method
        if model_method == "vanilla":
            # FIX: Filter the dictionary to ensure keys are tuples of length 1, satisfying ScoreDict1D
            mean_neg_a_1d: ScoreDict1D = {
                k: v for k, v in mean_neg_a.items() if len(k) == 1
            }
            mean_neg_L_1d: ScoreDict1D = {
                k: v for k, v in mean_neg_L.items() if len(k) == 1
            }
            best_params_a = self._find_best_1d(params_grid["alpha"], mean_neg_a_1d)
            best_params_L = self._find_best_1d(params_grid["alpha"], mean_neg_L_1d)
            self.cv_results[model_method]["alpha_a"][fold_id] = best_params_a["alpha"]
            self.cv_results[model_method]["alpha_L"][fold_id] = best_params_L["alpha"]

        elif model_method == "annotated":
            if interp_grid is None:
                raise ValueError("interp_grid is required for 'annotated' method.")
            best_params_a = self._find_best_2d(params_grid, mean_neg_a, interp_grid)
            best_params_L = self._find_best_2d(params_grid, mean_neg_L, interp_grid)
            self.cv_results[model_method]["params_a"][fold_id] = (
                best_params_a.get("alpha"),
                best_params_a.get("lambd"),
            )
            self.cv_results[model_method]["params_L"][fold_id] = (
                best_params_L.get("alpha"),
                best_params_L.get("lambd"),
            )

    def _find_best_1d(
        self, alphas: list[float], scores: dict[tuple[float], float]
    ) -> dict[str, float]:
        """Finds the best alpha using 1D interpolation."""
        y = [scores[(alpha,)] for alpha in alphas]
        f = interp1d(
            alphas, y, kind="quadratic", fill_value="extrapolate", assume_sorted=False
        )
        res = minimize(f, x0=np.mean(alphas), bounds=((min(alphas), max(alphas)),))
        print(f"Optimal alpha: {res.x[0]:.4f} (Score: {-res.fun:.4f})")
        return {"alpha": float(res.x[0])}

    def _find_best_2d(
        self,
        param_grid: dict[str, list[float]],
        scores: dict[tuple, float],
        interp_grid: dict[str, list[float]],
    ) -> dict[str, float]:
        """Finds the best (alpha, lambda) using 2D interpolation."""
        p_names = list(param_grid.keys())
        p1_vals, p2_vals = param_grid[p_names[0]], param_grid[p_names[1]]
        interp_p1, interp_p2 = interp_grid[p_names[0]], interp_grid[p_names[1]]

        score_matrix = np.array([
            scores.get((p1, p2), 0.0) for p2 in p2_vals for p1 in p1_vals
        ]).reshape(len(p2_vals), len(p1_vals))

        interp_func = RegularGridInterpolator(
            (p1_vals, p2_vals),
            score_matrix.T,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        X, Y = np.meshgrid(interp_p1, interp_p2, indexing="ij")
        points = np.array([X.ravel(), Y.ravel()]).T
        interpolated_values = interp_func(points).reshape(
            len(interp_p1), len(interp_p2)
        )

        interpolated_values = np.nan_to_num(interpolated_values, nan=np.inf)

        min_idx_flat = np.argmin(interpolated_values)
        min_idx_unraveled = np.unravel_index(min_idx_flat, interpolated_values.shape)

        best_p1 = interp_p1[min_idx_unraveled[0]]
        best_p2 = interp_p2[min_idx_unraveled[1]]
        min_val = interpolated_values[min_idx_unraveled]

        print(
            f"Optimal ({p_names[0]}, {p_names[1]}): ({best_p1:.4f}, {best_p2:.4f}) (Score: {-min_val:.4f})"
        )
        return {p_names[0]: best_p1, p_names[1]: best_p2}
