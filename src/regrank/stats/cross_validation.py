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
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import csr_matrix
try:
    import graph_tool.all as gt
except ModuleNotFoundError:
    print("graph_tool not found. Please install graph_tool.")
import warnings
from collections import defaultdict
from typing import Dict, Tuple, List, Any, Callable

from numba import njit  # type: ignore
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=RuntimeWarning)

# READABILITY: Renamed variables for clarity (M->A, r->ranking, b->beta).
# NOTE: These functions are inherently O(N^2) because the Bradley-Terry model
# defines probabilities over all pairs of nodes, not just connected ones.
# For very large, sparse graphs, this can be a bottleneck.
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
    
    return total_deviation / total_weight - 1.0

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
    return y

@njit(cache=True)
def compute_accuracy(A: np.ndarray, ranking: np.ndarray, beta_local: float, beta_global: float) -> Tuple[float, float]:
    """Computes local and global accuracy metrics."""
    total_weight = np.sum(A)
    if total_weight == 0:
        return 0.5, 0.0 # Default accuracy if there are no edges

    num_nodes = len(ranking)
    y_local = 0.0
    log_likelihood_global = 0.0

    for i in range(num_nodes):
        for j in range(num_nodes):
            rank_diff = ranking[i] - ranking[j]
            p_local = 1.0 / (1.0 + np.exp(-2 * beta_local * rank_diff))
            p_global = 1.0 / (1.0 + np.exp(-2 * beta_global * rank_diff))

            # Local accuracy (deviation-based)
            y_local += abs(A[i, j] - (A[i, j] + A[j, i]) * p_local)

            # Global accuracy (log-likelihood based)
            if p_global > 0 and p_global < 1:
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
    # READABILITY: Using a lambda with clearer variable names
    objective = lambda beta: negacc(adj_dense, ranking, beta)
    result = minimize_scalar(objective, bounds=(1e-6, 1000), method='bounded')
    return result.x

def betaGlobal(adj: csr_matrix, ranking: np.ndarray) -> float:
    """Finds the optimal global beta value."""
    adj_dense = adj.toarray()
    # READABILITY: Squaring the result of the objective function
    objective = lambda beta: f_objective(adj_dense, ranking, beta) ** 2
    result = minimize_scalar(objective, bounds=(1e-6, 1000), method='bounded')
    return result.x

class CrossValidation:
    """
    Performs k-fold cross-validation to find optimal regularization parameters
    for SpringRank models.
    """
    def __init__(self, g: gt.Graph, n_folds: int = 5, n_subfolds: int = 4, n_reps: int = 3, seed: int = 42):
        """
        Initializes the CrossValidation instance.

        Args:
            g (gt.Graph): The input graph.
            n_folds (int): Number of folds for the main cross-validation.
            n_subfolds (int): Number of sub-folds for nested validation.
            n_reps (int): Number of repetitions for the nested validation.
            seed (int): Random seed for reproducibility.
        """
        self.g = g
        self.n_folds = n_folds
        self.n_subfolds = n_subfolds
        self.n_reps = n_reps
        self.seed = seed
        self.model = None

        # --- Data Structures for CV Splits and Results ---
        # Main splits: fold_id -> edge_filter
        self.main_cv_splits: Dict[int, gt.EdgePropertyMap] = self.get_cv_realization(g, n_folds, seed=seed)

        # Nested splits: fold_id -> rep_id -> subfold_id -> edge_filter
        self.sub_cv_splits: Dict[int, Dict[int, Dict[int, gt.EdgePropertyMap]]] = defaultdict(lambda: defaultdict(dict))

        # Optimal parameters from CV
        self.cv_results: Dict[str, Dict[str, Dict[int, Any]]] = defaultdict(lambda: defaultdict(dict))

    @staticmethod
    def get_cv_realization(graph: gt.Graph, n_splits: int, seed: int = None) -> Dict[int, gt.EdgePropertyMap]:
        """
        Generates edge filters for k-fold cross-validation.

        Args:
            graph (gt.Graph): The graph to split.
            n_splits (int): The number of folds (k).
            seed (int): Random seed for the KFold split.

        Returns:
            A dictionary mapping fold index to a boolean edge property map,
            where `True` indicates the edge is in the training set for that fold.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        all_edges = graph.get_edges()
        edge_filter_dict = {}

        for idx, (train_indices, test_indices) in enumerate(kf.split(all_edges)):
            # Create a filter that includes all edges by default
            edge_filter = graph.new_edge_property("bool", val=True)
            # Set edges in the test set to False
            for edge_idx in test_indices:
                edge_filter[all_edges[edge_idx]] = False
            edge_filter_dict[idx] = edge_filter

        return edge_filter_dict

    def gen_all_train_validate_splits(self):
        """Generates all nested cross-validation splits for each main fold."""
        print("Generating all cross-validation splits...")
        rng = np.random.default_rng(self.seed)
        for fold_id in range(self.n_folds):
            main_train_view = gt.GraphView(self.g, efilt=self.main_cv_splits[fold_id])

            # Generate sub-splits for each repetition
            for rep in range(self.n_reps):
                sub_seed = rng.integers(0, 1e6)
                self.sub_cv_splits[fold_id][rep] = self.get_cv_realization(
                    main_train_view, self.n_subfolds, seed=sub_seed
                )
        print("Done.")

    # REFACTORED: Helper to run a single validation trial to reduce code duplication.
    def _run_single_validation(self, g_train: gt.GraphView, g_validate: gt.GraphView, params: Dict[str, float]) -> Tuple[float, float]:
        """
        Fits the model on g_train and evaluates accuracy on g_validate.

        Args:
            g_train: The graph to train on.
            g_validate: The graph to validate on.
            params: Model parameters (e.g., {'alpha': 0.1, 'lambd': 1.0}).

        Returns:
            A tuple of (local_accuracy, global_accuracy).
        """
        adj_train = gt.adjacency(g_train)
        adj_validate = gt.adjacency(g_validate)

        ranking = self.model.fit(g_train, printEvery=0, **params)['primal']

        # If all ranks are identical, accuracy is undefined or minimal.
        if np.std(ranking) < 1e-9:
            return 0.5, -np.inf

        b_local = betaLocal(adj_train, ranking)
        b_global = betaGlobal(adj_train, ranking)

        return compute_accuracy(adj_validate.toarray(), ranking, b_local, b_global)

    def train_and_validate(self, model, fold_id: int, params_grid: Dict[str, list], interp_grid: Dict[str, list] = None):
        """
        Performs nested cross-validation for a given main fold to find the best hyperparameters.

        Args:
            model: The ranking model instance (must have a `.fit` method).
            fold_id (int): The index of the main fold to process.
            params_grid (dict): A dictionary with parameter names as keys and lists of values to test.
                                E.g., {'alpha': [0.1, 1, 10]} for vanilla.
                                E.g., {'alpha': [0.1, 1], 'lambd': [0.1, 1]} for annotated.
            interp_grid (dict, optional): A dictionary with parameter names and fine-grained lists
                                          for interpolation-based optimization.
        """
        self.model = model
        main_train_view = gt.GraphView(self.g, efilt=self.main_cv_splits[fold_id])

        # --- Run validation for all parameter combinations ---
        # Store results as: neg_accuracy[param_tuple] = [list_of_scores]
        results_a = defaultdict(list)
        results_L = defaultdict(list)

        param_names = list(params_grid.keys())

        # Create all combinations of parameters to iterate over
        from itertools import product
        param_combinations = list(product(*params_grid.values()))

        for param_values in param_combinations:
            current_params = dict(zip(param_names, param_values))

            for rep in range(self.n_reps):
                for subfold in range(self.n_subfolds):
                    efilter = self.sub_cv_splits[fold_id][rep][subfold]

                    sub_train_view = gt.GraphView(main_train_view, efilt=efilter)
                    # Use numpy's logical_not for the validation set filter
                    sub_validate_view = gt.GraphView(main_train_view, efilt=np.logical_not(efilter.a))

                    sig_a, sig_L = self._run_single_validation(sub_train_view, sub_validate_view, current_params)

                    results_a[param_values].append(sig_a)
                    results_L[param_values].append(sig_L)

        # --- Average results and find optimal parameters ---
        # We want to minimize the *negative* of the accuracy/likelihood
        mean_neg_a = {k: -np.mean(v) for k, v in results_a.items()}
        mean_neg_L = {k: -np.mean(v) for k, v in results_L.items()}

        if self.model.method == "vanilla":
            best_params_a = self._find_best_1d(params_grid['alpha'], mean_neg_a)
            best_params_L = self._find_best_1d(params_grid['alpha'], mean_neg_L)
            self.cv_results[self.model.method]['alpha_a'][fold_id] = best_params_a['alpha']
            self.cv_results[self.model.method]['alpha_L'][fold_id] = best_params_L['alpha']

        elif self.model.method == "annotated":
            best_params_a = self._find_best_2d(params_grid, mean_neg_a, interp_grid)
            best_params_L = self._find_best_2d(params_grid, mean_neg_L, interp_grid)
            self.cv_results[self.model.method]['params_a'][fold_id] = (best_params_a['alpha'], best_params_a['lambd'])
            self.cv_results[self.model.method]['params_L'][fold_id] = (best_params_L['alpha'], best_params_L['lambd'])

    # REFACTORED: Helper for 1D hyperparameter optimization.
    def _find_best_1d(self, alphas: list, scores: dict) -> Dict[str, float]:
        """Finds the best alpha using 1D interpolation."""
        y = [scores[(alpha,)] for alpha in alphas]
        f = interp1d(alphas, y, kind="quadratic", fill_value="extrapolate", assume_sorted=True)
        res = minimize(f, x0=np.mean(alphas), bounds=((min(alphas), max(alphas)),))
        print(f"Optimal alpha: {res.x[0]:.4f} (Score: {-res.fun:.4f})")
        return {'alpha': res.x}

    # REFACTORED: Helper for 2D hyperparameter optimization.
    def _find_best_2d(self, param_grid: dict, scores: dict, interp_grid: dict) -> Dict[str, float]:
        """Finds the best (alpha, lambda) using 2D interpolation."""
        p_names = list(param_grid.keys())
        p1_vals, p2_vals = param_grid[p_names], param_grid[p_names]

        # Create a 2D array of scores
        score_matrix = np.array([scores[(p1, p2)] for p2 in p2_vals for p1 in p1_vals]).reshape(len(p2_vals), len(p1_vals))

        interp_func = RegularGridInterpolator((p1_vals, p2_vals), score_matrix.T, method="linear")

        p1_interp, p2_interp = interp_grid[p_names], interp_grid[p_names]
        X, Y = np.meshgrid(p1_interp, p2_interp, indexing="ij")
        points = np.array([X.ravel(), Y.ravel()]).T

        interpolated_values = interp_func(points).reshape(len(p1_interp), len(p2_interp))
        min_idx = np.unravel_index(np.argmin(interpolated_values), interpolated_values.shape)

        best_p1 = p1_interp[min_idx]
        best_p2 = p2_interp[min_idx]
        min_val = np.min(interpolated_values)

        print(f"Optimal ({p_names}, {p_names}): ({best_p1:.4f}, {best_p2:.4f}) (Score: {-min_val:.4f})")
        return {p_names: best_p1, p_names: best_p2}

    @staticmethod
    def _compute_score_per_tag(g_train: gt.Graph, ranking: np.ndarray, tag: Any) -> Tuple[float, Tuple[int, int]]:
        """
        Computes a custom utility-based score for nodes with a specific tag.

        Args:
            g_train (gt.Graph): The graph on which to compute the score. Must have a "goi" vertex property.
            ranking (np.ndarray): The ranking scores for all nodes.
            tag (Any): The specific tag ('group of interest') to evaluate.

        Returns:
            A tuple containing:
            - The final score for the tag.
            - A tuple of (number_of_nodes_with_tag, number_of_edges_connected_to_tag_nodes).
        """
        score = 0
        node_count = 0
        edge_count = 0

        # EFFICIENCY: Pre-calculate exponentials of rankings once
        exp_ranking = np.exp(ranking)

        # Helper to compute utility for a single node
        def _get_node_utility(node_idx: int) -> float:
            utility = 0
            # Sum utility from all neighbors (in and out)
            for neighbor in g_train.get_all_neighbors(node_idx):
                utility += exp_ranking[neighbor]
            return utility

        for v in g_train.vertices():
            v_idx = int(v)
            if g_train.vp["goi"][v] != tag:
                continue

            node_count += 1
            node_utility = _get_node_utility(v_idx)
            if node_utility == 0:
                continue

            rank_i = ranking[v_idx]

            # Out-edges (i -> j)
            for e in v.out_edges():
                edge_count += 1
                j_idx = int(e.target())
                rank_j = ranking[j_idx]
                edge_weight = g_train.ep["weights"][e] if "weights" in g_train.ep else 1.0

                # Reward if rank_i > rank_j, penalize otherwise
                sign = 1 if rank_i > rank_j else -1
                score += sign * edge_weight * exp_ranking[j_idx] / node_utility

            # In-edges (j -> i)
            for e in v.in_edges():
                edge_count += 1
                j_idx = int(e.source())
                rank_j = ranking[j_idx]
                edge_weight = g_train.ep["weights"][e] if "weights" in g_train.ep else 1.0

                # Reward if rank_i > rank_j, penalize otherwise
                sign = 1 if rank_i > rank_j else -1
                score += sign * edge_weight * exp_ranking[j_idx] / node_utility

        return score, (node_count, edge_count)

