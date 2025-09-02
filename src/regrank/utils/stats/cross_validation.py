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

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from itertools import product
from typing import Any, cast

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from .utils import betaGlobal, betaLocal, compute_accuracy

# --- Library Imports ---
try:
    import graph_tool.all as gt
except ModuleNotFoundError:
    print("graph_tool not found. Please install it to run this script.")
    gt = None

try:
    import optuna
except ModuleNotFoundError:
    print("Optuna not found. Run 'pip install optuna' to use the Optuna tuner.")
    optuna = None

try:
    from ax.service.managed_loop import optimize as ax_optimize
except ModuleNotFoundError:
    print("Ax not found. Run 'pip install ax-platform' to use the Ax tuner.")
    ax_optimize = None

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Helper Functions ---


def get_cv_realization(
    graph: gt.Graph, n_splits: int, seed: int | None = None
) -> dict[int, gt.EdgePropertyMap]:
    """Generates cross-validation splits for the edges of a graph."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_edges = list(graph.edges())
    edge_filter_dict = {}

    for idx, (_train_indices, test_indices) in enumerate(kf.split(all_edges)):
        edge_filter = graph.new_edge_property("bool", val=True)
        for edge_idx in test_indices:
            edge_filter[all_edges[edge_idx]] = False
        edge_filter_dict[idx] = edge_filter

    return edge_filter_dict


# --- Hyperparameter Tuner Abstraction ---


class HyperparameterTuner(ABC):
    """Abstract base class for hyperparameter tuners."""

    def __init__(
        self, evaluation_function: Callable, params_grid: dict, n_trials: int = 50
    ):
        self.evaluation_function = evaluation_function
        self.params_grid = params_grid
        self.n_trials = n_trials

    @abstractmethod
    def tune(self) -> tuple[dict[str, float], dict[str, float]]:
        """Run the tuning process."""
        pass


# --- Concrete Tuner Implementations ---


class GridSearchTuner(HyperparameterTuner):
    """Performs hyperparameter tuning using grid search with interpolation."""

    def __init__(
        self,
        evaluation_function: Callable,
        params_grid: dict,
        interp_grid: dict | None = None,
        **kwargs,
    ):
        super().__init__(evaluation_function, params_grid)
        self.interp_grid = interp_grid

    def tune(self) -> tuple[dict[str, float], dict[str, float]]:
        param_names = list(self.params_grid.keys())
        param_combinations = list(product(*self.params_grid.values()))

        results_a, results_L = self.evaluation_function(param_combinations)

        mean_neg_a = {k: -np.mean(v) for k, v in results_a.items()}
        mean_neg_L = {k: -np.mean(v) for k, v in results_L.items()}

        if len(param_names) == 1:
            best_params_a = self._find_best_1d(
                self.params_grid[param_names[0]], mean_neg_a
            )
            best_params_L = self._find_best_1d(
                self.params_grid[param_names[0]], mean_neg_L
            )
        elif len(param_names) == 2:
            if not self.interp_grid:
                raise ValueError("Interpolation grid is required for 2D grid search.")
            best_params_a = self._find_best_2d(
                self.params_grid, mean_neg_a, self.interp_grid
            )
            best_params_L = self._find_best_2d(
                self.params_grid, mean_neg_L, self.interp_grid
            )
        else:
            raise ValueError("Grid search is only supported for 1 or 2 parameters.")

        return dict(best_params_a), dict(best_params_L)

    def _find_best_1d(
        self, alphas: list[float], scores: dict[tuple[float], float]
    ) -> dict[str, float]:
        y = [scores[(alpha,)] for alpha in alphas]
        f = interp1d(
            alphas, y, kind="quadratic", fill_value="extrapolate", assume_sorted=False
        )
        res = minimize(f, x0=np.mean(alphas), bounds=((min(alphas), max(alphas)),))
        print(f"Optimal alpha: {res.x[0]:.4f} (Score: {-res.fun:.4f})")
        return {"alpha": float(res.x[0])}

    def _find_best_2d(
        self, param_grid: dict, scores: dict, interp_grid: dict
    ) -> dict[str, float]:
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
            fill_value=np.nan,
        )

        X, Y = np.meshgrid(interp_p1, interp_p2, indexing="ij")
        points = np.array([X.ravel(), Y.ravel()]).T
        interpolated_values = interp_func(points).reshape(
            len(interp_p1), len(interp_p2)
        )
        interpolated_values = np.nan_to_num(interpolated_values, nan=np.inf)

        min_idx_flat = np.argmin(interpolated_values)
        min_idx_unraveled = np.unravel_index(min_idx_flat, interpolated_values.shape)
        best_p1, best_p2 = (
            interp_p1[min_idx_unraveled[0]],
            interp_p2[min_idx_unraveled[1]],
        )
        min_val = interpolated_values[min_idx_unraveled]
        print(
            f"Optimal ({p_names[0]}, {p_names[1]}): ({best_p1:.4f}, {best_p2:.4f}) (Score: {-min_val:.4f})"
        )
        return {p_names[0]: best_p1, p_names[1]: best_p2}


class OptunaTuner(HyperparameterTuner):
    """Performs hyperparameter tuning using Optuna."""

    def tune(self) -> tuple[dict[str, float], dict[str, float]]:
        if not optuna:
            raise ImportError("Optuna is not installed.")

        study_a = optuna.create_study(direction="maximize")
        study_a.optimize(
            lambda trial: self._objective(trial, "accuracy_local"),
            n_trials=self.n_trials,
        )

        study_L = optuna.create_study(direction="maximize")
        study_L.optimize(
            lambda trial: self._objective(trial, "accuracy_global"),
            n_trials=self.n_trials,
        )

        print(
            f"Best params (local accuracy): {study_a.best_params} (Score: {study_a.best_value:.4f})"
        )
        print(
            f"Best params (global accuracy): {study_L.best_params} (Score: {study_L.best_value:.4f})"
        )
        return study_a.best_params, study_L.best_params

    def _objective(self, trial, metric: str) -> float:
        params = {}
        for name, values in self.params_grid.items():
            param_type = type(values[0])
            if param_type is float:
                params[name] = trial.suggest_float(name, min(values), max(values))
            elif param_type is int:
                params[name] = trial.suggest_int(name, min(values), max(values))
            else:
                params[name] = trial.suggest_categorical(name, values)

        score_a, score_L = self.evaluation_function(params)

        return (
            cast(float, score_a) if metric == "accuracy_local" else cast(float, score_L)
        )


class AxTuner(HyperparameterTuner):
    """Performs hyperparameter tuning using Ax."""

    def tune(self) -> tuple[dict[str, float], dict[str, float]]:
        if not ax_optimize:
            raise ImportError("Ax is not installed.")

        parameters = [
            {
                "name": name,
                "type": "range",
                "bounds": [min(b), max(b)],
                "value_type": "float" if isinstance(b[0], float) else "int",
            }
            for name, b in self.params_grid.items()
        ]

        best_params_a, best_values_a, _, _ = ax_optimize(
            parameters=parameters,
            evaluation_function=lambda p: self._evaluation_wrapper(p, "accuracy_local"),
            objective_name="accuracy_local",
            minimize=False,
            total_trials=self.n_trials,
        )
        best_params_L, best_values_L, _, _ = ax_optimize(
            parameters=parameters,
            evaluation_function=lambda p: self._evaluation_wrapper(
                p, "accuracy_global"
            ),
            objective_name="accuracy_global",
            minimize=False,
            total_trials=self.n_trials,
        )

        print(
            f"Best params (local accuracy): {best_params_a} (Score: {best_values_a[0]['accuracy_local']:.4f})"
        )
        print(
            f"Best params (global accuracy): {best_params_L} (Score: {best_values_L[0]['accuracy_global']:.4f})"
        )
        return best_params_a, best_params_L

    def _evaluation_wrapper(self, p: dict, metric: str) -> dict[str, float]:
        score_a, score_L = self.evaluation_function(p)
        return {metric: score_a if metric == "accuracy_local" else score_L}


# --- Main CrossValidation Class ---


class CrossValidation:
    """Performs cross-validation to find optimal hyperparameters."""

    TUNER_MAP = {"grid_search": GridSearchTuner, "optuna": OptunaTuner, "ax": AxTuner}

    def __init__(
        self,
        g: gt.Graph,
        n_folds: int = 5,
        n_subfolds: int = 4,
        n_reps: int = 3,
        seed: int = 42,
        goi: Any = None,  # Add goi parameter
    ):
        self.g = g
        self.n_folds = n_folds
        self.n_subfolds = n_subfolds
        self.n_reps = n_reps
        self.seed = seed
        self.goi = goi  # Store goi parameter
        self.model: Any | None = None
        self.main_cv_splits: dict[int, gt.EdgePropertyMap] = {}
        self.sub_cv_splits: dict[int, dict[int, dict[int, gt.EdgePropertyMap]]] = (
            defaultdict(lambda: defaultdict(dict))
        )
        self.cv_results: dict[str, dict[str, dict[int, Any]]] = defaultdict(
            lambda: defaultdict(dict)
        )

    def prepare_cv_splits(self):
        print("Generating all cross-validation splits...")
        self.main_cv_splits = get_cv_realization(self.g, self.n_folds, seed=self.seed)
        rng = np.random.default_rng(self.seed)
        for fold_id in range(self.n_folds):
            main_train_view = gt.GraphView(self.g, efilt=self.main_cv_splits[fold_id])
            for rep in range(self.n_reps):
                sub_seed = int(rng.integers(0, 1e6))
                self.sub_cv_splits[fold_id][rep] = get_cv_realization(
                    main_train_view, self.n_subfolds, seed=sub_seed
                )
        print("Done.")

    def _run_single_validation(
        self, params: dict[str, float], main_train_view, fold_id
    ) -> tuple[list[float], list[float]]:
        if self.model is None:
            raise RuntimeError("Model has not been set.")
        scores_a, scores_L = [], []
        for rep in range(self.n_reps):
            for subfold in range(self.n_subfolds):
                efilter = self.sub_cv_splits[fold_id][rep][subfold]
                sub_train_view = gt.GraphView(main_train_view, efilt=efilter)

                validate_filter_map = main_train_view.new_edge_property("bool")
                for edge in main_train_view.edges():
                    validate_filter_map[edge] = not efilter[edge]
                sub_validate_view = gt.GraphView(
                    main_train_view, efilt=validate_filter_map
                )

                adj_train = gt.adjacency(sub_train_view)
                adj_validate = gt.adjacency(sub_validate_view)

                # Add goi to params if it exists and the model needs it
                fit_params = params.copy()
                if self.goi is not None:
                    fit_params["goi"] = self.goi

                print("!!")
                print(fit_params)
                result = self.model.fit(sub_train_view, printEvery=0, **fit_params)
                print("???")
                ranking = result.get("primal", np.array([]))

                if np.std(ranking) < 1e-9:
                    scores_a.append(0.5)
                    scores_L.append(-np.inf)
                    continue

                b_local, b_global = (
                    betaLocal(adj_train, ranking),
                    betaGlobal(adj_train, ranking),
                )
                acc_local, acc_global = compute_accuracy(
                    adj_validate.toarray(), ranking, b_local, b_global
                )
                scores_a.append(acc_local)
                scores_L.append(acc_global)
        return scores_a, scores_L

    def _evaluation_for_tuner(
        self,
        param_combinations: list[tuple],
        main_train_view,
        fold_id,
        param_names: list[str],
        **kwargs,
    ) -> tuple[dict, dict]:
        results_a, results_L = defaultdict(list), defaultdict(list)
        for param_values in param_combinations:
            # current_params = dict(zip(param_names, param_values, strict=True))
            current_params = {
                **dict(zip(param_names, param_values, strict=True)),
                **kwargs,
            }

            print("000", current_params)

            scores_a, scores_L = self._run_single_validation(
                current_params, main_train_view, fold_id
            )
            results_a[param_values].extend(scores_a)
            results_L[param_values].extend(scores_L)
        return results_a, results_L

    def _evaluate_for_single_trial(
        self, params: dict, main_train_view, fold_id
    ) -> tuple[float, float]:
        """Evaluates a single set of parameters and returns the mean scores."""
        print("999")
        scores_a, scores_L = self._run_single_validation(
            params, main_train_view, fold_id
        )
        return float(np.mean(scores_a)), float(np.mean(scores_L))

    def train_and_validate(
        self, model: Any, fold_id: int, tuner_type: str, params_grid: dict, **kwargs
    ):
        self.model = model
        main_train_view = gt.GraphView(self.g, efilt=self.main_cv_splits[fold_id])

        tuner_class = self.TUNER_MAP.get(tuner_type)
        if not tuner_class:
            raise ValueError(f"Unknown tuner_type: {tuner_type}")

        if tuner_type == "grid_search":
            param_names = list(params_grid.keys())

            def grid_evaluation_function(combos):
                return self._evaluation_for_tuner(
                    combos, main_train_view, fold_id, param_names, kwargs=kwargs
                )

            evaluation_function = grid_evaluation_function
        else:

            def single_evaluation_function(params):
                return self._evaluate_for_single_trial(params, main_train_view, fold_id)

            evaluation_function = single_evaluation_function

        tuner = tuner_class(evaluation_function, params_grid, **kwargs)
        best_params_a, best_params_L = tuner.tune()
        model_method = getattr(self.model, "method", "unknown")
        if model_method == "legacy":
            self.cv_results[model_method]["alpha_a"][fold_id] = best_params_a["alpha"]
            self.cv_results[model_method]["alpha_L"][fold_id] = best_params_L["alpha"]
        elif model_method == "annotated":
            self.cv_results[model_method]["params_a"][fold_id] = tuple(
                best_params_a.values()
            )
            self.cv_results[model_method]["params_L"][fold_id] = tuple(
                best_params_L.values()
            )
