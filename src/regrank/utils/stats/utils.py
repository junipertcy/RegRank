from typing import cast

import numpy as np
from numba import njit  # type: ignore
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix


@njit(parallel=True, cache=True)
def negacc(A: np.ndarray, ranking: np.ndarray, beta: float) -> float:
    """Calculates the negative accuracy for beta optimization."""
    total_weight = np.sum(A)
    if total_weight == 0:
        return 0.0

    num_nodes = len(ranking)
    total_deviation = 0.0
    for i in range(num_nodes):
        for j in range(num_nodes):
            rank_diff = ranking[i] - ranking[j]
            p_ij = 1.0 / (1.0 + np.exp(-2 * beta * rank_diff))
            total_deviation += np.abs(A[i, j] - (A[i, j] + A[j, i]) * p_ij)

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
    return float(y)


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

    # Added type hints for mypy compliance
    def objective(beta: float) -> float:
        return cast(float, negacc(adj_dense, ranking, beta))

    result = minimize_scalar(objective, bounds=(1e-6, 1000), method="bounded")
    return float(result.x)


def betaGlobal(adj: csr_matrix, ranking: np.ndarray) -> float:
    """Finds the optimal global beta value."""
    adj_dense = adj.toarray()

    # Added type hints for mypy compliance
    def objective(beta: float) -> float:
        return cast(float, f_objective(adj_dense, ranking, beta) ** 2)

    result = minimize_scalar(objective, bounds=(1e-6, 1000), method="bounded")
    return float(result.x)
