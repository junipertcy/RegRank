# regrank/regularizers/branch_constrained.py

import logging
from typing import Any

import graph_tool.all as gt
import numpy as np
from omegaconf import DictConfig

from ..core.graph_utils import get_adjacency_from_data
from ..solvers.optimization import GradientDescentSolver
from .base_regularizer import BaseRegularizer

# Set up a logger for this module
logger = logging.getLogger(__name__)


class BranchConstrainedRegularizer(BaseRegularizer):
    """
    Implements the branch-constrained SpringRank regularizer.

    This regularizer adds a squared hinge loss penalty to the SpringRank objective
    to guide the ranking solution towards a specific branch of the linear extension tree.

    Objective: E(r) = E_SpringRank(r) + λ Σₘ max(0, -dₘ(r_{iₘ} - r_{jₘ}))²
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.solver = GradientDescentSolver(cfg.solver)

    def fit(self, data: gt.Graph, cfg: DictConfig) -> dict[str, Any]:
        """
        Fits the branch-constrained SpringRank model.

        Args:
            data: The graph data.
            cfg: The Hydra configuration object for this run.

        Returns:
            A dictionary containing the primal solution (rankings), convergence
            status, final objective value, and constraint analysis.
        """
        adj = get_adjacency_from_data(data)
        N = adj.shape[0]

        constraints = self._validate_constraints(cfg.regularizer.constraints, N)

        # Build system matrices required for gradient and objective calculations
        matrices = self._build_system_matrices(adj, cfg.alpha)

        # Get an initial solution, e.g., from the legacy SpringRank
        # This provides a much better starting point than random initialization
        r_initial = self._get_initial_solution(adj, cfg.alpha)

        logger.info(
            f"Starting branch-constrained SpringRank with {len(constraints)} constraints..."
        )

        # Optimize using the injected solver
        result = self.solver.optimize(
            initial_point=r_initial,
            gradient_fn=lambda r: self._compute_gradient(
                r, matrices, constraints, cfg.lambd
            ),
            objective_fn=lambda r: self._evaluate_objective(
                r, matrices, constraints, cfg.lambd
            ),
        )

        # Analyze constraint satisfaction on the final solution
        constraint_analysis = self._analyze_constraint_satisfaction(
            result.solution, constraints
        )

        return {
            "primal": result.solution,
            "constraints": constraints,
            "converged": result.converged,
            "final_objective": result.final_objective,
            "iterations": result.iterations,
            "constraint_analysis": constraint_analysis,
        }

    def _validate_constraints(
        self, constraints: list[tuple], num_nodes: int
    ) -> list[tuple]:
        """Validates the format and indices of branch constraints."""
        validated = []
        for const in constraints:
            if not (isinstance(const, list | tuple) and len(const) == 3):
                raise TypeError(
                    f"Constraint must be a tuple/list of length 3, got {const}"
                )

            i_m, j_m, d_m = const
            if not isinstance(i_m, int) or not isinstance(j_m, int):
                raise ValueError(
                    f"Node indices must be integers, got i_m={i_m}, j_m={j_m}."
                )
            if not (0 <= i_m < num_nodes and 0 <= j_m < num_nodes):
                raise ValueError(
                    f"Invalid node indices in constraint {const}. Must be in [0, {num_nodes - 1}]"
                )
            if d_m not in [-1, 1]:
                raise ValueError(f"Decision value d_m must be +1 or -1, got {d_m}")
            validated.append(const)
        return validated

    def _build_system_matrices(
        self, adj: np.ndarray, alpha: float
    ) -> dict[str, np.ndarray]:
        """Precomputes matrices for the SpringRank objective."""
        k_in = adj.sum(axis=0)
        k_out = adj.sum(axis=1)
        L = np.diag(k_in + k_out) - (adj + adj.T)
        L_reg = L + alpha * np.eye(adj.shape[0])
        b_sr = k_out - k_in
        return {"L_reg": L_reg, "b_sr": b_sr}

    def _get_initial_solution(self, adj: np.ndarray, alpha: float) -> np.ndarray:
        """Computes a standard SpringRank solution as a warm start."""
        from ..regularizers.legacy import SpringRankLegacy  # Avoid circular import

        result = SpringRankLegacy().compute_sr(adj, int(alpha))
        return np.array(result, dtype=float)

    def _compute_gradient(
        self,
        r: np.ndarray,
        matrices: dict[str, np.ndarray],
        constraints: list[tuple],
        lambd: float,
    ) -> np.ndarray:
        """Computes the gradient of the full objective function."""
        grad: np.ndarray = matrices["L_reg"] @ r - matrices["b_sr"]
        for i_m, j_m, d_m in constraints:
            diff = r[i_m] - r[j_m]
            violation = max(0, -d_m * diff)
            if violation > 0:
                grad_penalty = -2 * lambd * d_m * violation
                if abs(grad_penalty) < 1e6:
                    grad[i_m] += grad_penalty
                    grad[j_m] -= grad_penalty
        return grad

    def _evaluate_objective(
        self, r: np.ndarray, matrices: dict, constraints: list[tuple], lambd: float
    ) -> float:
        """Evaluates the full objective function with overflow protection."""
        try:
            residual = matrices["L_reg"] @ r - matrices["b_sr"]
            springrank_term = 0.5 * np.dot(residual, residual)
            penalty_term = 0.0
            for i_m, j_m, d_m in constraints:
                diff = r[i_m] - r[j_m]
                violation = max(0, -d_m * diff)
                penalty_term += violation**2

            total_obj = springrank_term + lambd * penalty_term
            return total_obj if np.isfinite(total_obj) else np.inf
        except (OverflowError, FloatingPointError):
            return np.inf

    def _analyze_constraint_satisfaction(
        self, r: np.ndarray, constraints: list[tuple]
    ) -> list[dict]:
        """Analyzes how well the final ranking satisfies the constraints."""
        analysis = []
        for i_m, j_m, d_m in constraints:
            diff = r[i_m] - r[j_m]
            violation = max(0, -d_m * diff)
            satisfied = violation == 0
            analysis.append({
                "constraint": (i_m, j_m, d_m),
                "difference": diff,
                "violation_magnitude": violation,
                "satisfied": satisfied,
            })
        return analysis

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
