# regrank/regularizers/branch_constrained.py

import logging
from typing import Any

import graph_tool.all as gt
import numpy as np
from omegaconf import DictConfig

from .. import solvers
from . import BaseRegularizer

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
        self.solver = solvers.get_solver(cfg.solver)

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
        adj = gt.adjacency(data).T.toarray()
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
        from .legacy import SpringRankLegacy  # Avoid circular import

        result = SpringRankLegacy().compute_sr(adj, alpha)
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
