# regrank/regularizers/huber.py

import logging
from typing import Any

import graph_tool.all as gt
import numpy as np
from omegaconf import DictConfig

# CVXPY is a specific dependency for this regularizer
try:
    import cvxpy as cp
except ImportError:
    # This allows the rest of the library to function if cvxpy is not installed,
    # and only fails when this specific regularizer is used.
    cp = None

from ..io.cvx import huber_cvx  # Assuming cvx problem definitions are in io
from .base_regularizer import BaseRegularizer

# Set up a logger for this module
logger = logging.getLogger(__name__)


class HuberRegularizer(BaseRegularizer):
    """
    Implements SpringRank with a Huber loss regularizer.

    This method is robust to outliers in the comparison data. It is solved using
    the convex optimization framework CVXPY.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        if cp is None:
            raise ImportError(
                "CVXPY is required to use the HuberRegularizer. "
                "Please install it via 'pip install cvxpy'."
            )

    def fit(self, data: gt.Graph, cfg: DictConfig) -> dict[str, Any]:
        """
        Fits the Huber-regularized SpringRank model.

        Args:
            data: The graph data.
            cfg: The Hydra configuration object for this run.

        Returns:
            A dictionary containing the primal solution (rankings) and the
            final primal objective value.
        """
        if not cfg.regularizer.get("cvxpy", True):
            # This check is for user clarity, as this implementation requires CVXPY.
            raise NotImplementedError(
                "First-order solver for Huber norm is not implemented. "
                "The 'cvxpy' flag must be True for this regularizer."
            )

        M = cfg.regularizer.M
        incl_reg = cfg.regularizer.incl_reg

        logger.info(
            f"Starting Huber Regularized SpringRank with M={M}, alpha={cfg.alpha}"
        )

        # 1. Define the CVXPY problem
        huber_problem, primal_variable = self._define_cvx_problem(
            data, cfg.alpha, M, incl_reg
        )

        # 2. Solve the problem
        try:
            huber_problem.solve(verbose=cfg.solver.get("verbose", False))
        except cp.SolverError as e:
            logger.warning(
                f"Default CVXPY solver failed ({e}). Attempting to solve with GUROBI."
            )
            # Fallback to a more powerful commercial solver if available
            try:
                huber_problem.solve(
                    solver=cp.GUROBI,
                    verbose=cfg.solver.get("verbose", False),
                    reltol=cfg.solver.get("tol", 1e-8),  # GUROBI specific params
                )
            except cp.SolverError as gurobi_e:
                logger.error(f"GUROBI solver also failed: {gurobi_e}")
                primal_variable.value = None  # Ensure it's None on failure

        # 3. Process and package the results
        return self._package_results(
            huber_problem, primal_variable, data.num_vertices()
        )

    def _define_cvx_problem(
        self, data: gt.Graph, alpha: float, M: float, incl_reg: bool
    ) -> tuple[cp.Problem, cp.Variable]:
        """Constructs the CVXPY problem object."""
        h_cvx = huber_cvx(data, alpha=alpha, M=M, incl_reg=incl_reg)
        primal_s = cp.Variable((data.num_vertices(), 1))
        objective = cp.Minimize(h_cvx.objective_fn_primal(primal_s))
        problem = cp.Problem(objective)
        return problem, primal_s

    def _package_results(
        self, problem: cp.Problem, primal_variable: cp.Variable, num_nodes: int
    ) -> dict[str, Any]:
        """Packages the results from the CVXPY solve into a standard dictionary."""
        if primal_variable.value is None:
            logger.warning(
                "CVXPY solver did not return a solution. Returning a zero vector for rankings."
            )
            primal_solution = np.zeros(num_nodes)
            objective_value = np.inf
        else:
            primal_solution = primal_variable.value.flatten()
            objective_value = problem.value

        return {
            "primal": primal_solution,
            "f_primal": objective_value,
        }
