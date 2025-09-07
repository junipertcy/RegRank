# regrank/regularizers/legacy.py

import logging
from typing import Any

import graph_tool.all as gt
import numpy as np
from omegaconf import DictConfig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr

# CVXPY is an optional dependency
try:
    import cvxpy as cp
except ImportError:
    cp = None

from ..core.base import SpringRankLegacy
from ..solvers import legacy_cvx
from ..utils.graph2mat import cast2sum_squares_form
from . import BaseRegularizer

# Set up a logger for this module
logger = logging.getLogger(__name__)


class LegacyRegularizer(BaseRegularizer):
    """
    Implements the original SpringRank algorithm with multiple solver options.

    This class can solve the standard SpringRank objective using:
    1. A direct Lagrange multiplier approach (for alpha=0).
    2. A regularized linear system (for alpha > 0).
    3. CVXPY for a convex optimization formulation.
    4. LSMR for a least-squares formulation.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.name = cfg.name
        # Store the solver method from the init config
        self.solver_method = cfg.solver_method
        self.legacy_solvers = SpringRankLegacy()

    def fit(
        self, data: gt.Graph | gt.GraphView | csr_matrix | np.ndarray, cfg: DictConfig
    ) -> dict[str, Any]:
        """
        Fits the legacy SpringRank model using the configured solver.

        Args:
            data: The graph data.
            cfg: The Hydra configuration object for this run.

        Returns:
            A dictionary containing the primal solution (rankings) and,
            if applicable, the final objective value.
        """
        # Use the solver method stored on the instance
        logger.info(f"Fitting Legacy SpringRank using solver: {self.solver_method}")

        # The 'cfg' passed here should be the top-level one containing 'alpha', etc.
        if self.solver_method == "cvxpy":
            return self._fit_cvxpy(data, cfg)
        elif self.solver_method == "bicgstab":
            return self._fit_bicgstab(data, cfg)
        elif self.solver_method == "lsmr":
            return self._fit_lsmr(data, cfg)
        else:
            raise ValueError(
                f"Unknown solver_method '{self.solver_method}' for LegacyRegularizer."
            )

    # In regrank/regularizers/legacy.py

    def _fit_cvxpy(
        self, data: gt.Graph | gt.GraphView | csr_matrix | np.ndarray, cfg: DictConfig
    ) -> dict[str, Any]:
        """Solves the SpringRank problem using CVXPY."""
        if cp is None:
            raise ImportError("CVXPY is required to use the 'cvxpy' solver.")

        # Get the CVXPY backend from the config. Defaults to None, which
        # tells CVXPY to use its default solver.
        cvxpy_backend = cfg.solver.get("backend", None)
        if cvxpy_backend:
            logger.info(f"Using CVXPY backend: {cvxpy_backend}")

        v_cvx = legacy_cvx(data, alpha=cfg.alpha)
        primal_s = cp.Variable((data.num_vertices(), 1))
        problem = cp.Problem(cp.Minimize(v_cvx.objective_fn_primal(primal_s)))
        # Pass the selected backend to the solve method.
        problem.solve(solver=cvxpy_backend, verbose=cfg.solver.get("verbose", False))

        if primal_s.value is None:
            logger.warning(
                f"CVXPY solver ({cvxpy_backend or 'default'}) did not return a solution."
            )
            primal = np.zeros(data.num_vertices())
        else:
            primal = primal_s.value.flatten()

        return {"primal": primal, "f_primal": problem.value}

    def _fit_bicgstab(
        self, data: gt.Graph | gt.GraphView | csr_matrix | np.ndarray, cfg: DictConfig
    ) -> dict[str, Any]:
        """Solves the SpringRank problem using direct linear system solvers."""
        if isinstance(data, csr_matrix | np.ndarray):
            adj = data
        else:
            adj = gt.adjacency(data)
        ranks = self.legacy_solvers.compute_sr(adj, cfg.alpha)
        return {"primal": ranks.flatten()}

    def _fit_lsmr(
        self, data: gt.Graph | gt.GraphView | csr_matrix | np.ndarray, cfg: DictConfig
    ) -> dict[str, Any]:
        """Solves the SpringRank problem by casting it to a least-squares form."""
        B, b = cast2sum_squares_form(data, alpha=cfg.alpha)
        b_array = b.toarray(order="C")
        primal = lsmr(B, b_array)[0]

        # Compute the final objective value
        final_objective = 0.5 * np.linalg.norm(B @ primal - b_array) ** 2

        return {"primal": primal.flatten(), "f_primal": final_objective}
