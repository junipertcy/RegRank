# regrank/regularizers/annotated.py

import logging
from typing import Any, cast

import graph_tool.all as gt
import numpy as np
from omegaconf import DictConfig

from ..losses import sum_squared_loss_conj
from ..regularizers import same_mean_reg
from ..solvers.optimization import gradientDescent
from .base_regularizer import BaseRegularizer

# Set up a logger for this module
logger = logging.getLogger(__name__)


class AnnotatedRegularizer(BaseRegularizer):
    """
    Implements the annotated SpringRank regularizer.

    This method uses group annotations on nodes (the 'goi' or group of interest)
    to apply a regularization penalty that encourages nodes within the same group
    to have similar ranks. It is solved using a dual-based proximal gradient
    descent algorithm.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.sslc = sum_squared_loss_conj()

    def fit(self, data: gt.Graph, cfg: DictConfig) -> dict[str, Any]:
        """
        Fits the annotated regularized SpringRank model.

        Args:
            data: The graph data.
            cfg: The Hydra configuration object for this run.

        Returns:
            A dictionary containing the primal and dual solutions, solver history,
            and final primal and dual objective values.
        """
        goi = cfg.regularizer.get("goi")
        if goi is None:
            raise ValueError(
                "The 'goi' parameter is required for the annotated regularizer."
            )

        logger.info(f"Starting Annotated Regularized SpringRank with goi='{goi}'")

        # 1. Setup the loss and regularization terms
        self._setup_loss_and_reg(data, cfg, goi)

        # 2. Configure and run the solver
        fo_setup = self._configure_solver(cfg)
        dual, history = gradientDescent(**fo_setup)

        # 3. Convert dual solution back to primal (rankings)
        primal = self.sslc.dual2primal(dual)

        # 4. Compute final objective values for analysis
        f_primal = self._evaluate_primal_objective(primal, cfg.lambd)
        f_dual = self.sslc.evaluate(dual)

        return {
            "primal": primal.flatten(),
            "dual": dual.flatten(),
            "fo_output": history,
            "f_primal": f_primal,
            "f_dual": f_dual,
        }

    def _setup_loss_and_reg(self, data: gt.Graph, cfg: DictConfig, goi: str):
        """Prepares the sum_squared_loss_conj object."""
        try:
            self.sslc.setup(data, alpha=cfg.alpha, goi=goi)
        except Exception as e:
            logger.error(f"Error during sslc setup: {e}")
            raise

        if not hasattr(self.sslc, "evaluate"):
            raise AttributeError("self.sslc.setup did not properly initialize.")

    def _configure_solver(self, cfg: DictConfig) -> dict[str, Any]:
        """Configures the dictionary of parameters for the gradientDescent solver."""
        solver_cfg = cfg.solver
        x0 = np.random.rand(self.sslc.ell.shape[0], 1).astype(np.float64)

        return {
            "f": lambda x: self.sslc.evaluate(x),
            "grad": lambda x: self.sslc.prox(x),
            "x0": x0,
            "prox": lambda x, t: same_mean_reg(lambd=cfg.lambd).prox(x, t),
            "prox_obj": lambda x: same_mean_reg(lambd=cfg.lambd).evaluate(x),
            "stepsize": 1.0 / solver_cfg.get("Lip_c", 1.0),
            "printEvery": solver_cfg.get("printEvery", 5000),
            "maxIters": solver_cfg.get("maxIters", 1e6),
            "tol": solver_cfg.get("tol", 1e-12),
            "saveHistory": True,
            "linesearch": solver_cfg.get("linesearch", True),
            "ArmijoLinesearch": solver_cfg.get("ArmijoLinesearch", False),
            "acceleration": solver_cfg.get("acceleration", True),
            "restart": solver_cfg.get("restart", 50),
        }

    def _evaluate_primal_objective(self, r: np.ndarray, lambd: float) -> float:
        """Computes the primal objective value: 0.5*||Br - b||² + λ*||Lr||₁."""
        r_col = r.reshape(-1, 1)
        residual = self.sslc.B @ r_col - self.sslc.b
        l2_term = 0.5 * np.dot(residual.T, residual)
        l1_term = lambd * np.linalg.norm(self.sslc.ell @ r_col, 1)
        return cast(float, (l2_term + l1_term).item())
