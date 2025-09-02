# regrank/regularizers/l1time.py

import logging
from typing import Any

import graph_tool.all as gt
import numpy as np
from omegaconf import DictConfig

from ..losses import sum_squared_loss_conj
from ..regularizers import same_mean_reg
from ..solvers.optimization import gradientDescent
from .base_regularizer import BaseRegularizer

# Set up a logger for this module
logger = logging.getLogger(__name__)


class L1TimeRegularizer(BaseRegularizer):
    """
    Implements a time-smoothing L1 regularizer for dynamic SpringRank.

    This method models the evolution of ranks over time by adding a penalty
    on the L1 norm of rank differences between consecutive time steps. It is
    solved using a dual-based proximal gradient descent algorithm.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.sslc = sum_squared_loss_conj()

    def fit(self, data: gt.Graph, cfg: DictConfig) -> dict[str, Any]:
        """
        Fits the time-L1 regularized SpringRank model.

        Args:
            data: The graph data, expected to have temporal information.
            cfg: The Hydra configuration object for this run.

        Returns:
            A dictionary containing the timewise rankings and solver history.
        """
        from_year = cfg.regularizer.from_year
        to_year = cfg.regularizer.to_year
        top_n = cfg.regularizer.top_n

        logger.info(
            f"Starting L1-Time Regularized SpringRank for years {from_year}-{to_year} on top {top_n} entities."
        )

        # 1. Setup the loss and regularization terms
        self._setup_loss_and_reg(data, cfg, from_year, to_year, top_n)

        # 2. Configure and run the solver
        fo_setup = self._configure_solver(cfg)
        dual_time, history = gradientDescent(**fo_setup)

        # 3. Convert dual solution back to primal (rankings)
        primal_time = self.sslc.dual2primal(dual_time)

        return {"timewise": primal_time.reshape(-1, top_n), "fo_output": history}

    def _setup_loss_and_reg(
        self, data: gt.Graph, cfg: DictConfig, from_year: int, to_year: int, top_n: int
    ):
        """Prepares the sum_squared_loss_conj object with all necessary data."""
        self.sslc.setup(
            data,
            alpha=cfg.alpha,
            lambd=cfg.lambd,
            from_year=from_year,
            to_year=to_year,
            top_n=top_n,
            method="time::l1",
        )

    def _configure_solver(self, cfg: DictConfig) -> dict[str, Any]:
        """Configures the dictionary of parameters for the gradientDescent solver."""
        solver_cfg = cfg.solver
        x0 = np.random.rand(self.sslc.ell.shape[0], 1).astype(np.float64)

        return {
            "f": lambda x: self.sslc.evaluate(x),
            "grad": lambda x: self.sslc.prox(x),
            "x0": x0,
            "prox": lambda x, t: same_mean_reg(lambd=1).prox(x, t),
            "prox_obj": lambda x: same_mean_reg(lambd=1).evaluate(x),
            "stepsize": 1.0 / solver_cfg.get("Lip_c", 1.0),
            "printEvery": solver_cfg.get("printEvery", 5000),
            "maxIters": solver_cfg.get("maxIters", 1e5),
            "tol": solver_cfg.get("tol", 1e-14),
            "saveHistory": True,
            "linesearch": solver_cfg.get("linesearch", True),
            "ArmijoLinesearch": solver_cfg.get("ArmijoLinesearch", False),
            "acceleration": solver_cfg.get("acceleration", True),
            "restart": solver_cfg.get("restart", 50),
        }
