# regrank/regularizers/l2time.py

import logging
from typing import Any

import graph_tool.all as gt
from omegaconf import DictConfig
from scipy.sparse.linalg import lsqr

from ..utils import cast2sum_squares_form_t
from . import BaseRegularizer

# Set up a logger for this module
logger = logging.getLogger(__name__)


class L2TimeRegularizer(BaseRegularizer):
    """
    Implements a time-smoothing L2 regularizer for dynamic SpringRank.

    This method models the evolution of ranks over time by adding a penalty
    on the L2 norm of rank differences between consecutive time steps. It is
    solved directly as a global least-squares problem.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def fit(self, data: gt.Graph, cfg: DictConfig) -> dict[str, Any]:
        """
        Fits the time-L2 regularized SpringRank model.

        Args:
            data: The graph data, expected to have temporal information.
            cfg: The Hydra configuration object for this run.

        Returns:
            A dictionary containing the timewise rankings.
        """
        from_year = cfg.regularizer.from_year
        to_year = cfg.regularizer.to_year
        top_n = cfg.regularizer.top_n

        logger.info(
            f"Starting L2-Time Regularized SpringRank for years {from_year}-{to_year} on top {top_n} entities."
        )

        # This function encapsulates the complex data transformation logic
        B, b, _ = cast2sum_squares_form_t(
            data,
            alpha=cfg.alpha,
            lambd=cfg.lambd,
            from_year=from_year,
            to_year=to_year,
            top_n=top_n,
        )

        # Solve the global least-squares problem using LSQR
        # LSQR is well-suited for large, sparse systems like this one.
        primal_time, istop, itn, _, _, _, _, _, _, _ = lsqr(B, b.toarray(order="C"))

        if istop != 1 and istop != 2:
            logger.warning(
                f"LSQR solver may not have converged. "
                f"Stop condition: {istop}, Iterations: {itn}"
            )

        # Reshape the flat vector of ranks into a (time, nodes) matrix
        timewise_rankings = primal_time.reshape(-1, top_n)

        return {"timewise": timewise_rankings}
