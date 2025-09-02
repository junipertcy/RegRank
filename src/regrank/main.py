# regrank/main.py

import logging
from typing import Any

import graph_tool.all as gt
from omegaconf import DictConfig, OmegaConf

from .core.base import BaseModel
from .regularizers import RegularizerFactory

# Set up a logger for this module
logger = logging.getLogger(__name__)


class SpringRank(BaseModel):
    """
    The main user-facing class for the Regularized-SpringRank library.

    This class acts as a dispatcher, using a configuration object to instantiate
    and run the appropriate regularizer (e.g., legacy, annotated, dictionary).

    Example usage with Hydra:

    @hydra.main(config_path="conf", config_name="config")
    def my_app(cfg: DictConfig) -> None:
        model = SpringRank(cfg)
        results = model.fit(my_graph_data)
        print(results['primal'])
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes the SpringRank model.

        Args:
            cfg: A Hydra (OmegaConf) configuration object that specifies which
                 regularizer to use and its parameters.
        """
        super().__init__(loss=cfg.get("loss"), cfg=cfg)
        self.cfg = cfg
        logger.info(f"Initializing SpringRank with regularizer: {cfg.regularizer.name}")

        # The factory creates the specific regularizer instance (e.g., L1Time, Legacy)
        # This is a key part of the modular design pattern.
        self.regularizer = RegularizerFactory.create(cfg.regularizer)
        self.result: dict[str, Any] = {}

    def fit(self, data: gt.Graph, **kwargs) -> dict[str, Any]:
        """
        Fits the configured SpringRank model to the provided graph data.

        This method delegates the actual fitting process to the specific regularizer
        object that was created during initialization.

        Args:
            data: The graph data, typically a graph_tool.Graph object.
            **kwargs: Runtime parameters that can override the initial configuration.
                      This is useful for quick experiments or adjustments without
                      changing config files.

        Returns:
            A dictionary containing the results of the fitting process, such as
            the primal solution (rankings), final objective value, etc. The exact
            contents depend on the regularizer used.
        """
        # Create a disposable copy of the config for this run, allowing runtime overrides.
        run_cfg = self.cfg.copy()

        # Merge any runtime kwargs into the config.
        # This provides flexibility for programmatic adjustments.
        if kwargs:
            logger.debug(f"Overriding config with runtime kwargs: {kwargs}")
            # Note: A more robust implementation might validate kwargs against the config schema.
            for key, value in kwargs.items():
                OmegaConf.update(run_cfg, key, value, merge=True)

        logger.info(f"Starting fit with regularizer: '{run_cfg.regularizer.name}'")

        # Delegate the actual fitting process to the chosen regularizer object.
        # The regularizer's `fit` method will contain the specific logic for
        # its algorithm (e.g., bicgstab, cvxpy, gradient descent).
        self.result = self.regularizer.fit(data, run_cfg)

        logger.info("Fitting process completed.")
        return self.result
