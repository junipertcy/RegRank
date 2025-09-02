# regularizers/__init__.py
from abc import ABC, abstractmethod
from typing import Any

from omegaconf import DictConfig

from .regularizers import same_mean_reg, zero_reg

__all__ = [
    "RegularizerFactory",
    "BaseRegularizer",
    "same_mean_reg",
    "zero_reg",
]


class BaseRegularizer(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @abstractmethod
    def fit(self, data, cfg: DictConfig) -> dict[str, Any]:
        pass


class RegularizerFactory:
    _regularizers = {
        "legacy": "LegacyRegularizer",
        "branch_constrained": "BranchConstrainedRegularizer",
        "dictionary": "DictionaryRegularizer",
        "huber": "HuberRegularizer",
    }

    @classmethod
    def create(cls, cfg: DictConfig) -> BaseRegularizer:
        name = cfg.name
        if name not in cls._regularizers:
            raise ValueError(f"Unknown regularizer: {name}")

        # Dynamic import and instantiation
        module = __import__(
            f"regrank.regularizers.{name}", fromlist=[cls._regularizers[name]]
        )
        regularizer_class: type[BaseRegularizer] = getattr(
            module, cls._regularizers[name]
        )
        return regularizer_class(cfg)
