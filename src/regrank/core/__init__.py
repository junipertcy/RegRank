# regrank/core/__init__.py

"""
Core models for network ranking.

This submodule provides the primary model implementations for the `regrank` package.
The main recommended class is SpringRank.
"""

from __future__ import annotations

import warnings

# Import the primary model directly for easy access
from .base import SpringRank

# Import the legacy model with a private alias to avoid direct use
from .base import SpringRankLegacy as _SpringRankLegacy
from .losses import huber_loss, sum_squared_loss, sum_squared_loss_conj

# Define the public API for this submodule.
# The primary, recommended model is listed first.
__all__ = [
    "SpringRank",
    "SpringRankLegacy",
    "sum_squared_loss",
    "sum_squared_loss_conj",
    "huber_loss",
]


# Create a wrapper for the legacy class that issues a warning upon use.
# This makes it clear to users that they should migrate to the new class.
class SpringRankLegacy(_SpringRankLegacy):
    """
    DEPRECATED: Legacy implementation of the SpringRank model.

    .. deprecated:: 0.x.x
       Use :class:`regrank.core.SpringRank` instead. This class will be
       removed in a future version.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            (
                "The 'SpringRankLegacy' class is deprecated and will be removed "
                "in a future version. Please use 'SpringRank' instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# Prepend the deprecation warning to the original docstring
if _SpringRankLegacy.__doc__ and SpringRankLegacy.__doc__:
    SpringRankLegacy.__doc__ += "\n\n" + _SpringRankLegacy.__doc__
elif _SpringRankLegacy.__doc__:
    SpringRankLegacy.__doc__ = _SpringRankLegacy.__doc__
