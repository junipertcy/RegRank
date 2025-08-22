# regrank/optimize/__init__.py

"""
Core optimization models for network ranking.

This submodule provides the primary model implementations for the `regrank` package.
The main recommended class is SpringRank.
"""
from __future__ import annotations

import warnings

# Import the primary model directly for easy access
from .models import SpringRank

# Import the legacy model with a private alias to avoid direct use
from .models import SpringRankLegacy as _SpringRankLegacy

# Define the public API for this submodule.
# The primary, recommended model is listed first.
__all__ = ["SpringRank", "SpringRankLegacy"]


# Create a wrapper for the legacy class that issues a warning upon use.
# This makes it clear to users that they should migrate to the new class.
class SpringRankLegacy(_SpringRankLegacy):
    """
    DEPRECATED: Legacy implementation of the SpringRank model.

    .. deprecated:: 0.x.x
       Use :class:`regrank.optimize.SpringRank` instead. This class will be
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

# Ensure the docstring from the original class is not lost
SpringRankLegacy.__doc__ = _SpringRankLegacy.__doc__

