"""
``rSpringRank.stats``
---------------------

This module contains miscellaneous statistical functions.

Summary
+++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   PeerInstitution
   PhDExchange
   CrossValidation

"""

from .cross_validation import CrossValidation
from .experiments import PeerInstitution, PhDExchange

__all__ = ["PeerInstitution", "PhDExchange", "CrossValidation"]
# __all__ = [s for s in dir() if not s.startswith('_')]
