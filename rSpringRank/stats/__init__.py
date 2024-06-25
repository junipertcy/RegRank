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
from .experiments import PeerInstitution, PhDExchange
from .cross_validation import CrossValidation

__all__ = ["PeerInstitution", "PhDExchange", "CrossValidation"]
# __all__ = [s for s in dir() if not s.startswith('_')]
