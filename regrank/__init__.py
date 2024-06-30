#!/usr/bin/env python3
#
# Regularized-SpringRank -- regularized methods for efficient ranking in networks
#
# Copyright (C) 2023 Tzu-Chi Yen <tzuchi.yen@colorado.edu>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import importlib as _importlib
import os

from regrank.datasets import *
from regrank.draw import *
from regrank.io import *
from regrank.optimize import *
from regrank.stats import *

# import warnings


__package__ = "regrank"
__title__ = "regrank: Regularized methods for efficient ranking in networks."
__description__ = ""
__copyright__ = "Copyright (C) 2023 Tzu-Chi Yen"
__license__ = "LGPL version 3 or above"
__author__ = """\n""".join(
    [
        "Tzu-Chi Yen <tzuchi.yen@colorado.edu>",
    ]
)
__URL__ = "https://github.com/junipertcy/regrank"

submodules = ["datasets", "io", "optimize", "stats", "draw"]
func_optimize = ["SpringRankLegacy", "SpringRank"]
dunder = [
    "__version__",
    "__package__",
    "__title__",
    "__description__",
    "__author__",
    "__URL__",
    "__license__",
    "__release__",
]
__all__ = (
    submodules
    + [
        "show_config",
    ]
    + dunder
)


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f"regrank.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'regrank' has no attribute '{name}'")


def show_config():
    """Show ``regrank`` build configuration."""
    print("uname:", " ".join(os.uname()))
