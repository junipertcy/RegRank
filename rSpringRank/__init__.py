#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


import warnings

import rSpringRank
import rSpringRank.datasets
import rSpringRank.draw
import rSpringRank.io
import rSpringRank.optimize
import rSpringRank.stats
from rSpringRank import *
from rSpringRank.datasets import *
from rSpringRank.draw import *
from rSpringRank.io import *
from rSpringRank.optimize import *
from rSpringRank.stats import *

__package__ = "rSpringRank"
__title__ = "rSpringRank: Regularized methods for efficient ranking in networks."
__description__ = ""
__copyright__ = "Copyright (C) 2023 Tzu-Chi Yen"
__license__ = "LGPL version 3 or above"
__author__ = """\n""".join(
    [
        "Tzu-Chi Yen <tzuchi.yen@colorado.edu>",
    ]
)
__URL__ = "https://github.com/junipertcy/rSpringRank"
__version__ = "0.2.28"
__release__ = "0.2.28"

submodules = ["datasets", "io", "optimize", "stats", "draw"]
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
        return _importlib.import_module(f"rSpringRank.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'rSpringRank' has no attribute '{name}'")


def show_config():
    """Show ``rSpringRank`` build configuration."""
    print("uname:", " ".join(os.uname()))
