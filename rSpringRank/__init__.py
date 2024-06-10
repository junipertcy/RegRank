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

__version__ = "0.0.1"

from .optimize import *
from .datasets import *
from .io import *
from .stats import *

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
__URL__ = ""
__version__ = "0.8.0"
__release__ = ""


submodules = ["datasets", "io", "optimize", "stats"]

__all__ = submodules + [
    "LowLevelCallable",
    "test",
    "show_config",
    "__version__",
]


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
