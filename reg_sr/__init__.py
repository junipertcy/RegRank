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

__version__ = "0.0.1"

from .fit import *
from .models import *
from .losses import *
from .regularizers import *
from .utils import *
from .experiments import *

__package__ = 'rSpringRank'
__title__ = 'rSpringRank: Regularized methods for efficient ranking in networks.'
__description__ = ''
__copyright__ = 'Copyright (C) 2023 Tzu-Chi Yen'
__license__ = "LGPL version 3 or above"
__author__ = """\n""".join([
    'Tzu-Chi Yen <tzuchi.yen@colorado.edu>',
])
__URL__ = ""
__version__ = '0.8.0'
__release__ = ''


__all__ = [
    "rSpringRank",
    "PhDExchange",
    "__author__",
    "__URL__",
    "__version__",
    "__copyright__"
]


