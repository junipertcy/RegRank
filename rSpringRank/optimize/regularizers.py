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
import cvxpy as cp
import numpy as np
from numpy.linalg import norm


class Regularizer:
    """
    Inputs:
        lambd (scalar > 0): regularization coefficient. Default value is 1.
    All regularizers implement the following functions:
    1. evaluate(theta). Evaluates the regularizer at theta.
    2. prox(t, nu, warm_start, pool): Evaluates the proximal operator of the regularizer at theta
    """

    def __init__(self, lambd=1):
        if type(lambd) in [int, float] and lambd < 0:
            raise ValueError("Regularization coefficient must be a nonnegative scalar.")

        self.lambd = lambd

    def evaluate(self, theta):
        raise NotImplementedError(
            "This method is not implemented for the parent class."
        )

    def prox(self, t, nu, warm_start, pool):
        raise NotImplementedError(
            "This method is not implemented for the parent class."
        )


#### Regularizers
class zero_reg(Regularizer):
    def __init__(self, lambd=1):
        super().__init__(lambd)
        self.lambd = lambd

    def evaluate(self, theta):
        return 0

    def prox(self, t, nu, warm_start, pool):
        return nu


class same_mean_reg(Regularizer):
    # this is the conjugate function.
    def __init__(self, lambd=1):
        if np.size(lambd) > 1 or lambd <= 0:
            raise ValueError("'lambd' must be a positive scalar")
        self.lambd = lambd

    def evaluate(self, theta):
        """Indicate if the input 'theta' is in the constraint set or not

        Args:
            theta (float): input value (in the dual space)

        Returns:
            float: 0 if in the constraint set, +inf otherwise
        """
        tol = 1e-8
        return 0.0 if norm(theta, ord=np.inf) <= self.lambd + tol else np.infty

    def evaluate_cvx(self, theta):
        return (
            0.0 if cp.norm(theta / self.lambd, 1) <= 1 else np.infty
        )  # double check the ord

    def prox(self, theta, t):  # see LinfBall.py
        if self.lambd == 0:
            return 0.0
        else:
            return theta / np.maximum(1, np.abs(theta) / self.lambd)
