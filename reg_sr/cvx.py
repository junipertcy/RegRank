import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np


import clarabel
from reg_sr.losses import *


class same_mean_cvx(object):
    def __init__(self, g, L):
        self.g = g
        self.L = L

        self.sslc = None
        self.ssl = None
        pass

    # these 2 below are for dual formulations
    def loss_fn(self, dual_v):
        if self.sslc is None:
            self.sslc = sum_squared_loss_conj()
            self.sslc.setup(self.g, alpha=1)
        return self.sslc.evaluate_cvx(dual_v)

    def objective_fn(self, dual_v):
        return self.loss_fn(dual_v)  # + lambd * regularizer(dual_v)

    # these three below are for primal formulations
    def loss_fn_primal(self, primal_s, alpha=1):
        if self.ssl is None:
            self.ssl = sum_squared_loss()
            self.ssl.setup(self.g, alpha=1)
        return self.ssl.evaluate_cvx(primal_s)

    def regularizer(self, primal_s):
        # ssl = sum_squared_loss()
        # ssl.setup(g, alpha=1)
        return cp.norm(self.L @ primal_s, 1)

    def objective_fn_primal(self, primal_s, lambd=1):
        return self.loss_fn_primal(primal_s) + lambd * self.regularizer(primal_s)


class vanilla_cvx(object):
    def __init__(self, g, alpha=1):
        self.g = g
        self.alpha = alpha

        self.ssl = sum_squared_loss()
        self.ssl.setup(self.g, alpha=self.alpha)

    def loss_fn_primal(self, primal_s):
        return self.ssl.evaluate_cvx(primal_s)

    def objective_fn_primal(self, primal_s):
        return self.loss_fn_primal(primal_s)


class huber_cvx(object):
    def __init__(self, g, alpha=1, M=1, incl_reg=False):
        self.g = g
        self.alpha = alpha
        self.M = M

        self.hl = huber_loss()
        self.hl.setup(self.g, alpha=self.alpha, M=self.M, incl_reg=incl_reg)

    def loss_fn_primal(self, primal_s):
        return self.hl.evaluate_cvx(primal_s)

    def objective_fn_primal(self, primal_s):
        return self.loss_fn_primal(primal_s)
