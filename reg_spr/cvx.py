import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np


import clarabel
from reg_spr.losses import *


class same_mean_cvx(object):
    def __init__(self, g, L):
        self.g = g
        self.L = L
        pass

    # these 2 below are for dual formulations
    def loss_fn(self, dual_v):
        sslc = sum_squared_loss_conj()
        sslc.setup(self.g, alpha=1)
        return sslc.evaluate_cvx(dual_v)

    def objective_fn(self, dual_v):
        return self.loss_fn(dual_v)  # + lambd * regularizer(dual_v)

    # these three below are for primal formulations
    def loss_fn_primal(self, primal_s, alpha=1):
        ssl = sum_squared_loss()
        ssl.setup(self.g, alpha=1)
        return ssl.evaluate_cvx(primal_s)

    def regularizer(self, primal_s):
        # ssl = sum_squared_loss()
        # ssl.setup(g, alpha=1)
        return cp.norm(self.L @ primal_s, 1)

    def objective_fn_primal(self, primal_s, lambd=10):
        return self.loss_fn_primal(primal_s) + lambd * self.regularizer(primal_s)


class vanilla_cvx(object):
    def __init__(self, g, alpha=1):
        self.g = g
        self.ssl = sum_squared_loss(compute_ell=False)
        self.ssl.setup(self.g, alpha=1)

        self.alpha = alpha
        pass

    def loss_fn_primal(self, primal_s):
        return self.ssl.evaluate_cvx(primal_s)

    def objective_fn_primal(self, primal_s):
        return self.loss_fn_primal(primal_s)


class huber_cvx(object):
    def __init__(self):
        pass

    def loss_fn_primal(self, primal_s, alpha=1):
        # It seems that there's no regularizer for the huber case
        # its loss is huber.
        pass

    def objective_fn_primal(self, primal_s, lambd=10):
        pass

