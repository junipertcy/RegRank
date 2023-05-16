import numpy as np
from numpy.linalg import norm
import cvxpy as cp


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
        raise NotImplementedError("This method is not implemented for the parent class.")

    def prox(self, t, nu, warm_start, pool):
        raise NotImplementedError("This method is not implemented for the parent class.")


#### Regularizers
class zero_reg(Regularizer):
    def __init__(self, lambd=0):
        super().__init__(lambd)
        self.lambd = lambd

    def evaluate(self, theta):
        return 0

    def prox(self, t, nu, warm_start, pool):
        return nu   


class same_mean_reg(Regularizer):
    def __init__(self, tau, lambd=0):
        super().__init__(lambd)
        self.lambd = lambd
        self.tau = tau

    def evaluate(self, theta):
        # return 0 if norm(theta / self.tau, ord=1) <= 1 else np.infty
        # may not be called very often
        # print( theta.shape )
        return 0 if norm(theta, ord=np.inf) <= self.tau else np.infty

    def evaluate_cvx(self, theta):
        return 0 if cp.norm(theta / self.tau, 1) <= 1 else np.infty
    
    def prox(self, theta, t): # see LinfBall.py
        if self.tau == 0:
            return 0*theta
        else:
            return theta / np.maximum(1, np.abs(theta) / self.tau)
    


        # # called very often
        # if self.tau >= norm(theta, ord='inf'):
        #     return theta  # already feasible
        # else:
        #     return theta / 
        # return theta - np.multiply(np.sign(theta), np.maximum(0, np.fabs(theta) - self.tau * t))

        #     # return theta - np.asarray(np.sign(theta)) * np.asarray(np.maximum(0, np.fabs(theta) - self.tau * t))
        # return theta - np.multiply(np.sign(theta), np.maximum(0, np.fabs(theta) - self.tau * t))

        # # return theta - cp.multiply(np.asarray(np.sign(theta)), np.asarray(np.maximum(0, np.fabs(theta) - self.tau * t)))
        # # return theta - cp.multiply(cp.sign(theta), cp.maximum(0, np.fabs(theta) - self.tau * t))

