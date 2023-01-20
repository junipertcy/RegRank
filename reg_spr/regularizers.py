import numpy as np


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
