import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import inv

from collections import Counter
from itertools import combinations
from math import comb
from numpy.linalg import norm
import graph_tool.all as gt
import cvxpy as cp
from reg_spr.utils import compute_cache_from_g


class Loss:
    """ """

    def __init__(self):
        pass

    def evaluate(self, theta):
        raise NotImplementedError(
            "This method is not implemented for the parent class."
        )

    def setup(self, data, K):
        """This function has any important setup required for the problem."""
        raise NotImplementedError(
            "This method is not implemented for the parent class."
        )

    def prox(self, t, nu, data, warm_start, pool):
        raise NotImplementedError(
            "This method is not implemented for the parent class."
        )

    def anll(self, data, G):
        return -np.mean(self.logprob(data, G))


class sum_squared_loss(Loss):
    """
    f(s) = || B @ s - b ||_2^2
    """

    def __init__(self, compute_ell=True):
        super().__init__()
        self.B = None
        self.b = None
        self.ell = None
        self.Bt_B_inv = None
        self.compute_ell = compute_ell

    def evaluate(self, theta):
        return 0.5 * norm(self.B @ theta - self.b) ** 2

    def evaluate_cvx(self, theta):
        return 0.5 * cp.norm(self.B @ theta - self.b) ** 2

    def setup(self, data, alpha):
        # data is graph_tool.Graph()
        cache = compute_cache_from_g(
            data, alpha=alpha, sparse=1, ell=self.compute_ell
        )
        self.B = cache["B"]
        self.b = cache["b"]
        self.ell = cache["ell"]
        self.Bt_B_inv = cache["Bt_B_inv"]

    def prox(self, theta):
        raise NotImplementedError(
            "This class is for the primal problem, and is only intended for CVXPY to solve."
        )

    def dual2primal(self, v):
        raise NotImplementedError(
            "This class is for the primal problem, and is only intended for CVXPY to solve."
        )

    def predict(self):
        pass

    def scores(self):
        pass

    def logprob(self):
        pass


class sum_squared_loss_conj(Loss):
    """
    Conjugate of ...
    f(s) = || B @ s - b ||_2^2
    """

    def __init__(self):
        super().__init__()
        self.B = None
        self.b = None
        self.ell = None
        self.Bt_B_inv = None
        self.Bt_B_invSqrt = None

    def find_Lipschitz_constant(self):
        L = norm(self.Bt_B_invSqrt @ self.ell.T, ord=2) ** 2

        return L

    def evaluate(self, theta):
        # B.T @ b should be of size (227, 1)
        # term_1 = 0.5 * (cp.norm(Bt_B_inv) ** 0.5) * cp.norm(-ell.T @ theta + B.T @ b) ** 2

        # term_1 = 0.5 * (norm(self.Bt_B_inv) ** 0.5) * norm(-self.ell.T @ theta + self.B.T @ self.b) ** 2
        term_1 = (
            0.5
            * norm(self.Bt_B_invSqrt @ (-self.ell.T @ theta + self.B.T @ self.b)) ** 2
        )

        term_2 = -0.5 * norm(self.b.todense()) ** 2
        return term_1 + term_2
        # return sum((theta @ data["B"] - data["Y"]) ** 2)

    def evaluate_cvx(self, theta):
        # B.T @ b should be of size (227, 1)
        # term_1 = 0.5 * (cp.norm(Bt_B_inv) ** 0.5) * cp.norm(-ell.T @ theta + B.T @ b) ** 2

        term_1 = (
            0.5
            * cp.norm(self.Bt_B_invSqrt @ (-self.ell.T @ theta + self.B.T @ self.b))
            ** 2
        )

        term_2 = -0.5 * cp.norm(self.b.todense()) ** 2
        return term_1 + term_2
        # return sum((theta @ data["B"] - data["Y"]) ** 2)

    def setup(self, data, alpha):
        cache = compute_cache_from_g(data, alpha=alpha, sparse=1)
        self.B = cache["B"]
        self.b = cache["b"]
        self.ell = cache["ell"]
        self.Bt_B_inv = cache["Bt_B_inv"]
        self.Bt_B_invSqrt = cache["Bt_B_invSqrt"]

    def prox(self, theta):
        return -self.ell @ self.Bt_B_inv @ (-self.ell.T @ theta + self.B.T @ self.b)

    def dual2primal(self, v):
        d = self.Bt_B_inv @ (-self.ell.T @ v + self.B.T @ self.b)
        return np.array(np.squeeze(d)).reshape(-1, 1)

    def predict(self):
        pass

    def scores(self):
        pass

    def logprob(self):
        pass
