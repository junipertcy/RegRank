import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import inv
from scipy.linalg import svd
from collections import Counter
from itertools import combinations
from math import comb
from numpy.linalg import norm
import graph_tool.all as gt
import cvxpy as cp


def compute_cache_from_g(
    g, alpha, sparse=True, regularization=True, lbd=None, ell=True
):
    A = gt.adjacency(g)
    # print(f"our method: adj = {A.todense()[:5,:5]}")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Are you sure that A is symmetric?")
    if type(A) not in [csr_matrix, csc_matrix]:
        raise TypeError(
            "Please make sure that A is of type `csr_matrix` or `csc_matrix` of scipy.sparse."
        )
    if regularization and lbd is None:
        raise ValueError("Please assign the regularization strength (lbd).")
    shape = A.shape[0]
    row = []
    col = []
    data = []
    row_b = []
    col_b = []
    data_b = []
    for ind in zip(*A.nonzero()):
        i, j = ind[0], ind[1]
        if j <= i:  # TODO: double check if equality should be here??
            _row = i * (shape - 1) + j
        else:
            _row = i * (shape - 1) + j - 1
        row.append(_row)
        row.append(_row)
        col.append(i)
        col.append(j)
        data.append(A[ind] ** 0.5)
        data.append(-A[ind] ** 0.5)

        row_b.append(_row)
        col_b.append(0)
        data_b.append(-alpha * A[ind] ** 0.5)
        # print(f"adding... {_row, i, A[ind] ** 0.5} & {_row, j, - A[ind] ** 0.5}")

    if regularization:
        row += list(range(shape**2 - shape, shape**2))
        col += list(range(shape))
        data += [lbd] * shape
        row_b += list(range(shape**2 - shape, shape**2))
        col_b += [0] * shape
        data_b += [0] * shape
        B = csr_matrix((data, (row, col)), shape=(shape**2, shape), dtype=np.float)
        b = csr_matrix((data_b, (row_b, col_b)), shape=(shape**2, 1), dtype=np.float)
    else:
        B = csr_matrix(
            (data, (row, col)), shape=(shape**2 - shape, shape), dtype=np.float
        )
        b = csr_matrix(
            (data_b, (row_b, col_b)), shape=(shape**2 - shape, 1), dtype=np.float
        )

    if not sparse:
        B = B.todense()
        b = b.todense()

    if ell:
        _ell = compute_ell(g)
    else:
        _ell = None

    Bt_B_inv = compute_Bt_B_inv(B, sparse=0)

    _, s, Vh = svd(B.todense(), full_matrices=False)
    Bt_B_invSqrt = Vh.T @ np.diag(1 / s) @ Vh

    return {
        "B": B,
        "b": b,
        "ell": _ell,
        "Bt_B_inv": Bt_B_inv,
        "Bt_B_invSqrt": Bt_B_invSqrt,
    }


def compute_Bt_B_inv(B, sparse=True):
    if not sparse:
        B = B.todense()
        return np.linalg.inv(np.dot(B.T, B))
    return inv(np.dot(B.T, B))


def grad_g_star(B, b, v):
    return np.dot(compute_Bt_B_inv(B), v + np.dot(B.T, b))


def compute_ell(g, arg="class"):
    ctr_classes = Counter(g.vp[arg])
    len_classes = len(ctr_classes)
    # print(f"ctr_classes: {ctr_classes}")
    comb_classes = combinations(ctr_classes, 2)
    mb = list(g.vp[arg])
    ell = np.zeros([comb(len_classes, 2), len(g.get_vertices())])

    for idx, (i, j) in enumerate(comb_classes):
        for vtx in g.vertices():
            vtx = vtx.__int__()
            if mb[vtx] == i:
                ell[idx][vtx] = -ctr_classes[i] ** -1
            elif mb[vtx] == j:
                ell[idx][vtx] = ctr_classes[j] ** -1
            else:
                ell[idx][vtx] = 0
    return ell


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
        cache = compute_cache_from_g(data, alpha=alpha, sparse=1, lbd=1, ell=self.compute_ell)
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
            * cp.norm(self.Bt_B_invSqrt @ (-self.ell.T @ theta + self.B.T @ self.b)) ** 2
        )

        term_2 = -0.5 * cp.norm(self.b.todense()) ** 2
        return term_1 + term_2
        # return sum((theta @ data["B"] - data["Y"]) ** 2)

    def setup(self, data, alpha):
        cache = compute_cache_from_g(data, alpha=alpha, sparse=1, lbd=1)
        self.B = cache["B"]
        self.b = cache["b"]
        self.ell = cache["ell"]
        self.Bt_B_inv = cache["Bt_B_inv"]
        self.Bt_B_invSqrt = cache["Bt_B_invSqrt"]

    def prox(self, theta):
        return -self.ell @ self.Bt_B_inv @ (-self.ell.T @ theta + self.B.T @ self.b)

    def dual2primal(self, v):
        d = self.Bt_B_inv @ (-self.ell.T @ v + self.B.T @ self.b)
        return np.squeeze(d).tolist()[0]

    def predict(self):
        pass

    def scores(self):
        pass

    def logprob(self):
        pass
