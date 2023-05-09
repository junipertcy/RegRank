import reg_spr.losses as losses
import reg_spr.regularizers as regularizers
import graph_tool.all as gt
from reg_spr.fit import fit_base_model

import numpy as np
from numba import jit
from scipy.sparse import spdiags, csr_matrix
from scipy.optimize import brentq
import scipy.sparse.linalg

import warnings
from scipy.sparse import SparseEfficiencyWarning

warnings.simplefilter("ignore", SparseEfficiencyWarning)


class BaseModel:
    def __init__(self, loss, reg=regularizers.zero_reg()):
        self.loss = loss
        self.local_reg = reg


class SpringRank:
    def __init__(self, alpha=0):
        self.alpha = alpha
        pass
        # self.change_base_model(BaseModel)

    def fit_scaled(self, data, scale=0.75):
        if type(data) == gt.Graph:
            adj = gt.adjacency(data)
        else:
            raise NotImplementedError
        # from Hunter's code
        ranks = self.get_ranks(adj)
        inverse_temperature = self.get_inverse_temperature(adj, ranks)
        scaling_factor = 1 / (np.log(scale / (1 - scale)) / (2 * inverse_temperature))
        scaled_ranks = self.scale_ranks(ranks, scaling_factor)

        info = {"rank": scaled_ranks}
        return info

    def fit(self, data):
        if type(data) == gt.Graph:
            adj = gt.adjacency(data)
        else:
            raise NotImplementedError
        print(f"bicgstab: adj = {adj.todense()[:5,:5]}")
        ranks = self.get_ranks(adj)

        info = {"rank": ranks}
        return info

    # below came from Hunter's code
    def get_ranks(self, A):
        """
        params:
        - A: a (square) np.ndarray

        returns:
        - ranks, np.array

        TODO:
        - support passing in other formats (eg a sparse matrix)
        """
        return self.compute_sr(A, self.alpha)

    def get_inverse_temperature(self, A, ranks):
        """given an adjacency matrix and the ranks for that matrix, calculates the
        temperature of those ranks"""
        betahat = brentq(self.eqs39, 0.01, 20, args=(ranks, A))
        return betahat

    @staticmethod
    def scale_ranks(ranks, scaling_factor):
        return ranks * scaling_factor

    @staticmethod
    def csr_SpringRank(A):
        """
        Main routine to calculate SpringRank by solving linear system
        Default parameters are initialized as in the standard SpringRank model

        Arguments:
            A: Directed network (np.ndarray, scipy.sparse.csr.csr_matrix)

        Output:
            rank: N-dim array, indeces represent the nodes' indices used in ordering the matrix A
        """

        N = A.shape[0]
        k_in = np.array(A.sum(axis=0))
        k_out = np.array(A.sum(axis=1).transpose())

        # form the graph laplacian
        operator = csr_matrix(spdiags(k_out + k_in, 0, N, N) - A - A.transpose())

        # form the operator A (from Ax=b notation)
        # note that this is the operator in the paper, but augmented
        # to solve a Lagrange multiplier problem that provides the constraint
        operator.resize(N + 1, N + 1)
        operator[N, 0] = 1
        operator[0, N] = 1

        # form the solution vector b (from Ax=b notation)
        solution_vector = np.append((k_out - k_in), np.array([0])).transpose()

        # perform the computations
        ranks = scipy.sparse.linalg.bicgstab(
            scipy.sparse.csr_matrix(operator), solution_vector
        )[0]

        return ranks[:-1]

    def compute_sr(self, A, alpha=0):
        """
        Solve the SpringRank system.
        If alpha = 0, solves a Lagrange multiplier problem.
        Otherwise, performs L2 regularization to make full rank.

        Arguments:
            A: Directed network (np.ndarray, scipy.sparse.csr.csr_matrix)
            alpha: regularization term. Defaults to 0.

        Output:
            ranks: Solution to SpringRank
        """

        if alpha == 0:
            rank = self.csr_SpringRank(A)
        else:
            print("Running bicgstab to solve Ax=b ...")
            N = A.shape[0]
            k_in = np.sum(A, 0)
            k_out = np.sum(A, 1)

            C = A + A.T
            D1 = np.diag(k_out + k_in)
            B = k_out - k_in
            # print(B)
            B = B @ np.ones([N, 1])
            # print(B)
            print(f"alpha = {alpha}")
            A = alpha * np.eye(N) + D1 - C
            A = scipy.sparse.csr_matrix(np.matrix(A))
            # print(f"shape of A: {A.shape}; shape of B: {B.shape};")
            rank = scipy.sparse.linalg.bicgstab(A, B)[0]

        return np.transpose(rank)

    # @jit(nopython=True)
    def eqs39(self, beta, s, A):
        N = A.shape[0]
        x = 0
        for i in range(N):
            for j in range(N):
                if A[i, j] == 0:
                    continue
                else:
                    x += (s[i] - s[j]) * (
                        A[i, j]
                        - (A[i, j] + A[j, i]) / (1 + np.exp(-2 * beta * (s[i] - s[j])))
                    )
        return x