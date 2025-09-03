import logging
import warnings
from collections import defaultdict
from typing import Any

import graph_tool.all as gt
import numpy as np
import scipy
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import brentq
from scipy.sparse import csr_matrix, spdiags

from ..regularizers import BaseRegularizer, RegularizerFactory, zero_reg
from ..utils.clustering import cluster_1d_array

logger = logging.getLogger(__name__)


class BaseModel:
    def __init__(self, loss, cfg: DictConfig, reg=None):
        if reg is None:
            reg = zero_reg()
        self.loss = loss
        self.local_reg = reg
        self.cfg = cfg

    @staticmethod
    def compute_summary(g, goi, sslc=None, dual_v=None, primal_s=None):
        if dual_v is not None and primal_s is not None:
            raise AttributeError("Only use either dual_v or primal_s.")
        elif dual_v is None and primal_s is None:
            raise AttributeError("You need some input data.")
        elif dual_v is not None:
            # We take firstOrderMethods.py output directly
            dual_v = np.array(dual_v).reshape(-1, 1)
            output = sslc.dual2primal(dual_v)
        else:
            output = primal_s
        node_metadata = np.array(list(g.vp[goi]))
        data_goi = defaultdict(list)
        for idx, _c in enumerate(node_metadata):
            data_goi[_c].append(output[idx])

        summary = {}
        keys = []
        diff_avgs = []
        for idx, key in enumerate(data_goi):
            keys.append(data_goi[key])
            diff_avgs.append(np.mean(data_goi[key]))
            summary[idx] = (key, len(data_goi[key]))

        summary["avg_clusters"], summary["keyid2clusterid"] = cluster_1d_array(
            diff_avgs
        )
        summary["goi"] = data_goi
        summary["rankings"] = output
        return summary


class SpringRank(BaseModel):
    """
    A class to compute SpringRank, a regularized ranking method for networks.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes the SpringRank object with a configuration.

        Args:
            cfg: A DictConfig object containing model and solver parameters.
        """
        self.cfg = cfg
        self.regularizer: BaseRegularizer = RegularizerFactory.create(cfg.regularizer)
        self.result: dict[str, Any] = {}

    def fit(self, data: Any, **kwargs) -> dict[str, Any]:
        """
        Main fitting method that delegates to a specific regularizer.

        This method computes the ranks based on the provided data and configuration.
        Runtime parameters can be passed as keyword arguments to override the
        initial configuration for this specific run.

        Args:
            data: The graph data to be ranked (e.g., a graph_tool.Graph).
            **kwargs: Runtime parameters to override the base configuration.
                      For example, `alpha=0.5` or `solver_max_iter=10000`.

        Returns:
            A dictionary containing the results, typically including the 'primal' ranks.
        """
        logger.info(f"Fitting with regularizer: {self.regularizer}")

        # Merge the initial config with any runtime kwargs
        run_cfg = self._merge_runtime_config(kwargs)

        # Delegate the fitting process to the selected regularizer
        self.result = self.regularizer.fit(data, run_cfg)
        return self.result

    def _merge_runtime_config(self, kwargs: dict[str, Any]) -> DictConfig:
        """
        Merges the base Hydra config with runtime keyword arguments.

        If no kwargs are provided, it returns the original configuration.
        Otherwise, it creates a new configuration object with the overrides.

        Args:
            kwargs: A dictionary of runtime parameters.

        Returns:
            A new DictConfig object with the merged parameters.
        """
        if not kwargs:
            return self.cfg

        # Create a mutable copy of the base config
        merged_cfg = OmegaConf.to_container(self.cfg, resolve=True)

        # Merge runtime kwargs into the copied config
        # This allows for nested dot notation like "solver.max_iter"
        override_cfg = OmegaConf.from_dotlist([f"{k}={v}" for k, v in kwargs.items()])
        merged_cfg = OmegaConf.merge(merged_cfg, override_cfg)

        return merged_cfg


class SpringRankLegacy:
    def __init__(self, alpha=0):
        warnings.warn(
            "The 'SpringRankLegacy' class is deprecated and will be removed "
            "in a future version. Please use 'regrank.SpringRank' with "
            "regularizer=legacy instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.alpha = alpha

    def fit_from_adjacency(self, adj):
        """Fit SpringRank directly from adjacency matrix."""
        ranks = self.compute_sr(adj, self.alpha)
        return {"rank": ranks.reshape(-1, 1)}

    def fit_scaled(self, data, scale=0.75):
        if type(data) is gt.Graph:
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
        if type(data) in [gt.Graph, gt.GraphView]:
            adj = gt.adjacency(data)
        else:
            raise NotImplementedError
        # print(f"bicgstab: adj = {adj.toarray()[:5,:5]}")
        ranks = self.get_ranks(adj)

        info = {"rank": ranks.reshape(-1, 1)}
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
            scipy.sparse.csr_matrix(operator), solution_vector, atol=1e-8
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
            return self.csr_SpringRank(A)
        else:
            N = A.shape[0]
            k_in = np.array(A.sum(axis=0)).flatten()
            k_out = np.array(A.sum(axis=1)).flatten()

            D = spdiags(k_out + k_in, 0, N, N)
            L = D - (A + A.T)

            B = k_out - k_in

            L_reg = L + alpha * scipy.sparse.eye(N)

            ranks = scipy.sparse.linalg.bicgstab(L_reg, B, atol=1e-8)[0]

            return ranks

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
