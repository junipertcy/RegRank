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
#
#
# This code is translated to Python from MATLAB code by ChatGPT.
# The MATLAB code was originally written by Daniel Larremore, at:
# https://github.com/cdebacco/SpringRank/blob/master/matlab/crossValidation.m

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize, minimize_scalar

try:
    import graph_tool.all as gt
except ModuleNotFoundError:
    print("graph_tool not found. Please install graph_tool.")
import warnings
from collections import defaultdict

from numba import njit
from scipy.interpolate import RegularGridInterpolator
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=RuntimeWarning)


@njit(parallel=True, cache=True)
def negacc(M, r, b):
    m = np.sum(M)
    n = len(r)
    y = 0
    for i in range(n):
        for j in range(n):
            d = r[i] - r[j]
            y += np.abs(
                M[i, j] - (M[i, j] + M[j, i]) * ((1 + np.exp(-2 * b * d)) ** (-1))
            )
    a = y / m - 1
    return a


@njit(cache=True)
def f(M, r, b):
    n = len(r)
    y = 0.0
    for i in range(n):
        for j in range(n):
            d = r[i] - r[j]
            pij = (1 + np.exp(-2 * b * d)) ** (-1)
            y += np.float64(d * (M[i, j] - (M[i, j] + M[j, i]) * pij))
    return y


@njit(cache=True)
def compute_accuracy(A, s, beta_local, beta_global):
    m = np.sum(A)
    n = len(s)
    y_local = 0
    a_global = 0
    for i in range(n):
        for j in range(n):
            d = s[i] - s[j]
            p_local = (1 + np.exp(-2 * beta_local * d)) ** (-1)
            p_global = (1 + np.exp(-2 * beta_global * d)) ** (-1)
            # for local accuracy
            y_local += abs(A[i, j] - (A[i, j] + A[j, i]) * p_local)
            # for global accuracy
            if p_global == 0 or p_global == 1:
                pass
            else:
                a_global += A[i, j] * np.log(p_global) + A[j, i] * np.log(1 - p_global)
    # cleanup
    a_local = 1 - 0.5 * y_local / m
    a_global /= m
    return a_local, a_global


# def compute_rum_accuracy(A, s):
#     g = gt.Graph()
#     g.add_edge_list(A)

#     m = np.sum(A)
#     n = len(s)
#     y_local = 0
#     a_global = 0
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 continue
#             if A[i, j] == 0 and A[j, i] == 0:
#                 continue
#             if A[i, j]
#             d = s[i] - s[j]
#             p_local = (1 + np.exp(-2 * beta_local * d)) ** (-1)
#             p_global = (1 + np.exp(-2 * beta_global * d)) ** (-1)
#             # for local accuracy
#             y_local += abs(A[i, j] - (A[i, j] + A[j, i]) * p_local)
#             # for global accuracy
#             if p_global == 0 or p_global == 1:
#                 pass
#             else:
#                 a_global += A[i, j] * np.log(p_global) + A[j, i] * np.log(1 - p_global)
#     # cleanup
#     a_local = 1 - 0.5 * y_local / m
#     a_global /= m
#     return a_local, a_global


def betaLocal(A, s):
    M = A.toarray()
    r = np.array(s, dtype=np.float64)
    b = minimize_scalar(lambda _: negacc(M, r, _), bounds=(1e-6, 1000)).x
    return b


def betaGlobal(A, s):
    M = A.toarray()
    r = np.array(s, dtype=np.float64)
    b = minimize_scalar(lambda _: f(M, r, _) ** 2, bounds=(1e-6, 1000)).x
    return b


class CrossValidation(object):
    def __init__(
        self,
        g,
        n_folds=5,
        n_subfolds=4,
        n_reps=3,
        seed=42,
        **kwargs,
    ) -> None:
        self.all_edges = g.get_edges()
        # self.G = G.copy()  # probably not needed, as we work on the edge indices
        self.g = g
        self.folds_per_rep = defaultdict(dict)
        self.n_folds = n_folds
        self.n_subfolds = n_subfolds
        self.main_cv_splits = self.get_cv_realization(g, n_folds, seed=seed)

        self.seed = seed
        self.subseeds = defaultdict(dict)
        self.n_reps = n_reps
        self.sub_cv_splits = defaultdict(
            dict
        )  # key structure: fold_id -> rep_id -> subfold_id -> edge_filter

        self.cv_alpha_a = defaultdict(dict)
        self.cv_alpha_L = defaultdict(dict)
        # for idx, (train, test) in enumerate(self.kf.split(self.all_edges)):
        #     [idx] = g.new_edge_property("bool", val=True)
        #     for _ in test:
        #         self.edge_filter_dict[idx][self.all_edges[_]] = False

    @staticmethod
    def get_cv_realization(graph, n_splits, seed=None):
        # we do not control the random state here,
        # but ensure that each realization creates 5 trials
        # and each trial fairly fitted with an alpha to evaluate the accuracy
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        all_edges = graph.get_edges()
        _edge_filter_dict = {}

        for idx, (train, test) in enumerate(kf.split(all_edges)):
            _edge_filter_dict[idx] = graph.new_edge_property("bool", val=True)
            for _ in test:
                _edge_filter_dict[idx][all_edges[_]] = False
        return _edge_filter_dict

    def gen_train_validate_splits(self, subgraph, seed=None, **kwargs):
        fold_id = kwargs.get("fold_id", None)
        self.subseeds[fold_id] = np.random.randint(0, int(1e3), self.n_reps)
        for rep in range(self.n_reps):
            self.sub_cv_splits[fold_id][rep] = self.get_cv_realization(
                subgraph, self.n_subfolds, seed=self.subseeds[fold_id][rep]
            )

    def gen_all_train_validate_splits(self):
        for fold in range(self.n_folds):
            subgraph = gt.GraphView(self.g, efilt=self.main_cv_splits[fold])
            self.gen_train_validate_splits(subgraph, self.n_reps, fold_id=fold)

    def train_and_validate(self, model, fold_id, params=None, interp=None, **kwargs):
        self.model = model
        gv = gt.GraphView(self.g, efilt=self.main_cv_splits[fold_id])
        if self.model.method == "vanilla":
            list_alpha = params
            self.y_a, self.y_L = [], []  # just temp vals, for debugging

            for alpha in list_alpha:
                sig_a_list = []
                sig_L_list = []
                for rep in range(self.n_reps):
                    # five sub-folds (between train and validate)
                    for subfold in range(self.n_subfolds):
                        efilter = self.sub_cv_splits[fold_id][rep][subfold]
                        gvv = gt.GraphView(gv, efilt=efilter)
                        gvv_inv = gt.GraphView(gv, efilt=np.logical_not(efilter))
                        adj_gvv = gt.adjacency(gvv)

                        ranking = self.model.fit(gvv, alpha=alpha, printEvery=0)[
                            "primal"
                        ]
                        bloc0 = betaLocal(adj_gvv, ranking)
                        bloc1 = betaGlobal(adj_gvv, ranking)
                        sig_a, sig_L = compute_accuracy(
                            gt.adjacency(gvv_inv).toarray(), ranking, bloc0, bloc1
                        )
                        sig_a_list.append(sig_a)
                        sig_L_list.append(sig_L)
                # we want to minimize the negative accuracy
                self.y_a.append(-np.mean(sig_a_list))
                # we want to minimize the negative log likelihood
                self.y_L.append(-np.mean(sig_L_list))

            f_a = interp1d(
                list_alpha,
                self.y_a,
                kind="quadratic",
                fill_value="extrapolate",
                assume_sorted=True,
            )
            f_L = interp1d(
                list_alpha,
                self.y_L,
                kind="quadratic",
                fill_value="extrapolate",
                assume_sorted=True,
            )
            bnds = ((0, None),)
            cv_alpha_a = minimize(f_a, x0=1, bounds=bnds).x[0]
            cv_alpha_L = minimize(f_L, x0=1, bounds=bnds).x[0]
            # print(cv_alpha_a, cv_alpha_L)
            self.cv_alpha_a["vanilla"][fold_id] = cv_alpha_a
            self.cv_alpha_L["vanilla"][fold_id] = cv_alpha_L

        elif self.model.method == "annotated":

            list_alpha, list_lambd = params
            list_alpha_interp, list_lambd_interp = interp

            self.y_a, self.y_L = [], []  # just temp vals, for debugging
            for alpha in list_alpha:
                _y_a = []
                _y_L = []
                for lambd in list_lambd:
                    sig_a_list = []
                    sig_L_list = []
                    for rep in range(self.n_reps):
                        # five sub-folds (between train and validate)
                        for subfold in range(self.n_subfolds):
                            efilter = self.sub_cv_splits[fold_id][rep][subfold]
                            gvv = gt.GraphView(gv, efilt=efilter)
                            gvv_inv = gt.GraphView(gv, efilt=np.logical_not(efilter))
                            adj_gvv = gt.adjacency(gvv)
                            ranking = self.model.fit(
                                gvv, alpha=alpha, lambd=lambd, printEvery=0
                            )["primal"]
                            adj_gvv = gt.adjacency(gvv)
                            bloc0 = betaLocal(adj_gvv, ranking)
                            bloc1 = betaGlobal(adj_gvv, ranking)
                            sig_a, sig_L = compute_accuracy(
                                gt.adjacency(gvv_inv).toarray(), ranking, bloc0, bloc1
                            )
                            sig_a_list.append(sig_a)
                            sig_L_list.append(sig_L)
                    _y_a.append(-np.mean(sig_a_list))
                    _y_L.append(-np.mean(sig_L_list))
                self.y_a.append(_y_a)
                self.y_L.append(_y_L)
            interp_a = RegularGridInterpolator(
                (list_alpha, list_lambd), self.y_a, method="linear"
            )
            interp_L = RegularGridInterpolator(
                (list_alpha, list_lambd), self.y_L, method="linear"
            )

            X, Y = np.meshgrid(list_alpha_interp, list_lambd_interp, indexing="ij")
            points = np.array([X.ravel(), Y.ravel()]).T
            interpolated_values_a = interp_a(points).reshape(100, 100)

            min_value_a = np.min(interpolated_values_a)
            min_index_a = np.unravel_index(
                np.argmin(interpolated_values_a), interpolated_values_a.shape
            )
            min_point_a = (
                list_alpha_interp[min_index_a[0]],
                list_lambd_interp[min_index_a[1]],
            )

            interpolated_values_L = interp_L(points).reshape(100, 100)

            min_value_L = np.min(interpolated_values_L)
            min_index_L = np.unravel_index(
                np.argmin(interpolated_values_L), interpolated_values_L.shape
            )
            min_point_L = (
                list_alpha_interp[min_index_L[0]],
                list_lambd_interp[min_index_L[1]],
            )

            print(f"(a) Minimum value: {min_value_a} at point: {min_point_a}")
            print(f"(L) Minimum value: {min_value_L} at point: {min_point_L}")
            self.cv_alpha_a["annotated"][fold_id] = min_point_a
            self.cv_alpha_L["annotated"][fold_id] = min_point_L
