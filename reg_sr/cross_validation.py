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
from scipy.sparse import issparse, find, triu
from scipy.optimize import minimize_scalar, minimize
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

from reg_sr.utils import *
from reg_sr.losses import *
from reg_sr.regularizers import *
from reg_sr.experiments import *
import reg_sr
from reg_sr.fit import rSpringRank
import graph_tool.all as gt

from numba import njit


def shuffle(arr, seed=42):
    np.random.seed(seed)
    np.random.shuffle(arr)
    return arr


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


def test(graph, params):
    return


def _get_folds(num_folds, M=None, seed=42):
    # Size of each fold
    if M is None:
        raise ValueError("M must be specified.")
    else:
        foldSize = M // num_folds

    # Shuffle interactions
    idx = shuffle(np.arange(M), seed=seed)

    # Build K-1 num_folds of equal size
    folds = []
    for f in range(num_folds - 1):
        folds.append(idx[(f * foldSize) : ((f + 1) * foldSize)])

    # Put the remainder in the final Kth fold
    folds.append(idx[((num_folds - 1) * foldSize) :])
    return folds


def get_folds(G, num_folds, seed=None):
    if seed is None:
        seed = np.random.randint(0, 1000)
    else:
        seed = int(seed)

    A = gt.adjacency(G)
    r, c, _, m = get_interacting_pairs(A)
    folds = _get_folds(num_folds, M=m, seed=seed)
    return folds, r, c, seed


def split(G, num_folds, ind_folds, seed=None, return_graph=True):
    G = G.copy()
    folds, r, c, seed = get_folds(G, num_folds, seed=seed)
    # print(f"seed = {seed}.")

    # Build the test set of indices
    test_i = r[folds[ind_folds]]
    test_j = c[folds[ind_folds]]
    test_ij = np.stack((test_i, test_j), axis=-1)
    test_ji = np.stack((test_j, test_i), axis=-1)
    train_mask = G.new_edge_property("bool", val=True)
    test_mask = G.new_edge_property("bool", val=False)

    # Build the training set by setting test set interactions to zero
    train_mask.a[test_ij] = False
    train_mask.a[test_ji] = False
    test_mask.a[test_ij] = True
    test_mask.a[test_ji] = True

    if return_graph:
        G.set_edge_filter(train_mask)
        train_G = G.copy()
        G.set_edge_filter(test_mask)
        test_G = G.copy()
        return train_G, test_G, test_G.num_edges()

    # Build the test set
    G.set_edge_filter(test_mask)
    test_G = gt.adjacency(G)

    numTestEdges = G.num_edges()

    # Train SpringRank on the TRAIN set
    G.set_edge_filter(train_mask)
    train_G = gt.adjacency(G)
    return train_G, test_G, numTestEdges


def get_interacting_pairs(A):
    # Find interacting pairs
    r, c, v = find(triu(A + A.T))
    # Number of interacting pairs
    return r, c, v, len(v)


class CrossValidation(object):
    def __init__(self, G, model) -> None:
        self.G = G.copy()
        self.model = model

    def train_and_validate(self, num_folds, reps, params=None):
        # alpha_v = params.get("vanilla", np.logspace(-1, 2, 10))
        # alpha_a, lambd_a = params.get("annotated", (np.logspace(-1, 2, 10), np.logspace(-1, 2, 10)))
        if self.model.method == "vanilla":
            alpha_v = params
        elif self.model.method == "annotated":
            alpha_a, lambd_a = params

        list_alpha, y_a, y_L = [], [], []
        for alpha in alpha_v:
            list_alpha.append(alpha)
            _sig_a, _sig_L = self.train_and_validate_sub(num_folds, reps, params=alpha)
            y_a.append(-np.mean(_sig_a.reshape(1, -1)))
            y_L.append(-np.mean(_sig_L.reshape(1, -1)))
            # print(alpha, "-->", y_a, y_L)
        f_a = interp1d(list_alpha, y_a, kind='quadratic', fill_value='extrapolate', assume_sorted=True)
        f_L = interp1d(list_alpha, y_L, kind='quadratic', fill_value='extrapolate', assume_sorted=True)
        print(y_a)
        print(y_L)
        bnds = ((0, None),)
        cv_alpha_a = minimize(f_a, x0=1, bounds=bnds).x[0]
        cv_alpha_L = minimize(f_L, x0=1, bounds=bnds).x[0]
        print(f"cv_alpha_a = {cv_alpha_a}. || cv_alpha_L = {cv_alpha_L}.")


    def train_and_validate_sub(self, num_folds, reps, params=None):
        # Preallocate
        sig_a = np.zeros(reps * num_folds)
        sig_L = np.zeros(reps * num_folds)

        if self.model.method == "vanilla":
            alpha_v = params
        elif self.model.method == "annotated":
            alpha_a, lambd_a = params

        # Iterate over reps
        for rep in range(reps):
            seed = np.random.randint(0, 1000)
            # Iterate over folds
            for ind_folds in range(num_folds):
                # print(
                #     "Cross validation progress: Rep {}/{}, Fold {}/{}.".format(
                #         rep + 1, reps, ind_folds + 1, num_folds
                #     )
                # )

                # Bookkeeping
                foldrep = ind_folds + rep * num_folds
                train_G, validate_G, numTestEdges = split(self.G, num_folds, ind_folds, seed=seed)

                if self.model.method == "vanilla":
                    ranking = self.model.fit(train_G, alpha=alpha_v)["primal"]
                elif self.model.method == "annotated":
                    ranking = self.model.fit(train_G, alpha=alpha_a, lambd=lambd_a)["primal"]

                train_A = gt.adjacency(train_G)
                bloc0 = self.betaLocal(train_A, ranking)
                bglob0 = self.betaGlobal(train_A, ranking)

                # SpringRank accuracies on VALIDATION set
                validate_A = gt.adjacency(validate_G)
                validate_A = np.array(validate_A.todense(), dtype=np.float64)
                sig_a[foldrep] = self.localAccuracy(validate_A, ranking, bloc0)
                sig_L[foldrep] = (
                    -self.globalAccuracy(validate_A, ranking, bglob0) / numTestEdges
                )
        return sig_a, sig_L

    @staticmethod
    @njit(cache=True)
    def localAccuracy(A, s, b):
        m = np.sum(A)
        n = len(s)
        y = 0
        for i in range(n):
            for j in range(n):
                d = s[i] - s[j]
                p = (1 + np.exp(-2 * b * d)) ** (-1)
                y = y + abs(A[i, j] - (A[i, j] + A[j, i]) * p)
        # cleanup
        a = 1 - 0.5 * y / m
        return a

    @staticmethod
    @njit(cache=True)
    def globalAccuracy(A, s, b):
        n = len(s)
        y = 0
        for i in range(n):
            for j in range(n):
                d = s[i] - s[j]
                p = (1 + np.exp(-2 * b * d)) ** (-1)
                if p == 0 or p == 1:
                    pass
                else:
                    y = y + A[i, j] * np.log(p) + A[j, i] * np.log(1 - p)
        return y

    @staticmethod
    def betaLocal(A, s):
        M = np.array(A.todense(), dtype=np.float64)
        r = np.array(s, dtype=np.float64)
        b = minimize_scalar(lambda _: negacc(M, r, _), bounds=(1e-6, 1000)).x
        return b

    @staticmethod
    def betaGlobal(A, s):
        M = np.array(A.todense(), dtype=np.float64)
        r = np.array(s, dtype=np.float64)
        b = minimize_scalar(lambda _: f(M, r, _) ** 2, bounds=(1e-6, 1000)).x
        return b
