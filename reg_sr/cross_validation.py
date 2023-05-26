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

from numba import njit


def shuffle(arr):
    np.random.shuffle(arr)
    return arr


def crossValidation(G, folds, reps):
    A = gt.adjacency(G)

    # Find interacting pairs
    r, c, v = find(triu(A + A.T))

    # Number of interacting pairs
    M = len(v)

    # Size of each fold
    foldSize = M // folds

    # Preallocate
    sig_a = np.zeros((reps * folds, 2))
    sig_L = np.zeros((reps * folds, 2))

    # Iterate over reps
    for rep in range(reps):
        # Shuffle interactions
        idx = shuffle(np.arange(M))

        # Build K-1 folds of equal size
        fold = []
        for f in range(folds - 1):
            fold.append(idx[(f * foldSize) : ((f + 1) * foldSize)])

        # Put the remainder in the final Kth fold
        fold.append(idx[((folds - 1) * foldSize) :])

        # Iterate over folds
        for f in range(folds):
            # Print
            print(
                "Cross validation progress: Rep {}/{}, Fold {}/{}.".format(
                    rep + 1, reps, f + 1, folds
                )
            )

            # Bookkeeping
            foldrep = f + rep * folds

            # Build the test set of indices
            test_i = r[fold[f]]
            test_j = c[fold[f]]
            test_ij = np.stack((test_i, test_j), axis=-1)
            test_ji = np.stack((test_j, test_i), axis=-1)
            train_mask = G.new_edge_property("bool", val=True)
            test_mask = G.new_edge_property("bool", val=False)

            # Build the training set by setting test set interactions to zero
            train_mask.a[test_ij] = False
            train_mask.a[test_ji] = False
            test_mask.a[test_ij] = True
            test_mask.a[test_ji] = True

            # Build the test set
            G.set_edge_filter(test_mask)
            TEST = gt.adjacency(G)

            numTestEdges = G.num_edges()

            # Train SpringRank on the TRAIN set
            G.set_edge_filter(train_mask)
            TRAIN = gt.adjacency(G)

            # x, y_a, y_L = [], [], []
            # for _alpha in np.logspace(-1, 2, 10):
            #     x.append(_alpha)
            #     _sig_a, _sig_L = _crossValidation(G, 2, 1, _alpha)
            #     y_a.append(-np.mean(_sig_a.reshape(1, -1)))
            #     # y_L.append(-np.mean(_sig_L.reshape(1, -1)))
            #     print(_alpha, "-->", y_a, y_L)
            # f_a = interp1d(x, y_a, kind='linear', fill_value="extrapolate")
            # # f_L = interp1d(x, y_L, kind='linear', fill_value="extrapolate")
            # cv_alpha = minimize(f_a, x0=0).x[0]
            cv_alpha = 0.01
            rsp = rSpringRank(method="vanilla")
            s0 = rsp.fit(G, alpha=cv_alpha)["primal"]

            # s0 = springRank(TRAIN)
            bloc0 = betaLocal(TRAIN, s0)
            bglob0 = betaGlobal(TRAIN, s0)

            # SpringRank accuracies on TEST set
            TEST = np.array(TEST.todense(), dtype=np.float64)
            sig_a[foldrep, 0] = localAccuracy(TEST, s0, bloc0)
            sig_L[foldrep, 0] = -globalAccuracy(TEST, s0, bglob0) / numTestEdges

            # # Train regularized SpringRank on the TRAIN set
            # x, y_a, y_L = [], [], []
            # for _alpha in np.logspace(-1, 2, 10):
            #     x.append(_alpha)
            #     _sig_a, _sig_L = _crossValidation2(G, 2, 1, _alpha)
            #     y_a.append(-np.mean(_sig_a.reshape(1, -1)))
            #     # y_L.append(-np.mean(_sig_L.reshape(1, -1)))
            #     print(_alpha, "-->", y_a, y_L)
            # f_a = interp1d(x, y_a, kind='linear', fill_value="extrapolate")
            # # f_L = interp1d(x, y_L, kind='linear', fill_value="extrapolate")
            # cv_alpha = minimize(f_a, x0=0).x[0]
            cv_alpha = 0.01
            rsp = rSpringRank(method="annotated")
            s1 = rsp.fit(G, alpha=cv_alpha, lambd=10)["primal"]
            # s2 = springRankFull(TRAIN, 2)
            bloc1 = betaLocal(TRAIN, s1)
            bglob1 = betaGlobal(TRAIN, s1)

            # # Regularized SpringRank accuracies on TEST set
            sig_a[foldrep, 1] = localAccuracy(TEST, s1, bloc1)
            sig_L[foldrep, 1] = -globalAccuracy(TEST, s1, bglob1) / numTestEdges
    return sig_a, sig_L


def _crossValidation(G, folds, reps, alpha):
    A = gt.adjacency(G)
    r, c, v = find(triu(A + A.T))
    M = len(v)
    foldSize = M // folds
    sig_a = np.zeros((reps * folds, 1))
    sig_L = np.zeros((reps * folds, 1))
    np.random.seed(0)
    for rep in range(reps):
        idx = shuffle(np.arange(M))
        fold = []
        for f in range(folds - 1):
            fold.append(idx[(f * foldSize) : ((f + 1) * foldSize)])
        fold.append(idx[((folds - 1) * foldSize) :])
        for f in range(folds):
            print(
                "training (+validating) progress: Rep {}/{}, Fold {}/{}.".format(
                    rep + 1, reps, f + 1, folds
                )
            )
            foldrep = f + rep * folds

            test_i = r[fold[f]]
            test_j = c[fold[f]]
            test_ij = np.stack((test_i, test_j), axis=-1)
            test_ji = np.stack((test_j, test_i), axis=-1)
            train_mask = G.new_edge_property("bool", val=True)
            validation_mask = G.new_edge_property("bool", val=False)
            train_mask.a[test_ij] = False
            train_mask.a[test_ji] = False
            validation_mask.a[test_ij] = True
            validation_mask.a[test_ji] = True

            G.set_edge_filter(validation_mask)
            TEST = gt.adjacency(G)

            numTestEdges = G.num_edges()

            G.set_edge_filter(train_mask)
            TRAIN = gt.adjacency(G)

            rsp = rSpringRank(method="vanilla")
            s0 = rsp.fit(G, alpha=alpha)["primal"]

            bloc0 = betaLocal(TRAIN, s0)
            bglob0 = betaGlobal(TRAIN, s0)

            # SpringRank accuracies on TEST set
            TEST = np.array(TEST.todense(), dtype=np.float64)
            sig_a[foldrep, 0] = localAccuracy(TEST, s0, bloc0)
            sig_L[foldrep, 0] = -globalAccuracy(TEST, s0, bglob0) / numTestEdges

    return sig_a, sig_L


# def _crossValidation2(G, folds, reps, alpha):
#     A = gt.adjacency(G)
#     r, c, v = find(triu(A + A.T))
#     M = len(v)
#     foldSize = M // folds
#     sig_a = np.zeros((reps * folds, 1))
#     sig_L = np.zeros((reps * folds, 1))
#     np.random.seed(0)
#     for rep in range(reps):
#         idx = shuffle(np.arange(M))
#         fold = []
#         for f in range(folds - 1):
#             fold.append(idx[(f * foldSize) : ((f + 1) * foldSize)])
#         fold.append(idx[((folds - 1) * foldSize) :])
#         for f in range(folds):
#             print(
#                 "training (+validating) progress: Rep {}/{}, Fold {}/{}.".format(
#                     rep + 1, reps, f + 1, folds
#                 )
#             )
#             foldrep = f + rep * folds

#             test_i = r[fold[f]]
#             test_j = c[fold[f]]
#             test_ij = np.stack((test_i, test_j), axis=-1)
#             test_ji = np.stack((test_j, test_i), axis=-1)
#             train_mask = G.new_edge_property("bool", val=True)
#             validation_mask = G.new_edge_property("bool", val=False)
#             train_mask.a[test_ij] = False
#             train_mask.a[test_ji] = False
#             validation_mask.a[test_ij] = True
#             validation_mask.a[test_ji] = True

#             G.set_edge_filter(validation_mask)
#             TEST = gt.adjacency(G)

#             numTestEdges = G.num_edges()

#             G.set_edge_filter(train_mask)
#             TRAIN = gt.adjacency(G)

#             rsp = rSpringRank(method="annotated")
#             s0 = rsp.fit(G, alpha=alpha, lambd=1)["primal"]

#             bloc0 = betaLocal(TRAIN, s0)
#             bglob0 = betaGlobal(TRAIN, s0)

#             # SpringRank accuracies on TEST set
#             TEST = np.array(TEST.todense(), dtype=np.float64)
#             sig_a[foldrep, 0] = localAccuracy(TEST, s0, bloc0)
#             sig_L[foldrep, 0] = -globalAccuracy(TEST, s0, bglob0) / numTestEdges

#     return sig_a, sig_L


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


def betaLocal(A, s):
    M = np.array(A.todense(), dtype=np.float64)
    r = np.array(s, dtype=np.float64)
    b = minimize_scalar(lambda _: negacc(M, r, _), bounds=(1e-6, 1000)).x
    return b


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


def betaGlobal(A, s):
    M = np.array(A.todense(), dtype=np.float64)
    r = np.array(s, dtype=np.float64)
    b = minimize_scalar(lambda _: f(M, r, _) ** 2, bounds=(1e-6, 1000)).x
    return b


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
