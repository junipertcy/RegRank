import graph_tool.all as gt
import numpy as np
from numba import jit
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import inv
from itertools import combinations
from math import comb
from collections import Counter


# from numpy.random import default_rng
# import scipy.sparse

def compute_spearman_correlation(g, s):
    return 



def render_ijwt(
    path="./etc/prestige_reinforcement/data/PhD Exchange Network Data/PhD_exchange.txt",
    delimiter=" ",
):
    g = gt.Graph()
    eweight = g.new_ep("double")
    etime = g.new_ep("int")

    name2id = dict()
    time2id = dict()
    nameid = 0
    timeid = 0

    with open(path, "r") as f:
        for line in f:
            ijwt = line.replace("\n", "").split(delimiter)[:4]

            try:
                name2id[ijwt[0]]
            except KeyError:
                name2id[ijwt[0]] = nameid
                nameid += 1

            try:
                name2id[ijwt[1]]
            except KeyError:
                name2id[ijwt[1]] = nameid
                nameid += 1

            try:
                time2id[ijwt[3]]
            except KeyError:
                time2id[ijwt[3]] = timeid
                timeid += 1

            g.add_edge_list(
                [(name2id[ijwt[0]], name2id[ijwt[1]], ijwt[2], time2id[ijwt[3]])],
                eprops=[eweight, etime],
            )
    g.edge_properties["eweight"] = eweight
    g.edge_properties["etime"] = etime
    return g


def filter_by_time(g, time):
    mask_e = g.new_edge_property("bool")
    for edge in g.edges():
        if g.ep["etime"][edge] == time:
            mask_e[edge] = 1
        else:
            mask_e[edge] = 0
    return mask_e


def D_operator(s):
    # output = np.zeros(n**2)
    n = len(s)
    output = np.zeros(n**2 - n)  # if we avoid the zero rows
    k = 0
    for i in range(n):
        for j in range(i):
            output[k] = s[i] - s[j]
            k += 1
        # Avoid letting j = i since that's just s[i] - s[i] so it's a zero row
        for j in range(i + 1, n):
            output[k] = s[i] - s[j]
            k += 1
        # for j in range(n):
        #     # k = i + j*n
        #     output[k] = s[i] - s[j]
        #     k += 1
    return output


def D_operator_reg_t_sparse(a, s):
    if type(a) is not csr_matrix:
        raise TypeError("Please use a `csr_matrix` of scipy.sparse.")
    n = a.shape[0]
    output_t = np.zeros(n)  # if we avoid the zero rows

    for ind in zip(*a.nonzero()):
        i, j = ind[0], ind[1]
        if i < j:
            k = n * i + j - i - 1
        elif i > j:
            k = n * i + j - i

        output_t[i] += (a[i, j] ** 0.5) * s[k]
        output_t[j] -= (a[i, j] ** 0.5) * s[k]
    return output_t


@jit(nopython=True)
def D_operator_reg_t(a, s):
    n = len(a)
    output_t = np.zeros(n)  # if we avoid the zero rows
    k = 0
    for i in range(n):
        for j in range(i):
            output_t[i] += (a[i, j] ** 0.5) * s[k]
            output_t[j] -= (a[i, j] ** 0.5) * s[k]
            k += 1
        # Avoid letting j = i since that's just s[i] - s[i] so it's a zero row
        for j in range(i + 1, n):
            output_t[i] += (a[i, j] ** 0.5) * s[k]
            output_t[j] -= (a[i, j] ** 0.5) * s[k]
            k += 1
    return output_t


def D_operator_reg_sparse(a, s):
    if type(a) is not csr_matrix:
        raise TypeError("Please use a `csr_matrix` of scipy.sparse.")
    n = a.shape[0]
    output = np.zeros(n**2 - n)  # if we avoid the zero rows
    for ind in zip(*a.nonzero()):
        i, j = ind[0], ind[1]
        if i < j:
            k = n * i + j - i - 1
        elif i > j:
            k = n * i + j - i
        output[k] = (a[i, j] ** 0.5) * (s[i] - s[j])
    return output


@jit(nopython=True)
def D_operator_reg(a, s):
    n = len(a)
    output = np.zeros(n**2 - n)  # if we avoid the zero rows
    k = 0
    for i in range(n):
        for j in range(i):
            output[k] = (a[i, j] ** 0.5) * (s[i] - s[j])
            k += 1
        # Avoid letting j = i since that's just s[i] - s[i] so it's a zero row
        for j in range(i + 1, n):
            output[k] = (a[i, j] ** 0.5) * (s[i] - s[j])
            k += 1
    return output


def D_operator_b_sparse(a):
    if type(a) is not csr_matrix:
        raise TypeError("Please use a `csr_matrix` of scipy.sparse.")
    n = a.shape[0]
    output = np.zeros(n**2 - n)  # if we avoid the zero rows
    for ind in zip(*a.nonzero()):
        i, j = ind[0], ind[1]
        if i < j:
            k = n * i + j - i - 1
        elif i > j:
            k = n * i + j - i
        output[k] = a[ind] ** 0.5
    return output


@jit(nopython=True)
def D_operator_b(a):
    n = len(a)
    output = np.zeros(n**2 - n)  # if we avoid the zero rows
    # k = n
    k = 0
    for i in range(n):
        for j in range(i):
            output[k] = a[i, j] ** 0.5

            # raise Exception(i, j, k, n * i + j - i - 1)
            k += 1
        # Avoid letting j = i since that's just s[i] - s[i] so it's a zero row
        for j in range(i + 1, n):
            output[k] = a[i, j] ** 0.5
            k += 1
    return output


def implicit2explicit(f, a, m, n):
    """assumes f(x) is a linear operator (x has size n)
    so it can be represented f(x) = A*x for some matrix x
    (for now, assume A is square for simplicity)
    A = A * identity
    """
    e = np.zeros(n)  # length n vector
    A = np.zeros((m, n))  # (n ** 2 - n) x n matrix
    for i in range(n):
        # Loop over columns of identity
        e[i] = 1
        output = f(a, e)
        A[:, i] = output
        e[i] = 0
    return A
