# import graph_tool.all as gt
import numpy as np
from numba import jit
from scipy.sparse import csr_matrix, csc_matrix, issparse
from scipy.sparse.linalg import inv
from itertools import combinations
from math import comb
from collections import Counter
from scipy.linalg import svd
from scipy.sparse.linalg import svds
import linecache
from random import sample

from logging import getLogger

logger = getLogger(__name__)
# from numpy.random import default_rng
# import scipy.sparse


def cast2sum_squares_form_t(
    g, alpha, lambd, from_year=1960, to_year=1961, top_n=70, separate=False
):
    """Operator to linearize the sum of squares loss function.

    Args:
        g (_type_): _description_
        alpha (_type_): _description_
        lambd (_type_): _description_
        from_year (int, optional): _description_. Defaults to 1960.
        to_year (int, optional): _description_. Defaults to 1961.
        top_n (int, optional): _description_. Defaults to 70.
        separate (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_
        ValueError: _description_
        TypeError: _description_

    Returns:
        _type_: _description_
    """
    if type(g) is not gt.Graph:
        raise TypeError("g should be of type `graph_tool.Graph`.")
    if from_year >= to_year:
        raise ValueError("from_year should be smaller than to_year")

    row, col, data = [], [], []
    row_b, col_b, data_b = [], [], []
    if separate:
        row_T, col_T, data_T = [], [], []
    T = to_year - from_year + 1
    for t in range(0, T):
        u = filter_by_year(
            g, from_year=from_year + t, to_year=from_year + t + 1, top_n=top_n
        )
        A = gt.adjacency(u)
        shape = A.shape[0]

        if A.shape[0] != A.shape[1]:
            raise ValueError("Are you sure that A is asymmetric?")
        if type(A) not in [csr_matrix, csc_matrix]:
            raise TypeError(
                "Please make sure that A is of type `csr_matrix` or `csc_matrix` of scipy.sparse."
            )
        for ind in zip(*A.nonzero()):
            i, j = ind[0], ind[1]
            if i == j:
                continue
            if j < i:
                _row = i * (shape - 1) + j
            else:
                _row = i * (shape - 1) + j - 1
            _row_t = _row + t * (shape**2)
            i_t = i + t * shape
            j_t = j + t * shape

            row.append(_row_t)
            col.append(i_t)
            data.append(-A[ind] ** 0.5)  # TODO: check sign
            row.append(_row_t)
            col.append(j_t)
            data.append(A[ind] ** 0.5)

            # constant term
            row_b.append(_row_t)
            col_b.append(0)
            data_b.append(-A[ind] ** 0.5)

        row += [
            _ for _ in range((t + 1) * (shape**2) - shape, (t + 1) * (shape**2))
        ]
        col += [_ for _ in range(t * shape, (t + 1) * shape)]
        data += [alpha**0.5] * shape

        # Note that you do not need to specify zeros, since the default value is zero.
        # row_b += [
        #     _ for _ in range((t + 1) * (shape**2) - shape, (t + 1) * (shape**2))
        # ]
        # col_b += [0] * shape
        # data_b += [0] * shape

        # regularize-over-time term
        if t < T - 1:
            _row = [
                _
                for _ in range(
                    T * shape**2 + t * shape, T * shape**2 + shape + t * shape
                )
            ]
            _col_t = [_ for _ in range(t * shape, (t + 1) * shape)]
            _col_t_plus_1 = [_ for _ in range((t + 1) * shape, ((t + 1) + 1) * shape)]
            if separate:
                shift = T * shape**2
                row_T += [(_ - shift) for _ in _row]
                col_T += _col_t
                data_T += [lambd**0.5] * shape

                row_T += [(_ - shift) for _ in _row]
                col_T += _col_t_plus_1
                data_T += [-(lambd**0.5)] * shape
            else:
                row += _row
                col += _col_t
                data += [lambd**0.5] * shape

                row += _row
                col += _col_t_plus_1
                data += [-(lambd**0.5)] * shape
    if separate:
        B = csr_matrix(
            (data, (row, col)),
            shape=(T * shape**2, T * shape),
            dtype=np.float64,
        )
        b = csr_matrix(
            (data_b, (row_b, col_b)),
            shape=(T * shape**2, 1),
            dtype=np.float64,
        )
        B_T = csr_matrix(
            (data_T, (row_T, col_T)),
            shape=((T - 1) * shape, T * shape),
            dtype=np.float64,
        )
        return B, b, B_T
    else:
        B = csr_matrix(
            (data, (row, col)),
            shape=(T * shape**2 + (T - 1) * shape, T * shape),
            dtype=np.float64,
        )
        b = csr_matrix(
            (data_b, (row_b, col_b)),
            shape=(T * shape**2 + (T - 1) * shape, 1),
            dtype=np.float64,
        )
        return B, b, None


def cast2sum_squares_form(data, alpha, regularization=True):
    """
    This is how we linearize the objective function:
    B_ind  i  j
    0      0  1
    1      0  2
    2      0  3
    3      1  0
    4      1  2
    5      1  3
    6      2  0
    ...
    11     3  2
    12     0  0
    13     1  1
    14     2  2
    15     3  3
    """
    if type(data) is gt.Graph or type(data) is gt.GraphView:
        A = gt.adjacency(data)
    elif type(data) is csr_matrix:
        A = data
    else:
        raise TypeError(
            "Please make sure that data is of type `graph_tool.Graph` or `csr_matrix` of scipy.sparse."
        )
    # print(f"our method: adj = {A.todense()[:5,:5]}")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Are you sure that A is asymmetric?")
    if type(A) not in [csr_matrix, csc_matrix]:
        raise TypeError(
            "Please make sure that A is of type `csr_matrix` or `csc_matrix` of scipy.sparse."
        )
    shape = A.shape[0]
    row, col, data = [], [], []
    row_b, col_b, data_b = [], [], []
    for ind in zip(*A.nonzero()):
        i, j = ind[0], ind[1]
        if i == j:
            # logger.warning(
            #     "WARNING: self-loop detected in the adjacency matrix. Ignoring..."
            # )
            continue
        if j < i:
            _row = i * (shape - 1) + j
        else:
            _row = i * (shape - 1) + j - 1
        row.append(_row)
        col.append(i)
        data.append(-A[ind] ** 0.5)  # TODO: check sign
        row.append(_row)
        col.append(j)
        data.append(A[ind] ** 0.5)

        row_b.append(_row)
        col_b.append(0)
        data_b.append(-A[ind] ** 0.5)

    if regularization:
        row += [_ for _ in range(shape**2 - shape, shape**2)]
        col += [_ for _ in range(shape)]
        data += [alpha**0.5] * shape

        # Note that you do not need to specify zeros, since the default value is zero.
        # row_b += [_ for _ in range(shape**2 - shape, shape**2)]
        # col_b += [0] * shape
        # data_b += [0] * shape
        B = csr_matrix((data, (row, col)), shape=(shape**2, shape), dtype=np.float64)
        b = csr_matrix(
            (data_b, (row_b, col_b)), shape=(shape**2, 1), dtype=np.float64
        )
    else:
        # logger.warning("WARNING: no regularization is used. Are you sure?")
        B = csr_matrix(
            (data, (row, col)), shape=(shape**2 - shape, shape), dtype=np.float64
        )
        b = csr_matrix(
            (data_b, (row_b, col_b)), shape=(shape**2 - shape, 1), dtype=np.float64
        )
    return B, b


def compute_cache_from_data_t(
    data, alpha=1, lambd=1, from_year=1960, to_year=1961, top_n=70
):
    B, b, _ell = cast2sum_squares_form_t(
        data,
        alpha,
        lambd=lambd,
        from_year=from_year,
        to_year=to_year,
        top_n=top_n,
        separate=True,
    )

    # for timewise setting, we always use sparse matrix
    sparse = True

    # somehow, sparse Bt_B_inv runs faster
    Bt_B_inv = compute_Bt_B_inv(B, sparse=True)
    # _, s, Vh = svd(B.todense(), full_matrices=False)
    if type(B) is not csr_matrix:
        _, s, Vh = svd(B.todense(), full_matrices=False)
    else:
        # this is much slower than the dense counterpart
        _, s, Vh = svds(
            B, k=min(B.shape) - 1, solver="arpack"
        )  # implement svd for sparse matrix
    Bt_B_invSqrt = Vh.T @ np.diag(1 / s) @ Vh

    return {
        "B": B,
        "b": b,
        "ell": _ell,
        "Bt_B_inv": Bt_B_inv,
        "Bt_B_invSqrt": Bt_B_invSqrt,
    }


def compute_cache_from_data(data, alpha, regularization=True):
    """_summary_

    Args:

    data (_type_): _description_

    alpha (_type_): _description_

    regularization (bool, optional): _description_. Defaults to True.

    Returns:

    dictionary: _description_

    """
    import datetime

    B, b = cast2sum_squares_form(data, alpha, regularization=regularization)
    _ell = compute_ell(data)
    Bt_B_inv = compute_Bt_B_inv(B, sparse=True)  # expensive step
    if not issparse(B):  # we do not go this path
        _, s, Vh = svd(B, full_matrices=False)
    else:
        _, s, Vh = svds(
            B, k=min(B.shape) - 1, solver="arpack"
        )  # implement svd for sparse matrix
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
        return np.linalg.inv(B.T @ B)
    return inv(B.T @ B)


def grad_g_star(B, b, v):
    return np.dot(compute_Bt_B_inv(B), v + np.dot(B.T, b))


def compute_ell(g, sparse=True):
    if (type(g) is not gt.Graph) and (type(g) is not gt.GraphView):
        raise TypeError("g should be of type `graph_tool.Graph`.")
    try:
        ctr_classes = Counter(g.vp["goi"])
    except KeyError:
        return None
    len_classes = len(ctr_classes)
    comb_classes = combinations(ctr_classes, 2)
    mb = list(g.vp["goi"])

    if sparse:
        row, col, data = [], [], []
    else:
        ell = np.zeros([comb(len_classes, 2), len(g.get_vertices())])
    for idx, (i, j) in enumerate(comb_classes):
        for _, vtx in enumerate(g.vertices()):
            # sometimes we feed g as a gt.GraphView
            # in this case, vtx will return the (unfiltered) vertex id
            if mb[_] == i:
                if sparse:
                    row.append(idx)
                    col.append(_)
                    data.append(-ctr_classes[i] ** -1)
                else:
                    ell[idx][_] = -ctr_classes[i] ** -1
            elif mb[_] == j:
                if sparse:
                    row.append(idx)
                    col.append(_)
                    data.append(ctr_classes[j] ** -1)
                else:
                    ell[idx][_] = ctr_classes[j] ** -1
    if sparse:
        ell = csr_matrix(
            (data, (row, col)), shape=(comb(len_classes, 2), len(g.get_vertices()))
        )
    return ell


def compute_spearman_correlation(g, s):
    return


def render_ijwt(
    path="./data/PhD Exchange Network Data/PhD_exchange.txt",
    delimiter=" ",
):
    g = gt.Graph()
    vname = g.new_vp("string")
    vindex = g.new_vp("int")
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
                [
                    (name2id[ijwt[1]], name2id[ijwt[0]], ijwt[2], time2id[ijwt[3]])
                ],  # note the source / target order
                eprops=[eweight, etime],
            )
    g.edge_properties["eweight"] = eweight
    g.edge_properties["etime"] = etime
    id2name = {v: k for k, v in name2id.items()}

    school_name = lambda n: linecache.getline(
        "./data/PhD Exchange Network Data/school_names.txt", n
    ).replace("\n", "")[:-1]
    # print(school_name(165))  # >> University of Michigan
    for vertex in g.vertices():
        vname[vertex] = school_name(int(id2name[vertex]))
        # print(vname[vertex], vertex, id2name[vertex])
        vindex[vertex] = vertex.__int__()

    g.vertex_properties["vname"] = vname
    g.vertex_properties["vindex"] = vindex
    # print(name2id)

    return g


def filter_by_time(g, time):
    mask_e = g.new_edge_property("bool")
    for edge in g.edges():
        if g.ep["etime"][edge] == time:
            mask_e[edge] = 1
        else:
            mask_e[edge] = 0
    return mask_e


def add_erroneous_edges(g, nid=0, times=1, method="single_point_mutation"):
    if method == "single_point_mutation":
        # scenario 1: add "single point mutation"
        for _ in range(int(times)):
            for node in range(g.num_vertices()):
                if node != nid:
                    # nid always endorsed by others
                    g.add_edge(node, nid) 

    elif method == "random_edges":
        # scenario 2: add random edges
        for _ in range(int(times)):
            src, tar = sample(range(g.num_vertices()), 2)
            g.add_edge(src, tar)
    else:
        raise NotImplementedError(
            "method should be either `single_point_mutation` or `random_edges`."
        )
    return g


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


# for use of the PhD Exchange data set
def filter_by_year(g, from_year=1946, to_year=2006, top_n=70):
    if to_year <= from_year:
        raise ValueError("to_year must be greater than from_year")

    from_year_ind = from_year - 1946
    to_year_ind = to_year - 1946

    eb = g.ep["etime"]
    cond_0 = eb.a >= from_year_ind
    cond_1 = eb.a < to_year_ind  # notice that no equal sign here

    cond = cond_0 & cond_1

    # todo: check if "in" or "out" (I think it is "in")
    node_indices = np.argsort(g.degree_property_map("in").a, axis=0)[-top_n:]
    vcond = lambda v: g.vp["vindex"][v] in node_indices

    return gt.GraphView(g, efilt=cond, vfilt=lambda v: vcond(v))
