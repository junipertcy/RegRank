from scipy.sparse import csc_matrix, csr_matrix, find

try:
    import graph_tool.all as gt
except ModuleNotFoundError:
    print("graph_tool not found. Please install graph_tool.")

from collections import Counter
from itertools import combinations
from math import comb

import numpy as np
from numba import njit, types
from numba.typed import Dict
from scipy.linalg import sqrtm
from scipy.sparse.linalg import inv


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
        for ind in zip(*find(A)):
            i, j, val = ind[0], ind[1], ind[2]
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
            data.append(-(val**0.5))  # TODO: check sign
            row.append(_row_t)
            col.append(j_t)
            data.append(val**0.5)

            # constant term
            row_b.append(_row_t)
            col_b.append(0)
            data_b.append(-(val**0.5))

        row += [_ for _ in range((t + 1) * (shape**2) - shape, (t + 1) * (shape**2))]
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
        B = csc_matrix(
            (data, (row, col)),
            shape=(T * shape**2, T * shape),
            dtype=np.float64,
        )
        b = csc_matrix(
            (data_b, (row_b, col_b)),
            shape=(T * shape**2, 1),
            dtype=np.float64,
        )
        B_T = csc_matrix(
            (data_T, (row_T, col_T)),
            shape=((T - 1) * shape, T * shape),
            dtype=np.float64,
        )
        return B, b, B_T
    else:
        B = csc_matrix(
            (data, (row, col)),
            shape=(T * shape**2 + (T - 1) * shape, T * shape),
            dtype=np.float64,
        )
        b = csc_matrix(
            (data_b, (row_b, col_b)),
            shape=(T * shape**2 + (T - 1) * shape, 1),
            dtype=np.float64,
        )
        return B, b, None


def cast2sum_squares_form(data, alpha, regularization=True):  # TODO: this is slow
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
    elif type(data) is csc_matrix:
        A = data

    # print(f"our method: adj = {A.toarray()[:5,:5]}")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Are you sure that A is asymmetric?")
    if type(A) not in [csr_matrix, csc_matrix]:
        raise TypeError(
            "Please make sure that A is of type `csr_matrix` or `csc_matrix` of scipy.sparse."
        )
    shape = A.shape[0]
    A_nonzero = A.nonzero()
    num_nonzero = A_nonzero[0].shape[0]
    if regularization:
        row, col, data = [
            np.zeros(num_nonzero * 2 + shape, dtype=np.float64, order="C")
            for _ in range(3)
        ]
    else:
        row, col, data = [
            np.zeros(num_nonzero * 2, dtype=np.float64, order="C") for _ in range(3)
        ]
    row_b, col_b, data_b = [
        np.zeros(num_nonzero, dtype=np.float64, order="C") for _ in range(3)
    ]
    counter_B = counter_b = 0
    # data_iter = iter(A.data)
    for ind in zip(*find(A)):
        i, j, val = ind[0], ind[1], ind[2]
        if i == j:
            # logger.warning(
            #     "WARNING: self-loop detected in the adjacency matrix. Ignoring..."
            # )
            continue
        if j < i:
            _row = i * (shape - 1) + j
        else:
            _row = i * (shape - 1) + j - 1

        row[counter_B] = _row
        col[counter_B] = i
        # val = next(data_iter)
        data[counter_B] = -(val**0.5)  # TODO: check sign
        counter_B += 1

        row[counter_B] = _row
        col[counter_B] = j
        data[counter_B] = val**0.5
        counter_B += 1

        row_b[counter_b] = _row
        col_b[counter_b] = 0
        data_b[counter_b] = -(val**0.5)
        counter_b += 1

    if regularization:
        __ = shape**2 - shape
        for _ in range(shape**2 - shape, shape**2):
            row[counter_B] = _
            col[counter_B] = _ - __
            data[counter_B] = alpha**0.5
            counter_B += 1
        B = csc_matrix((data, (row, col)), shape=(shape**2, shape), dtype=np.float64)
        b = csc_matrix((data_b, (row_b, col_b)), shape=(shape**2, 1), dtype=np.float64)
    else:
        B = csc_matrix(
            (data, (row, col)), shape=(shape**2 - shape, shape), dtype=np.float64
        )
        b = csc_matrix(
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
    Bt_B_inv = compute_Bt_B_inv(B)
    Bt_B_invSqrt = sqrtm(Bt_B_inv.toarray(order="C"))

    return {
        "B": B,
        "b": b,
        "ell": _ell,
        "Bt_B_inv": Bt_B_inv,
        "Bt_B_invSqrt": Bt_B_invSqrt,
    }


def compute_cache_from_data(data, alpha, regularization=True, **kwargs):
    """_summary_

    Args:

    data (_type_): _description_

    alpha (_type_): _description_

    regularization (bool, optional): _description_. Defaults to True.

    Returns:

    dictionary: _description_

    """
    goi = kwargs.get("goi", None)
    B, b = cast2sum_squares_form(data, alpha, regularization=regularization)
    _ell = compute_ell(data, key=goi)
    Bt_B_inv = compute_Bt_B_inv(B)
    Bt_B_invSqrt = sqrtm(Bt_B_inv.toarray(order="C"))  # C-order

    return {
        "B": B,  # in csc_matrix format and also is sparse
        "b": b,  # in csc_matrix format and also is sparse
        "ell": _ell,  # in csc_matrix format and is also sparse
        "Bt_B_inv": Bt_B_inv,  # in csc_matrix format, but it's actually dense
        "Bt_B_invSqrt": Bt_B_invSqrt,  # in np.ndarray for and is also dense
    }


def compute_Bt_B_inv(B):
    return inv((B.T @ B).tocsc())  # only for sparse matrices


def grad_g_star(B, b, v):
    return np.dot(compute_Bt_B_inv(B), v + B.T @ b)


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

    def vcond(v):
        return g.vp["vindex"][v] in node_indices

    return gt.GraphView(g, efilt=cond, vfilt=lambda v: vcond(v))


def compute_ell(g, key=None):
    if (type(g) is not gt.Graph) and (type(g) is not gt.GraphView):
        raise TypeError("g should be of type `graph_tool.Graph`.")
    if key is None:
        return None
    try:
        ctr_classes = Counter(g.vp[key])
    except KeyError:
        raise AttributeError(
            f"Key: {key} not found. Please provide a 'group of interest' for the vertex property, "
            + "so we can compute the annotated rankings."
        )
    _ctr_classes = dict(Counter({k: v**-1 for k, v in ctr_classes.items()}))
    ctr_classes = Dict.empty(key_type=types.int64, value_type=types.float64)
    for _k, value in _ctr_classes.items():
        ctr_classes[hash(_k)] = value
    len_classes = len(ctr_classes)
    comb_classes = np.array(
        list(combinations(ctr_classes, 2)), dtype=np.int64, order="C"
    )
    mb = np.array([hash(_) for _ in g.vp[key]], dtype=np.int64, order="C")
    dim_0 = comb(len_classes, 2)
    dim_1 = len(g.get_vertices())
    row = np.zeros((len_classes - 1) * dim_1, dtype=np.int64, order="C")
    col = np.zeros((len_classes - 1) * dim_1, dtype=np.int64, order="C")
    data = np.zeros((len_classes - 1) * dim_1, dtype=np.float64, order="C")
    counter = 0
    if dim_0 >= 185360:
        raise ValueError(
            "We currently do not support this many metatdata. "
            + "(You many need more than 256 GB on-board memory for this.) "
            + "Please choose a smaller GOI to analyze."
        )

    @njit
    def _compute_ell(row, col, data, counter, n_nodes, ctr_classes):
        for idx, (i, j) in enumerate(comb_classes):
            for _ in range(n_nodes):
                # sometimes we feed g as a gt.GraphView
                # in this case, vtx will return the (unfiltered) vertex id
                if mb[_] == i:
                    row[counter] = idx
                    col[counter] = _
                    data[counter] = -ctr_classes[i]
                    counter += 1
                elif mb[_] == j:
                    row[counter] = idx
                    col[counter] = _
                    data[counter] = ctr_classes[j]
                    counter += 1

    _compute_ell(row, col, data, counter, dim_1, ctr_classes)

    ell = csc_matrix(
        (data, (row, col)),
        shape=(dim_0, dim_1),
        dtype=np.float64,
    )
    return ell
