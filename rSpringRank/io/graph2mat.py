from scipy.sparse import csr_matrix, csc_matrix, find
import graph_tool.all as gt
import numpy as np
from scipy.linalg import sqrtm
from scipy.sparse.linalg import inv
from itertools import combinations
from math import comb
from collections import Counter


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
    elif type(data) is csr_matrix:
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
            np.zeros(num_nonzero * 2 + shape, dtype=np.float64) for _ in range(3)
        ]
    else:
        row, col, data = [np.zeros(num_nonzero * 2, dtype=np.float64) for _ in range(3)]
    row_b, col_b, data_b = [np.zeros(num_nonzero, dtype=np.float64) for _ in range(3)]
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
        B = csr_matrix((data, (row, col)), shape=(shape**2, shape), dtype=np.float64)
        b = csr_matrix((data_b, (row_b, col_b)), shape=(shape**2, 1), dtype=np.float64)
    else:
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
    Bt_B_inv = compute_Bt_B_inv(B)
    Bt_B_invSqrt = sqrtm(Bt_B_inv.toarray())

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
    B, b = cast2sum_squares_form(data, alpha, regularization=regularization)
    _ell = compute_ell(data)
    Bt_B_inv = compute_Bt_B_inv(B)
    Bt_B_invSqrt = sqrtm(Bt_B_inv.toarray())

    return {
        "B": B,  # in csr_matrix format and also is sparse
        "b": b,  # in csr_matrix format and also is sparse
        "ell": _ell,  # in csr_matrix format and is also sparse
        "Bt_B_inv": Bt_B_inv,  # in csr_matrix format, but it's actually dense
        "Bt_B_invSqrt": Bt_B_invSqrt,  # in np.ndarray for and is also dense
    }


def compute_Bt_B_inv(B):
    return inv(B.T @ B)  # only for sparse matrices


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
        ell = np.zeros([comb(len_classes, 2), len(g.get_vertices())], dtype=np.float64)
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
            (data, (row, col)),
            shape=(comb(len_classes, 2), len(g.get_vertices())),
            dtype=np.float64,
        )
    return ell
