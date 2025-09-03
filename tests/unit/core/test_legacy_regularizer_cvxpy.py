import numpy as np
import pytest
from omegaconf import OmegaConf

from regrank.regularizers.legacy import LegacyRegularizer
from regrank.utils import generate_dag
from regrank.utils.datasets import small_graph

try:
    import cvxpy as cp
except ImportError:
    cp = None

# Ignore the specific UserWarning from the SCS solver.
# This filter applies to all tests within this file.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Converting A to a CSC.*:UserWarning",
    "ignore:Converting P to a CSC.*:UserWarning",
)


# Test with general-purpose conic solvers. OSQP is too specialized
# and can fail the problem reduction chain.
# Just for fun: We are adding commercial solvers here.
# Note that they may be free for personal or academic use.
@pytest.mark.parametrize(
    "backend",
    ["ECOS", "CLARABEL", "SCS", "CVXOPT", "PROXQP", "MOSEK", "GUROBI", "CPLEX"],
)
def test_legacy_regularizer_cvxpy_small_graph(backend):
    """
    Tests the legacy regularizer with CVXPY on the small graph,
    parameterized by the backend solver.
    """
    if cp is None:
        pytest.skip("CVXPY not installed, skipping test.")

    if backend not in cp.installed_solvers():
        pytest.skip(f"CVXPY backend '{backend}' not installed, skipping test.")

    cfg = OmegaConf.create({
        "alpha": 0.1,
        "regularizer": {"name": "legacy", "solver_method": "cvxpy"},
        "solver": {
            "verbose": False,
            "backend": backend,
        },
    })

    regularizer = LegacyRegularizer(cfg.regularizer)
    graph_data = small_graph()

    result = regularizer.fit(graph_data, cfg)

    assert "primal" in result
    assert result["primal"].shape == (graph_data.num_vertices(),)

    expected_ranks = np.array([-1.28630705, -0.41493776, 0.41493776, 1.28630705])
    assert np.allclose(result["primal"], expected_ranks, atol=1e-2)


@pytest.mark.parametrize(
    "backend",
    ["ECOS", "CLARABEL", "SCS", "CVXOPT", "PROXQP", "MOSEK", "GUROBI", "CPLEX"],
)
def test_legacy_regularizer_cvxpy_larger_dag(backend):
    """
    Tests the legacy regularizer with CVXPY on a larger DAG,
    parameterized by the backend solver.
    """
    if cp is None:
        pytest.skip("CVXPY not installed, skipping test.")

    if backend not in cp.installed_solvers():
        pytest.skip(f"CVXPY backend '{backend}' not installed, skipping test.")

    cfg = OmegaConf.create({
        "alpha": 0.5,
        "regularizer": {"name": "legacy", "solver_method": "cvxpy"},
        "solver": {
            "verbose": False,
            "backend": backend,
        },
    })

    regularizer = LegacyRegularizer(cfg.regularizer)
    graph_data, _ = generate_dag(10, 30, seed=69)

    result = regularizer.fit(graph_data, cfg)

    assert "primal" in result
    assert result["primal"].shape == (graph_data.num_vertices(),)

    expected_ranks = np.array([
        -0.31913913,
        0.22970351,
        -0.49676754,
        -0.76805986,
        0.47619272,
        -0.15233272,
        0.71925216,
        0.62366846,
        -0.62886078,
        0.31634318,
    ])
    assert np.allclose(result["primal"], expected_ranks, atol=1e-2)
