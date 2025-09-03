import numpy as np
from omegaconf import OmegaConf

from regrank.regularizers.legacy import LegacyRegularizer
from regrank.utils import generate_dag
from regrank.utils.datasets import small_graph


def test_legacy_regularizer_lsmr():
    """Tests the legacy regularizer with the lsmr solver."""
    cfg = OmegaConf.create({
        "alpha": 0.1,
        "regularizer": {"name": "legacy", "solver_method": "lsmr"},
        # Add a solver block for consistency
        "solver": {"verbose": False},
    })

    # FIX: Pass only the 'regularizer' sub-config to the constructor.
    regularizer = LegacyRegularizer(cfg.regularizer)
    graph_data = small_graph()

    # The 'fit' method receives the full config to access 'alpha'.
    result = regularizer.fit(graph_data, cfg)

    assert "primal" in result
    assert result["primal"].shape == (graph_data.num_vertices(),)
    expected_ranks = np.array([-1.28630705, -0.41493776, 0.41493776, 1.28630705])
    assert np.allclose(result["primal"], expected_ranks, atol=1e-2)


def test_legacy_regularizer_lsmr_larger_dag():
    """Tests the legacy regularizer with the lsmr solver on a larger DAG."""
    cfg = OmegaConf.create({
        "alpha": 0.5,
        "regularizer": {"name": "legacy", "solver_method": "lsmr"},
        "solver": {"verbose": False},
    })

    # FIX: Pass the correct sub-config to the constructor.
    regularizer = LegacyRegularizer(cfg.regularizer)

    # The seed ensures the test is reproducible.
    graph_data, _ = generate_dag(10, 30, seed=69)

    # The 'fit' method still receives the full config.
    result = regularizer.fit(graph_data, cfg)

    assert "primal" in result
    assert result["primal"].shape == (graph_data.num_vertices(),)

    # All solvers should produce the same primal solution for the same problem.
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
