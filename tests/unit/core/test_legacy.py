# tests/unit/regularizers/test_legacy.py

import numpy as np
from omegaconf import OmegaConf

from regrank.regularizers.legacy import LegacyRegularizer
from regrank.utils.datasets import small_graph


def test_legacy_regularizer_bicgstab():
    """Tests the legacy regularizer with the bicgstab solver."""
    # Create a minimal config for this specific test
    cfg = OmegaConf.create({
        "alpha": 0.1,
        "regularizer": {"name": "legacy", "solver_method": "bicgstab"},
    })

    # Instantiate the regularizer directly
    regularizer = LegacyRegularizer(cfg)

    # Get test data
    graph_data = small_graph()

    # Run the fit method
    result = regularizer.fit(graph_data, cfg)

    # Assertions
    assert "primal" in result
    assert result["primal"].shape == (graph_data.num_vertices(),)
    # You could assert against a known, pre-calculated result for this small graph
    expected_ranks = np.array([
        -1.28630705,
        -0.41493776,
        0.41493776,
        1.28630705,
    ])  # Example values
    assert np.allclose(result["primal"], expected_ranks, atol=1e-2)
