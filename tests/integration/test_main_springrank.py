# tests/integration/test_main_springrank.py

from omegaconf import OmegaConf

from regrank.main import SpringRank  # Import the main dispatcher class
from regrank.utils.datasets import small_graph


def test_springrank_dispatcher_with_legacy():
    """
    Tests that the main SpringRank class can correctly dispatch to and run
    the legacy regularizer.
    """
    # Use a config similar to what a user would have
    cfg = OmegaConf.create({
        "alpha": 1.0,
        "regularizer": {
            "name": "legacy",
            "solver_method": "lsmr",  # Test a different solver
        },
    })

    # 1. Initialize the main SpringRank model
    model = SpringRank(cfg)

    # 2. Get data
    graph_data = small_graph()

    # 3. Run the full fit process
    results = model.fit(graph_data)

    # 4. Check the output
    assert "primal" in results
    assert "f_primal" in results
    assert results["primal"].shape == (graph_data.num_vertices(),)
