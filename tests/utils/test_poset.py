from collections import Counter

import numpy as np
import pymongo
import pytest

# Try to import the main module. If it fails, pytest will skip all tests.
# This assumes your code is saved in a file named `poset.py`.
poset = pytest.importorskip("regrank.utils.poset")

# in tests/conftest.py


@pytest.fixture(scope="session")
def mongo_service():
    print("--- ENTERING mongo_service fixture ---")
    client = None  # Initialize client to None
    try:
        # Replace with your actual connection URL
        auth_url = "mongodb://localhost:27017"
        client = pymongo.MongoClient(auth_url, serverSelectionTimeoutMS=1000)

        # The ismaster command is cheap and does not require auth.
        client.admin.command("ismaster")
        print("--> MongoDB is responsive! Handing over to tests.")
        yield client
    finally:
        if client:
            print("<-- Closing MongoDB connection. ---")
            client.close()


# --- Fixtures for test data ---
@pytest.fixture
def A_simple_cycle():
    """Adjacency matrix that produces a 3-cycle: 0->1, 1->2, 2->0."""
    return np.array([
        [0, 2, 1],  # R01 > 0, R02 < 0
        [1, 0, 2],  # R12 > 0
        [2, 1, 0],
    ])


@pytest.fixture
def A_acyclic_graph():
    """Adjacency matrix producing an acyclic graph: 0->1, 0->2, 1->2."""
    return np.array([[0, 2, 2], [1, 0, 2], [1, 1, 0]])


@pytest.fixture
def A_no_relations():
    """Symmetric matrix, produces no relations as R is the zero matrix."""
    return np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])


# --- Test Functions ---


def test_potential_poset_from_adjacency(
    A_simple_cycle, A_acyclic_graph, A_no_relations
):
    """Tests the extraction of cover relations from a matrix."""
    # Test case 1: Matrix producing a cyclic graph
    covers_cycle = poset.potential_poset_from_adjacency(A_simple_cycle)
    assert Counter(covers_cycle) == Counter([(0, 1), (2, 0), (1, 2)])

    # Test case 2: Matrix producing an acyclic graph
    covers_acyclic = poset.potential_poset_from_adjacency(A_acyclic_graph)
    assert Counter(covers_acyclic) == Counter([(0, 1), (0, 2), (1, 2)])

    # Test case 3: Symmetric matrix should yield no relations
    covers_none = poset.potential_poset_from_adjacency(A_no_relations)
    assert len(covers_none) == 0


def test_has_cycle():
    """Tests the DFS-based cycle detection algorithm."""
    # A simple 3-cycle should be detected
    assert poset.has_cycle([(0, 1), (1, 2), (2, 0)], n=3) is True

    # A line graph is acyclic
    assert poset.has_cycle([(0, 1), (1, 2)], n=3) is False

    # A self-loop is a cycle
    assert poset.has_cycle([(0, 0)], n=1) is True

    # Disjoint components, one of which has a cycle
    assert poset.has_cycle([(0, 1), (1, 0), (2, 3)], n=4) is True

    # An empty graph has no cycles
    assert poset.has_cycle([], n=3) is False


def test_break_cycles_exact():
    """Tests the ILP-based exact cycle breaking."""
    # Test case 1: A simple 3-cycle
    covers_cycle = [(0, 1), (1, 2), (2, 0)]
    n = 3
    kept, dropped = poset.break_cycles_exact(covers_cycle, n)

    assert not poset.has_cycle(kept, n), "The resulting graph should be acyclic."
    assert len(dropped) == 1, "Exactly one edge must be dropped for a 3-cycle."
    assert Counter(kept + dropped) == Counter(covers_cycle), (
        "The union of kept and dropped edges must match the original."
    )

    # Test case 2: A graph with two disjoint 2-cycles
    covers_two_cycles = [(0, 1), (1, 0), (2, 3), (3, 2)]
    n = 4
    kept, dropped = poset.break_cycles_exact(covers_two_cycles, n)

    assert not poset.has_cycle(kept, n)
    assert len(dropped) == 2, "Exactly two edges must be dropped."

    # Test case 3: An already acyclic graph should not be changed
    covers_acyclic = [(0, 1), (1, 2), (0, 2)]
    n = 3
    kept, dropped = poset.break_cycles_exact(covers_acyclic, n)

    assert len(dropped) == 0, "No edges should be dropped from an acyclic graph."
    assert Counter(kept) == Counter(covers_acyclic)


def test_is_linear_extension():
    """Tests the SageMath-based linear extension checker."""
    # Define a simple poset: 0 ≺ 1 ≺ 2
    covers = [(0, 1), (1, 2)]

    # A valid topological sort
    valid_order = [0, 1, 2]
    assert poset.is_linear_extension(covers, valid_order) is True

    # An invalid ordering (violates both relations)
    invalid_order = [2, 1, 0]
    assert poset.is_linear_extension(covers, invalid_order) is False

    # Another invalid ordering (violates one relation)
    another_invalid = [1, 0, 2]
    assert poset.is_linear_extension(covers, another_invalid) is False

    # Any ordering is a linear extension of an empty poset
    assert poset.is_linear_extension([], [2, 0, 1]) is True
