# /tests/unit/utils/test_named_poset.py
import os

import pytest

from regrank.utils import NamedPoset


@pytest.fixture
def poset():
    """Provides a fresh NamedPoset instance for each test."""
    return NamedPoset()


def test_initialization(poset):
    """Test the constructor and default values."""
    assert poset.relation == "≼"
    assert poset.g.num_vertices() == 0
    assert poset.g.num_edges() == 0
    assert len(poset.name_to_vertex) == 0


def test_add_element_single(poset):
    """Test adding a single new element."""
    poset.add_element("a")
    assert poset.g.num_vertices() == 1
    assert "a" in poset.name_to_vertex
    vertex_a = poset.name_to_vertex["a"]
    assert poset.g.vp.name[vertex_a] == "a"


def test_add_element_duplicate(poset):
    """Test that adding an existing element does not create a new vertex."""
    poset.add_element("a")
    initial_vertex_count = poset.g.num_vertices()
    initial_vertex_obj = poset.name_to_vertex["a"]

    poset.add_element("a")
    assert poset.g.num_vertices() == initial_vertex_count
    assert poset.name_to_vertex["a"] == initial_vertex_obj


def test_add_chain_creates_vertices_and_edges(poset):
    """Test that add_chain correctly adds vertices and transitive edges."""
    chain = ["a", "b", "c"]
    poset.add_chain(chain)

    # Check vertices
    assert poset.g.num_vertices() == 3
    for name in chain:
        assert name in poset.name_to_vertex

    # Check edges for transitivity: a->b, a->c, b->c
    v_a = poset.name_to_vertex["a"]
    v_b = poset.name_to_vertex["b"]
    v_c = poset.name_to_vertex["c"]

    assert poset.g.edge(v_a, v_b) is not None
    assert poset.g.edge(v_b, v_c) is not None
    assert poset.g.edge(v_a, v_c) is not None
    assert poset.g.num_edges() == 3


def test_add_overlapping_chains(poset):
    """Test adding chains that share elements."""
    poset.add_chain(["a", "b", "d"])  # a->b, a->d, b->d
    poset.add_chain(["a", "c", "d"])  # a->c, c->d

    assert poset.g.num_vertices() == 4
    # Expected edges: (a,b), (a,d), (b,d), (a,c), (c,d)
    assert poset.g.num_edges() == 5, (
        f"Expected 5 edges, but got {poset.print_named_edges()}"
    )


def test_assign_ranking_single(poset):
    """Test assigning a ranking to a single, existing element."""
    poset.add_element("x")
    poset.assign_ranking("x", 5.5)
    v_x = poset.name_to_vertex["x"]
    assert poset.g.vp.true_ranking[v_x] == 5.5


def test_assign_ranking_list(poset):
    """Test assigning rankings from a list to multiple elements."""
    elements = ["x", "y", "z"]
    rankings = [1.0, 2.5, 3.8]
    poset.add_chain(elements)
    poset.assign_ranking(elements, rankings)

    for name, rank in zip(elements, rankings, strict=True):
        v = poset.name_to_vertex[name]
        assert poset.g.vp.true_ranking[v] == rank


def test_assign_ranking_creates_new_element_with_warning(poset):
    """Test that assign_ranking creates a vertex if it doesn't exist and warns."""
    with pytest.warns(UserWarning, match="Vertex 'new_node' not found. Creating it."):
        poset.assign_ranking("new_node", 99.0)

    assert poset.g.num_vertices() == 1
    assert "new_node" in poset.name_to_vertex
    v_new = poset.name_to_vertex["new_node"]
    assert poset.g.vp.true_ranking[v_new] == 99.0


def test_assign_ranking_raises_errors_for_invalid_input(poset):
    """Test that assign_ranking raises appropriate errors for mismatched/bad inputs."""
    with pytest.raises(ValueError, match="Length mismatch: 2 names vs 1 rankings."):
        poset.assign_ranking(["a", "b"], [1.0])

    with pytest.raises(ValueError, match="Ranking must be a float or int when assigning to single node."):
        poset.assign_ranking("a", [1.0, 2.0])

    with pytest.raises(ValueError, match="When names is a list, rankings must also be a list."):
        poset.assign_ranking(["a", "b"], 1.0)


def test_draw_poset_creates_file(poset, tmp_path):
    """Test that draw_poset successfully creates an image file."""
    output_file = tmp_path / "test_poset.png"

    poset.add_chain(["a", "b", "c"])
    poset.assign_ranking(["a", "b", "c"], [1.0, 2.0, 3.0])

    poset.draw_poset(output_filename=str(output_file))

    assert output_file.is_file()
    assert os.path.getsize(output_file) > 0


def test_draw_poset_on_empty_graph(poset, capsys):
    """Test that drawing an empty graph prints a message and does not error."""
    poset.draw_poset()
    captured = capsys.readouterr()
    assert "Graph is empty. Nothing to draw." in captured.out


def test_draw_poset_with_all_zero_rankings(poset, tmp_path):
    """Test drawing when all ranks are 0 to ensure no normalization errors."""
    output_file = tmp_path / "zero_rank_poset.png"

    poset.add_chain(["x", "y"])
    poset.assign_ranking(["x", "y"], [0.0, 0.0])

    # This should run without errors and create a file
    poset.draw_poset(output_filename=str(output_file))

    assert output_file.is_file()
    assert os.path.getsize(output_file) > 0
