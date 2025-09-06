import warnings

import matplotlib

matplotlib.use("cairo")

import graph_tool.all as gt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyArrowPatch
from scipy.sparse import csr_matrix


class HandlerArrow(HandlerPatch):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # Create an arrow patch from left to right
        p = FancyArrowPatch(
            (0, 0.5 * height),
            (width, 0.5 * height),
            arrowstyle="-|>",
            mutation_scale=15,  # Controls arrow head size
            lw=1.5,
            color="dimgray",
        )
        return [p]


class NamedPoset:
    def __init__(
        self,
        relation: str = "≼",
    ):
        """Initializes a new NamedPoset object."""
        self.relation = relation
        self.g = gt.Graph(directed=True)
        self.name_prop = self.g.new_vertex_property("string")
        self.g.vp["name"] = self.name_prop
        self.true_rank_prop = self.g.new_vertex_property("double")
        self.g.vp["true_ranking"] = self.true_rank_prop
        self.name_to_vertex = {}

    def add_element(self, name):
        """Adds an element with a given name if it doesn't already exist."""
        if name not in self.name_to_vertex:
            v = self.g.add_vertex()
            self.name_prop[v] = name
            self.name_to_vertex[name] = v
        return self.g

    def add_chain(self, chain: list[str]):
        """
        Adds a total order over a subset of elements, forming a chain.
        This implies that for a chain [a, b, c], a ≼ b ≼ c.
        """
        [self.add_element(name) for name in chain]
        for i, name_i in enumerate(chain):
            for _, name_j in enumerate(chain[i + 1 :], start=i + 1):
                source_v = self.name_to_vertex[name_i]
                target_v = self.name_to_vertex[name_j]
                if self.g.edge(source_v, target_v) is None:
                    self.g.add_edge(source_v, target_v)
        return self.g

    def assign_ranking(self, names, rankings):
        """
        Assign true rankings to nodes.
        Parameters:
        - names: Either a string (single node name) or list of strings (node names)
        - rankings: Either a float (single ranking) or list of floats (rankings)
        """
        # Handle single node assignment
        if isinstance(names, str):
            if not isinstance(rankings, float | int):
                raise ValueError(
                    "Ranking must be a float or int when assigning to single node."
                )
            # Get or create vertex
            if names not in self.name_to_vertex:
                warnings.warn(f"Vertex '{names}' not found. Creating it.", stacklevel=2)
                self.add_element(names)
            vertex = self.name_to_vertex[names]
            self.true_rank_prop[vertex] = float(rankings)
        # Handle list assignment
        elif isinstance(names, list):
            if not isinstance(rankings, list):
                raise ValueError("When names is a list, rankings must also be a list.")
            if len(names) != len(rankings):
                raise ValueError(
                    f"Length mismatch: {len(names)} names vs {len(rankings)} rankings."
                )
            for name, ranking in zip(names, rankings, strict=False):
                if not isinstance(name, str):
                    raise ValueError(f"All names must be strings, got {type(name)}")
                if not isinstance(ranking, float | int):
                    raise ValueError(
                        f"All rankings must be float or int, got {type(ranking)}"
                    )
                # Get or create vertex
                if name not in self.name_to_vertex:
                    warnings.warn(
                        f"Vertex '{name}' not found. Creating it.", stacklevel=2
                    )
                    self.add_element(name)
                vertex = self.name_to_vertex[name]
                # print(
                #     f"Assigning ranking {ranking} to vertex {name} (id {int(vertex)})"
                # )
                self.true_rank_prop[vertex] = float(ranking)
        else:
            raise TypeError("Names must be either a string or a list of strings.")

    def draw_poset(self, output_filename: str = "poset_visualization.png"):
        """
        Draws the poset using graph-tool and saves it to a file.


        Args:
            output_filename (str): The name of the file to save the image to.
        """
        if self.g.num_vertices() == 0:
            print("Graph is empty. Nothing to draw.")
            return
        colorbar = True
        ranks = np.array([self.true_rank_prop[v] for v in self.g.vertices()])
        if np.all(ranks == 0.0):
            norm = mcolors.Normalize(vmin=0, vmax=1)
            colorbar = False
        else:
            norm = mcolors.Normalize(vmin=ranks.min(), vmax=ranks.max())

        cmap = plt.get_cmap("YlGn")

        vertex_colors = self.g.new_vertex_property("vector<double>")
        for v in self.g.vertices():
            vertex_colors[v] = cmap(norm(self.true_rank_prop[v]))

        pos = gt.sfdp_layout(
            self.g,
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        gt.graph_draw(
            self.g,
            pos=pos,
            vertex_text=self.g.vp["name"],
            vertex_color="black",
            vertex_fill_color=vertex_colors,
            vertex_size=0.2,  # Smaller vertex size
            vertex_font_size=0.2,  # Adjusted font size
            vertex_halo=False,
            vertex_pen_width=0.01,
            edge_pen_width=0.025,
            edge_marker_size=0.1,
            mplfig=fig,
        )

        if colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(
                sm, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8
            )
            cbar.set_label("Ordering", fontsize=12)

        ax.legend(
            handles=[FancyArrowPatch((0, 0), (1, 0), color="black")],
            labels=[self.relation],
            handler_map={FancyArrowPatch: HandlerArrow()},
            loc="lower center",
            fontsize=12,
            frameon=False,
        )
        # ax.set_title("Poset Visualization", fontsize=16)
        ax.axis("off")
        fig.savefig(output_filename, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Poset visualization saved to '{output_filename}'")

    def print_named_edges(self):
        """Print all edges with their names."""
        for edge in self.g.edges():
            source_name = self.name_prop[edge.source()]
            target_name = self.name_prop[edge.target()]
            print(f"{source_name} -> {target_name}")

    def get_ranking(self, name):
        """Get the true ranking of a node by name."""
        if "true_ranking" not in self.g.vp:
            return None

        vertex = self.get_vertex_by_name(name)
        if vertex is None:
            return None

        return self.g.vp["true_ranking"][vertex]

    def get_rankings(self, chain):
        """List all nodes with their true rankings."""
        if "true_ranking" not in self.g.vp:
            raise ValueError("No rankings assigned yet.")
        rankings = []
        for named_element in chain:
            vertex = self.get_vertex_by_name(named_element)
            if vertex is not None:
                ranking = self.g.vp["true_ranking"][vertex]
                rankings.append(ranking)
        return rankings

    def get_vertex_by_name(self, name):
        """Get vertex descriptor by name."""
        return self.name_to_vertex.get(name)

    def get_name_by_vertex(self, vertex):
        """Get name by vertex descriptor."""
        return self.name_prop[vertex]

    def get_vertex_index(self, name):
        """Get the integer index of a vertex by name."""
        vertex = self.name_to_vertex.get(name)
        return int(vertex) if vertex is not None else None

    def list_vertices(self):
        """List all vertices with their names and indices."""
        for name, vertex in self.name_to_vertex.items():
            print(f"Vertex {int(vertex)}: {name}")

    def list_edges(self):
        """List all edges with names."""
        for edge in self.g.edges():
            source_name = self.name_prop[edge.source()]
            target_name = self.name_prop[edge.target()]
            weight = self.weight_prop[edge] if hasattr(self, "weight_prop") else 1.0
            print(f"{source_name} -> {target_name} (weight: {weight})")

    # Delegate attribute access to the underlying graph
    def __getattr__(self, name):
        return getattr(self.g, name)

    @property
    def graph(self):
        """Access the underlying graph_tool Graph object."""
        return self.g

    def to_bt_matrix(self) -> csr_matrix:
        """
        Convert a NamedGraph with assigned rankings into a Bradley-Terry model matrix.

        Only generate BT probabilities for node pairs (i,j) where an edge exists between them.

        Returns:
            A scipy.sparse.csr_matrix of shape (N, N) where N is the number of vertices.
            M[i, j] represents the Bradley-Terry probability that node i beats node j:
            P(i beats j) = exp(s_i) / (exp(s_i) + exp(s_j))
            where s_i is the ranking/skill score of node i.
            Only entries for connected node pairs are non-zero.
        """

        N = self.num_vertices()
        vertex_list = list(self.g.vertices())
        for v in vertex_list:
            print(f"  vertex {v} (name = {self.g.vp['name'][v]}) has true ranking: {self.g.vp['true_ranking'][v]}")

        # Extract rankings for all nodes
        if "true_ranking" not in self.g.vp:
            raise ValueError("Graph vertices do not have 'true_ranking' property.")

        ranks = np.zeros(N)
        for idx, v in enumerate(vertex_list):
            ranks[idx] = self.g.vp["true_ranking"][v]

        # Prepare data for sparse matrix construction
        rows = []
        cols = []
        data = []

        # Only iterate over existing edges in the graph
        for edge in self.g.edges():
            i = int(edge.source())
            j = int(edge.target())

            s_i = ranks[i]
            s_j = ranks[j]

            # Bradley-Terry probabilities
            exp_i = np.exp(s_i)
            exp_j = np.exp(s_j)

            # Probability that i beats j
            p_i_beats_j = exp_i / (exp_i + exp_j)
            rows.append(i)
            cols.append(j)
            data.append(p_i_beats_j)

            # Probability that j beats i
            p_j_beats_i = exp_j / (exp_i + exp_j)
            rows.append(j)
            cols.append(i)
            data.append(p_j_beats_i)

        # Create and return the sparse matrix
        bt_matrix = csr_matrix((data, (rows, cols)), shape=(N, N))
        return bt_matrix
