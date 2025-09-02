import graph_tool.all as gt


class NamedGraph:
    def __init__(self, directed=True):
        self.g = gt.Graph(directed=directed)

        # Create vertex property for storing names
        self.name_prop = self.g.new_vertex_property("string")
        self.g.vp["name"] = self.name_prop

        # Dictionary to map names to vertex descriptors
        self.name_to_vertex = {}
        self.vertex_counter = 0

    def add_vertex(self, name):
        """Add a vertex with a given name if it doesn't exist."""
        if name in self.name_to_vertex:
            return self.name_to_vertex[name]

        # Add new vertex
        v = self.g.add_vertex()
        self.name_prop[v] = name
        self.name_to_vertex[name] = v

        return v

    def add_edge(self, source_name, target_name, weight=1.0):
        """Add an edge between two named vertices."""
        # Ensure both vertices exist
        source_v = self.add_vertex(source_name)
        target_v = self.add_vertex(target_name)

        # Add the edge
        edge = self.g.add_edge(source_v, target_v)

        # Optionally add weight property
        if not hasattr(self, "weight_prop"):
            self.weight_prop = self.g.new_edge_property("double")
            self.g.ep["weight"] = self.weight_prop

        self.weight_prop[edge] = weight
        return edge

    def assign_ranking(self, names, rankings):
        """
        Assign true rankings to nodes.

        Parameters:
        - names: Either a string (single node name) or list of strings (node names)
        - rankings: Either a float (single ranking) or list of floats (rankings)
        """
        # Create or get the vertex property "true_ranking"
        if "true_ranking" not in self.g.vp:
            self.true_rank_prop = self.g.new_vertex_property("double")
            self.g.vp["true_ranking"] = self.true_rank_prop
        else:
            self.true_rank_prop = self.g.vp["true_ranking"]

        # Handle single node assignment
        if isinstance(names, str):
            if not isinstance(rankings, float | int):
                raise ValueError(
                    "Ranking must be a float when assigning to single node."
                )

            # Get or create vertex
            vertex = self.get_vertex_by_name(names)
            if vertex is None:
                vertex = self.add_vertex(names)

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
                vertex = self.get_vertex_by_name(name)
                if vertex is None:
                    vertex = self.add_vertex(name)

                self.true_rank_prop[vertex] = float(ranking)

        else:
            raise TypeError("Names must be either a string or a list of strings.")

    def get_ranking(self, name):
        """Get the true ranking of a node by name."""
        if "true_ranking" not in self.g.vp:
            return None

        vertex = self.get_vertex_by_name(name)
        if vertex is None:
            return None

        return self.g.vp["true_ranking"][vertex]

    def list_rankings(self):
        """List all nodes with their true rankings."""
        if "true_ranking" not in self.g.vp:
            print("No rankings assigned yet.")
            return

        print("Node rankings:")
        for name, vertex in self.name_to_vertex.items():
            ranking = self.g.vp["true_ranking"][vertex]
            print(f"  {name}: {ranking}")

    def get_vertex_by_name(self, name):
        """Get vertex descriptor by name."""
        return self.name_to_vertex.get(name, None)

    def get_name_by_vertex(self, vertex):
        """Get name by vertex descriptor."""
        return self.name_prop[vertex]

    def get_vertex_index(self, name):
        """Get the integer index of a vertex by name."""
        vertex = self.get_vertex_by_name(name)
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
