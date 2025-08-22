import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import sys

    sys.path.append("/Users/tzuchi/Documents/Workspace/regrank/")

    import regrank as rr
    from regrank.models import SpringRank
    from regrank.draw import plot_hist, print_summary_table
    return SpringRank, rr


@app.cell
def _(rr):
    # g = rr.datasets.us_air_traffic()
    # g = rr.datasets.at_migrations()
    g = rr.datasets.parakeet()
    print(g)
    return (g,)


@app.cell
def _(g):
    g.list_properties()
    return


@app.cell
def _(SpringRank, g):
    # Assuming 'model' is an instance of your SpringRank class
    model = SpringRank(method="porder")

    # Run the fit method, providing the new hyperparameters 'k' and 'lambda_x'
    # 'data' can be your graph-tool object or a NumPy array
    # 'b_vec' is optional.
    results = model.fit(g, k=50, lambda_x=0.01)

    # Access the results
    pruned_A = results["primal_A"]
    ranks_x = results["primal_x"]

    print("Pruned Matrix A:")
    print(pruned_A)
    print("\nFinal Ranks x:")
    print(ranks_x)
    return (pruned_A,)


@app.cell
def _(pruned_A):
    import numpy as np
    from graph_tool.all import (
        Graph,
        graph_draw,
        label_largest_component,
        GraphView,
        sfdp_layout,
    )


    def plot_largest_component(adj_matrix):
        """
        Creates a graph from an adjacency matrix, finds its largest connected component,
        and plots it using graph-tool.

        Args:
            adj_matrix (np.ndarray): A square NumPy array representing the adjacency matrix
                                     of an undirected graph.
        """
        # Create an undirected graph-tool Graph object
        g = Graph(directed=False)

        # Add vertices to the graph
        num_vertices = adj_matrix.shape[0]
        g.add_vertex(num_vertices)

        # Add edges by iterating through the upper triangle of the adjacency matrix
        # This avoids adding duplicate edges for an undirected graph
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                if adj_matrix[i, j] != 0:
                    g.add_edge(g.vertex(i), g.vertex(j))

        # Find the largest connected component and get a property map
        # The map assigns '1' to vertices in the largest component and '0' to others
        l = label_largest_component(g)

        # Create a GraphView that filters for the largest component
        largest_comp_view = GraphView(g, vfilt=l)

        # Generate a layout for the largest component using the sfdp algorithm
        # This layout is often effective for revealing the structure of the graph
        pos = sfdp_layout(largest_comp_view)

        # Draw the graph with aesthetic customizations
        graph_draw(
            largest_comp_view,
            pos=pos,
            output_size=(600, 600),
            vertex_fill_color="deepskyblue",  # A pleasant node color
            vertex_size=15,
            vertex_pen_width=1.5,
            edge_color="dimgray",  # A softer edge color
            edge_pen_width=1.2,
            bg_color="white",  # A clean white background
            output=None,
        )  # Set to a file name like "graph.png" to save


    # --- Example Usage ---
    if __name__ == "__main__":
        # Create an example adjacency matrix with two separate components
        adj = np.array(
            [
                [0, 1, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 1, 0],
            ]
        )

        # Generate and display the plot of the largest component
        plot_largest_component(pruned_A)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
