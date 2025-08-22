import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np

    np.set_printoptions(precision=2, suppress=True)

    import sys

    sys.path.append("/Users/tzuchi/Documents/Workspace/regrank/")
    # sys.path.append("..")

    from collections import defaultdict, Counter
    import graph_tool.all as gt
    import matplotlib.pyplot as plt
    import regrank

    from regrank.models.pg import prune_fixed_matrix

    n_dim = 8
    k_sparsity = 23  # Keep top 8 entries in the upper triangle

    np.random.seed(41)
    # This is the original matrix whose values must be preserved.
    A_initial = np.random.rand(n_dim, n_dim)

    # Create a plausible b vector for the example
    x_plausible = np.random.randn(n_dim)
    b_vec = A_initial @ x_plausible

    # Run the corrected pruning algorithm
    x3, A3 = prune_fixed_matrix(A_initial, b_vec, k=k_sparsity)

    # --- VERIFICATION ---
    print("--- Verification of Results ---")
    np.set_printoptions(precision=3, suppress=True)

    # The symmetric version of the original matrix that the algorithm uses
    A_orig_symm = (A_initial + A_initial.T) / 2
    np.fill_diagonal(A_orig_symm, 0)

    print("\nOriginal Symmetric Matrix (A_orig_symm):")
    print(A_orig_symm)

    print("\nFinal Pruned Matrix (A3):")
    print(A3)

    # 1. Check if the non-zero values in A3 are identical to A_orig_symm
    non_zero_mask = A3 != 0
    values_preserved = np.allclose(A3[non_zero_mask], A_orig_symm[non_zero_mask])

    print("\n---------------------------------")
    print(f"Values Preserved: {values_preserved}")
    print(f"Is Symmetric:     {np.allclose(A3, A3.T)}")
    print(f"Sparsity (k):     {np.count_nonzero(np.triu(A3, 1))}/{k_sparsity}")
    print("---------------------------------")

    # This should assert to True
    assert values_preserved
    return A3, A_initial


@app.cell
def _(A_initial):
    A_initial
    return


@app.cell
def _(A3):
    A3
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
