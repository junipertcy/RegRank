import numpy as np
from numpy.linalg import inv


def ridge_regression(A, b, lam):
    """Solves the Ridge Regression problem: (A'A + lambda*I)x = A'b."""
    n = A.shape[1]
    identity_matrix = np.eye(n)
    try:
        return inv(A.T @ A + lam * identity_matrix) @ (A.T @ b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A.T @ A + lam * identity_matrix) @ (A.T @ b)


def prune_fixed_matrix(A_init, b, k, lambda_x=0.1, max_iters=100, tol=1e-6):
    """
    Correctly prunes a fixed initial matrix A_init to be sparse and symmetric
    by learning an optimal binary mask M. The non-zero values of A_init are
    guaranteed to be unchanged.

    Args:
        A_init (np.ndarray): The initial dense matrix to be pruned.
        b (np.ndarray): The target vector.
        k (int): The number of non-zero entries to keep in the upper triangle.
        lambda_x (float): Regularization parameter for x.
        max_iters (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        np.ndarray: The optimized vector x.
        np.ndarray: The final pruned, sparse, and symmetric matrix A.
    """
    # 1. Store the original matrix. It will NOT be modified.
    #    Also, enforce symmetry and zero-diagonal on this base matrix.
    A_orig = (A_init + A_init.T) / 2
    np.fill_diagonal(A_orig, 0)

    n = A_orig.shape[0]
    x = np.zeros(n)

    # 2. Initialize a binary mask. This is the only part of A that "learns".
    mask = np.ones_like(A_orig)
    np.fill_diagonal(mask, 0)

    for it in range(max_iters):
        # 3. Construct the current sparse A using the mask and ORIGINAL values.
        #    This is the key step that ensures values are never changed.
        A = mask * A_orig

        # Step 1: Solve for x with the current sparse A
        x_new = ridge_regression(A, b, lambda_x)

        # Step 2: Use the gradient only to get importance scores for the mask.
        grad_A = 2 * (A @ x_new - b)[:, np.newaxis] @ x_new[np.newaxis, :]

        # We only need to consider the upper triangle for selection
        upper_tri_indices = np.triu_indices(n, 1)
        importance_scores = np.abs(grad_A[upper_tri_indices])

        # Determine the threshold to keep the top k scores
        if k < len(importance_scores):
            threshold = np.partition(importance_scores, -k)[-k]
        else:
            threshold = -1

        # Create the new mask based on the top-k scores
        new_mask = np.zeros_like(mask)
        keep_mask = importance_scores >= threshold

        row_indices, col_indices = upper_tri_indices
        i_keep = row_indices[keep_mask]
        j_keep = col_indices[keep_mask]

        new_mask[i_keep, j_keep] = 1
        new_mask = new_mask + new_mask.T  # Ensure symmetry

        # Check for convergence
        if np.allclose(new_mask, mask) and np.linalg.norm(x_new - x) < tol:
            break

        mask = new_mask
        x = x_new

    # 4. The final matrix is guaranteed to be a sub-matrix of A_orig.
    A_final = mask * A_orig
    return x, A_final

