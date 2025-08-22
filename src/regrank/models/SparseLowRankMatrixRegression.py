import numpy as np
from .pg import (
    singular_value_thresholding,
    soft_thresholding,
    project_nonnegative,
    project_symmetric,
    project_zero_diagonal,
    svd,
)


class SparseLowRankMatrixRegression:
    def __init__(
        self,
        b,
        x_init,
        A_init,
        lambda1,
        lambda2,
        lambda3,
        alpha,
        max_iter=100,
        tol=1e-5,
    ):
        """
        Initializes the optimizer.

        Args:
            b (np.array): The target vector.
            x_init (np.array): Initial guess for vector x.
            A_init (np.array): Initial guess for matrix A.
            lambda1 (float): Regularization parameter for x (Ridge).
            lambda2 (float): Regularization parameter for L1 sparsity on A's upper triangle.
            lambda3 (float): Regularization parameter for nuclear norm of A.
            alpha (float): Step size for the A update (gradient descent part).
            max_iter (int): Maximum number of iterations.
            tol (float): Tolerance for convergence.
        """
        self.b = b
        self.x = x_init
        self.A = A_init
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.n = A_init.shape[0]

    def update_x(self):
        """
        Updates vector x using Ridge Regression (closed-form solution).
        Problem: minimize_x ||A@x - b||^2 + lambda1 * ||x||^2
        """
        AtA = self.A.T @ self.A
        reg_term = self.lambda1 * np.eye(self.n)
        Atb = self.A.T @ self.b
        self.x = np.linalg.solve(AtA + reg_term, Atb)

    def update_A(self):
        """
        Updates matrix A using a proximal gradient step with projections.
        Problem: minimize_A ||A@x - b||^2 + lambda2*||A_upper_tri||_1 + lambda3*||A||_*
        Subject to A=A^T, A>=0, A_diag=0
        """
        # 1. Gradient step on the data fidelity term
        grad = 2 * (self.A @ self.x - self.b)[:, np.newaxis] @ self.x[np.newaxis, :]
        A_temp = self.A - self.alpha * grad

        # 2. Apply proximal operator for nuclear norm
        A_prox = singular_value_thresholding(A_temp, self.alpha * self.lambda3)

        # 3. Apply proximal operator for L1 sparsity on the upper triangular part
        upper_idx = np.triu_indices(
            self.n, k=1
        )  # Get indices for upper triangle, excluding diagonal
        A_prox[upper_idx] = soft_thresholding(
            A_prox[upper_idx], self.alpha * self.lambda2
        )
        # Note: we only apply soft-thresholding to the upper triangle here.
        # Symmetry will be enforced next, propagating sparsity to the lower triangle.

        # 4. Enforce structural constraints via projections (order matters for performance/numerical stability)
        A_prox = project_zero_diagonal(A_prox)
        A_prox = project_symmetric(
            A_prox
        )  # This makes lower triangle match upper and propagates sparsity
        A_prox = project_nonnegative(A_prox)  # Ensures all elements are non-negative

        self.A = A_prox

    def objective(self):
        """Calculates the current value of the objective function."""
        data_fidelity = np.linalg.norm(self.A @ self.x - self.b) ** 2
        reg_x = self.lambda1 * np.linalg.norm(self.x) ** 2
        reg_sparse = self.lambda2 * np.sum(
            np.abs(np.triu(self.A, 1))
        )  # L1 on upper triangle
        reg_nuclear = self.lambda3 * np.sum(
            svd(self.A, compute_uv=False)
        )  # Nuclear norm
        return data_fidelity + reg_x + reg_sparse + reg_nuclear

    def solve(self):
        """Runs the alternating minimization algorithm."""
        prev_obj = np.inf
        for it in range(self.max_iter):
            self.update_x()
            self.update_A()
            obj = self.objective()
            print(f"Iteration {it + 1}: Objective = {obj:.6f}")
            if abs(prev_obj - obj) < self.tol:
                print(f"Converged after {it + 1} iterations.")
                break
            prev_obj = obj
        return self.A, self.x
