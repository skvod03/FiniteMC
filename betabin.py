import numpy as np
import scipy.integrate as spi
from scipy.integrate import quad_vec
from scipy.stats import binom, beta as beta_dist
import matplotlib.pyplot as plt


class BetaBinomialMarkovChain:
    def __init__(self, x_0, n, alpha, beta):
        self.x_0 = float(x_0)
        self.n = int(n)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.chain = [self.x_0]
        self.state = self.x_0

        # fixed k-grid for the binomial mixture
        self.k = np.arange(self.n + 1)

    def SimulateNext(self):
        k = np.random.binomial(self.n, self.state)
        self.state = np.random.beta(self.alpha + k, self.beta + self.n - k)
        self.chain.append(self.state)

    # ----------------------------
    # Helper kernel (vectorized)
    # ----------------------------
    def kernel_xy(self, x, y):
        """
        Vectorized transition kernel K(x,y).

        Accepts:
          x: scalar or array-like of shape (Nx,)
          y: scalar or array-like of shape (Ny,)

        Returns:
          K: array of shape (Nx, Ny) if both are arrays,
             or shape (Ny,) if x is scalar,
             or shape (Nx,) if y is scalar,
             or scalar if both are scalars.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # Ensure at least 1D for broadcasting; remember original scalar-ness
        x_scalar = (x.ndim == 0)
        y_scalar = (y.ndim == 0)
        if x_scalar:
            x = x[None]
        if y_scalar:
            y = y[None]

        # Shapes:
        #   x[:, None] -> (Nx, 1)
        #   k[None, :] -> (1, n+1)
        #   w -> (Nx, n+1)
        w = binom.pmf(self.k[None, :], self.n, x[:, None])

        # Beta table over y:
        #   y[None, :] -> (1, Ny)
        #   k[:, None] -> (n+1, 1)
        #   B -> (n+1, Ny)
        B = beta_dist.pdf(
            y[None, :],
            self.alpha + self.k[:, None],
            self.beta + self.n - self.k[:, None]
        )

        # Mixture:
        #   (Nx, n+1) @ (n+1, Ny) -> (Nx, Ny)
        K = w @ B

        # Squeeze back to natural shapes
        if x_scalar and y_scalar:
            return float(K[0, 0])
        if x_scalar:
            return K[0, :]          # (Ny,)
        if y_scalar:
            return K[:, 0]          # (Nx,)
        return K                    # (Nx, Ny)

    # Keep old name if you want a scalar-friendly wrapper
    def Kernel(self, x, y):
        return self.kernel_xy(x, y)

    def Stationary(self, x):
        return beta_dist.pdf(x, self.alpha, self.beta)

    # ----------------------------
    # Next distribution via quad_vec
    # ----------------------------
    def NextDistribution_quadvec(self, y_grid, dist, *params):
        """
        Computes f_next(y) = ∫_0^1 K(x,y) dist(x; params) dx for y in y_grid.

        Uses one quad_vec call returning a vector.
        """
        y_grid = np.asarray(y_grid, dtype=float)

        def integrand(x):
            # kernel_xy(x, y_grid) returns shape (Ny,) since x is scalar
            return dist(x, *params) * self.kernel_xy(x, y_grid)

        res, err = quad_vec(integrand, 0.0, 1.0)
        return res, err

    def RunChain(self, steps):
        for _ in range(int(steps)):
            self.SimulateNext()



