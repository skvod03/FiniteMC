import numpy as np  # type: ignore
from numpy.polynomial.legendre import leggauss  # type: ignore
from scipy.sparse.linalg import eigs  # type: ignore
from quadrature import QUAD_RULES



# ============================================================
# Grid discretization with pluggable quadrature rule
# ============================================================

def OneDGridAprox(Chain, h, m, quad_rule):
    """
    Build grid approximation and compute stationary distribution for the discretized chain.

    Parameters
    ----------
    Chain : object
        Must provide Chain.kernel_xy(x, y) vectorized in x and y.
    h : float
        Cell width (approximately).
    m : int
        Quadrature nodes per destination cell.
        Requirements depend on quad_rule:
          - trapezoid: m>=2
          - simpson: m odd >=3
          - clenshaw_curtis: m>=2
          - legendre: m>=1
    quad_rule : str or callable
        If str, must be one of QUAD_RULES keys.
        If callable, must return (xi, wi) on [-1,1].

    Returns
    -------
    pi : (N,) array
        Stationary distribution over cells (cell masses), sum(pi)=1.
    x_grid : (N,) array
        Cell midpoints.
    """
    # --- set up grid ---
    pts = np.linspace(0.0, 1.0, int(1.0 / h) + 1)
    left = pts[:-1]
    right = pts[1:]
    x_grid = (left + right) / 2.0
    intervals = np.vstack((left, right)).T
    N = x_grid.size

    # --- choose quadrature rule on [-1,1] ---
    if isinstance(quad_rule, str):
        if quad_rule not in QUAD_RULES:
            raise ValueError(f"Unknown quad_rule='{quad_rule}'. Choose from {list(QUAD_RULES)}.")
        xi, wi = QUAD_RULES[quad_rule](m)
    else:
        xi, wi = quad_rule(m)

    # --- map nodes to each destination interval ---
    a = intervals[:, 0]
    b = intervals[:, 1]
    half = (b - a) / 2.0
    mid = (a + b) / 2.0
    t = mid[:, None] + half[:, None] * xi[None, :]      # (N,m)

    # --- evaluate K(x_i, y_{ell,j}) for all i,ell,j ---
    y_flat = t.ravel()                                   # (N*m,)
    K_all = Chain.kernel_xy(x_grid, y_flat)              # should be (N, N*m)

    K_reshaped = K_all.reshape(N, N, len(xi))            # (N, N, m)

    # --- quadrature sum over j, multiply by half-length per destination cell ---
    P = (K_reshaped @ wi) * half[None, :]                # (N,N)

    # --- stationary distribution of P ---
    def stationary_sparse(P_sparse):
        w, v = eigs(P_sparse.T, k=1, sigma=1.0)  # shift-invert near 1
        pi = np.real(v[:, 0])
        pi = np.abs(pi)
        pi /= pi.sum()
        return pi, w[0]

    pi = stationary_sparse(P)[0]
    return pi, x_grid


# ============================================================
# Evaluation helpers 
# ============================================================

def Eh_pi(pi, x_grid, g):
    return np.sum(pi * g(x_grid))

def test_functions(name, *params):
    if name == "moment":
        r = params[0]
        return lambda x, r=r: x**r
    if name == "beta_mixed":
        p, q = params
        return lambda x, p=p, q=q: (x**p) * ((1-x)**q)
    if name == "logx":
        return np.log
    if name == "log1mx":
        return lambda x: np.log(1-x)


