import numpy as np  # type: ignore
from numpy.polynomial.legendre import leggauss  # type: ignore
from scipy.sparse.linalg import eigs  # type: ignore


# ============================================================
# Quadrature rules on [-1, 1]
# Each rule returns (xi, wi) with:
#   ∫_{-1}^1 f(x) dx  ≈  Σ_j wi[j] * f(xi[j])
# ============================================================

def quad_gauss_legendre(m: int):
    """Gauss–Legendre (exact for polys up to degree 2m-1)."""
    xi, wi = leggauss(m)
    return xi.astype(float), wi.astype(float)

def quad_trapezoid(m: int):
    """
    Trapezoid on [-1,1] with m nodes.
    Note: very low order but fast.
    """
    if m < 2:
        raise ValueError("Trapezoid needs m>=2.")
    xi = np.linspace(-1.0, 1.0, m)
    h = xi[1] - xi[0]
    wi = np.full(m, h)
    wi[0] *= 0.5
    wi[-1] *= 0.5
    return xi, wi

def quad_simpson(m: int):
    """
    Composite Simpson on [-1,1] with m nodes (m must be odd and >= 3).
    """
    if m < 3 or (m % 2 == 0):
        raise ValueError("Simpson needs m odd and >= 3.")
    xi = np.linspace(-1.0, 1.0, m)
    h = xi[1] - xi[0]
    wi = np.ones(m)
    wi[1:-1:2] = 4.0
    wi[2:-2:2] = 2.0
    wi *= h / 3.0
    return xi, wi


def quad_clenshaw_curtis(m: int):
    """
    Clenshaw–Curtis quadrature on [-1,1] with m nodes.
    Returns (xi, wi) such that ∫_{-1}^1 f(x) dx ≈ Σ wi[j] f(xi[j]).
    """
    if m < 2:
        raise ValueError("Clenshaw–Curtis needs m>=2.")

    N = m - 1
    theta = np.pi * np.arange(m) / N
    xi = np.cos(theta)

    wi = np.zeros(m, dtype=float)

    if N == 1:
        # 2-point trapezoid on [-1,1]
        wi[:] = 1.0
        return xi.astype(float), wi.astype(float)

    ii = np.arange(1, N)          # interior indices
    v = np.ones(N - 1, dtype=float)

    if N % 2 == 0:
        # N even
        wi[0] = 1.0 / (N**2 - 1.0)
        wi[-1] = wi[0]
        for k in range(1, N//2):
            v -= 2.0 * np.cos(2.0 * k * theta[ii]) / (4.0 * k**2 - 1.0)
        v -= np.cos(N * theta[ii]) / (N**2 - 1.0)
    else:
        # N odd
        wi[0] = 1.0 / (N**2)
        wi[-1] = wi[0]
        for k in range(1, (N + 1)//2):
            v -= 2.0 * np.cos(2.0 * k * theta[ii]) / (4.0 * k**2 - 1.0)

    wi[ii] = 2.0 * v / N
    return xi.astype(float), wi.astype(float)


# ------------------------------------------------------------
# Rule registry 
# ------------------------------------------------------------
QUAD_RULES = {
    "legendre": quad_gauss_legendre,
    "trapezoid": quad_trapezoid,
    "simpson": quad_simpson,
    "clenshaw_curtis": quad_clenshaw_curtis
}
