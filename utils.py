import numpy as np
from scipy import special
from scipy.special import roots_jacobi, beta as beta_fn


# ----------------------------
# Helpers
# ----------------------------
def Phi_complex(z):
    """Standard normal CDF extended to complex arguments via erf."""
    return 0.5 * (1.0 + special.erf(z / np.sqrt(2.0)))


def beta_cf_jacobi(t, a, b, m=120):
    """
    Robust CF of Beta(a,b) on [0,1] computed from scratch:

        φ(t) = 1/B(a,b) ∫_0^1 exp(i t x) x^{a-1} (1-x)^{b-1} dx

    Uses Gauss–Jacobi quadrature on [-1,1], which is tailor-made for Beta weights.

    Parameters
    ----------
    t : float or ndarray
        Real frequencies.
    a, b : float
        Beta parameters (>0).
    m : int
        Quadrature nodes. Increase for larger max|t| if needed.

    Returns
    -------
    complex scalar if t scalar, else complex ndarray with same shape as t.
    """
    t_arr = np.asarray(t, dtype=float)
    scalar = (t_arr.ndim == 0)
    t_flat = t_arr.reshape(-1)

    # Jacobi weights: (1-u)^(alpha) (1+u)^(beta) on [-1,1]
    # With x=(u+1)/2, Beta weight becomes (1-u)^(b-1) (1+u)^(a-1) up to factor.
    u, w = roots_jacobi(m, b - 1.0, a - 1.0)  # u,w length m
    x = 0.5 * (u + 1.0)                       # map to [0,1]

    factor = 2.0 ** (-(a + b - 1.0))
    expo = np.exp(1j * np.outer(x, t_flat))   # (m, Nt)
    integral = factor * (w[:, None] * expo).sum(axis=0)  # (Nt,)
    out = integral / beta_fn(a, b)

    if scalar:
        return out[0].item()
    return out.reshape(t_arr.shape)

def analytic_stationary_mean_var(lam, innov_mean, innov_var):
    """
    For X_{n+1} = (1-lam) X_n + lam Z_{n+1} with Z independent of X_n:

      E[X] = E[Z]
      Var(X) = (lam^2 Var(Z)) / (1 - (1-lam)^2) = (lam/(2-lam)) Var(Z)
    """
    rho = 1.0 - lam
    mean = innov_mean
    var = (lam ** 2) * innov_var / (1.0 - rho ** 2)
    return mean, var