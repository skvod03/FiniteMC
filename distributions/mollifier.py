from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss

# ============================================================
# Smooth bump mollifier on [0,1]:
#   m_0(x) = exp( 1/(x(x-1)) ) for x in (0,1), else 0
# Note: 1/(x(x-1)) = -1/(x(1-x)) < 0 on (0,1), so this is a C^∞ bump.
# We normalize it to get a probability density on [0,1].
# ============================================================

def bump_unnormalized(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = (x > 0.0) & (x < 1.0)
    # exp(1/(x(x-1))) = exp(-1/(x(1-x)))
    z = x[mask]
    out[mask] = np.exp(-1.0 / (z * (1.0 - z)))
    return out

def gauss_legendre_integrate_01(f, m: int = 200) -> float:
    """
    ∫_0^1 f(x) dx using m-point Gauss–Legendre on [-1,1] mapped to [0,1].
    """
    xi, wi = leggauss(m)  # nodes/weights on [-1,1]
    x = 0.5 * (xi + 1.0)
    return 0.5 * np.sum(wi * f(x))

@dataclass(frozen=True)
class BumpMollifier01:
    """
    Probability distribution on [0,1] with density proportional to exp(1/(x(x-1))).

    Provides:
      - sample(size)
      - cf(t)   = E[e^{itU}]
      - mean(), var()
    """
    m_quad: int = 200
    grid_n: int = 20000
    rng: np.random.Generator = np.random.default_rng()

    def __post_init__(self):
        # Precompute normalization and CDF grid for fast sampling
        xg = np.linspace(0.0, 1.0, self.grid_n, dtype=float)
        ug = bump_unnormalized(xg)

        # Normalize using trapezoid on the fine grid (fast + accurate here)
        Z = np.trapz(ug, xg)
        if Z <= 0.0 or not np.isfinite(Z):
            raise ValueError("Normalization failed for bump mollifier.")

        pg = ug / Z
        cdf = np.cumsum(pg)
        # Convert cumulative sum to proper integral-like CDF via dx scaling and normalize
        dx = xg[1] - xg[0]
        cdf = cdf * dx
        cdf[-1] = 1.0

        object.__setattr__(self, "_xg", xg)
        object.__setattr__(self, "_pg", pg)
        object.__setattr__(self, "_Z", Z)
        object.__setattr__(self, "_cdf", cdf)

        # Precompute mean/var from the grid (stable)
        mu = np.trapz(xg * pg, xg)
        m2 = np.trapz((xg**2) * pg, xg)
        var = m2 - mu**2
        object.__setattr__(self, "_mu", float(mu))
        object.__setattr__(self, "_var", float(var))

    def sample(self, size=None):
        u = self.rng.random(size=size)
        # inverse CDF via interpolation on the precomputed grid
        return np.interp(u, self._cdf, self._xg)

    def cf(self, t):
        """
        φ_U(t) = ∫_0^1 e^{itx} m(x) dx (normalized).
        Uses Gauss–Legendre on [0,1] (very accurate for smooth integrands).
        """
        t = np.asarray(t, dtype=float)

        # vectorized quadrature over t by evaluating the integrand on nodes
        xi, wi = leggauss(self.m_quad)
        x = 0.5 * (xi + 1.0)  # nodes in [0,1]
        w = 0.5 * wi          # weights scaled by 1/2 from mapping

        m0 = bump_unnormalized(x) / self._Z  # normalized density at nodes

        # integrand: e^{itx} m(x)
        # outer(x, t) gives shape (m_quad, Nt) when t is 1D
        if t.ndim == 0:
            return np.sum(w * m0 * np.exp(1j * t * x))
        else:
            return (w * m0) @ np.exp(1j * np.outer(x, t))

    def mean(self) -> float:
        return self._mu

    def var(self) -> float:
        return self._var


# ============================================================
# Mollified distribution:
#   Y = (1-alpha) X + alpha U
# where X is your original compact [0,1] distribution
# and U is the bump mollifier on [0,1], independent of X.
#
# This is a "mollified" / smoothed version because its density is
# a scaled mixture integral (a convolution-type smoothing):
#   f_Y(y) = ∫ f_X(x) * (1/alpha) f_U((y-(1-alpha)x)/alpha) dx
# (with appropriate support restrictions).
#
# CF is cheap:
#   φ_Y(t) = φ_X((1-alpha)t) * φ_U(alpha t).
# ============================================================

@dataclass(frozen=True)
class Mollified01:
    """
    Wraps any base distribution on [0,1] (with sample/cf/mean/var),
    and returns the mollified random variable:
        Y = (1-alpha) X + alpha U,
    where U ~ bump mollifier on [0,1] independent of X.

    Required interface for base:
      - base.sample(size=None)
      - base.cf(t)
      - base.mean()
      - base.var()
    """
    base: object
    alpha: float
    moll: object = None
    rng: np.random.Generator = np.random.default_rng()

    def __post_init__(self):
        a = float(self.alpha)
        if not (0.0 < a < 1.0):
            raise ValueError("alpha must be in (0,1).")
        if self.moll is None:
            object.__setattr__(self, "moll", BumpMollifier01(rng=self.rng))

        # sanity: base must have required methods
        for name in ("sample", "cf", "mean", "var"):
            if not hasattr(self.base, name):
                raise TypeError(f"base must implement `{name}()`")

    def sample(self, size=None):
        """
        If size is None -> scalar
        Else -> array of shape (size,)
        """
        X = self.base.sample(size=size)
        U = self.moll.sample(size=size)
        return (1.0 - self.alpha) * X + self.alpha * U

    def cf(self, t):
        """
        φ_Y(t) = φ_X((1-alpha)t) * φ_U(alpha t)
        """
        t = np.asarray(t, dtype=float)
        return self.base.cf((1.0 - self.alpha) * t) * self.moll.cf(self.alpha * t)

    def mean(self) -> float:
        return (1.0 - self.alpha) * float(self.base.mean()) + self.alpha * float(self.moll.mean())

    def var(self) -> float:
        # independence: Var((1-a)X + aU) = (1-a)^2 Var(X) + a^2 Var(U)
        return ((1.0 - self.alpha) ** 2) * float(self.base.var()) + (self.alpha ** 2) * float(self.moll.var())

