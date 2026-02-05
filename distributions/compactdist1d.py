from dataclasses import dataclass
import numpy as np # type: ignore
from scipy import special # type: ignore
from scipy.stats import truncnorm # type: ignore
from utils import Phi_complex, beta_cf_jacobi # type: ignore

# ----------------------------
# Innovation distributions on [0,1]
# ----------------------------
@dataclass(frozen=True)
class Beta:
    """
    Beta(a,b) innovation on [0,1].
    """
    a: float
    b: float
    m_cf: int = 120
    rng: np.random.Generator = np.random.default_rng()

    def sample(self, size=None):
        return self.rng.beta(self.a, self.b, size=size)

    def cf(self, t):
        return beta_cf_jacobi(t, self.a, self.b, m=self.m_cf)

    def mean(self) -> float:
        return self.a / (self.a + self.b)

    def var(self) -> float:
        a, b = self.a, self.b
        return (a * b) / ((a + b) ** 2 * (a + b + 1))


@dataclass(frozen=True)
class TruncNormal01:
    """Normal(mu,sigma^2) truncated to [0,1], with closed-form CF."""
    mu: float
    sigma: float
    rng: np.random.Generator = np.random.default_rng()

    @property
    def _a_std(self) -> float:
        return (0.0 - self.mu) / self.sigma

    @property
    def _b_std(self) -> float:
        return (1.0 - self.mu) / self.sigma

    @property
    def _dist(self):
        return truncnorm(self._a_std, self._b_std, loc=self.mu, scale=self.sigma)

    def sample(self, size=None):
        return self._dist.rvs(size=size, random_state=self.rng)

    def cf(self, t):
        """
        φ(t) = exp(i t mu - 0.5 sigma^2 t^2)/Z * [Φ(b_std - i sigma t) - Φ(a_std - i sigma t)]
        where Z = Φ(b_std) - Φ(a_std).
        """
        t = np.asarray(t, dtype=float)
        a = self._a_std
        b = self._b_std
        Z = special.ndtr(b) - special.ndtr(a)  # Φ for real args

        zb = b - 1j * self.sigma * t
        za = a - 1j * self.sigma * t

        pref = np.exp(1j * t * self.mu - 0.5 * (self.sigma ** 2) * (t ** 2))
        return (pref / Z) * (Phi_complex(zb) - Phi_complex(za))

    def mean(self) -> float:
        return float(self._dist.mean())

    def var(self) -> float:
        return float(self._dist.var())
    

# compactdist2d.py
from dataclasses import dataclass
import numpy as np  # type: ignore

# Reuse your existing 1D innovations:
#   Beta, TruncNormal01
# They must expose: sample(), cf(t), mean(), var()

@dataclass(frozen=True)
class Independent2D:
    """
    Minimal 2D innovation wrapper for independent components on [0,1]^2.

    If Z = (Z1, Z2) with Z1 ⟂ Z2, then for u=(u1,u2):
        φ_Z(u) = E[exp(i(u1 Z1 + u2 Z2))] = φ_{Z1}(u1) * φ_{Z2}(u2).

    This wrapper is designed to plug into AR1Compact2D:
      - sample() -> shape (2,)
      - cf(u) where u has shape (...,2) -> complex array shape u.shape[:-1]
      - mean() -> shape (2,)
      - var()  -> shape (2,)   (diagonal variances)
      - cov()  -> shape (2,2)  (diagonal covariance)
    """
    d1: object
    d2: object
    rng: np.random.Generator = np.random.default_rng()

    def sample(self, size=None):
        """
        Returns:
          if size is None: array shape (2,)
          else: array shape (size, 2)
        """
        z1 = self.d1.sample(size=size)
        z2 = self.d2.sample(size=size)
        return np.stack([z1, z2], axis=-1)

    def cf(self, u):
        """
        u: array-like with last dimension 2, shape (...,2)
        returns: complex array with shape u.shape[:-1]
        """
        u = np.asarray(u, dtype=float)
        if u.shape[-1] != 2:
            raise ValueError("u must have last dimension 2, i.e. shape (...,2).")

        u1 = u[..., 0]
        u2 = u[..., 1]
        return self.d1.cf(u1) * self.d2.cf(u2)

    def mean(self):
        return np.array([float(self.d1.mean()), float(self.d2.mean())], dtype=float)

    def var(self):
        return np.array([float(self.d1.var()), float(self.d2.var())], dtype=float)

    def cov(self):
        v = self.var()
        return np.diag(v)


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # assuming your 1D classes live in compactdist.py
    from compactdist import Beta, TruncNormal01  # type: ignore

    rng = np.random.default_rng(0)

    # Independent Beta innovations in each coordinate
    z1 = Beta(a=1.2, b=1.7, m_cf=120, rng=rng)
    z2 = Beta(a=2.0, b=3.0, m_cf=120, rng=rng)
    innov2d = Independent2D(z1, z2, rng=rng)

    # Quick sanity checks
    u = np.array([[0.1, 0.2], [1.0, -0.5]])  # shape (2,2)
    print("cf(u):", innov2d.cf(u))           # should be shape (2,)
    print("mean:", innov2d.mean())           # shape (2,)
    print("cov:\n", innov2d.cov())           # 2x2 diagonal

    # Sample
    print("one sample:", innov2d.sample())          # (2,)
    print("many samples:", innov2d.sample(size=5))  # (5,2)

