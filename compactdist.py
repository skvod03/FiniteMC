from dataclasses import dataclass
import numpy as np
from scipy import special
from scipy.stats import truncnorm
from utils import Phi_complex, beta_cf_jacobi

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
