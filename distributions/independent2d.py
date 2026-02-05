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




