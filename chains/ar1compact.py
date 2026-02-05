import numpy as np # type: ignore

# ----------------------------
# AR(1) chain on [0,1]
# ----------------------------
class AR1Compact1D:
    """
    X_{n+1} = rho X_n + lam Z_{n+1}, with rho = 1-lam and Z in [0,1].

    The stationary CF is:
        φ_X(t) = Π_{k>=0} φ_Z(lam * rho^k * t)

    We approximate using first K terms plus optional quadratic tail correction,
    then invert via FFT on a padded interval [0,L).
    """
    def __init__(self, x0, lam, innov):
        self.x0 = float(x0)
        self.lambda_ = float(lam)
        if not (0.0 < self.lambda_ < 1.0):
            raise ValueError("Need 0 < lambda < 1.")
        self.rho = 1.0 - self.lambda_
        self.innov = innov

        self.state = self.x0
        self.chain = [self.x0]

        self.innov_mean = float(self.innov.mean())
        self.innov_var = float(self.innov.var())

    def simulate_next(self):
        z = float(self.innov.sample())
        self.state = self.rho * self.state + self.lambda_ * z
        self.chain.append(self.state)

    def innovation_cf(self, t):
        return self.innov.cf(t)

    def stationary_density_cf_fft(
        self,
        *,
        N=2**12,
        L=4.0,
        K=None,
        eps_u=1e-8,
    ):
        """
        Compute stationary density on [0,1] via CF product + inverse FFT.

        Key practical choices:
          - Use L>1 padding (default L=4) to reduce t_max and avoid wrap-around.
          - Optionally apply a Lanczos window in frequency domain to reduce Gibbs ripples.
          - Phase unwrapping is usually NOT needed when phi is computed smoothly,
            but it's available as a toggle.

        Returns x,f on [0,1]. If return_cf=True, also returns (t_centered, phiX_centered, K).
        """
        lam = self.lambda_
        rho = self.rho

        if N % 2 != 0:
            raise ValueError("N must be even for centered FFT grids.")

        dx = L / N
        t_max = np.pi / dx  # ~ pi * N / L

        # Choose K so that |lam*rho^K*t_max| <= eps_u
        if K is None:
            if rho == 0.0:
                K = 0
            else:
                target = eps_u / (abs(lam) * max(1.0, abs(t_max)))
                if target >= 1.0:
                    K = 0
                else:
                    K = int(np.ceil(np.log(target) / np.log(abs(rho))))
                    K = max(K, 0)

        # Centered frequency grid: k=-N/2,...,N/2-1
        kk = np.arange(-N // 2, N // 2)
        t = 2.0 * np.pi * kk / L  # (N,)

        # Arguments u_m(t) = lam * rho^m * t
        m = np.arange(K + 1)
        U = (lam * (rho ** m))[:, None] * t[None, :]  # (K+1, N)

        # Evaluate innovation CF on all U
        PhiZ = self.innovation_cf(U)  # expect (K+1, N), complex

        # Build log φ_X(t) in polar form
        abs_PhiZ = np.maximum(np.abs(PhiZ), 1e-300)
        logabs = np.log(abs_PhiZ)  # (K+1, N)
        ang = np.angle(PhiZ)       # (K+1, N)

        log_phiX = np.sum(logabs, axis=0) + 1j * np.sum(ang, axis=0)  # (N,)

        # Optional quadratic tail correction
        mu = self.innov_mean
        var = self.innov_var
        rp = rho ** (K + 1)

        # Tail1: i mu t * rho^(K+1)
        tail1 = 1j * mu * t * rp
        # Tail2: -(1/2) var t^2 * lam^2 * rho^(2(K+1)) / (1-rho^2)
        tail2 = -0.5 * var * (t ** 2) * (lam ** 2) * (rp ** 2) / (1.0 - rho ** 2)

        log_phiX += (tail1 + tail2)

        phiX_centered = np.exp(log_phiX)

        # Enforce Hermitian symmetry explicitly 
        i0 = N // 2
        phiX_centered[i0] = 1.0 + 0.0j
        for j in range(1, i0):
            phiX_centered[i0 - j] = np.conj(phiX_centered[i0 + j])


        # Inverse FFT: f(x_n) ≈ (1/L) Σ φ(t_k) e^{-i t_k x_n}
        phi_fft_order = np.fft.ifftshift(phiX_centered)
        f_full = (1.0 / L) * np.fft.fft(phi_fft_order)
        f_full = np.real(f_full)

        x_full = np.arange(N) * dx  # on [0,L)

        # Restrict to [0,1] (true state space)
        mask = (x_full >= 0.0) & (x_full <= 1.0)
        x = x_full[mask]
        f = f_full[mask]

        # IMPORTANT: do NOT clip negative values before moment checks.
        # The Lanczos window should already greatly reduce negative ripples.
        #if renormalize:
        mass = np.trapz(f, x)
        f = f / mass
        return x, f
    


import numpy as np
from scipy.linalg import solve_discrete_lyapunov

class AR1Compact2D:
    """
    2D compact AR(1)-type chain on [0,1]^2:

        X_{n+1} = A X_n + B Z_{n+1},

    where:
      - X_n in R^2 (you choose parameters so the chain lives in [0,1]^2),
      - Z_{n+1} in [0,1]^2 are i.i.d. "innovations",
      - A is a stable 2x2 matrix (spectral radius < 1),
      - B is a 2x2 matrix (often diagonal scaling).

    Stationary characteristic function:
        φ_X(t) = Π_{k>=0} φ_Z( B^T (A^T)^k t )

    We approximate by truncating to k=0..K and inverting via 2D FFT on [0,L1)×[0,L2).
    """

    def __init__(self, x0, A, B, innov):
        x0 = np.asarray(x0, dtype=float).reshape(2,)
        A = np.asarray(A, dtype=float).reshape(2, 2)
        B = np.asarray(B, dtype=float).reshape(2, 2)

        # basic stability check (not strict proof, but a good guardrail)
        eigA = np.linalg.eigvals(A)
        if np.max(np.abs(eigA)) >= 1.0:
            raise ValueError(f"A must be stable (spectral radius < 1). Got eigs={eigA}")

        self.x0 = x0
        self.A = A
        self.B = B
        self.innov = innov

        self.state = x0.copy()
        self.chain = [x0.copy()]

        # innovation mean/cov used for analytic checks (optional)
        # Expect innov.mean() -> shape (2,)
        #        innov.cov()  -> shape (2,2) (or implement var() for diagonal-only)
        self.innov_mean = np.asarray(self.innov.mean(), dtype=float).reshape(2,)
        if hasattr(self.innov, "cov"):
            self.innov_cov = np.asarray(self.innov.cov(), dtype=float).reshape(2, 2)
        else:
            # fallback: treat as independent with known variances
            v = np.asarray(self.innov.var(), dtype=float).reshape(2,)
            self.innov_cov = np.diag(v)

    # ----------------------------
    # Simulation
    # ----------------------------
    def simulate_next(self):
        z = np.asarray(self.innov.sample(), dtype=float).reshape(2,)
        self.state = self.A @ self.state + self.B @ z
        self.chain.append(self.state.copy())

    # ----------------------------
    # Innovation CF interface
    # ----------------------------
    def innovation_cf(self, u):
        """
        u: array with last dimension 2, e.g. (..., 2)
        returns: complex array with shape u.shape[:-1]
        """
        return self.innov.cf(u)

    # ----------------------------
    # Analytic stationary mean/cov (for validation)
    # ----------------------------
    def analytic_mean_cov(self):
        """
        Stationary mean solves: m = A m + B mZ  =>  (I-A)m = B mZ
        Stationary covariance solves discrete Lyapunov:
            S = A S A^T + B SZ B^T
        """
        I = np.eye(2)
        mean = np.linalg.solve(I - self.A, self.B @ self.innov_mean)
        Q = self.B @ self.innov_cov @ self.B.T
        cov = solve_discrete_lyapunov(self.A, Q)  # solves S = A S A^T + Q
        return mean, cov

    # ----------------------------
    # Stationary density via CF product + 2D FFT
    # ----------------------------
    def stationary_density_cf_fft(
        self,
        *,
        N=(2**10, 2**10),
        L=(4.0, 4.0),
        K=None,
        eps_u=1e-8,
        unwrap_phase=False,
        renormalize=True,
        return_cf=False,
    ):
        """
        Compute stationary density on [0,1]^2 via truncated CF product + 2D FFT.

        Parameters
        ----------
        N : (N1,N2)
            FFT grid sizes along x1, x2. Each must be even.
        L : (L1,L2)
            Padding box size. Larger L reduces wrap-around contamination on [0,1]^2.
        K : int or None
            Truncation level for CF product. If None, chosen by eps_u and max frequency.
        eps_u : float
            Threshold for smallest retained CF argument magnitude.
        unwrap_phase : bool
            Usually not needed if CF is computed smoothly; available as a toggle.
        renormalize : bool
            Renormalize the returned density on [0,1]^2 to integrate to 1.
        return_cf : bool
            If True, also return (t1_grid, t2_grid, phi_centered, K).

        Returns
        -------
        x1, x2, f   (and optionally t1, t2, phi, K)
        where:
          x1: (Nx_keep,) grid on [0,1]
          x2: (Ny_keep,) grid on [0,1]
          f : (Nx_keep, Ny_keep) density values on that grid
        """
        N1, N2 = map(int, N)
        L1, L2 = map(float, L)

        if (N1 % 2) or (N2 % 2):
            raise ValueError("Both N1 and N2 must be even for centered FFT grids.")

        dx1 = L1 / N1
        dx2 = L2 / N2

        # Nyquist-ish max frequency magnitudes in each dimension
        tmax1 = np.pi / dx1
        tmax2 = np.pi / dx2

        # Choose K so that max || B^T (A^T)^K t || is small on the FFT frequency box.
        # We use a conservative bound based on operator norms.
        if K is None:
            AT = self.A.T
            BT = self.B.T

            # conservative bound: ||B^T|| * ||(A^T)^K|| * ||t|| <= eps_u
            # approximate ||t|| by sqrt(tmax1^2 + tmax2^2)
            tmax = np.sqrt(tmax1**2 + tmax2**2)
            normB = np.linalg.norm(BT, ord=2)
            normA = np.linalg.norm(AT, ord=2)

            if normA == 0.0:
                K = 0
            else:
                target = eps_u / (max(1.0, normB * tmax))
                if target >= 1.0:
                    K = 0
                else:
                    # want (normA)^K <= target  => K >= log(target)/log(normA)
                    # (since normA<1 for stable A usually)
                    K = int(np.ceil(np.log(target) / np.log(normA)))
                    K = max(K, 0)

        # Centered frequency grids
        k1 = np.arange(-N1 // 2, N1 // 2)
        k2 = np.arange(-N2 // 2, N2 // 2)
        t1 = 2.0 * np.pi * k1 / L1
        t2 = 2.0 * np.pi * k2 / L2

        # Build 2D mesh of frequency vectors t = (t1,t2)
        T1, T2 = np.meshgrid(t1, t2, indexing="ij")  # shapes (N1,N2)
        T = np.stack([T1, T2], axis=-1)               # shape (N1,N2,2)

        # Precompute matrices M_k = B^T (A^T)^k for k=0..K
        AT = self.A.T
        BT = self.B.T
        M = []
        Pk = np.eye(2)
        for _ in range(K + 1):
            M.append(BT @ Pk)   # 2x2
            Pk = Pk @ AT
        M = np.stack(M, axis=0)  # (K+1,2,2)

        # Evaluate innovation CF at U_k = M_k @ t (with t as column vector).
        # Using tensordot: (N1,N2,2) • (2,2) -> (N1,N2,2)
        # for each k, with matrix transpose as needed.
        # We'll compute all k in a loop (K is usually not huge).
        logabs_sum = np.zeros((N1, N2), dtype=float)
        ang_sum = np.zeros((N1, N2), dtype=float)

        for kk in range(K + 1):
            Mk = M[kk]  # (2,2)
            U = np.tensordot(T, Mk.T, axes=([2], [0]))  # (N1,N2,2) = T @ Mk
            PhiZ = self.innovation_cf(U)                # (N1,N2), complex

            absPhi = np.maximum(np.abs(PhiZ), 1e-300)
            logabs_sum += np.log(absPhi)

            ang = np.angle(PhiZ)
            if unwrap_phase:
                # unwrap in a way consistent across k: unwrap against the accumulated phase
                # (simple option) unwrap along k by carrying a reference is more complex;
                # here we just unwrap relative to previous sum (works well if phases are smooth).
                ang = np.unwrap(np.stack([ang_sum, ang], axis=0), axis=0)[1]
            ang_sum += ang

        log_phi = logabs_sum + 1j * ang_sum
        phi_centered = np.exp(log_phi).astype(np.complex128)

        # Enforce Hermitian symmetry (2D): φ(-t) = conj(φ(t))
        # This is usually already true up to rounding, but we can enforce numerically.
        # For a centered grid, we can enforce by mapping indices (i,j) -> (-i,-j).
        i0, j0 = N1 // 2, N2 // 2
        phi_centered[i0, j0] = 1.0 + 0.0j  # φ(0)=1

        # Enforce conjugate symmetry by iterating over half the grid
        for i in range(N1):
            ii = (- (i - i0)) + i0
            ii %= N1
            for j in range(N2):
                jj = (- (j - j0)) + j0
                jj %= N2
                # enforce only one direction to avoid double-writing
                if (i > ii) or (i == ii and j > jj):
                    continue
                phi_centered[ii, jj] = np.conj(phi_centered[i, j])

        # Inverse FFT:
        # f(x) ≈ (1/(L1 L2)) Σ φ(t_k) exp(-i t_k·x)
        phi_fft_order = np.fft.ifftshift(phi_centered, axes=(0, 1))
        f_full = (1.0 / (L1 * L2)) * np.fft.fft2(phi_fft_order)
        f_full = np.real(f_full)

        x1_full = np.arange(N1) * dx1  # [0,L1)
        x2_full = np.arange(N2) * dx2  # [0,L2)

        # Restrict to [0,1]^2
        n1_keep = min(N1, int(np.floor(1.0 / dx1)) + 1)
        n2_keep = min(N2, int(np.floor(1.0 / dx2)) + 1)

        x1_keep = x1_full[:n1_keep]
        x2_keep = x2_full[:n2_keep]
        f_keep = f_full[:n1_keep, :n2_keep]

        if renormalize:
            # normalize over [0,1]^2 using iterated trapezoid
            mass = np.trapz(np.trapz(f_keep, x2_keep, axis=1), x1_keep, axis=0)
            if mass != 0.0:
                f_keep = f_keep / mass

        if return_cf:
            return x1_keep, x2_keep, f_keep, (t1, t2), phi_centered, K
        return x1_keep, x2_keep, f_keep

    
