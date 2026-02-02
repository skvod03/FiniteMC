import numpy as np

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

        # Enforce Hermitian symmetry explicitly (numerical hygiene)
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
    
