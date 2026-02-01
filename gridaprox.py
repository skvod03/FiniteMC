import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse.linalg import eigs

def OneDGridAprox(Chain, h, m=10):
    pts = np.linspace(0, 1, int(1 / h) + 1)
    left = pts[:-1]
    right = pts[1:]
    x_grid = (left + right) / 2
    intervals = np.vstack((left, right)).T
    N = x_grid.size

    # Quadrature on each destination interval
    xi, wi = leggauss(m)
    a = intervals[:, 0]
    b = intervals[:, 1]
    half = (b - a) / 2.0                               # (N,)
    mid  = (a + b) / 2.0                               # (N,)
    t = mid[:, None] + half[:, None] * xi[None, :]     # (N,m) nodes per dest cell

    # We want P[i,ell] = half[ell] * sum_j wi[j] * K(x_i, t[ell,j])
    # Build all pairs (x_i, y_{ell,j}) in one shot by flattening destination nodes
    y_flat = t.ravel()                                  # (N*m,)
    # Evaluate K for all x_i against all y_flat -> (N, N*m)
    K_all = Chain.kernel_xy(x_grid, y_flat)    # expect broadcasting to (N, N*m)

    # Reshape to (N, N, m): source i, dest ell, quad j
    K_reshaped = K_all.reshape(N, N, m)

    # Weighted sum over j then multiply by half-length of each dest interval
    P = (K_reshaped @ wi) * half[None, :]               # (N,N)

    def stationary_sparse(P_sparse):
        w, v = eigs(P_sparse.T, k=1, sigma=1.0)  # shift-invert near 1
        pi = np.real(v[:, 0])
        pi = np.abs(pi)
        pi /= pi.sum()
        return pi, w[0]
    
    pi = stationary_sparse(P)[0]

    return pi, x_grid                               

def Eh_pi(pi, x_grid, g):
    # pi is cell mass, sum(pi)=1
    return np.sum(pi * g(x_grid))

def test_functions(name, *params):
    if name == "moment":
        r = params[0]
        return lambda x, r = r: x**r
    if name == "beta_mixed":
        p, q = params
        return lambda x, p=p, q=q: (x**p) * ((1-x)**q)
    if name == "logx": 
        return np.log
    if name == "log1mx":
        return lambda x: np.log(1-x)
