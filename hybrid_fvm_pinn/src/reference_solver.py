import numpy as np
from .weno5 import weno5_flux_derivative


def laplacian_periodic(u: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(u, -1) - 2*u + np.roll(u, 1)) / (dx*dx)


def ssp_rk3(u, dt, rhs):
    u1 = u + dt*rhs(u)
    u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1))
    u3 = (1/3)*u + (2/3)*(u2 + dt*rhs(u2))
    return u3


def stable_dt_weno(u: np.ndarray, dx: float, mu: float, cfl: float) -> float:
    umax = float(np.max(np.abs(u)))
    dt_conv = np.inf if umax == 0 else cfl * dx / umax
    dt_diff = np.inf if mu == 0 else 0.45 * dx*dx / mu
    return min(dt_conv, dt_diff)


def solve_reference_weno(u0: np.ndarray, x: np.ndarray, T: float, mu: float,
                         cfl: float = 0.4, dt: float | None = None,
                         save_every: int = 20):
    dx = float(x[1] - x[0])
    u = u0.copy()
    t = 0.0
    n = 0
    times = [t]
    snaps = [u.copy()]

    def rhs(v):
        # v_t + (f(v))_x = mu v_xx  => v_t = - (f(v))_x + mu v_xx
        conv = -weno5_flux_derivative(v, dx)
        diff = mu * laplacian_periodic(v, dx)
        return conv + diff

    while t < T - 1e-15:
        dt_eff = dt if dt is not None else stable_dt_weno(u, dx, mu, cfl)
        if t + dt_eff > T:
            dt_eff = T - t
        u = ssp_rk3(u, dt_eff, rhs)
        t += dt_eff
        n += 1
        if (n % save_every) == 0 or t >= T - 1e-15:
            times.append(t)
            snaps.append(u.copy())

    return np.array(times), snaps
