import numpy as np
from .pde_burgers import flux, max_wave_speed


def rusanov_flux(uL: np.ndarray, uR: np.ndarray) -> np.ndarray:
    a = max_wave_speed(uL, uR)
    return 0.5 * (flux(uL) + flux(uR)) - 0.5 * a * (uR - uL)


def laplacian_central(u: np.ndarray, dx: float, bc: str) -> np.ndarray:
    if bc == "periodic":
        um = np.roll(u, 1)
        up = np.roll(u, -1)
    elif bc == "dirichlet":
        # Dirichlet handled by padding ghost values = boundary values
        um = np.empty_like(u)
        up = np.empty_like(u)
        um[1:] = u[:-1]
        um[0] = u[0]
        up[:-1] = u[1:]
        up[-1] = u[-1]
    else:
        raise ValueError(f"Unknown bc={bc}")
    return (up - 2.0*u + um) / (dx*dx)


def rhs_fv_viscous_burgers(u: np.ndarray, dx: float,
                           mu: float, bc: str) -> np.ndarray:
    """
    Semi-discrete FV: u_t = -(F_{i+1/2}-F_{i-1/2})/dx + mu*u_xx
    Here: convection via interface fluxes; viscosity via central Laplacian.
    """
    if bc == "periodic":
        uL = u
        uR = np.roll(u, -1)
        F_iphalf = rusanov_flux(uL, uR)  # i+1/2 for i=0..N-1
        F_imhalf = np.roll(F_iphalf, 1)  # i-1/2
    elif bc == "dirichlet":
        # For Dirichlet, treat boundary fluxes with boundary states
        N = u.size
        u_ext = np.empty(N + 2, dtype=u.dtype)
        u_ext[1:-1] = u
        u_ext[0] = u[0]
        u_ext[-1] = u[-1]
        uL = u_ext[1:-1]
        uR = u_ext[2:]
        F_iphalf = rusanov_flux(uL, uR)
        # i-1/2 uses u_{i-1} and u_i
        uL2 = u_ext[:-2]
        uR2 = u_ext[1:-1]
        F_imhalf = rusanov_flux(uL2, uR2)
    else:
        raise ValueError(f"Unknown bc={bc}")

    convection = -(F_iphalf - F_imhalf) / dx
    diffusion = mu * laplacian_central(u, dx, bc)
    return convection + diffusion


def ssp_rk3_step(u: np.ndarray, dt: float, rhs, *rhs_args) -> np.ndarray:
    """
    SSP-RK3:
      u1 = u + dt L(u)
      u2 = 3/4 u + 1/4 (u1 + dt L(u1))
      u3 = 1/3 u + 2/3 (u2 + dt L(u2))
    """
    u1 = u + dt * rhs(u, *rhs_args)
    u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1, *rhs_args))
    u3 = (1.0/3.0)*u + (2.0/3.0)*(u2 + dt*rhs(u2, *rhs_args))
    return u3


def stable_timestep(u: np.ndarray, dx: float, mu: float, cfl: float) -> float:
    umax = float(np.max(np.abs(u)))
    # convective restriction
    dt_conv = np.inf if umax == 0.0 else cfl * dx / umax
    # diffusive restriction (for explicit Laplacian)
    dt_diff = np.inf if mu == 0.0 else 0.45 * dx*dx / mu
    return min(dt_conv, dt_diff)


def solve_fv(u0: np.ndarray, x: np.ndarray, T: float, mu: float,
             bc: str = "periodic", cfl: float = 0.4, dt: float | None = None,
             save_every: int = 10):
    """
    Returns times, snapshots list
    """
    dx = float(x[1] - x[0])
    u = u0.copy()
    t = 0.0
    n = 0
    times = [t]
    snaps = [u.copy()]

    while t < T - 1e-15:
        dt_eff = dt if dt is not None else stable_timestep(u, dx, mu, cfl)
        if t + dt_eff > T:
            dt_eff = T - t
        u = ssp_rk3_step(u, dt_eff, rhs_fv_viscous_burgers, dx, mu, bc)
        t += dt_eff
        n += 1
        if (n % save_every) == 0 or t >= T - 1e-15:
            times.append(t)
            snaps.append(u.copy())

    return np.array(times), snaps
