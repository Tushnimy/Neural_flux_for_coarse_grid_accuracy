import numpy as np
import tensorflow as tf
from src.fv1d import laplacian_central, ssp_rk3_step, stable_timestep
from src.pde_burgers import flux as physical_flux, max_wave_speed


def rusanov_interface_flux(uL: np.ndarray, uR: np.ndarray) -> np.ndarray:
    a = max_wave_speed(uL, uR)
    return 0.5 * (physical_flux(uL) + physical_flux(uR)) - 0.5 * a * (uR - uL)


@tf.function
def _nn_flux_batch(model, uL, uR, log_mu):
    inp = tf.concat([uL, uR, log_mu], axis=1)
    return model(inp)


def nn_interface_flux(model, uL: np.ndarray, uR: np.ndarray, mu: float) -> np.ndarray:
    # uL,uR shape (N,)
    uL_tf = tf.convert_to_tensor(uL.reshape(-1, 1), dtype=tf.float32)
    uR_tf = tf.convert_to_tensor(uR.reshape(-1, 1), dtype=tf.float32)
    log_mu_tf = tf.fill([uL_tf.shape[0], 1], tf.cast(tf.math.log(mu), tf.float32))
    fhat = _nn_flux_batch(model, uL_tf, uR_tf, log_mu_tf)
    return fhat.numpy().reshape(-1)


def rhs_fv(u: np.ndarray, dx: float, mu: float, bc: str, iface_flux_fn) -> np.ndarray:
    # periodic assumed; extend if you need Dirichlet
    if bc != "periodic":
        raise NotImplementedError("rhs_fv currently implemented for periodic BCs only.")
    uL = u
    uR = np.roll(u, -1)
    F_iphalf = iface_flux_fn(uL, uR)          # i+1/2
    F_imhalf = np.roll(F_iphalf, 1)           # i-1/2
    dudt = -(F_iphalf - F_imhalf) / dx
    dudt += mu * laplacian_central(u, dx, bc="periodic")
    return dudt


def simulate_steps(
    u0: np.ndarray,
    x: np.ndarray,
    mu: float,
    nsteps: int,
    model=None,
    dt: float | None = None,
    cfl: float = 0.4,
    bc: str = "periodic",
    save_every: int = 10,
):
    """
    Simulate viscous Burgers on a coarse FV grid for a fixed number of RK3 steps.
    If model is None -> Rusanov flux.
    Returns: times (np.ndarray), snaps (list[np.ndarray])
    """
    dx = float(x[1] - x[0])
    u = u0.copy()
    t = 0.0
    times = [t]
    snaps = [u.copy()]

    if model is None:
        def iface(uL, uR): return rusanov_interface_flux(uL, uR)
    else:
        def iface(uL, uR): return nn_interface_flux(model, uL, uR, mu)

    for n in range(1, nsteps + 1):
        dt_eff = dt if dt is not None else stable_timestep(u, dx, mu, cfl)
        u = ssp_rk3_step(u, dt_eff, lambda uu, *_: rhs_fv(uu, dx, mu, bc, iface))
        t += dt_eff
        if (n % save_every) == 0 or n == nsteps:
            times.append(t)
            snaps.append(u.copy())

    return np.array(times), snaps
