import numpy as np


def flux(u: np.ndarray) -> np.ndarray:
    # inviscid Burgers flux f(u) = 0.5 u^2
    return 0.5 * u**2


def max_wave_speed(uL: np.ndarray, uR: np.ndarray) -> np.ndarray:
    # For scalar Burgers: a(u)=f'(u)=u, local speed is max(|uL|,|uR|)
    return np.maximum(np.abs(uL), np.abs(uR))
