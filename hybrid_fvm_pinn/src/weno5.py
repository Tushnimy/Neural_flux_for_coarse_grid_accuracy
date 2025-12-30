import numpy as np


def lf_split_flux(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Lax-Friedrichs flux splitting:
      f(u) = f^+(u) + f^-(u)
      f^± = 0.5 (f(u) ± alpha u)
    alpha = max |f'(u)| = max|u| for Burgers
    """
    f = 0.5*u*u
    alpha = float(np.max(np.abs(u)))
    fp = 0.5*(f + alpha*u)
    fm = 0.5*(f - alpha*u)
    return fp, fm, alpha


def weno5_reconstruct_plus(fp: np.ndarray) -> np.ndarray:
    """
    Reconstruct numerical flux at i+1/2 using left-biased WENO5 on fp.
    Returns F_{i+1/2} for i=0..N-1
    (periodic assumed; caller supplies padded fp).
    """
    eps = 1e-6
    # stencil: fp[i-2], fp[i-1], fp[i], fp[i+1], fp[i+2]
    f_im2 = fp[:-4]
    f_im1 = fp[1:-3]
    f_i   = fp[2:-2]
    f_ip1 = fp[3:-1]
    f_ip2 = fp[4:]

    # Candidate polynomials (Jiang-Shu)
    p0 = (1/3)*f_im2 - (7/6)*f_im1 + (11/6)*f_i
    p1 = (-1/6)*f_im1 + (5/6)*f_i + (1/3)*f_ip1
    p2 = (1/3)*f_i + (5/6)*f_ip1 - (1/6)*f_ip2

    # Smoothness indicators beta
    b0 = (13/12)*(f_im2 - 2*f_im1 + f_i)**2 + (1/4)*(f_im2 - 4*f_im1 + 3*f_i)**2
    b1 = (13/12)*(f_im1 - 2*f_i + f_ip1)**2 + (1/4)*(f_im1 - f_ip1)**2
    b2 = (13/12)*(f_i - 2*f_ip1 + f_ip2)**2 + (1/4)*(3*f_i - 4*f_ip1 + f_ip2)**2

    d0, d1, d2 = 0.1, 0.6, 0.3
    a0 = d0 / (eps + b0)**2
    a1 = d1 / (eps + b1)**2
    a2 = d2 / (eps + b2)**2
    w0 = a0 / (a0 + a1 + a2)
    w1 = a1 / (a0 + a1 + a2)
    w2 = a2 / (a0 + a1 + a2)
    return w0*p0 + w1*p1 + w2*p2


def weno5_reconstruct_minus(fm: np.ndarray) -> np.ndarray:
    """
    Reconstruct numerical flux at i+1/2 using right-biased WENO5 on fm
    (equivalently left-biased on reversed direction).
    Caller supplies padded fm.
    """
    eps = 1e-6
    # stencil for right-biased: fm[i+3],fm[i+2],fm[i+1],fm[i],fm[i-1]
    f_ip3 = fm[5:]
    f_ip2 = fm[4:-1]
    f_ip1 = fm[3:-2]
    f_i   = fm[2:-3]
    f_im1 = fm[1:-4]

    p0 = (1/3)*f_ip3 - (7/6)*f_ip2 + (11/6)*f_ip1
    p1 = (-1/6)*f_ip2 + (5/6)*f_ip1 + (1/3)*f_i
    p2 = (1/3)*f_ip1 + (5/6)*f_i - (1/6)*f_im1

    b0 = (13/12)*(f_ip3 - 2*f_ip2 + f_ip1)**2 + (1/4)*(f_ip3 - 4*f_ip2 + 3*f_ip1)**2
    b1 = (13/12)*(f_ip2 - 2*f_ip1 + f_i)**2 + (1/4)*(f_ip2 - f_i)**2
    b2 = (13/12)*(f_ip1 - 2*f_i + f_im1)**2 + (1/4)*(3*f_ip1 - 4*f_i + f_im1)**2

    d0, d1, d2 = 0.1, 0.6, 0.3
    a0 = d0 / (eps + b0)**2
    a1 = d1 / (eps + b1)**2
    a2 = d2 / (eps + b2)**2
    w0 = a0 / (a0 + a1 + a2)
    w1 = a1 / (a0 + a1 + a2)
    w2 = a2 / (a0 + a1 + a2)
    return w0*p0 + w1*p1 + w2*p2


def weno5_flux_derivative(u: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute d/dx f(u) using WENO5 + LF flux splitting, periodic BC.
    """
    fp, fm, _ = lf_split_flux(u)

    pad = 4
    fp_pad = np.r_[fp[-pad:], fp, fp[:pad]]
    fm_pad = np.r_[fm[-pad:], fm, fm[:pad]]

    Fp = weno5_reconstruct_plus(fp_pad)   
    Fm = weno5_reconstruct_minus(fm_pad)

    N = u.size
    start = 2  
    Fp_c = Fp[start:start+N]
    Fm_c = Fm[start:start+N]
    F = Fp_c + Fm_c

    # flux difference: (F_{i+1/2} - F_{i-1/2})/dx with periodic shift
    F_imhalf = np.roll(F, 1)
    return (F - F_imhalf) / dx
