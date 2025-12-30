import numpy as np
import h5py
from tqdm import tqdm
from src.reference_solver import solve_reference_weno


def random_ic_fourier(x: np.ndarray, K: int = 6, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = np.zeros_like(x)
    for k in range(1, K+1):
        ak = rng.normal(scale=1.0/k)
        bk = rng.normal(scale=1.0/k)
        u += ak*np.sin(2*np.pi*k*(x - x.min())/(x.max()-x.min())) + bk*np.cos(2*np.pi*k*(x - x.min())/(x.max()-x.min()))
    u = u / (np.max(np.abs(u)) + 1e-12)
    return u


def sample_mu(rng: np.random.Generator, mu_min: float, mu_max: float) -> float:
    # log-uniform
    return float(np.exp(rng.uniform(np.log(mu_min), np.log(mu_max))))


def extract_interface_pairs(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uL = u
    uR = np.roll(u, -1)
    return uL, uR


def make_dataset_h5(
    out_path: str,
    num_cases: int = 200,
    N_fine: int = 4000,
    domain=(-1.0, 1.0),
    T: float = 0.5,
    mu_min: float = 1e-3,
    mu_max: float = 1e-1,
    save_stride: int = 20,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    a, b = domain
    x = np.linspace(a, b, N_fine, endpoint=False)

    with h5py.File(out_path, "w") as f:
        # store per-case trajectories in groups (simpler than a giant ragged array)
        cases_grp = f.create_group("cases")

        for c in tqdm(range(num_cases), desc="Generating cases"):
            mu = sample_mu(rng, mu_min, mu_max)
            u0 = random_ic_fourier(x, K=6, seed=int(rng.integers(1_000_000)))

            times, snaps = solve_reference_weno(u0, x, T=T, mu=mu, cfl=0.4, save_every=save_stride)

            cg = cases_grp.create_group(f"case_{c:05d}")
            cg.create_dataset("x", data=x, compression="gzip")
            cg.create_dataset("times", data=times, compression="gzip")

            U = np.stack(snaps, axis=0)  # [Nt, N]
            cg.create_dataset("U", data=U, compression="gzip")
            cg.attrs["mu"] = mu

            # also store interface samples for decoupled training
            # take a subset of time slices to limit size
            take = np.linspace(0, U.shape[0]-1, num=min(10, U.shape[0]), dtype=int)
            UL_list, UR_list = [], []
            for idx in take:
                u = U[idx]
                uL, uR = extract_interface_pairs(u)
                UL_list.append(uL)
                UR_list.append(uR)
            UL = np.concatenate(UL_list)
            UR = np.concatenate(UR_list)
            MU = np.full_like(UL, mu)

            cg.create_dataset("uL", data=UL, compression="gzip")
            cg.create_dataset("uR", data=UR, compression="gzip")
            cg.create_dataset("mu_vec", data=MU, compression="gzip")
