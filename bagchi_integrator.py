import numpy as np
import matplotlib.pyplot as plt

def bagchi_step(Omega, Delta, eps):
    N = Omega.shape[0]
    for parity in (0, 1):
        new = Omega.copy()
        for j in range(parity, N, 2):
            jp = (j + 1) % N
            jm = (j - 1) % N
            B = np.array([
                Omega[jp, 0] + Omega[jm, 0],
                Omega[jp, 1] + Omega[jm, 1],
                Delta * (Omega[jp, 2] + Omega[jm, 2])
            ], dtype=float)
            Bnorm = np.linalg.norm(B)
            if Bnorm < 1e-12:
                new[j] = Omega[j]
                continue
            bhat = B / Bnorm
            phi = Bnorm * eps
            Oj = Omega[j]
            rotated = (Oj * np.cos(phi)
                       + np.cross(Oj, bhat) * np.sin(phi)
                       + bhat * (bhat.dot(Oj)) * (1 - np.cos(phi)))
            n = np.linalg.norm(rotated)
            if n > 0:
                rotated /= n
            new[j] = rotated
        Omega = new
    return Omega

def init_helix(N, theta, q):
    return np.array([
        [np.cos(q*j) * np.sin(theta),
         np.sin(q*j) * np.sin(theta),
         np.cos(theta)]
        for j in range(N)
    ], dtype=float)

def energy(Omega, Delta):
    xp = np.roll(Omega, -1, axis=0)
    return float(np.dot(Omega[:,0], xp[:,0])
                 + np.dot(Omega[:,1], xp[:,1])
                 + Delta * np.dot(Omega[:,2], xp[:,2]))

def Sz_total(Omega):
    return float(Omega[:,2].sum())

def simulate(N, theta, q, Delta, eps, T, record=False, max_frames=200):
    Omega = init_helix(N, theta, q)
    E0 = energy(Omega, Delta)
    Sz0 = Sz_total(Omega)
    steps = int(np.round(T / eps))
    times = np.arange(1, steps + 1, dtype=float) * eps
    errE, errSz = [], []
    frames = []
    stride = max(1, steps // max_frames)
    for k in range(steps):
        Omega = bagchi_step(Omega, Delta, eps)
        if (k + 1) % stride == 0 and record:
            frames.append(Omega.copy())
        e = energy(Omega, Delta)
        sz = Sz_total(Omega)
        errE.append(abs(e - E0) / max(1e-15, abs(E0)))
        errSz.append(abs(sz - Sz0) / max(1e-15, abs(Sz0)))
    frames = np.array(frames) if record and frames else None
    return times, np.array(errE), np.array(errSz), frames

N = 24
theta = np.pi/4
q = np.pi/3
T = 20.0
eps_values = [0.1, 0.05, 0.02, 0.01]
deltas = [0.5, 2.0]

for Delta in deltas:
    plt.figure(figsize=(6, 4))
    for eps in eps_values:
        times, errE, errSz, _ = simulate(N, theta, q, Delta, eps, T, record=False)
        plt.loglog(times, np.maximum(errE, 1e-16), label=f'ε={eps}')
    plt.axhline(1e-2, linestyle='--', color='tab:blue', label='1% threshold')
    plt.xlabel('Time')
    plt.ylabel('Relative energy error')
    plt.title(f'Energy error, Δ={Delta}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"energy_error_Delta{str(Delta).replace('.', 'p')}.png", dpi=220)
    plt.show()

    plt.figure(figsize=(6, 4))
    for eps in eps_values:
        times, errE, errSz, _ = simulate(N, theta, q, Delta, eps, T, record=False)
        plt.loglog(times, np.maximum(errSz, 1e-16), label=f'ε={eps}')
    plt.axhline(1e-2, linestyle='--', color='tab:blue', label='1% threshold')
    plt.xlabel('Time')
    plt.ylabel('Relative Sz error')
    plt.title(f'Sz drift, Δ={Delta}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"sz_error_Delta{str(Delta).replace('.', 'p')}.png", dpi=220)
    plt.show()

    eps_show = min(eps_values)
    times_show, _, _, frames = simulate(N, theta, q, Delta, eps_show, T, record=True, max_frames=200)
    if frames is not None:
        Sx = frames[..., 0]
        Sy = frames[..., 1]
        Sz = frames[..., 2]
        F = Sx.shape[0]
        t_axis = np.linspace(0, T, F)
        fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
        im0 = axs[0].imshow(Sx, aspect='auto', origin='lower',
                            extent=[0, N-1, t_axis[0], t_axis[-1]])
        axs[0].set_ylabel('time')
        axs[0].set_title(f'Sx(j,t), Δ={Delta}, ε={eps_show}')
        fig.colorbar(im0, ax=axs[0], pad=0.01)
        im1 = axs[1].imshow(Sy, aspect='auto', origin='lower',
                            extent=[0, N-1, t_axis[0], t_axis[-1]])
        axs[1].set_ylabel('time')
        axs[1].set_title('Sy(j,t)')
        fig.colorbar(im1, ax=axs[1], pad=0.01)
        im2 = axs[2].imshow(Sz, aspect='auto', origin='lower',
                            extent=[0, N-1, t_axis[0], t_axis[-1]])
        axs[2].set_ylabel('time')
        axs[2].set_xlabel('site j')
        axs[2].set_title('Sz(j,t)')
        fig.colorbar(im2, ax=axs[2], pad=0.01)
        fig.tight_layout()
        plt.savefig(f"helix_components_Delta{str(Delta).replace('.', 'p')}_eps{str(eps_show).replace('.', 'p')}.png", dpi=220)
        plt.show()
