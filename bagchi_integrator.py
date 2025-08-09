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
    return float(np.dot(Omega[:,0], xp[:,0]) +
                 np.dot(Omega[:,1], xp[:,1]) +
                 Delta * np.dot(Omega[:,2], xp[:,2]))

def Sz_total(Omega):
    return float(Omega[:,2].sum())

N = 24
theta = np.pi/4
q = np.pi/3
T = 20.0
eps_values = [0.1, 0.05, 0.02, 0.01]
deltas = [0.5, 2.0]

for Delta in deltas:
    plt.figure(figsize=(6, 4))
    for eps in eps_values:
        Omega = init_helix(N, theta, q)
        E0 = energy(Omega, Delta)
        steps = int(np.round(T / eps))
        times = np.arange(1, steps + 1, dtype=float) * eps
        errE = []
        for _ in range(steps):
            Omega = bagchi_step(Omega, Delta, eps)
            rel_err_E = abs(energy(Omega, Delta) - E0) / max(1e-15, abs(E0))
            errE.append(rel_err_E)
        plt.loglog(times, errE, label=f'ε={eps}')
    plt.axhline(1e-2, linestyle='--', color='tab:blue', label='1% threshold')
    plt.xlabel('Time')
    plt.ylabel('Relative energy error')
    plt.title(f'Bagchi: Energy error Δ={Delta}')
    plt.legend()
    plt.tight_layout()
    outname = f"bagchi_energy_error_Delta{str(Delta).replace('.', 'p')}.png"
    plt.savefig(outname, dpi=220)
    plt.show()
