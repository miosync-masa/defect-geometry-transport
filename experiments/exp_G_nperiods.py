"""
G2 supplement: how many periods are needed before a finite defect array tracks the infinite-superlattice stop bands.
  W=4 wire, U=8t, N_d = 2,3,4,6,10 defects on row 1 with period a=1..8, E_F swept; corr(G, Bloch fraction).
"""
import numpy as np, json
from _paths import *   # noqa
from negf_toy import Device

W2, L2, U = 4, 60, 8.0
Es = np.linspace(-2.6, -0.2, 49); periods = np.arange(1, 9)
Hp = np.zeros((W2, W2))
for r in range(W2 - 1): Hp[r, r + 1] = Hp[r + 1, r] = -1
def bloch(E, a):
    M = np.eye(2 * W2, dtype=complex)
    for c in range(a):
        Hc = Hp.copy()
        if c == 0: Hc[1, 1] += U
        T = np.zeros((2 * W2, 2 * W2), complex); T[:W2, :W2] = Hc - E * np.eye(W2); T[:W2, W2:] = -np.eye(W2); T[W2:, :W2] = np.eye(W2); M = T @ M
    lam = np.abs(np.linalg.eigvals(M)); return int(np.sum(np.abs(lam - 1) < 1e-6)) // 2
eps2 = -2 * np.cos(np.arange(1, W2 + 1) * np.pi / (W2 + 1))
Nopen = np.array([np.sum(np.abs(E - eps2) < 2) for E in Es])
Nb = np.array([[bloch(E, a) / max(n, 1) for E, n in zip(Es, Nopen)] for a in periods])
out = {}
for Nd in [2, 3, 4, 6, 10]:
    Gm = np.zeros((len(periods), len(Es)))
    for j, E in enumerate(Es):
        d0 = Device(W2, L2, E=E, eta=1e-6); T0 = d0.transmission(); G0, GL, GR = d0.greens(); gl = GL[:W2, :W2]; gr = GR[-W2:, -W2:]
        for i, a in enumerate(periods):
            coords = [(int(30 - (Nd - 1) / 2 * a + k * a), 1) for k in range(Nd)]; coords = [(c, r) for c, r in coords if 0 <= c < L2]
            idx = [d0.idx(c, r) for c, r in coords]; nd = len(idx); Gdd = G0[np.ix_(idx, idx)]
            T = np.linalg.solve(np.eye(nd) - Gdd * U, U * np.eye(nd)); GLR = G0[:W2, -W2:] + G0[:W2, idx] @ T @ G0[idx, -W2:]
            Gm[i, j] = np.real(np.trace(gl @ GLR @ gr @ GLR.conj().T)) / T0
    stop = Nb < 1 - 1e-9
    out[Nd] = {"corr": float(np.corrcoef(Gm.ravel(), Nb.ravel())[0, 1]), "G_stop": float(Gm[stop].mean()), "G_pass": float(Gm[~stop].mean())}
    print(f"N_defects={Nd:2d}: corr(G, Bloch frac)={out[Nd]['corr']:+.3f}  G in stop/pass={out[Nd]['G_stop']:.3f}/{out[Nd]['G_pass']:.3f}")
json.dump({str(k): v for k, v in out.items()}, open("results_G_nperiods.json", "w"), indent=1)
