"""
Stage 0c — permutation-only disorder (phonon analogue of Exp A).
  Fix the MULTISET of W interface bond strengths {κ_r}; enumerate all W! arrangements across the W rows.
  Jensen/tail effects are frozen (same multiset) — only ORDER changes.
  Measures: spread of G across permutations; spectral distance to DMM per permutation and for the
  permutation-averaged spectrum; comparison with the uniform interface at the same arithmetic mean;
  and the mode-overlap explanation: corr(G, Σ_r κ_r φ_n(r)²) for the lowest transverse modes.
"""
import numpy as np, json, itertools
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _paths import *   # noqa

W = 6
MAT = {"A": dict(m=4.0, k=1.0), "B": dict(m=1.0, k=9.0)}

def lead_blocks(mat):
    m, k = mat["m"], mat["k"]; H00 = np.zeros((W, W)); H01 = -(k / m) * np.eye(W)
    for r in range(W):
        ntrans = (1 if r > 0 else 0) + (1 if r + 1 < W else 0); H00[r, r] = (2 + ntrans) * k / m
        if r + 1 < W: H00[r, r + 1] = H00[r + 1, r] = -(k / m)
    return H00, H01
def surface_gf(E, H00, H01, eta=1e-5):
    z = (E + 1j * eta) * np.eye(W); es, eb, a, b = H00.astype(complex).copy(), H00.astype(complex).copy(), H01.astype(complex).copy(), H01.conj().T.copy()
    for _ in range(300):
        g = np.linalg.inv(z - eb); agb = a @ g @ b; bga = b @ g @ a; es = es + agb; eb = eb + agb + bga; a = a @ g @ a; b = b @ g @ b
        if np.linalg.norm(a) < 1e-13 and np.linalg.norm(b) < 1e-13: break
    return np.linalg.inv(z - es)
def n_open(E, mat):
    m, k = mat["m"], mat["k"]; n = np.arange(0, W); eps = (k / m) * (4 - 2 * np.cos(n * np.pi / W)); return int(np.sum(np.abs(E - eps) < 2 * (k / m)))
def dmm(E):
    nA, nB = n_open(E, MAT["A"]), n_open(E, MAT["B"]); return nA * nB / (nA + nB) if nA + nB > 0 else 0.0

# device: 6 columns A | 6 columns B ; the A–B bonds (W of them) carry κ_r
NA, NB = 6, 6; L = NA + NB; N = W * L
cols = ["A"] * NA + ["B"] * NB; mass = np.array([MAT[c]["m"] for c in cols for _ in range(W)], float)
kAB = np.sqrt(MAT["A"]["k"] * MAT["B"]["k"])
def idx(c, r): return c * W + r
def dyn_base():
    Dm = np.zeros((N, N))
    for c in range(L):
        kc = MAT[cols[c]]["k"]
        for r in range(W):
            i = idx(c, r)
            if r + 1 < W: j = idx(c, r + 1); Dm[i, j] = Dm[j, i] = -kc / np.sqrt(mass[i] * mass[j]); Dm[i, i] += kc / mass[i]; Dm[j, j] += kc / mass[j]
            if c + 1 < L and cols[c] == cols[c + 1]: j = idx(c + 1, r); Dm[i, j] = Dm[j, i] = -kc / np.sqrt(mass[i] * mass[j]); Dm[i, i] += kc / mass[i]; Dm[j, j] += kc / mass[j]
    kA = MAT["A"]["k"]; kB = MAT["B"]["k"]
    for r in range(W): Dm[r, r] += kA / mass[r]; Dm[N - W + r, N - W + r] += kB / mass[N - W + r]
    return Dm
D0 = dyn_base()
def with_interface(kappas):
    Dm = D0.copy(); c = NA - 1
    for r in range(W):
        i, j = idx(c, r), idx(c + 1, r); k = kAB * kappas[r]
        Dm[i, j] = Dm[j, i] = -k / np.sqrt(mass[i] * mass[j]); Dm[i, i] += k / mass[i]; Dm[j, j] += k / mass[j]
    return Dm

wmaxA = np.sqrt(8 * MAT["A"]["k"] / MAT["A"]["m"]); omegas = np.linspace(0.02, wmaxA * 1.02, 120); Es = omegas ** 2
eta = 1e-5; HA = lead_blocks(MAT["A"]); HB = lead_blocks(MAT["B"])
cache = []
for E in Es:
    gA = surface_gf(E, *HA, eta); gB = surface_gf(E, *HB, eta)
    cA = -MAT["A"]["k"] / np.sqrt(mass[0] * MAT["A"]["m"]); cB = -MAT["B"]["k"] / np.sqrt(mass[-1] * MAT["B"]["m"])
    SA = np.zeros((N, N), complex); SB = np.zeros((N, N), complex); SA[:W, :W] = cA * gA * cA; SB[-W:, -W:] = cB * gB * cB
    cache.append((SA, SB, 1j * (SA - SA.conj().T), 1j * (SB - SB.conj().T)))
def spectrum(kappas):
    Dm = with_interface(kappas); out = np.zeros(len(Es))
    for n, E in enumerate(Es):
        SA, SB, GL, GR = cache[n]; G = np.linalg.inv((E + 1j * eta) * np.eye(N) - Dm - SA - SB)
        out[n] = max(np.real(np.trace(GL @ G @ GR @ G.conj().T)), 0.0)
    return out
def conductance(Xi, T=1.0):
    x = omegas / T; dn = (x / T) * np.exp(x) / np.expm1(x) ** 2; dn[~np.isfinite(dn)] = 0
    return np.trapezoid(omegas * Xi * dn, omegas) / (2 * np.pi * W)
Xi_dmm = np.array([dmm(E) for E in Es]); G_dmm = conductance(Xi_dmm); norm = np.trapezoid(Xi_dmm, omegas)
def dist(Xi): return np.trapezoid(np.abs(Xi - Xi_dmm), omegas) / norm

# transverse mode profiles of the A lead (free boundary): φ_n(r) ∝ cos(nπ(r+1/2)/W)
phi = np.array([np.cos(n * np.pi * (np.arange(W) + 0.5) / W) for n in range(W)]); phi /= np.linalg.norm(phi, axis=1, keepdims=True)

rng = np.random.default_rng(5)
multisets = {"uniform": np.ones(W)}
for name, s in [("low-spread (σ=0.3)", 0.3), ("high-spread (σ=1.0)", 1.0)]:
    v = np.exp(s * rng.normal(size=W) - s ** 2 / 2); multisets[name] = v / v.mean()      # arithmetic mean exactly 1
results = {}
for name, ks in multisets.items():
    perms = [np.array(ks)] if name == "uniform" else [np.array(p) for p in set(itertools.permutations(np.round(ks, 10)))]
    Xis = np.array([spectrum(p) for p in perms]); Gs = np.array([conductance(x) for x in Xis]); ds = np.array([dist(x) for x in Xis])
    Xbar = Xis.mean(0)
    ov = np.array([[np.sum(p * phi[n] ** 2) for n in range(3)] for p in perms])              # mode-weighted bond strength, n=0,1,2
    corr = [float(np.corrcoef(Gs, ov[:, n])[0, 1]) if len(perms) > 2 else float("nan") for n in range(3)]
    results[name] = {"multiset": list(map(float, ks)), "n_perm": len(perms), "G_min": float(Gs.min()), "G_max": float(Gs.max()), "G_mean": float(Gs.mean()), "G_std": float(Gs.std()),
                     "G_over_Gdmm_mean": float(Gs.mean() / G_dmm), "dist_per_perm_mean": float(ds.mean()), "dist_per_perm_std": float(ds.std()), "dist_perm_avg_spectrum": float(dist(Xbar)),
                     "corr_G_modeoverlap_n0_n1_n2": corr, "Xbar": Xbar.tolist(), "Gs": Gs.tolist()}
    print(f"{name:20s} perms={len(perms):3d}  G: {Gs.min():.4f}–{Gs.max():.4f} (×{Gs.max()/max(Gs.min(),1e-12):.2f}) mean/G_DMM={Gs.mean()/G_dmm:.3f} | dist per-perm {ds.mean():.3f}±{ds.std():.3f}, perm-avg spectrum {dist(Xbar):.3f} | corr(G, Σκφ²) n=0,1,2: {np.round(corr,2)}")
json.dump({"omegas": omegas.tolist(), "Xi_dmm": Xi_dmm.tolist(), "G_dmm": G_dmm, "results": results}, open("stage0c_permutation.json", "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
for name, c in zip(multisets, ["k", "C0", "C3"]):
    R = results[name]
    if R["n_perm"] > 1: ax[0].hist(np.array(R["Gs"]) / G_dmm, bins=30, alpha=0.6, color=c, label=f"{name} ({R['n_perm']} perms)")
    else: ax[0].axvline(R["G_mean"] / G_dmm, color=c, lw=2, label=name)
    ax[1].plot(omegas, R["Xbar"], color=c, label=f"{name}: permutation-averaged Ξ")
ax[0].axvline(1, color="gray", ls=":", label="DMM"); ax[0].set_xlabel("G / G_DMM"); ax[0].set_ylabel("count"); ax[0].legend(fontsize=7); ax[0].set_title("same multiset of bond strengths, all orderings")
ax[1].plot(omegas, Xi_dmm, "--", color="gray", label="DMM"); ax[1].set_xlabel("ω"); ax[1].set_ylabel("Ξ(ω)"); ax[1].legend(fontsize=7); ax[1].set_title("order-averaged spectrum vs DMM")
R = results["high-spread (σ=1.0)"]; perms = [np.array(p) for p in set(itertools.permutations(np.round(multisets["high-spread (σ=1.0)"], 10)))]
ov0 = [np.sum(p * phi[0] ** 2) for p in perms]; ov1 = [np.sum(p * phi[1] ** 2) for p in perms]
sc = ax[2].scatter(ov1, R["Gs"], c=ov0, s=8, cmap="viridis"); plt.colorbar(sc, ax=ax[2], label="Σκ_r φ_0(r)²")
ax[2].set_xlabel("Σ_r κ_r φ_1(r)²  (bond strength seen by mode 1)"); ax[2].set_ylabel("G"); ax[2].set_title("high-spread: G vs mode-weighted bond strength")
plt.suptitle("Stage 0c: permutation-only disorder — order alone, Jensen tail frozen")
plt.tight_layout(); plt.savefig(FIGDIR + "figX_phonon_permutation.png", dpi=130); plt.close(); print("done")
