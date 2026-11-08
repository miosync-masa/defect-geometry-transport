"""
Stage 0b — does coherent AGF approach DMM as interface disorder erases phase history?
  Interface bonds (and interlayer masses) randomised: k_r = k̄·exp(σξ − σ²/2) (arithmetic mean fixed), m_r = m̄·exp(σξ − σ²/2).
  Configs: direct A|B, A|I(6)|B. Metrics vs σ: spectral distance ∫|Ξ_AGF − Ξ_DMM|dω / ∫Ξ_DMM dω, G/G_DMM at T=1,
  and the Fabry–Pérot contrast of Ξ(ω) in the interlayer case.
"""
import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _paths import *   # noqa

W = 6
MAT = {"A": dict(m=4.0, k=1.0), "B": dict(m=1.0, k=9.0), "I": dict(m=2.45, k=2.45)}
rng = np.random.default_rng(3)

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

def build(layers, sigma, rng):
    """returns per-site mass, bond dict; disorder on bonds that touch a dissimilar material and on interlayer masses"""
    cols = []
    for mat, n in layers: cols += [mat] * n
    L = len(cols); N = W * L
    mass = np.array([MAT[c]["m"] for c in cols for _ in range(W)], float)
    for c in range(L):
        if cols[c] == "I":
            for r in range(W): mass[c * W + r] *= np.exp(sigma * rng.normal() - sigma ** 2 / 2)
    def idx(c, r): return c * W + r
    K = {}
    for c in range(L):
        kc = MAT[cols[c]]["k"]
        for r in range(W):
            if r + 1 < W: K[(idx(c, r), idx(c, r + 1))] = kc
            if c + 1 < L:
                kb = np.sqrt(kc * MAT[cols[c + 1]]["k"])
                if cols[c] != cols[c + 1]: kb *= np.exp(sigma * rng.normal() - sigma ** 2 / 2)
                K[(idx(c, r), idx(c + 1, r))] = kb
    kA = np.array([np.sqrt(MAT[cols[0]]["k"] * MAT["A"]["k"]) * (np.exp(sigma * rng.normal() - sigma ** 2 / 2) if cols[0] != "A" else 1.0) for _ in range(W)])
    kB = np.array([np.sqrt(MAT[cols[-1]]["k"] * MAT["B"]["k"]) * (np.exp(sigma * rng.normal() - sigma ** 2 / 2) if cols[-1] != "B" else 1.0) for _ in range(W)])
    return cols, mass, K, L, N, kA, kB

def spectrum(layers, sigma, rng, Es, eta=1e-5):
    cols, mass, K, L, N, kA, kB = build(layers, sigma, rng)
    Dm = np.zeros((N, N))
    for (i, j), k in K.items():
        Dm[i, j] = Dm[j, i] = -k / np.sqrt(mass[i] * mass[j]); Dm[i, i] += k / mass[i]; Dm[j, j] += k / mass[j]
    for r in range(W): Dm[r, r] += kA[r] / mass[r]; Dm[N - W + r, N - W + r] += kB[r] / mass[N - W + r]
    cA = -kA / np.sqrt(mass[:W] * MAT["A"]["m"]); cB = -kB / np.sqrt(mass[-W:] * MAT["B"]["m"])
    HA = lead_blocks(MAT["A"]); HB = lead_blocks(MAT["B"]); out = []
    for E in Es:
        gA = surface_gf(E, *HA, eta); gB = surface_gf(E, *HB, eta)
        SA = np.zeros((N, N), complex); SB = np.zeros((N, N), complex)
        SA[:W, :W] = np.diag(cA) @ gA @ np.diag(cA); SB[-W:, -W:] = np.diag(cB) @ gB @ np.diag(cB)
        G = np.linalg.inv((E + 1j * eta) * np.eye(N) - Dm - SA - SB); GL = 1j * (SA - SA.conj().T); GR = 1j * (SB - SB.conj().T)
        out.append(max(np.real(np.trace(GL @ G @ GR @ G.conj().T)), 0.0))
    return np.array(out)

def conductance(Xi, omegas, T):
    x = omegas / T; dn = (x / T) * np.exp(x) / np.expm1(x) ** 2; dn[~np.isfinite(dn)] = 0
    return np.trapezoid(omegas * Xi * dn, omegas) / (2 * np.pi * W)

wmaxA = np.sqrt(8 * MAT["A"]["k"] / MAT["A"]["m"]); omegas = np.linspace(0.02, wmaxA * 1.02, 140); Es = omegas ** 2
Xi_dmm = np.array([dmm(E) for E in Es]); G_dmm = conductance(Xi_dmm, omegas, 1.0); norm = np.trapezoid(Xi_dmm, omegas)
sigmas = [0.0, 0.15, 0.3, 0.5, 0.8, 1.2]; NS = 10
configs = {"direct A|B": [("A", 6), ("B", 6)], "A|I(6)|B": [("A", 6), ("I", 6), ("B", 6)]}
res = {name: {"sigma": sigmas, "dist": [], "dist_std": [], "G_ratio": [], "G_ratio_std": [], "contrast": [], "mean_spec": {}} for name in configs}
for name, layers in configs.items():
    for s in sigmas:
        Xis = np.array([spectrum(layers, s, rng, Es) for _ in range(1 if s == 0 else NS)])
        Xbar = Xis.mean(0)
        d = [np.trapezoid(np.abs(x - Xi_dmm), omegas) / norm for x in Xis]            # sample-wise spectral distance
        dbar = np.trapezoid(np.abs(Xbar - Xi_dmm), omegas) / norm                      # ensemble-averaged spectrum vs DMM
        g = [conductance(x, omegas, 1.0) / G_dmm for x in Xis]
        contrast = float(np.std(np.diff(Xbar)) / max(Xbar.mean(), 1e-9))              # Fabry–Pérot fringe contrast proxy
        res[name]["dist"].append(float(np.mean(d))); res[name]["dist_std"].append(float(np.std(d)))
        res[name]["G_ratio"].append(float(np.mean(g))); res[name]["G_ratio_std"].append(float(np.std(g)))
        res[name]["contrast"].append(contrast); res[name]["mean_spec"][str(s)] = Xbar.tolist(); res[name].setdefault("dist_ens", []).append(float(dbar))
        print(f"{name:11s} σ={s:4.2f}: spectral dist (per sample) {np.mean(d):.3f}±{np.std(d):.3f}  (ensemble-avg spectrum) {dbar:.3f}   G/G_DMM {np.mean(g):.3f}±{np.std(g):.3f}   fringe contrast {contrast:.3f}")
json.dump({"omegas": omegas.tolist(), "Xi_dmm": Xi_dmm.tolist(), "res": res}, open("stage0b_disorder_to_dmm.json", "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
for name, c in zip(configs, ["C0", "C3"]):
    R = res[name]
    ax[0].errorbar(R["sigma"], R["dist"], yerr=R["dist_std"], fmt="o-", color=c, label=f"{name} (per sample)")
    ax[0].plot(R["sigma"], R["dist_ens"], "s--", color=c, alpha=0.6, label=f"{name} (ensemble-averaged Ξ)")
    ax[1].errorbar(R["sigma"], R["G_ratio"], yerr=R["G_ratio_std"], fmt="o-", color=c, label=name)
ax[0].set_xlabel("interface disorder σ (lognormal, mean fixed)"); ax[0].set_ylabel("∫|Ξ_AGF − Ξ_DMM| dω / ∫Ξ_DMM dω"); ax[0].legend(fontsize=8); ax[0].set_title("spectral distance to DMM vs disorder")
ax[1].axhline(1, color="k", lw=0.8); ax[1].set_xlabel("σ"); ax[1].set_ylabel("G_AGF / G_DMM  (T=1)"); ax[1].legend(); ax[1].set_title("conductance relative to DMM")
R = res["A|I(6)|B"]
for s, c in zip(["0.0", "0.3", "1.2"], ["C3", "C1", "C2"]): ax[2].plot(omegas, R["mean_spec"][s], color=c, label=f"A|I|B  σ={s}")
ax[2].plot(omegas, Xi_dmm, "k--", label="DMM"); ax[2].set_xlabel("ω"); ax[2].set_ylabel("Ξ(ω) (ensemble mean)"); ax[2].legend(fontsize=8); ax[2].set_title("Fabry–Pérot fringes washed out by disorder?")
plt.suptitle("Stage 0b: is DMM the limit where interface disorder erases phase history?")
plt.tight_layout(); plt.savefig(FIGDIR + "figW_phonon_disorder_to_dmm.png", dpi=130); plt.close(); print("done")
