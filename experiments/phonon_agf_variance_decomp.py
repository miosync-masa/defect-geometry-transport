"""
Stage 0d — U-shape test: order-width vs disorder strength with the multiset frozen (variance decomposition).
  A | I(6 columns, W=6 rows) | B, mass disorder on the 36 interlayer sites: m = m̄·exp(σξ − σ²/2).
  For each σ: draw M multisets; within each, sample P permutations of three kinds
     full        — permute all 36 masses
     within-col  — permute masses among the 6 rows of each column   (order parallel to the interface)
     within-row  — permute masses among the 6 columns of each row   (order along transport)
  Metrics vs σ:  ⟨G⟩/G_DMM (mean response), within-multiset std(log G) and 5–95% width (order effect),
                 between-multiset std of mean log G (distribution effect), distance of ensemble-mean Ξ to DMM.
"""
import numpy as np, json, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _paths import *   # noqa

W, NA, NI, NB = 6, 5, 6, 5
MAT = {"A": dict(m=4.0, k=1.0), "B": dict(m=1.0, k=9.0), "I": dict(m=2.45, k=2.45)}
L = NA + NI + NB; N = W * L; cols = ["A"] * NA + ["I"] * NI + ["B"] * NB
def idx(c, r): return c * W + r
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

# bonds (fixed, geometric mean at dissimilar contacts); only interlayer MASSES are disordered
bonds = {}
for c in range(L):
    kc = MAT[cols[c]]["k"]
    for r in range(W):
        if r + 1 < W: bonds[(idx(c, r), idx(c, r + 1))] = kc
        if c + 1 < L: bonds[(idx(c, r), idx(c + 1, r))] = np.sqrt(kc * MAT[cols[c + 1]]["k"])
kA = MAT["A"]["k"]; kB = MAT["B"]["k"]
base_mass = np.array([MAT[c]["m"] for c in cols for _ in range(W)], float)
I_sites = np.array([idx(c, r) for c in range(NA, NA + NI) for r in range(W)])      # 36 interlayer sites (column-major)

wmaxA = np.sqrt(8 * MAT["A"]["k"] / MAT["A"]["m"]); omegas = np.linspace(0.03, wmaxA * 1.02, 90); Es = omegas ** 2; eta = 1e-5
HA = lead_blocks(MAT["A"]); HB = lead_blocks(MAT["B"]); cache = []
for E in Es:
    gA = surface_gf(E, *HA, eta); gB = surface_gf(E, *HB, eta)
    cA = -kA / np.sqrt(base_mass[0] * MAT["A"]["m"]); cB = -kB / np.sqrt(base_mass[-1] * MAT["B"]["m"])
    SA = np.zeros((N, N), complex); SB = np.zeros((N, N), complex); SA[:W, :W] = cA * gA * cA; SB[-W:, -W:] = cB * gB * cB
    cache.append((SA + SB, 1j * (SA - SA.conj().T), 1j * (SB - SB.conj().T)))
def spectrum(mass):
    Dm = np.zeros((N, N))
    for (i, j), k in bonds.items():
        Dm[i, j] = Dm[j, i] = -k / np.sqrt(mass[i] * mass[j]); Dm[i, i] += k / mass[i]; Dm[j, j] += k / mass[j]
    for r in range(W): Dm[r, r] += kA / mass[r]; Dm[N - W + r, N - W + r] += kB / mass[N - W + r]
    out = np.zeros(len(Es))
    for n, E in enumerate(Es):
        S, GL, GR = cache[n]; G = np.linalg.inv((E + 1j * eta) * np.eye(N) - Dm - S)
        out[n] = max(np.real(np.trace(GL @ G @ GR @ G.conj().T)), 0.0)
    return out
def conductance(Xi, T=1.0):
    x = omegas / T; dn = (x / T) * np.exp(x) / np.expm1(x) ** 2; dn[~np.isfinite(dn)] = 0
    return np.trapezoid(omegas * Xi * dn, omegas) / (2 * np.pi * W)
Xi_dmm = np.array([dmm(E) for E in Es]); G_dmm = conductance(Xi_dmm); norm = np.trapezoid(Xi_dmm, omegas)
def dist(Xi): return np.trapezoid(np.abs(Xi - Xi_dmm), omegas) / norm

def place(multiset_grid):
    """multiset_grid: (NI, W) array of interlayer masses (column, row) -> full mass array"""
    m = base_mass.copy(); m[I_sites] = multiset_grid.reshape(-1); return m
def shuffle(grid, kind, rng):
    g = grid.copy()
    if kind == "full": return rng.permutation(g.reshape(-1)).reshape(NI, W)
    if kind == "within-col":
        for c in range(NI): g[c] = rng.permutation(g[c])
        return g
    if kind == "within-row":
        for r in range(W): g[:, r] = rng.permutation(g[:, r])
        return g

rng = np.random.default_rng(11)
sigmas = [0.1, 0.3, 0.5, 0.8, 1.2, 1.6]; M, P = 12, 40; kinds = ["full", "within-col", "within-row"]
t0 = time.time(); out = {}
for s in sigmas:
    multisets = [MAT["I"]["m"] * np.exp(s * rng.normal(size=(NI, W)) - s ** 2 / 2) for _ in range(M)]
    rec = {k: {"within_std": [], "within_p90": [], "mean_logG": []} for k in kinds}; Xsum = np.zeros(len(Es)); nX = 0; logG_all = []
    for ms in multisets:
        for kind in kinds:
            logs = []
            for _ in range(P):
                Xi = spectrum(place(shuffle(ms, kind, rng))); g = conductance(Xi); logs.append(np.log(max(g, 1e-12)))
                if kind == "full": Xsum += Xi; nX += 1; logG_all.append(logs[-1])
            logs = np.array(logs); rec[kind]["within_std"].append(float(logs.std())); rec[kind]["within_p90"].append(float(np.percentile(logs, 95) - np.percentile(logs, 5))); rec[kind]["mean_logG"].append(float(logs.mean()))
    Xbar = Xsum / nX; logG_all = np.array(logG_all)
    res = {"meanG_over_dmm": float(np.exp(np.mean(logG_all)) / G_dmm), "meanG_arith_over_dmm": float(np.mean(np.exp(logG_all)) / G_dmm),
           "dist_ensemble": float(dist(Xbar)), "between_std_full": float(np.std(rec["full"]["mean_logG"])), "total_std_full": float(logG_all.std())}
    for k in kinds: res[f"within_std_{k}"] = float(np.mean(rec[k]["within_std"])); res[f"within_p90_{k}"] = float(np.mean(rec[k]["within_p90"]))
    out[s] = res
    print(f"σ={s:3.1f}: ⟨G⟩/G_DMM(geo) {res['meanG_over_dmm']:.3f} | within std(logG): full {res['within_std_full']:.3f}  col {res['within_std_within-col']:.3f}  row {res['within_std_within-row']:.3f} | between std {res['between_std_full']:.3f}  total {res['total_std_full']:.3f} | DMM dist {res['dist_ensemble']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
json.dump({str(k): v for k, v in out.items()}, open("stage0d_variance_decomp.json", "w"), indent=1)

S = np.array(sigmas)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].plot(S, [out[s]["meanG_over_dmm"] for s in sigmas], "o-", color="C3", label="⟨G⟩/G_DMM (geometric mean)")
ax[0].plot(S, [out[s]["meanG_arith_over_dmm"] for s in sigmas], "s--", color="C3", alpha=0.5, label="arithmetic mean")
ax[0].axhline(1, color="gray", ls=":"); ax0b = ax[0].twinx(); ax0b.plot(S, [out[s]["dist_ensemble"] for s in sigmas], "^-", color="k", label="distance of ⟨Ξ⟩ to DMM"); ax0b.set_ylabel("∫|⟨Ξ⟩−Ξ_DMM| / ∫Ξ_DMM")
ax[0].set_xlabel("interlayer mass disorder σ"); ax[0].set_ylabel("⟨G⟩ / G_DMM"); ax[0].set_title("mean response (mountain?) and DMM distance (valley?)"); ax[0].legend(loc="upper left", fontsize=8); ax0b.legend(loc="upper right", fontsize=8)
for kind, c in zip(kinds, ["C0", "C1", "C2"]): ax[1].plot(S, [out[s][f"within_std_{kind}"] for s in sigmas], "o-", color=c, label=f"within-multiset std(log G): {kind}")
ax[1].plot(S, [out[s]["between_std_full"] for s in sigmas], "s--", color="k", label="between-multiset std of mean log G (distribution)")
ax[1].set_xlabel("σ"); ax[1].set_ylabel("std(log G)"); ax[1].set_title("order effect (within) vs distribution effect (between)"); ax[1].legend(fontsize=7)
ax[2].plot(S, [out[s]["within_p90_full"] for s in sigmas], "o-", color="C0", label="5–95% width of log G (full shuffle)")
ax[2].plot(S, [out[s]["within_std_full"] ** 2 / max(out[s]["total_std_full"] ** 2, 1e-12) for s in sigmas], "d-", color="C4", label="fraction of variance due to order")
ax[2].set_xlabel("σ"); ax[2].set_title("order width and order share of variance"); ax[2].legend(fontsize=8)
plt.suptitle("Stage 0d: with the multiset frozen, does the order-width form a valley where the mean forms a mountain?")
plt.tight_layout(); plt.savefig(FIGDIR + "figY_phonon_variance_decomp.png", dpi=130); plt.close(); print("done", f"{time.time()-t0:.0f}s")
