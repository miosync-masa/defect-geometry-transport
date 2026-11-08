"""
Stage 0 — scalar-phonon AGF on a quasi-1D harmonic lattice with DISSIMILAR leads (A | interlayer | B).
Reuses the electron-NEGF skeleton: dynamical matrix D = M^{-1/2} K M^{-1/2}, "energy" = omega^2,
onsite = sum_k/m, hopping = -k/sqrt(m_i m_j). Leads A, B have their own (m, k); Sancho-Rubio per lead.

Outputs (results/data/stage0_phonon.json, results/figures/figV_phonon_stage0.png):
  * transmission Xi(omega), N_eff(omega), D_eff(omega) for direct A/B and A/I/B
  * G(T) = (1/A_cross) ∫ dω/2π ħω Xi(ω) ∂n_BE/∂T   (model units: ħ=k_B=1, lattice const 1)
  * DMM in the same model: Xi_DMM = N_A N_B / (N_A + N_B) (mode counting, elastic)
  * interlayer scans: impedance Z_I = sqrt(m_I k_I) and thickness d
"""
import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _paths import *   # noqa

W = 6                      # ribbon width (sites)
HBAR = KB = 1.0

# ---------- materials (model units) ----------
# A: heavy, soft (Cu-like)   B: light, stiff (diamond-like)   I: interlayer (TiC-like, intermediate)
MAT = {"A": dict(m=4.0, k=1.0), "B": dict(m=1.0, k=9.0), "I": dict(m=2.45, k=2.45)}   # Z_A=2, Z_B=3, Z_I=sqrt(6)
def omega_max(mat): return np.sqrt(8 * mat["k"] / mat["m"])   # 2D square lattice: max omega^2 = 8k/m (incl. transverse)

def lead_blocks(mat):
    m, k = mat["m"], mat["k"]
    H00 = np.zeros((W, W)); H01 = -(k / m) * np.eye(W)
    for r in range(W):
        ntrans = (1 if r > 0 else 0) + (1 if r + 1 < W else 0)
        H00[r, r] = (2 + ntrans) * k / m             # open-boundary ribbon: 2 longitudinal + actual transverse bonds
        if r + 1 < W: H00[r, r + 1] = H00[r + 1, r] = -(k / m)
    return H00, H01

def surface_gf(E, H00, H01, eta=1e-5):
    z = (E + 1j * eta) * np.eye(W)
    es, eb, a, b = H00.astype(complex).copy(), H00.astype(complex).copy(), H01.astype(complex).copy(), H01.conj().T.copy()
    for _ in range(300):
        g = np.linalg.inv(z - eb); agb = a @ g @ b; bga = b @ g @ a
        es = es + agb; eb = eb + agb + bga; a = a @ g @ a; b = b @ g @ b
        if np.linalg.norm(a) < 1e-13 and np.linalg.norm(b) < 1e-13: break
    return np.linalg.inv(z - es)

def n_open(E, mat):
    """number of propagating lead modes at omega^2 = E (transverse quantisation)"""
    m, k = mat["m"], mat["k"]; n = np.arange(0, W)                 # free (Neumann) transverse boundary: n = 0..W-1
    eps = (k / m) * (4 - 2 * np.cos(n * np.pi / W))                 # transverse part + 2k/m longitudinal centre
    return int(np.sum(np.abs(E - eps) < 2 * (k / m)))

def device(layers):
    """layers: list of (mat, ncols). Returns per-site mass array and spring dict; interface springs = geometric mean."""
    cols = [];
    for mat, n in layers: cols += [mat] * n
    L = len(cols); mass = np.array([MAT[c]["m"] for c in cols for _ in range(W)])
    def idx(c, r): return c * W + r
    K = {}
    for c in range(L):
        kc = MAT[cols[c]]["k"]
        for r in range(W):
            if r + 1 < W: K[(idx(c, r), idx(c, r + 1))] = kc
            if c + 1 < L: K[(idx(c, r), idx(c + 1, r))] = np.sqrt(kc * MAT[cols[c + 1]]["k"])
    return cols, mass, K, L

def transmission(E, layers, eta=1e-5):
    cols, mass, K, L = device(layers); N = W * L
    Dm = np.zeros((N, N))
    for (i, j), k in K.items():
        Dm[i, j] = Dm[j, i] = -k / np.sqrt(mass[i] * mass[j]); Dm[i, i] += k / mass[i]; Dm[j, j] += k / mass[j]
    # couple edges to leads (spring = geometric mean with lead), add the missing onsite share
    kA = np.sqrt(MAT[cols[0]]["k"] * MAT["A"]["k"]); kB = np.sqrt(MAT[cols[-1]]["k"] * MAT["B"]["k"])
    for r in range(W): Dm[r, r] += kA / mass[r]; Dm[N - W + r, N - W + r] += kB / mass[N - W + r]
    gA = surface_gf(E, *lead_blocks(MAT["A"]), eta); gB = surface_gf(E, *lead_blocks(MAT["B"]), eta)
    # lead self-energies with mixed coupling  -kA/sqrt(m_edge m_A)
    cA = -kA / np.sqrt(mass[0] * MAT["A"]["m"]); cB = -kB / np.sqrt(mass[-1] * MAT["B"]["m"])
    SA = np.zeros((N, N), complex); SB = np.zeros((N, N), complex)
    SA[:W, :W] = cA * gA * cA; SB[-W:, -W:] = cB * gB * cB
    G = np.linalg.inv((E + 1j * eta) * np.eye(N) - Dm - SA - SB)
    GL = 1j * (SA - SA.conj().T); GR = 1j * (SB - SB.conj().T)
    Xi = np.real(np.trace(GL @ G @ GR @ G.conj().T))
    # transmission eigenvalues -> N_eff, D_eff
    gl = GL[:W, :W]; gr = GR[-W:, -W:]; wl, vl = np.linalg.eigh(gl); sql = (vl * np.sqrt(np.clip(wl, 0, None))) @ vl.conj().T
    Gdr = G[:W, -W:]; Aeig = np.clip(np.real(np.linalg.eigvalsh(sql @ Gdr @ gr @ Gdr.conj().T @ sql)), 0, None)
    S1, S2 = Aeig.sum(), (Aeig ** 2).sum()
    return max(Xi, 0.0), (S1 ** 2 / S2 if S2 > 0 else 0.0), (S2 / S1 if S1 > 0 else 0.0)

def dmm(E):
    nA, nB = n_open(E, MAT["A"]), n_open(E, MAT["B"])
    return nA * nB / (nA + nB) if nA + nB > 0 else 0.0

def conductance(Xi, omegas, T):
    x = HBAR * omegas / (KB * T); dn = (x / T) * np.exp(x) / np.expm1(x) ** 2 / 1.0   # ∂n/∂T = (x/T) e^x/(e^x-1)^2
    dn[~np.isfinite(dn)] = 0
    return np.trapezoid(HBAR * omegas * Xi * dn, omegas) / (2 * np.pi * W)

# ---------- spectra ----------
wmax = max(omega_max(MAT["A"]), omega_max(MAT["B"]), omega_max(MAT["I"]))
omegas = np.linspace(0.02, wmax, 160); Es = omegas ** 2
configs = {"direct A|B": [("A", 6), ("B", 6)], "A|I(4)|B": [("A", 6), ("I", 4), ("B", 6)], "A|I(10)|B": [("A", 6), ("I", 10), ("B", 6)]}
spec = {}
for name, layers in configs.items():
    rows = np.array([transmission(E, layers) for E in Es]); spec[name] = rows
    print(f"{name:12s} ∫Xi dω = {np.trapezoid(rows[:,0], omegas):.3f}")
Xi_dmm = np.array([dmm(E) for E in Es]); print(f"{'DMM':12s} ∫Xi dω = {np.trapezoid(Xi_dmm, omegas):.3f}")
# clean A|A reference (perfect lattice, all channels open)
MAT_save = dict(MAT);

# ---------- G(T) ----------
Ts = np.linspace(0.05, 3.0, 40)
GT = {name: np.array([conductance(spec[name][:, 0], omegas, T) for T in Ts]) for name in configs}
GT["DMM"] = np.array([conductance(Xi_dmm, omegas, T) for T in Ts])
Tref = 1.0; iT = np.argmin(np.abs(Ts - Tref))
ratio_I4 = GT["A|I(4)|B"][iT] / GT["direct A|B"][iT]; ratio_I10 = GT["A|I(10)|B"][iT] / GT["direct A|B"][iT]
print(f"G(T=1): direct {GT['direct A|B'][iT]:.4f}  A|I4|B {GT['A|I(4)|B'][iT]:.4f} (×{ratio_I4:.2f})  A|I10|B {GT['A|I(10)|B'][iT]:.4f} (×{ratio_I10:.2f})  DMM {GT['DMM'][iT]:.4f}")

# ---------- interlayer impedance scan (thickness 4) ----------
Zs = np.linspace(1.0, 5.0, 17); ZA = np.sqrt(MAT["A"]["m"] * MAT["A"]["k"]); ZB = np.sqrt(MAT["B"]["m"] * MAT["B"]["k"])
scanZ = []
for Z in Zs:
    MAT["I"] = dict(m=Z, k=Z)       # Z_I = sqrt(m k) = Z, keep sound speed ratio fixed (k/m = 1)
    Xi = np.array([transmission(E, configs["A|I(4)|B"])[0] for E in Es]); scanZ.append(conductance(Xi, omegas, Tref))
MAT["I"] = dict(m=2.45, k=2.45)
scanZ = np.array(scanZ) / GT["direct A|B"][iT]
print("impedance scan: best Z_I =", round(Zs[np.argmax(scanZ)], 2), " (geometric mean sqrt(Z_A Z_B) =", round(np.sqrt(ZA * ZB), 2), ")  max ratio =", round(scanZ.max(), 2))

# ---------- thickness scan ----------
ds = [1, 2, 3, 4, 6, 8, 10, 14]; scanD = []
for d in ds:
    Xi = np.array([transmission(E, [("A", 6), ("I", d), ("B", 6)])[0] for E in Es]); scanD.append(conductance(Xi, omegas, Tref))
scanD = np.array(scanD) / GT["direct A|B"][iT]
print("thickness scan ratios:", dict(zip(ds, np.round(scanD, 2))))

json.dump({"omegas": omegas.tolist(), "spec": {k: v.tolist() for k, v in spec.items()}, "Xi_dmm": Xi_dmm.tolist(), "Ts": Ts.tolist(),
           "GT": {k: v.tolist() for k, v in GT.items()}, "ratio_I4": ratio_I4, "ratio_I10": ratio_I10,
           "Zscan": {"Z": Zs.tolist(), "ratio": scanZ.tolist(), "Zgeo": float(np.sqrt(ZA * ZB))}, "Dscan": {"d": ds, "ratio": scanD.tolist()}},
          open("stage0_phonon.json", "w"), indent=1)

fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
for name, c in zip(configs, ["C0", "C1", "C3"]):
    ax[0, 0].plot(omegas, spec[name][:, 0], color=c, label=name); ax[0, 1].plot(omegas, spec[name][:, 1], color=c, label=name); ax[0, 2].plot(omegas, spec[name][:, 2], color=c, label=name)
ax[0, 0].plot(omegas, Xi_dmm, "k--", label="DMM (mode counting)"); ax[0, 0].set_ylabel("phonon transmission Ξ(ω)"); ax[0, 0].legend(fontsize=8)
ax[0, 1].set_ylabel("N_eff(ω)  (effective transmitting channels)"); ax[0, 2].set_ylabel("D_eff(ω)  (per-channel transparency)")
for a in ax[0]: a.set_xlabel("ω"); a.axvline(omega_max(MAT["A"]), color="C0", ls=":", lw=0.8);
for name, c in zip(list(configs) + ["DMM"], ["C0", "C1", "C3", "k"]):
    ax[1, 0].plot(Ts, GT[name], color=c, ls="--" if name == "DMM" else "-", label=name)
ax[1, 0].set_xlabel("T (model units)"); ax[1, 0].set_ylabel("G(T) per width"); ax[1, 0].legend(fontsize=8); ax[1, 0].set_title("thermal boundary conductance")
ax[1, 1].plot(Zs, scanZ, "o-"); ax[1, 1].axvline(np.sqrt(ZA * ZB), color="gray", ls=":", label="√(Z_A Z_B)"); ax[1, 1].axhline(1, color="k", lw=0.8)
ax[1, 1].set_xlabel("interlayer impedance Z_I = √(m_I k_I)"); ax[1, 1].set_ylabel("G(A|I|B) / G(A|B)  at T=1"); ax[1, 1].legend(); ax[1, 1].set_title("mode bridging vs interlayer impedance (d=4)")
ax[1, 2].plot(ds, scanD, "s-"); ax[1, 2].axhline(1, color="k", lw=0.8); ax[1, 2].set_xlabel("interlayer thickness d (layers)"); ax[1, 2].set_ylabel("G ratio at T=1"); ax[1, 2].set_title("thickness dependence (Z_I = 2)")
plt.suptitle("Stage 0: scalar-phonon AGF with dissimilar leads — does an ordered interlayer bridge modes, and where does it beat DMM?")
plt.tight_layout(); plt.savefig(FIGDIR + "figV_phonon_stage0.png", dpi=130); plt.close()
print("done")
