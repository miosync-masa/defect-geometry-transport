"""
Exp B-series: 「外から電気的摂動を与えたとき、内部で何が変わるか」を観測窓にする
  B-0' : N_eff·D_eff 厳密分解（ご主人さま提案）で図C/図Aを再計算
  B0   : 微小バイアス応答  δn_i = (A_L,ii − A_R,ii)/2 ,  A_α = G Γ_α G†  (injectivity)
         局所電気化学ポテンシャル μ_i = (A_L μ_L + A_R μ_R)_ii / (A_L + A_R)_ii
  B1   : Hartree/Poisson 自己無撞着（Thomas–Fermi 線形遮蔽）で静電 φ_i を更新
単位: e = 1, バイアス V (t 単位), 電荷 δn は V/(2π) 単位 → δn_i = (A_L−A_R)_ii/2 と定義
"""
import numpy as np, numpy.linalg as la, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _paths import *   # noqa: cwd -> results/data, src on path
from negf_toy import Device, current_field

W, L, E, U_DEF = 20, 40, -1.1, 8.0
rng = np.random.default_rng(7)

def fresh(): return Device(W, L, E=E, eta=1e-6)
def sites(d, coords): return [d.idx(c, r) for c, r in coords]

CONF = {
    "wall ⊥": [(20, r) for r in range(4, 16)],
    "random": [(int(c), int(r)) for c, r in zip(rng.integers(12, 28, 12), rng.integers(0, W, 12))],
    "wall ∥": [(c, 10) for c in range(14, 26)],
}
d0 = fresh(); T0, eig0 = d0.transmission(return_eigs=True); N0 = d0.n_open_channels()

# ---------------- B-0': 厳密 N_eff·D_eff ----------------
def neff_deff(eig):
    e = np.clip(eig, 0, None); S1, S2 = e.sum(), (e**2).sum()
    return S1**2 / S2, S2 / S1

out = {"T0": T0, "N0": N0, "exact_decomp": {}}
print(f"clean: T0={T0:.3f} N0={N0}  N_eff={neff_deff(eig0)[0]:.2f} D_eff={neff_deff(eig0)[1]:.3f}")
for name, coords in CONF.items():
    d = fresh(); d.set_onsite(sites(d, coords), U_DEF); T, eig = d.transmission(return_eigs=True)
    Ne, De = neff_deff(eig)
    out["exact_decomp"][name] = {"G/G0": T / T0, "N_eff/N0": Ne / N0, "D_eff": De}
    print(f"{name:8s} G/G0={T/T0:.3f}  N_eff/N0={Ne/N0:.3f}  D_eff={De:.3f}  check={Ne/N0*De:.3f}")

# 図C を厳密分解で引き直す
BETA = 3.0; vs = np.linspace(0, 1, 26); rows = []
for v in vs:
    d = fresh(); d.scale_bonds_across_column(20, np.exp(-BETA * v)); T, eig = d.transmission(return_eigs=True)
    Ne, De = neff_deff(eig); rows.append((T / T0, Ne / N0, De))
rows = np.array(rows); out["figC_exact"] = {"v": vs.tolist(), "G": rows[:, 0].tolist(), "Neff": rows[:, 1].tolist(), "Deff": rows[:, 2].tolist()}
# 乱れ粒界も
rowsd = []
for v in vs:
    acc = []
    for s in range(8):
        d = fresh()
        for r in range(W):
            d.set_bond(d.idx(20, r), d.idx(21, r), np.exp(-BETA * v * max(0, 1 + rng.normal(0, 0.6))))
        T, eig = d.transmission(return_eigs=True); Ne, De = neff_deff(eig); acc.append((T / T0, Ne / N0, De))
    rowsd.append(np.mean(acc, axis=0))
rowsd = np.array(rowsd)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a, R, ttl in [(ax[0], rows, "ordered GB"), (ax[1], rowsd, "disordered GB (mean of 8)")]:
    a.plot(vs, R[:, 0], "k-", lw=2, label="G/G0")
    a.plot(vs, R[:, 1], "o-", label="N_eff / N0  (effective channel participation)")
    a.plot(vs, R[:, 2], "s-", label="D_eff  (per-channel transparency)")
    a.plot(vs, R[:, 1] * R[:, 2], "r:", lw=1.2, label="N_eff/N0 × D_eff (identity check)")
    a.set_xlabel("GB excess-volume proxy v"); a.set_title(f"exact N·D decomposition — {ttl}"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIGDIR + "figC2_exact_decomp.png", dpi=140); plt.close()

# ---------------- B0: 微小バイアス応答 ----------------
def response(d):
    G, GL, GR = d.greens()
    AL = np.real(np.diag(G @ GL @ G.conj().T))   # 左 injectivity (×2π LDOS 寄与)
    AR = np.real(np.diag(G @ GR @ G.conj().T))
    dn = 0.5 * (AL - AR)                          # μ_L = E+V/2, μ_R = E−V/2 の線形応答電荷 (V/2π 単位)
    mu = 0.5 * (AL - AR) / (AL + AR + 1e-12)      # 局所電気化学ポテンシャル (V 単位, +1/2 … −1/2)
    ldos = (AL + AR) / (2 * np.pi)
    return AL, AR, dn, mu, ldos

def grid(x): return x.reshape(L, W)

fig, axs = plt.subplots(3, 3, figsize=(15, 9.5))
B0 = {}
for i, (name, coords) in enumerate(CONF.items()):
    d = fresh(); d.set_onsite(sites(d, coords), U_DEF); T = d.transmission()
    AL, AR, dn, mu, ldos = response(d)
    # 清浄系との差分（欠陥が「作った」応答）
    _, _, dn0, mu0, _ = response(d0)
    ddn = dn - dn0
    cs = np.array(coords)
    vmax = np.abs(ddn).max()
    im = axs[i, 0].imshow(grid(ddn).T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axs[i, 0].scatter(cs[:, 0], cs[:, 1], s=10, c="k", marker="x"); plt.colorbar(im, ax=axs[i, 0])
    axs[i, 0].set_title(f"{name}  G/G0={T/T0:.3f}   δn − δn_clean  (bias-induced charge)")
    im = axs[i, 1].imshow(grid(mu).T, origin="lower", cmap="viridis", vmin=-0.5, vmax=0.5, aspect="auto")
    axs[i, 1].scatter(cs[:, 0], cs[:, 1], s=10, c="w", marker="x"); plt.colorbar(im, ax=axs[i, 1])
    axs[i, 1].set_title("local electrochemical potential μ_i / V")
    axs[i, 2].plot(grid(mu).mean(axis=1), "k-", label="⟨μ⟩_y(x)")
    axs[i, 2].plot(grid(dn).mean(axis=1) / np.abs(grid(dn).mean(axis=1)).max() * 0.5, "r--", label="⟨δn⟩_y(x) (scaled)")
    axs[i, 2].axvline(20, color="gray", ls=":"); axs[i, 2].set_ylim(-0.55, 0.55); axs[i, 2].legend(fontsize=8)
    axs[i, 2].set_title("row-averaged profiles along transport")
    # 定量: 双極子, 電位降下, 応答の空間不均一性
    X = np.repeat(np.arange(L), W)
    dip = float(np.sum(ddn * (X - 19.5)))
    drop = float(grid(mu)[:14].mean() - grid(mu)[26:].mean())
    B0[name] = {"G/G0": T / T0, "sum|ddn|": float(np.abs(ddn).sum()), "dipole": dip,
                "mu_drop_across_center": drop, "std_ddn": float(ddn.std()), "corr(ddn,ldos)": float(np.corrcoef(np.abs(ddn), ldos)[0, 1])}
    print(name, {k: round(v, 4) for k, v in B0[name].items()})
out["B0"] = B0
plt.tight_layout(); plt.savefig(FIGDIR + "figE_B0_response_maps.png", dpi=130); plt.close()

# ---------------- B1: Hartree 自己無撞着 ----------------
# 3D クーロン核を 2D 格子上で (ソフト化) : K_ij = U0 / sqrt(r_ij^2 + 1)
XY = np.array([(c, r) for c in range(L) for r in range(W)], float)
R2 = ((XY[:, None, :] - XY[None, :, :])**2).sum(-1)
def kernel(U0): return U0 / np.sqrt(R2 + 1.0)

def self_consistent(coords, U0=0.5, V=1.0, iters=80, mix=0.1):
    d = fresh(); d.set_onsite(sites(d, coords), U_DEF)
    eps0 = d.eps.copy()
    K = kernel(U0)
    phi = np.zeros(d.N); hist = []
    _, _, dn_clean, _, _ = response(d0)
    for it in range(iters):
        d.eps = eps0 + phi
        AL, AR, dn, mu, ldos = response(d)
        rho = V * (dn - dn_clean) / (2 * np.pi)          # 欠陥がつくる余剰電荷 (清浄系の線形ランプを引く)
        # TF 線形遮蔽: φ = K(ρ − ν φ)  →  (1 + Kν) φ = Kρ
        nu = np.diag(ldos)
        phi_new = la.solve(np.eye(d.N) + K @ nu, K @ rho)
        err = np.abs(phi_new - phi).max(); hist.append(err)
        phi = (1 - mix) * phi + mix * phi_new
        if err < 1e-6: break
    d.eps = eps0 + phi
    T_sc = d.transmission()
    d.eps = eps0; T_bare = d.transmission()
    _, _, _, mu_sc, _ = response(Device.__new__(Device)) if False else (None, None, None, None, None)
    d.eps = eps0 + phi; _, _, dn_sc, mu_sc, _ = response(d)
    return phi, hist, T_bare, T_sc, mu_sc, dn_sc - dn_clean

fig, axs = plt.subplots(3, 3, figsize=(15, 9.5)); B1 = {}
for i, (name, coords) in enumerate(CONF.items()):
    phi, hist, Tb, Tsc, mu_sc, ddn_sc = self_consistent(coords)
    cs = np.array(coords); vmax = np.abs(phi).max()
    im = axs[i, 0].imshow(grid(phi).T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axs[i, 0].scatter(cs[:, 0], cs[:, 1], s=10, c="k", marker="x"); plt.colorbar(im, ax=axs[i, 0])
    axs[i, 0].set_title(f"{name}: self-consistent electrostatic φ_i / V   (G: {Tb/T0:.3f} → {Tsc/T0:.3f})")
    axs[i, 1].plot(grid(phi).mean(axis=1), "b-", label="⟨φ⟩_y (electrostatic, defect-induced)")
    axs[i, 1].plot(grid(mu_sc).mean(axis=1), "k-", label="⟨μ⟩_y (electrochemical)")
    axs[i, 1].axvline(20, color="gray", ls=":"); axs[i, 1].legend(fontsize=8); axs[i, 1].set_title("φ vs μ along transport")
    axs[i, 2].semilogy(hist, "o-"); axs[i, 2].set_title("SC convergence (max|Δφ|)"); axs[i, 2].set_xlabel("iteration")
    B1[name] = {"G_bare": Tb / T0, "G_sc": Tsc / T0, "phi_drop_center": float(grid(phi)[:14].mean() - grid(phi)[26:].mean()),
                "phi_max": float(np.abs(phi).max()), "iters": len(hist), "final_err": float(hist[-1])}
    print(name, {k: round(v, 4) for k, v in B1[name].items()})
out["B1"] = B1
plt.tight_layout(); plt.savefig(FIGDIR + "figF_B1_selfconsistent.png", dpi=130); plt.close()
json.dump(out, open("results_B.json", "w"), indent=1, ensure_ascii=False)
print("done")
