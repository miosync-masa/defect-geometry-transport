"""
第7報 Exp G: 幾何 → モード結合 → 多重散乱 → 伝導 の階層を閉じる
  G1: T 行列スペクトル  M = I − G0_dd U  の最小特異値・固有値分布 と grid 残差の対応
  G2: 有限超格子 (F2 の暗帯) vs 無限周期系 (transfer matrix の Bloch バンド / 複素バンド)
  G3: 弱散乱 U=0.5,1,2 で F2 再計算 → 単純 Bragg 条件が回復する領域
  G4: 有限温度平均 ∫dE(−∂f/∂E)G(E)（W=20 の 390 構造全部、厳密 T 行列で E 掃引）
"""
import numpy as np, json, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from _paths import *   # noqa: cwd -> results/data, src on path
from negf_toy import Device

W, L, E0, U_DEF = 20, 40, -1.1, 8.0
t0 = time.time()
raw = json.load(open("coords_E.json"))[str(E0)]
df = pd.read_csv("dataset_F.csv")
coords_all = [[tuple(c) for c in r["coords"]] for r in raw]

def clean(E, W_=W, L_=L):
    d = Device(W_, L_, E=E, eta=1e-6); T0 = d.transmission(); G0, GL, GR = d.greens()
    return d, T0, G0, GL[:W_, :W_], GR[-W_:, -W_:]

def G_exact(d, T0, G0, gl, gr, coords, U=U_DEF):
    idx = [d.idx(c, r) for c, r in coords]; nd = len(idx)
    Gdd = G0[np.ix_(idx, idx)]; M = np.eye(nd) - Gdd * U
    T = np.linalg.solve(M, U * np.eye(nd))
    G_LR = G0[:d.W, -d.W:] + G0[:d.W, idx] @ T @ G0[idx, -d.W:]
    return np.real(np.trace(gl @ G_LR @ gr @ G_LR.conj().T)) / T0, M

# ================= G1: T 行列スペクトル =================
d, T0, G0, gl, gr = clean(E0)
spec = []
for coords in coords_all:
    g, M = G_exact(d, T0, G0, gl, gr, coords)
    sv = np.linalg.svd(M, compute_uv=False); ev = np.linalg.eigvals(M)
    # 最小特異値のモード（集団散乱状態）の参加比
    _, _, Vh = np.linalg.svd(M); vmin = np.abs(Vh[-1]) ** 2; pr = 1.0 / np.sum(vmin ** 2) / len(vmin)
    spec.append({"sv_min": sv.min(), "log_det": np.sum(np.log(sv)), "sv_min_norm": sv.min() / sv.max(),
                 "n_small_ev": int(np.sum(np.abs(ev) < 0.5)), "ev_min_abs": np.abs(ev).min(),
                 "pr_collective": pr, "mean_log_sv": np.mean(np.log(sv))})
spec = pd.DataFrame(spec); df = pd.concat([df.reset_index(drop=True), spec], axis=1)
print(f"G1 spectra done ({time.time()-t0:.0f}s)")

# M2'' の残差（未知族）とスペクトルの対応
y = np.log(np.clip(1 - df["G"], 1e-4, None)); tr = df["split"] == "train"
geom = [c for c in df if c.startswith(("M0", "M1", "M2"))]; sq = [c for c in df if c.startswith(f"SqE{E0}_")]; amn = [c for c in df if c.startswith("A_")]
M2pp = geom + sq + amn
reg = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0).fit(df.loc[tr, M2pp], y[tr])
df["resid"] = reg.predict(df[M2pp]) - y            # 正: 過大予測 (抵抗を多く見積り), 負: 過小 (grid の症状)
nov = df["split"] == "test_novel"
print("G1: corr(resid, spectral) on novel families:")
for c in ["sv_min", "log_det", "sv_min_norm", "n_small_ev", "ev_min_abs", "pr_collective"]:
    r_all = np.corrcoef(df.loc[nov, "resid"], df.loc[nov, c])[0, 1]
    r_grid = np.corrcoef(df.loc[nov & (df.fam == "grid"), "resid"], df.loc[nov & (df.fam == "grid"), c])[0, 1]
    print(f"   {c:14s} novel r={r_all:+.3f}   grid-only r={r_grid:+.3f}")
# 族ごとの σ_min 分布
print("G1: median sv_min by family:", df.groupby("fam")["sv_min"].median().round(3).to_dict())
print("G1: median log|det| by family:", df.groupby("fam")["log_det"].median().round(2).to_dict())
# スペクトル特徴を回帰に足したら
for name, cols in [("M2''", M2pp), ("M2''+spec", M2pp + ["sv_min", "log_det", "n_small_ev", "pr_collective"])]:
    r2s, grs = [], []
    for seed in range(5):
        rg = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=seed).fit(df.loc[tr, cols], y[tr])
        p = rg.predict(df.loc[nov, cols]); yy = y[nov].values; r2s.append(1 - ((p - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
        mg = (df.loc[nov, "fam"] == "grid").values; grs.append(np.sqrt(((p[mg] - yy[mg]) ** 2).mean()))
    print(f"   {name:10s} novel R² {np.mean(r2s):.3f}±{np.std(r2s):.3f}  grid rmse {np.mean(grs):.3f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for fam, c in zip(sorted(df.fam.unique()), plt.cm.tab10.colors):
    m = df.fam == fam
    ax[0].scatter(df.loc[m, "log_det"], df.loc[m, "G"], s=10, color=c, label=fam, alpha=0.7)
    ax[1].scatter(np.log10(df.loc[m, "sv_min"]), df.loc[m, "G"], s=10, color=c, alpha=0.7)
    if nov[m].any(): ax[2].scatter(np.log10(df.loc[m & nov, "sv_min"]), df.loc[m & nov, "resid"], s=14, color=c, label=fam)
ax[0].set_xlabel("log|det(I − G0 U)|"); ax[0].set_ylabel("G/G0"); ax[0].legend(fontsize=7); ax[0].set_title("G1: T-matrix determinant vs conductance")
ax[1].set_xlabel("log10 σ_min(I − G0 U)"); ax[1].set_ylabel("G/G0"); ax[1].set_title("smallest singular value (collective resonance)")
ax[2].axhline(0, color="k", lw=0.8); ax[2].set_xlabel("log10 σ_min"); ax[2].set_ylabel("M2'' residual (pred − true)"); ax[2].set_title("novel-family residual vs σ_min"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIGDIR + "figQ_G1_Tmatrix_spectrum.png", dpi=130); plt.close()

# ================= G2/G3: 有限超格子 vs 無限周期系, 弱 U =================
W2, L2, Nd = 4, 40, 6
Es = np.linspace(-2.6, -0.2, 97); periods = np.arange(1, 9)
def transfer_bloch(E, a, U):
    """1 セル (a 列, 行 1 に欠陥 1 個) の転送行列 → 伝播 Bloch 状態数, 最小減衰率"""
    Hp = np.zeros((W2, W2));
    for r in range(W2 - 1): Hp[r, r + 1] = Hp[r + 1, r] = -1.0
    Mcell = np.eye(2 * W2, dtype=complex)
    for c in range(a):
        Hc = Hp.copy()
        if c == 0: Hc[1, 1] += U
        T = np.zeros((2 * W2, 2 * W2), complex); T[:W2, :W2] = Hc - E * np.eye(W2); T[:W2, W2:] = -np.eye(W2); T[W2:, :W2] = np.eye(W2)
        Mcell = T @ Mcell
    lam = np.linalg.eigvals(Mcell)
    prop = np.sum(np.abs(np.abs(lam) - 1) < 1e-6)
    decay = np.min(np.abs(np.log(np.abs(lam[np.abs(lam) > 1 + 1e-6])))) / a if np.any(np.abs(lam) > 1 + 1e-6) else 0.0
    return prop // 2, decay      # 右進行の本数, 最小 Im K
n2 = np.arange(1, W2 + 1); eps2 = -2 * np.cos(n2 * np.pi / (W2 + 1))
def bragg_dist(a, E):
    dmin = np.inf
    for e_ in eps2:
        if abs(E - e_) < 2:
            k = np.arccos(-(E - e_) / 2)
            if k > 1e-3:
                for order in [1, 2, 3]: dmin = min(dmin, abs(a - order * np.pi / k))
    return dmin
G23 = {}
for U in [0.5, 1.0, 2.0, 8.0]:
    Gmap = np.zeros((len(periods), len(Es))); Nb = np.zeros_like(Gmap); Dk = np.zeros_like(Gmap)
    for j, Ee in enumerate(Es):
        d2, TT0, G02, gl2, gr2 = clean(Ee, W2, L2)
        for i, a in enumerate(periods):
            coords = [(int(20 - 2.5 * a + k * a), 1) for k in range(Nd)]; coords = [(c, r) for c, r in coords if 0 <= c < L2]
            Gmap[i, j], _ = G_exact(d2, TT0, G02, gl2, gr2, coords, U)
            Nb[i, j], Dk[i, j] = transfer_bloch(Ee, a, U)
    dist = np.array([[bragg_dist(a, Ee) for Ee in Es] for a in periods]); m = np.isfinite(dist)
    Nopen = np.array([np.sum(np.abs(Ee - eps2) < 2) for Ee in Es])[None, :].repeat(len(periods), 0)
    stop = (Nb < Nopen)    # 無限周期系で少なくとも 1 本の自由チャネルがストップバンド
    G23[U] = {"Gmap": Gmap, "Nb": Nb, "Dk": Dk, "dist": dist, "stop": stop,
              "corr_bragg": float(np.corrcoef(Gmap[m], dist[m])[0, 1]),
              "G_near_bragg": float(Gmap[m & (dist < 0.25)].mean()), "G_far_bragg": float(Gmap[m & (dist > 0.75)].mean()),
              "corr_bloch": float(np.corrcoef(Gmap.ravel(), (Nb / np.maximum(Nopen, 1)).ravel())[0, 1]),
              "G_in_stop": float(Gmap[stop].mean()) if stop.any() else np.nan, "G_pass": float(Gmap[~stop].mean()),
              "corr_decay": float(np.corrcoef(Gmap.ravel(), Dk.ravel())[0, 1])}
    print(f"G2/G3 U={U}: corr(G, Bragg dist)={G23[U]['corr_bragg']:+.3f}  G near/far Bragg={G23[U]['G_near_bragg']:.3f}/{G23[U]['G_far_bragg']:.3f} | "
          f"corr(G, Bloch fraction)={G23[U]['corr_bloch']:+.3f}  G in stop/pass band={G23[U]['G_in_stop']:.3f}/{G23[U]['G_pass']:.3f}  corr(G, −ImK)={G23[U]['corr_decay']:+.3f} ({time.time()-t0:.0f}s)")

fig, axs = plt.subplots(4, 3, figsize=(16, 13))
for i, U in enumerate([0.5, 1.0, 2.0, 8.0]):
    R = G23[U]
    im = axs[i, 0].imshow(R["Gmap"], origin="lower", aspect="auto", extent=[Es[0], Es[-1], 0.5, 8.5], cmap="viridis", vmin=0, vmax=1); plt.colorbar(im, ax=axs[i, 0])
    axs[i, 0].set_title(f"U={U}: finite superlattice G/G0 (6 defects)"); axs[i, 0].set_ylabel("period a")
    for e_, c in zip(eps2, ["w", "orange", "cyan"]):
        Eo = Es[np.abs(Es - e_) < 2]; k_ = np.arccos(np.clip(-(Eo - e_) / 2, -1, 1)); a_ = np.pi / np.where(k_ > 1e-3, k_, np.nan); mm = (a_ >= 0.5) & (a_ <= 8.5)
        axs[i, 0].plot(Eo[mm], a_[mm], "--", color=c, lw=1)
    Nopen = np.array([np.sum(np.abs(Ee - eps2) < 2) for Ee in Es])[None, :].repeat(len(periods), 0)
    im = axs[i, 1].imshow(R["Nb"] / np.maximum(Nopen, 1), origin="lower", aspect="auto", extent=[Es[0], Es[-1], 0.5, 8.5], cmap="viridis", vmin=0, vmax=1); plt.colorbar(im, ax=axs[i, 1])
    axs[i, 1].set_title("infinite superlattice: propagating Bloch bands / open channels")
    im = axs[i, 2].imshow(R["Dk"], origin="lower", aspect="auto", extent=[Es[0], Es[-1], 0.5, 8.5], cmap="magma"); plt.colorbar(im, ax=axs[i, 2])
    axs[i, 2].set_title("complex band: min Im K (decay per site)")
for a in axs[-1]: a.set_xlabel("E_F")
plt.tight_layout(); plt.savefig(FIGDIR + "figR_G2G3_superlattice.png", dpi=120); plt.close()

# ================= G4: 有限温度平均（W=20, 390 構造）=================
Egrid = np.linspace(E0 - 0.5, E0 + 0.5, 41)
GE = np.zeros((len(coords_all), len(Egrid)))
for j, Ee in enumerate(Egrid):
    dE, T0E, G0E, glE, grE = clean(Ee)
    for i, coords in enumerate(coords_all): GE[i, j], _ = G_exact(dE, T0E, G0E, glE, grE, coords)
print(f"G4 energy sweep done ({time.time()-t0:.0f}s)")
np.save("G4_GE.npy", GE)
def thermal(kT):
    x = (Egrid - E0) / kT; w = np.exp(x) / (1 + np.exp(x)) ** 2 / kT; w /= np.trapezoid(w, Egrid)
    return np.trapezoid(GE * w[None, :], Egrid, axis=1)
G4 = {}
for kT in [0.0, 0.03, 0.06, 0.12]:
    g = df["G"].values if kT == 0 else thermal(kT); yT = np.log(np.clip(1 - g, 1e-4, None))
    r2s, grs, gbs = [], [], []
    for seed in range(5):
        rg = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=seed).fit(df.loc[tr, M2pp], yT[tr])
        p = rg.predict(df.loc[nov, M2pp]); yy = yT[nov]; r2s.append(1 - ((p - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
        mg = (df.loc[nov, "fam"] == "grid").values; grs.append(np.sqrt(((p[mg] - yy[mg]) ** 2).mean())); gbs.append((p[mg] - yy[mg]).mean())
    # 族内の G 分散（干渉の指標）: grid vs random
    G4[kT] = {"novel_r2": float(np.mean(r2s)), "grid_rmse": float(np.mean(grs)), "grid_bias": float(np.mean(gbs)),
              "std_G_grid": float(np.std(g[(df.fam == "grid").values])), "std_G_random": float(np.std(g[(df.fam == "random").values])),
              "mean_G_grid": float(np.mean(g[(df.fam == "grid").values])), "mean_G_random": float(np.mean(g[(df.fam == "random").values]))}
    print(f"G4 kT={kT}: novel R² {G4[kT]['novel_r2']:.3f}  grid rmse {G4[kT]['grid_rmse']:.3f} bias {G4[kT]['grid_bias']:+.3f} | std G grid/random {G4[kT]['std_G_grid']:.3f}/{G4[kT]['std_G_random']:.3f}  mean {G4[kT]['mean_G_grid']:.3f}/{G4[kT]['mean_G_random']:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for fam, c in [("grid", "C1"), ("random", "C0"), ("arc", "C2")]:
    m = (df.fam == fam).values
    ax[0].plot(Egrid, GE[m].mean(0), color=c, label=f"{fam} (mean of family)"); ax[0].fill_between(Egrid, GE[m].mean(0) - GE[m].std(0), GE[m].mean(0) + GE[m].std(0), color=c, alpha=0.15)
ax[0].axvline(E0, color="k", ls=":"); ax[0].set_xlabel("E"); ax[0].set_ylabel("G/G0"); ax[0].legend(); ax[0].set_title("G(E) by family (W=20, 390 structures, exact T-matrix)")
kTs = [0.0, 0.03, 0.06, 0.12]
ax[1].plot(kTs, [G4[k]["novel_r2"] for k in kTs], "o-", label="novel R² (M2'')"); ax[1].plot(kTs, [G4[k]["grid_rmse"] for k in kTs], "s-", label="grid RMSE"); ax[1].plot(kTs, [-G4[k]["grid_bias"] for k in kTs], "^-", label="−grid bias")
ax[1].set_xlabel("kT (units of t)"); ax[1].legend(); ax[1].set_title("G4: does thermal averaging remove the grid anomaly?")
plt.tight_layout(); plt.savefig(FIGDIR + "figS_G4_thermal.png", dpi=130); plt.close()

json.dump({"G1_corr": {c: {"novel": float(np.corrcoef(df.loc[nov, 'resid'], df.loc[nov, c])[0, 1])} for c in ["sv_min", "log_det", "n_small_ev", "pr_collective"]},
           "G23": {str(U): {k: v for k, v in R.items() if not isinstance(v, np.ndarray)} for U, R in G23.items()},
           "G4": {str(k): v for k, v in G4.items()}}, open("results_G.json", "w"), indent=1)
df.to_csv("dataset_G.csv", index=False)
print("done", f"{time.time()-t0:.0f}s")
