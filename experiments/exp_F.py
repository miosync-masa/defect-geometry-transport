"""
第6報 Exp F: 構造因子から散乱行列へ
  F1: チャネル結合幾何 A_mn = Σ_j φ_m(y_j)φ_n(y_j) e^{i(k_n∓k_m)x_j} 由来の特徴量 (M2″)
  F3: 多重散乱の階層  ISA（独立散乱体, 加法的=PCC）→ pair（2体）→ full（厳密 T 行列 = NEGF と一致）
      T_dd = U (1 − G0_dd U)^{-1}  を欠陥サイト上で厳密に構成, Fisher–Lee で G/G0 を出す
  F5: seed 反復で改善が分割に依存しないか
  F2: W=4（3チャネル）で周期×E_F マップ → Bragg 線に追従するか
データ: coords_E.json (E_F=-1.1, 390 構造, NEGF 済み)
"""
import numpy as np, json, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from _paths import *   # noqa: cwd -> results/data, src on path
from negf_toy import Device

W, L, E, U_DEF = 20, 40, -1.1, 8.0
t0 = time.time()
d0 = Device(W, L, E=E, eta=1e-6); T0 = d0.transmission(); N0 = d0.n_open_channels()
G0, GL, GR = d0.greens()          # 清浄系グリーン関数（全サイト）
G0_LR = G0[:W, -W:]; gl = GL[:W, :W]; gr = GR[-W:, -W:]

def transmission_from_block(G_LR):
    return np.real(np.trace(gl @ G_LR @ gr @ G_LR.conj().T))
assert abs(transmission_from_block(G0_LR) - T0) < 1e-6

# ---------- チャネル量 ----------
nn = np.arange(1, W + 1); eps_n = -2 * np.cos(nn * np.pi / (W + 1))
is_open = np.abs(E - eps_n) < 2 - 1e-9
kx = np.where(is_open, np.arccos(np.clip(-(E - eps_n) / 2, -1, 1)), 0.0)
kappa = np.where(~is_open, np.arccosh(np.clip(np.abs(E - eps_n) / 2, 1, None)), 0.0)   # 減衰モード
v = 2 * np.sin(kx)                                                                       # 群速度 (t=1)
def phi(n, y): return np.sqrt(2 / (W + 1)) * np.sin(n * np.pi * (y + 1) / (W + 1))
PHI = np.array([[phi(n, y) for y in range(W)] for n in nn])     # (W modes, W sites)

def amn_features(coords):
    P = np.array(coords); xs, ys = P[:, 0].astype(float), P[:, 1].astype(int)
    op = np.where(is_open)[0]; ev = np.where(~is_open)[0]
    Phi_d = PHI[:, ys]                                   # (modes, defects)
    f = {}
    # 開⇄開: 反射 (k_n + k_m) と 前方 (k_n − k_m)
    Ab = np.zeros((len(op), len(op)), complex); Af = np.zeros_like(Ab)
    for a, n in enumerate(op):
        for b, m in enumerate(op):
            ov = Phi_d[m] * Phi_d[n]
            Ab[b, a] = np.sum(ov * np.exp(1j * (kx[n] + kx[m]) * xs)) / np.sqrt(v[n] * v[m])
            Af[b, a] = np.sum(ov * np.exp(1j * (kx[n] - kx[m]) * xs)) / np.sqrt(v[n] * v[m])
    off = ~np.eye(len(op), dtype=bool)
    f["A_back_diag"] = float(np.mean(np.abs(np.diag(Ab)) ** 2))      # チャネル内後方散乱
    f["A_back_off"] = float(np.mean(np.abs(Ab[off]) ** 2))           # チャネル間反射
    f["A_fwd_off"] = float(np.mean(np.abs(Af[off]) ** 2))            # チャネル間前方混合
    f["A_fwd_diag"] = float(np.mean(np.abs(np.diag(Af)) ** 2))       # 前方位相（≈ (Σ|φ|²)²）
    f["A_born_R"] = float(np.sum(np.abs(Ab) ** 2))                   # Born 1 次の総反射
    # 開→減衰モードへの結合（共鳴・準束縛の種）
    Ae = np.zeros((len(ev), len(op)), complex)
    for a, n in enumerate(op):
        for b, m in enumerate(ev):
            Ae[b, a] = np.sum(Phi_d[m] * Phi_d[n] * np.exp(1j * kx[n] * xs)) / np.sqrt(v[n] * max(np.sinh(kappa[m]) * 2, 1e-3))
    f["A_evan"] = float(np.mean(np.abs(Ae) ** 2))
    # 最強結合チャネル対の集中度
    f["A_back_max"] = float(np.abs(Ab).max() ** 2)
    return f

# ---------- 多重散乱の階層（厳密 T 行列） ----------
def multiple_scattering(coords):
    idx = [d0.idx(c, r) for c, r in coords]; nd = len(idx)
    Gdd = G0[np.ix_(idx, idx)]; U = U_DEF * np.eye(nd)
    G_Ld = G0[:W, idx]; G_dR = G0[idx, -W:]
    def Tn(Tmat): return transmission_from_block(G0_LR + G_Ld @ Tmat @ G_dR) / T0
    # ISA: 各サイトの t_j = U/(1 − G0_jj U), サイト間多重散乱なし
    tj = np.diag(U_DEF / (1 - np.diag(Gdd) * U_DEF))
    G_isa = Tn(tj)
    # pair: t + t G_offdiag t
    Goff = Gdd - np.diag(np.diag(Gdd))
    G_pair = Tn(tj + tj @ Goff @ tj)
    # full: 厳密
    Tfull = np.linalg.solve(np.eye(nd) - Gdd @ U, U)
    G_full = Tn(Tfull)
    # Born 1次（素の U）
    G_born1 = Tn(U)
    return {"G_isa": G_isa, "G_pair": G_pair, "G_full": G_full, "G_born1": G_born1}

# ---------- データ読み込みと特徴量付加 ----------
raw = json.load(open("coords_E.json"))[str(E)]
rows = []
for r in raw:
    coords = [tuple(c) for c in r["coords"]]
    f = {k: v_ for k, v_ in r.items() if k != "coords"}
    f.update(amn_features(coords)); f.update(multiple_scattering(coords)); rows.append(f)
df = pd.DataFrame(rows)
print(f"features done ({time.time()-t0:.0f}s); exact-T vs NEGF max|ΔG| = {np.abs(df['G_full']-df['G']).max():.2e}")

# F3: 階層の予測力（回帰なし, 直接）
y = np.log(np.clip(1 - df["G"], 1e-4, None))
print("\nF3 multiple-scattering hierarchy (direct, no regression) — RMSE of ln(1−G) by family:")
F3 = {}
for lvl in ["G_born1", "G_isa", "G_pair", "G_full"]:
    yp = np.log(np.clip(1 - df[lvl], 1e-4, None))
    F3[lvl] = {}
    for fam in sorted(df["fam"].unique()):
        m = df["fam"] == fam
        F3[lvl][fam] = float(np.sqrt(np.mean((yp[m] - y[m]) ** 2)))
    F3[lvl]["ALL"] = float(np.sqrt(np.mean((yp - y) ** 2)))
    print(f"  {lvl:8s}", {k: round(v_, 3) for k, v_ in F3[lvl].items()})

# F1 + F5: 記述子階層 × seed
geom = [c for c in df if c.startswith(("M0", "M1", "M2"))]
sq = [c for c in df if c.startswith(f"SqE{E}_")]
amn = [c for c in df if c.startswith("A_")]
sets = {"M2 (real space)": geom, "M2' (+S(q))": geom + sq, "M2'' (+A_mn)": geom + sq + amn,
        "M2''+ISA": geom + sq + amn + ["G_isa"], "M2''+ISA+pair": geom + sq + amn + ["G_isa", "G_pair"]}
tr = df["split"] == "train"
res = {}
for name, cols in sets.items():
    res[name] = {"novel_r2": [], "same_r2": [], "grid_rmse": [], "grid_bias": []}
    for seed in range(5):
        reg = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=seed).fit(df.loc[tr, cols], y[tr])
        for sp, key in [("test_novel", "novel_r2"), ("test_same", "same_r2")]:
            m = df["split"] == sp; p = reg.predict(df.loc[m, cols]); yy = y[m].values
            res[name][key].append(1 - ((p - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
        m = (df["split"] == "test_novel") & (df["fam"] == "grid"); p = reg.predict(df.loc[m, cols]); yy = y[m].values
        res[name]["grid_rmse"].append(float(np.sqrt(((p - yy) ** 2).mean()))); res[name]["grid_bias"].append(float((p - yy).mean()))
    print(f"{name:16s} novel R² {np.mean(res[name]['novel_r2']):.3f}±{np.std(res[name]['novel_r2']):.3f} | same {np.mean(res[name]['same_r2']):.3f} | grid rmse {np.mean(res[name]['grid_rmse']):.3f} bias {np.mean(res[name]['grid_bias']):+.3f}")
reg = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0).fit(df.loc[tr, sets["M2'' (+A_mn)"]], y[tr])
imp = sorted(zip(reg.feature_importances_, sets["M2'' (+A_mn)"]), reverse=True)[:10]
print("importance M2'':", [(n, round(float(v_), 3)) for v_, n in imp])
json.dump({"F3": F3, "F1F5": {k: {kk: [float(x) for x in vv] for kk, vv in v_.items()} for k, v_ in res.items()},
           "importance": [(n, float(v_)) for v_, n in imp]}, open("results_F.json", "w"), indent=1)
df.drop(columns=[]).to_csv("dataset_F.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
names = list(sets)
ax[0].errorbar(range(len(names)), [np.mean(res[n]["novel_r2"]) for n in names], yerr=[np.std(res[n]["novel_r2"]) for n in names], fmt="o-", label="novel families")
ax[0].errorbar(range(len(names)), [np.mean(res[n]["same_r2"]) for n in names], yerr=[np.std(res[n]["same_r2"]) for n in names], fmt="s--", label="same families")
ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=20, fontsize=8); ax[0].set_ylabel("R² (5 seeds)"); ax[0].legend(); ax[0].set_title("F1/F5: descriptor hierarchy")
ax[1].bar(range(len(names)), [np.mean(res[n]["grid_rmse"]) for n in names], yerr=[np.std(res[n]["grid_rmse"]) for n in names], color="C3")
ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, rotation=20, fontsize=8); ax[1].set_ylabel("RMSE on grid family"); ax[1].set_title("grid residual")
fams = sorted(df["fam"].unique()); lv = ["G_born1", "G_isa", "G_pair", "G_full"]
for i, fam in enumerate(fams):
    ax[2].plot(lv, [F3[l][fam] for l in lv], "o-", label=fam)
ax[2].set_yscale("log"); ax[2].set_ylabel("RMSE ln(1−G) (direct)"); ax[2].legend(fontsize=7, ncol=2); ax[2].set_title("F3: multiple-scattering hierarchy per family")
plt.tight_layout(); plt.savefig(FIGDIR + "figO_F_hierarchy.png", dpi=130); plt.close()

# ---------- F2: 少数チャネル (W=4) 周期 × E_F マップ ----------
W2, L2 = 4, 40
Es = np.linspace(-2.6, -0.2, 49); periods = np.arange(1, 9); Nd = 6
Gmap = np.zeros((len(periods), len(Es)))
for j, Ee in enumerate(Es):
    dd0 = Device(W2, L2, E=Ee, eta=1e-6); TT0 = dd0.transmission()
    for i, a in enumerate(periods):
        coords = [(int(20 - 2.5 * a + k * a), 1) for k in range(Nd)]; coords = [(c, r) for c, r in coords if 0 <= c < L2]
        d = Device(W2, L2, E=Ee, eta=1e-6); d.set_onsite([d.idx(c, r) for c, r in coords], U_DEF)
        Gmap[i, j] = d.transmission() / max(TT0, 1e-9)
print(f"F2 map done ({time.time()-t0:.0f}s)")
fig, ax = plt.subplots(figsize=(9, 4.5))
im = ax.imshow(Gmap, origin="lower", aspect="auto", extent=[Es[0], Es[-1], periods[0] - 0.5, periods[-1] + 0.5], cmap="viridis", vmin=0, vmax=1)
plt.colorbar(im, label="G/G0")
n2 = np.arange(1, W2 + 1); eps2 = -2 * np.cos(n2 * np.pi / (W2 + 1))
for n_, e_ in zip(n2, eps2):
    Eo = Es[np.abs(Es - e_) < 2]; k_ = np.arccos(np.clip(-(Eo - e_) / 2, -1, 1))
    for order in [1, 2]:
        a_star = order * np.pi / np.where(k_ > 1e-3, k_, np.nan)      # 2k a = 2π·order
        ax.plot(Eo, a_star, "w--" if order == 1 else "w:", lw=1)
ax.set_xlabel("E_F"); ax.set_ylabel("longitudinal period a (6 defects, row 1)"); ax.set_title("F2: W=4 (≤3 channels). dashed: Bragg a = π·order/k_n(E)")
plt.tight_layout(); plt.savefig(FIGDIR + "figP_F2_period_energy_map.png", dpi=130); plt.close()
np.save("F2_Gmap.npy", Gmap)
print("done", f"{time.time()-t0:.0f}s")
