"""
第8報 Exp H: 局所周期性 → 有効ユニットセル → Bloch 伝播条件 を W=20 の一般配置の記述子に
  H1: 重みなし自己相関から周期候補（対照）
  H3: モード重み付き自己相関  C_n(Δ) = Σ_{j,j'} φ_n(y_j)² φ_n(y_j')² δ(x_j − x_j' − Δ)
  H4: 周期候補 a* で有効ユニットセル（行ごとの占有率 → U_eff,r = U·occ_r）を構成し
      2W×2W 転送行列で Bloch 伝播状態数・最小減衰率を計算 → 特徴量
  回帰: M2'' + H 特徴で未知族 R², grid 残差がどう動くか（5 seed）
"""
import numpy as np, json, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from _paths import *   # noqa: cwd -> results/data, src on path

W, L, E0, U_DEF = 20, 40, -1.1, 8.0
t0 = time.time()
df = pd.read_csv("dataset_G.csv")
raw = json.load(open("coords_E.json"))[str(E0)]
coords_all = [[tuple(c) for c in r["coords"]] for r in raw]

nn = np.arange(1, W + 1); eps_n = -2 * np.cos(nn * np.pi / (W + 1))
is_open = np.abs(E0 - eps_n) < 2 - 1e-9; N_OPEN = int(is_open.sum())
kx = np.where(is_open, np.arccos(np.clip(-(E0 - eps_n) / 2, -1, 1)), 0.0); vel = 2 * np.sin(kx)
PHI2 = np.array([[2 / (W + 1) * np.sin(n * np.pi * (y + 1) / (W + 1)) ** 2 for y in range(W)] for n in nn])  # φ_n(y)²
Hp = np.zeros((W, W))
for r in range(W - 1): Hp[r, r + 1] = Hp[r + 1, r] = -1.0
DMAX = 10

def autocorr(xs, w, dmax=DMAX):
    C = np.zeros(dmax + 1)
    for d in range(1, dmax + 1):
        for j in range(len(xs)):
            for jp in range(len(xs)):
                if xs[j] - xs[jp] == d: C[d] += w[j] * w[jp]
    return C

def bloch(E, a, Ueff_rows):
    """周期 a、最初の列に diag(Ueff_rows) を持つセルの転送行列 → (伝播 Bloch 数/2, 最小 Im K among slowest decaying)"""
    M = np.eye(2 * W, dtype=complex)
    for c in range(a):
        Hc = Hp.copy()
        if c == 0: Hc = Hc + np.diag(Ueff_rows)
        T = np.zeros((2 * W, 2 * W), complex); T[:W, :W] = Hc - E * np.eye(W); T[:W, W:] = -np.eye(W); T[W:, :W] = np.eye(W)
        M = T @ M
    lam = np.linalg.eigvals(M); al = np.abs(lam)
    nprop = int(np.sum(np.abs(al - 1) < 1e-6)) // 2
    # 伝播しなかった「本来開いている」チャネル分の減衰: 1 に最も近い |λ|>1 の log
    grow = al[al > 1 + 1e-6]
    kmin = float(np.min(np.log(grow)) / a) if grow.size else 0.0
    return nprop, kmin

def h_features(coords):
    P = np.array(coords); xs = P[:, 0].astype(int); ys = P[:, 1].astype(int); n = len(xs)
    f = {}
    # --- H1: 重みなし ---
    C0 = autocorr(xs, np.ones(n)); a1 = int(np.argmax(C0[1:]) + 1)
    f["H1_period"] = a1; f["H1_peak"] = float(C0[a1] / max(C0[1:].mean(), 1e-9))
    # --- H3: モード重み付き（開チャネルを 1/v_n で重み） ---
    Cm = np.zeros(DMAX + 1); per_mode = []
    for ni in np.where(is_open)[0]:
        w = PHI2[ni, ys]; Cn = autocorr(xs, w) / max(w.sum() ** 2, 1e-12)
        Cm += Cn / vel[ni]; per_mode.append(int(np.argmax(Cn[1:]) + 1) if Cn[1:].max() > 0 else 0)
    a3 = int(np.argmax(Cm[1:]) + 1) if Cm[1:].max() > 0 else 1
    f["H3_period"] = a3; f["H3_peak"] = float(Cm[a3] / max(Cm[1:].mean(), 1e-12))
    f["H3_mode_agree"] = float(np.mean(np.array(per_mode) == a3))        # モード間で周期が一致する割合
    f["H3_Cpeak_abs"] = float(Cm[a3])
    # --- H4: 有効ユニットセル（H3 の周期）---
    span = max(xs.max() - xs.min() + 1, a3); ncell = span / a3
    occ = np.bincount(ys, minlength=W) / ncell                           # 行ごとの 1 セル当たり占有率
    Ueff = np.minimum(occ, 1.0) * U_DEF
    npb, kmin = bloch(E0, a3, Ueff)
    f["H4_bloch_frac"] = npb / N_OPEN; f["H4_kmin"] = kmin
    # 二値化版（占有率 ≥ 0.5 の行に完全な欠陥）
    Ub = (occ >= 0.5) * U_DEF
    npb2, kmin2 = bloch(E0, a3, Ub); f["H4b_bloch_frac"] = npb2 / N_OPEN; f["H4b_kmin"] = kmin2
    # E_F 近傍平均（±0.1）
    fr = [bloch(E0 + dE, a3, Ueff)[0] / N_OPEN for dE in (-0.1, 0.1)]; f["H4_bloch_frac_avg"] = float((fr[0] + fr[1] + f["H4_bloch_frac"]) / 3)
    # 対照: H1 の周期で同じ計算
    occ1 = np.bincount(ys, minlength=W) / (max(xs.max() - xs.min() + 1, a1) / a1)
    f["H4_bloch_frac_H1"] = bloch(E0, a1, np.minimum(occ1, 1) * U_DEF)[0] / N_OPEN
    # 真の周期（grid 用の検証）: x の最頻差分
    dx = np.diff(np.unique(xs)); f["true_dx"] = int(np.bincount(dx).argmax()) if dx.size else 0
    return f

H = pd.DataFrame([h_features(c) for c in coords_all]); df = pd.concat([df.reset_index(drop=True), H], axis=1)
print(f"H features done ({time.time()-t0:.0f}s)")

# 周期の再発見（grid）
g = df[df.fam == "grid"]
print("grid: H1 period == true dx:", round((g.H1_period == g.true_dx).mean(), 2), " H3 period == true dx:", round((g.H3_period == g.true_dx).mean(), 2))
print("H3_peak median by family:", df.groupby("fam")["H3_peak"].median().round(2).to_dict())
print("H4_bloch_frac median by family:", df.groupby("fam")["H4_bloch_frac"].median().round(3).to_dict())
print("H4b_bloch_frac median by family:", df.groupby("fam")["H4b_bloch_frac"].median().round(3).to_dict())

# 残差との相関
nov = df["split"] == "test_novel"; gridm = nov & (df.fam == "grid")
print("corr with M2'' residual (novel / grid-only):")
for c in ["H1_peak", "H3_peak", "H3_mode_agree", "H4_bloch_frac", "H4_kmin", "H4b_bloch_frac", "H4_bloch_frac_avg", "H4_bloch_frac_H1"]:
    print(f"   {c:18s} {np.corrcoef(df.loc[nov, 'resid'], df.loc[nov, c])[0,1]:+.3f} / {np.corrcoef(df.loc[gridm, 'resid'], df.loc[gridm, c])[0,1]:+.3f}")
print("corr with G itself (all):", {c: round(np.corrcoef(df["G"], df[c])[0, 1], 3) for c in ["H4_bloch_frac", "H4b_bloch_frac", "H4_kmin", "H3_peak"]})

# 回帰
y = np.log(np.clip(1 - df["G"], 1e-4, None)); tr = df["split"] == "train"
geom = [c for c in df if c.startswith(("M0", "M1", "M2"))]; sq = [c for c in df if c.startswith(f"SqE{E0}_")]; amn = [c for c in df if c.startswith("A_")]
M2pp = geom + sq + amn
sets = {"M2''": M2pp,
        "M2''+H1 (geom period)": M2pp + ["H1_peak", "H4_bloch_frac_H1"],
        "M2''+H3 (mode period)": M2pp + ["H3_peak", "H3_mode_agree", "H3_Cpeak_abs"],
        "M2''+H3+H4 (Bloch)": M2pp + ["H3_peak", "H3_mode_agree", "H3_Cpeak_abs", "H4_bloch_frac", "H4_kmin", "H4b_bloch_frac", "H4b_kmin", "H4_bloch_frac_avg"],
        "M2''+H4 only": M2pp + ["H4_bloch_frac", "H4_kmin", "H4b_bloch_frac", "H4b_kmin", "H4_bloch_frac_avg"]}
res = {}
for name, cols in sets.items():
    r = {"novel": [], "same": [], "grid_rmse": [], "grid_bias": [], "fam": {}}
    for seed in range(5):
        reg = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=seed).fit(df.loc[tr, cols], y[tr])
        for sp, key in [("test_novel", "novel"), ("test_same", "same")]:
            m = df["split"] == sp; p = reg.predict(df.loc[m, cols]); yy = y[m].values
            r[key].append(1 - ((p - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
        p = reg.predict(df.loc[nov, cols]); yy = y[nov].values
        for fam in df.loc[nov, "fam"].unique():
            mm = (df.loc[nov, "fam"] == fam).values; r["fam"].setdefault(fam, []).append(float(np.sqrt(((p[mm] - yy[mm]) ** 2).mean())))
        mg = (df.loc[nov, "fam"] == "grid").values; r["grid_rmse"].append(float(np.sqrt(((p[mg] - yy[mg]) ** 2).mean()))); r["grid_bias"].append(float((p[mg] - yy[mg]).mean()))
    res[name] = r
    print(f"{name:24s} novel R² {np.mean(r['novel']):.3f}±{np.std(r['novel']):.3f} | same {np.mean(r['same']):.3f} | grid rmse {np.mean(r['grid_rmse']):.3f} bias {np.mean(r['grid_bias']):+.3f} | fam rmse {{{', '.join(f'{k}:{np.mean(v):.3f}' for k,v in r['fam'].items())}}}")
reg = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0).fit(df.loc[tr, sets["M2''+H3+H4 (Bloch)"]], y[tr])
imp = sorted(zip(reg.feature_importances_, sets["M2''+H3+H4 (Bloch)"]), reverse=True)[:10]
print("importance:", [(n, round(float(v), 3)) for v, n in imp])
json.dump({k: {kk: (vv if kk != "fam" else {f: list(map(float, x)) for f, x in vv.items()}) for kk, vv in v.items()} for k, v in res.items()}, open("results_H.json", "w"), indent=1)
df.to_csv("dataset_H.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
names = list(sets)
ax[0].errorbar(range(len(names)), [np.mean(res[n]["novel"]) for n in names], yerr=[np.std(res[n]["novel"]) for n in names], fmt="o-", label="novel R²")
ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=20, fontsize=7); ax[0].axhline(0.96, color="gray", ls=":", label="NEGF upper bound (M3b)"); ax[0].legend(); ax[0].set_title("H: adding periodic-propagation descriptors")
ax[1].bar(range(len(names)), [np.mean(res[n]["grid_rmse"]) for n in names], yerr=[np.std(res[n]["grid_rmse"]) for n in names], color="C3")
ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, rotation=20, fontsize=7); ax[1].set_title("grid RMSE")
for fam, c in zip(sorted(df.fam.unique()), plt.cm.tab10.colors):
    m = df.fam == fam; ax[2].scatter(df.loc[m, "H4_bloch_frac"] + np.random.uniform(-0.01, 0.01, m.sum()), df.loc[m, "G"], s=10, color=c, label=fam, alpha=0.7)
ax[2].set_xlabel("H4: propagating Bloch fraction of effective unit cell"); ax[2].set_ylabel("G/G0"); ax[2].legend(fontsize=7); ax[2].set_title("Bloch fraction vs conductance")
plt.tight_layout(); plt.savefig(FIGDIR + "figT_H_local_period.png", dpi=130); plt.close()
print("done", f"{time.time()-t0:.0f}s")
