"""
第9報 Exp I: レジーム依存の補完を予測実験で示す
  新族 longgrid（x 方向 6〜8 周期・全幅・行周期 b）。訓練 a∈{2,4}, b∈{2,4}／テスト a∈{3,5}, b∈{3,5}（未見周期）
  H4c: 実パターンのユニットセル（a3 列の窓内の実欠陥配置）で転送行列 → Bloch 伝播割合
  評価: test_same / test_novel（少数周期）/ test_longgrid（多周期）で M2'' vs M2''+H4
"""
import numpy as np, json, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from _paths import *   # noqa: cwd -> results/data, src on path
from negf_toy import Device
import exp_D_fams as F

W, L, E0, U_DEF = 20, 40, -1.1, 8.0
t0 = time.time()
d0 = Device(W, L, E=E0, eta=1e-6); T0 = d0.transmission(); N0 = d0.n_open_channels()
G0, GL, GR = d0.greens(); gl = GL[:W, :W]; gr = GR[-W:, -W:]; G0_LR = G0[:W, -W:]
nn = np.arange(1, W + 1); eps_n = -2 * np.cos(nn * np.pi / (W + 1)); is_open = np.abs(E0 - eps_n) < 2 - 1e-9
kx = np.where(is_open, np.arccos(np.clip(-(E0 - eps_n) / 2, -1, 1)), 0.0); kappa = np.where(~is_open, np.arccosh(np.clip(np.abs(E0 - eps_n) / 2, 1, None)), 0.0)
vel = 2 * np.sin(kx); ky = nn * np.pi / (W + 1)
PHI = np.array([[np.sqrt(2 / (W + 1)) * np.sin(n * np.pi * (y + 1) / (W + 1)) for y in range(W)] for n in nn]); PHI2 = PHI ** 2
Hp = np.zeros((W, W))
for r in range(W - 1): Hp[r, r + 1] = Hp[r + 1, r] = -1.0

def G_exact(coords):
    idx = [d0.idx(c, r) for c, r in coords]; nd = len(idx); Gdd = G0[np.ix_(idx, idx)]
    T = np.linalg.solve(np.eye(nd) - Gdd * U_DEF, U_DEF * np.eye(nd))
    G_LR = G0_LR + G0[:W, idx] @ T @ G0[idx, -W:]
    return np.real(np.trace(gl @ G_LR @ gr @ G_LR.conj().T)) / T0

# ---------- 既存特徴（S(q), A_mn）を再実装 ----------
def structure_factor(P, qx, qy):
    ph = np.exp(1j * (np.outer(qx, P[:, 0]) + np.outer(qy, P[:, 1]))); return np.abs(ph.sum(1)) ** 2 / len(P) ** 2
def sq_features(coords):
    P = np.array(coords, float); op = np.where(is_open)[0]; k = kx[op]; kyo = ky[op]
    qx = (k[:, None] + k[None, :]).ravel(); qy1 = (kyo[:, None] - kyo[None, :]).ravel(); qy2 = (kyo[:, None] + kyo[None, :]).ravel(); qxf = (k[:, None] - k[None, :]).ravel()
    qg = np.linspace(0.3, np.pi, 24); QX, QY = np.meshgrid(qg, qg); Sg = structure_factor(P, QX.ravel(), QY.ravel())
    return {f"SqE{E0}_S_back": structure_factor(P, 2 * k, 0 * k).mean(), f"SqE{E0}_S_inter": 0.5 * (structure_factor(P, qx, qy1).mean() + structure_factor(P, qx, qy2).mean()),
            f"SqE{E0}_S_fwd": 0.5 * (structure_factor(P, qxf, qy1).mean() + structure_factor(P, qxf, qy2).mean()), f"SqE{E0}_S_back_max": structure_factor(P, 2 * k, 0 * k).max(),
            f"SqE{E0}_Sgrid_max": Sg.max(), f"SqE{E0}_Sgrid_mean": Sg.mean()}
def amn_features(coords):
    P = np.array(coords); xs, ys = P[:, 0].astype(float), P[:, 1].astype(int); op = np.where(is_open)[0]; ev = np.where(~is_open)[0]; Pd = PHI[:, ys]
    Ab = np.zeros((len(op), len(op)), complex); Af = np.zeros_like(Ab)
    for a, n in enumerate(op):
        for b, m in enumerate(op):
            ov = Pd[m] * Pd[n]; Ab[b, a] = np.sum(ov * np.exp(1j * (kx[n] + kx[m]) * xs)) / np.sqrt(vel[n] * vel[m]); Af[b, a] = np.sum(ov * np.exp(1j * (kx[n] - kx[m]) * xs)) / np.sqrt(vel[n] * vel[m])
    off = ~np.eye(len(op), dtype=bool)
    Ae = np.zeros((len(ev), len(op)), complex)
    for a, n in enumerate(op):
        for b, m in enumerate(ev): Ae[b, a] = np.sum(Pd[m] * Pd[n] * np.exp(1j * kx[n] * xs)) / np.sqrt(vel[n] * max(np.sinh(kappa[m]) * 2, 1e-3))
    return {"A_back_diag": float(np.mean(np.abs(np.diag(Ab)) ** 2)), "A_back_off": float(np.mean(np.abs(Ab[off]) ** 2)), "A_fwd_off": float(np.mean(np.abs(Af[off]) ** 2)),
            "A_fwd_diag": float(np.mean(np.abs(np.diag(Af)) ** 2)), "A_born_R": float(np.sum(np.abs(Ab) ** 2)), "A_evan": float(np.mean(np.abs(Ae) ** 2)), "A_back_max": float(np.abs(Ab).max() ** 2)}

# ---------- H 特徴（H3 周期 + H4 VCA + H4c 実パターン） ----------
DMAX = 10
def autocorr(xs, w):
    C = np.zeros(DMAX + 1); D = xs[:, None] - xs[None, :]
    for d in range(1, DMAX + 1): C[d] = np.sum(np.outer(w, w)[D == d])
    return C
def bloch_cell(E, cell_cols):
    """cell_cols: list (長さ a) of onsite 配列(W) → (伝播 Bloch 数, 最小 log|λ|/a)"""
    a = len(cell_cols); M = np.eye(2 * W, dtype=complex)
    for Uc in cell_cols:
        T = np.zeros((2 * W, 2 * W), complex); T[:W, :W] = Hp + np.diag(Uc) - E * np.eye(W); T[:W, W:] = -np.eye(W); T[W:, :W] = np.eye(W); M = T @ M
    al = np.abs(np.linalg.eigvals(M)); nprop = int(np.sum(np.abs(al - 1) < 1e-6)) // 2
    grow = al[al > 1 + 1e-6]; return nprop, (float(np.min(np.log(grow)) / a) if grow.size else 0.0)
def h_features(coords):
    P = np.array(coords); xs = P[:, 0].astype(int); ys = P[:, 1].astype(int); f = {}
    Cm = np.zeros(DMAX + 1)
    for ni in np.where(is_open)[0]:
        w = PHI2[ni, ys]; Cm += autocorr(xs, w) / max(w.sum() ** 2, 1e-12) / vel[ni]
    a3 = int(np.argmax(Cm[1:]) + 1) if Cm[1:].max() > 0 else 1
    f["H3_period"] = a3; f["H3_peak"] = float(Cm[a3] / max(Cm[1:].mean(), 1e-12))
    span = xs.max() - xs.min() + 1; f["H_nperiods"] = span / a3
    occ = np.bincount(ys, minlength=W) / max(span / a3, 1)
    npb, kmin = bloch_cell(E0, [np.minimum(occ, 1) * U_DEF] + [np.zeros(W)] * (a3 - 1)); f["H4_bloch_frac"] = npb / N0; f["H4_kmin"] = kmin
    # H4c: 実パターン — 最も欠陥の多い a3 列の窓をセルに
    best = None
    for x0 in range(xs.min(), xs.max() - a3 + 2):
        m = (xs >= x0) & (xs < x0 + a3)
        if best is None or m.sum() > best[0]: best = (m.sum(), x0)
    x0 = best[1]; cell = [np.zeros(W) for _ in range(a3)]
    for xj, yj in zip(xs, ys):
        if x0 <= xj < x0 + a3: cell[xj - x0][yj] = U_DEF
    npc, kminc = bloch_cell(E0, cell); f["H4c_bloch_frac"] = npc / N0; f["H4c_kmin"] = kminc
    fr = [bloch_cell(E0 + dE, cell)[0] / N0 for dE in (-0.1, 0.1)]; f["H4c_bloch_frac_avg"] = float((fr[0] + fr[1] + f["H4c_bloch_frac"]) / 3)
    # 「有限周期の立ち上がり」を掛けた指標: 周期数で重み
    f["H4c_x_nper"] = f["H4c_bloch_frac"] * min(f["H_nperiods"] / 6.0, 1.0) + (1 - min(f["H_nperiods"] / 6.0, 1.0))
    return f

def all_features(coords, fam, split):
    f = F.geom_features(coords); f.update(sq_features(coords)); f.update(amn_features(coords)); f.update(h_features(coords))
    f.update({"G": G_exact(coords), "fam": fam, "split": split, "coords": coords}); return f

# ---------- 既存 390 + longgrid ----------
raw = json.load(open("coords_E.json"))[str(E0)]
rows = [all_features([tuple(c) for c in r["coords"]], r["fam"], r["split"]) for r in raw]
print(f"existing 390 re-featured ({time.time()-t0:.0f}s)")
rng = np.random.default_rng(99)
def longgrid(a, b, nper, rng):
    x0 = rng.integers(1, max(2, L - a * nper - 1)); y0 = rng.integers(0, b)
    return [(x0 + i * a, y) for i in range(nper) for y in range(y0, W, b)]
cnt = {"train": 0, "test_longgrid": 0}
for a, b, split in [(2, 2, "train"), (2, 4, "train"), (4, 2, "train"), (4, 4, "train"), (3, 3, "test_longgrid"), (3, 5, "test_longgrid"), (5, 3, "test_longgrid"), (5, 5, "test_longgrid")]:
    for nper in [6, 7, 8]:
        if a * nper > L - 3: continue
        for _ in range(8 if split == "train" else 6):
            rows.append(all_features(longgrid(a, b, nper, rng), "longgrid", split)); cnt[split] += 1
# 訓練にも少数周期 grid 相当がないと不公平なので、longgrid の「短い版」(nper=3,4) も訓練に少し
for a, b in [(2, 2), (4, 4), (2, 4), (4, 2)]:
    for nper in [3, 4]:
        for _ in range(4): rows.append(all_features(longgrid(a, b, nper, rng), "shortgrid", "train"))
print(f"longgrid added: {cnt} ({time.time()-t0:.0f}s)")
df = pd.DataFrame(rows); df.drop(columns=["coords"]).to_csv("dataset_I.csv", index=False)
lg = df[df.fam == "longgrid"]
print("longgrid G range:", round(lg.G.min(), 3), "-", round(lg.G.max(), 3), " H_nperiods median:", round(lg.H_nperiods.median(), 1), " H3 period recovered:", round((lg.H3_period == [min(np.diff(np.unique([c[0] for c in cc]))) for cc in lg.coords]).mean(), 2))
for fam in ["longgrid", "grid", "random"]:
    m = df.fam == fam
    print(f"  corr(G, H4c_bloch_frac) in {fam}: {np.corrcoef(df.loc[m,'G'], df.loc[m,'H4c_bloch_frac'])[0,1]:+.3f}   corr(G, H4_bloch_frac VCA): {np.corrcoef(df.loc[m,'G'], df.loc[m,'H4_bloch_frac'])[0,1]:+.3f}   corr(G, A_fwd_diag): {np.corrcoef(df.loc[m,'G'], df.loc[m,'A_fwd_diag'])[0,1]:+.3f}")

# ---------- 回帰 ----------
y = np.log(np.clip(1 - df["G"], 1e-4, None)); tr = df["split"] == "train"
geom = [c for c in df if c.startswith(("M0", "M1", "M2"))]; sq = [c for c in df if c.startswith("SqE")]; amn = [c for c in df if c.startswith("A_")]
M2pp = geom + sq + amn
H4c = ["H4c_bloch_frac", "H4c_kmin", "H4c_bloch_frac_avg", "H4c_x_nper", "H_nperiods", "H3_peak"]
sets = {"M2''": M2pp, "M2''+H4(VCA)": M2pp + ["H4_bloch_frac", "H4_kmin", "H3_peak", "H_nperiods"], "M2''+H4c(actual cell)": M2pp + H4c}
splits = ["test_same", "test_novel", "test_longgrid"]
res = {}
for name, cols in sets.items():
    res[name] = {s: {"r2": [], "rmse": []} for s in splits}
    for seed in range(5):
        reg = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=seed).fit(df.loc[tr, cols], y[tr])
        for s in splits:
            m = df["split"] == s; p = reg.predict(df.loc[m, cols]); yy = y[m].values
            res[name][s]["r2"].append(1 - ((p - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()); res[name][s]["rmse"].append(float(np.sqrt(((p - yy) ** 2).mean())))
    print(f"{name:24s} " + " | ".join(f"{s}: R² {np.mean(res[name][s]['r2']):.3f}±{np.std(res[name][s]['r2']):.3f} rmse {np.mean(res[name][s]['rmse']):.3f}" for s in splits))
reg = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0).fit(df.loc[tr, sets["M2''+H4c(actual cell)"]], y[tr])
imp = sorted(zip(reg.feature_importances_, sets["M2''+H4c(actual cell)"]), reverse=True)[:10]; print("importance:", [(n, round(float(v), 3)) for v, n in imp])
# longgrid だけで学習→longgrid テスト（レジーム内の純粋比較）
trl = tr & (df.fam.isin(["longgrid", "shortgrid"])); ml = df["split"] == "test_longgrid"
for name, cols in sets.items():
    r2s = []
    for seed in range(5):
        reg = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=seed).fit(df.loc[trl, cols], y[trl]); p = reg.predict(df.loc[ml, cols]); yy = y[ml].values
        r2s.append(1 - ((p - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
    print(f"  [grid-only training] {name:24s} test_longgrid R² {np.mean(r2s):.3f}±{np.std(r2s):.3f}")
json.dump({k: {s: {kk: list(map(float, vv)) for kk, vv in v.items()} for s, v in r.items()} for k, r in res.items()}, open("results_I.json", "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
names = list(sets); x = np.arange(len(splits)); wd = 0.26
for i, n in enumerate(names):
    ax[0].bar(x + (i - 1) * wd, [np.mean(res[n][s]["r2"]) for s in splits], wd, yerr=[np.std(res[n][s]["r2"]) for s in splits], label=n)
ax[0].set_xticks(x); ax[0].set_xticklabels(["same\n(few periods)", "novel\n(few periods)", "longgrid\n(6–8 periods, unseen a,b)"]); ax[0].set_ylabel("R²"); ax[0].legend(fontsize=8); ax[0].set_title("I: regime-dependent value of the Bloch descriptor")
for fam, c in [("longgrid", "C3"), ("grid", "C1"), ("random", "C0"), ("cluster", "C2")]:
    m = df.fam == fam; ax[1].scatter(df.loc[m, "H4c_bloch_frac"] + np.random.uniform(-0.01, 0.01, m.sum()), df.loc[m, "G"], s=10, color=c, label=fam, alpha=0.7)
ax[1].set_xlabel("H4c: Bloch propagating fraction (actual unit cell)"); ax[1].set_ylabel("G/G0"); ax[1].legend(fontsize=8); ax[1].set_title("Bloch fraction vs G by family")
ax[2].scatter(df.loc[df.fam == "longgrid", "H_nperiods"], df.loc[df.fam == "longgrid", "G"], c=df.loc[df.fam == "longgrid", "H4c_bloch_frac"], cmap="viridis", s=14); ax[2].set_xlabel("number of periods"); ax[2].set_ylabel("G/G0"); ax[2].set_title("longgrid: G vs periods (color = Bloch fraction)")
plt.tight_layout(); plt.savefig(FIGDIR + "figU_I_regime.png", dpi=130); plt.close()
print("done", f"{time.time()-t0:.0f}s")
