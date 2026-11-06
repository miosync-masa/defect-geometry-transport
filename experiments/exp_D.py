"""
Exp D (G3 の最小版): 未知配置に対する予測実験  ΔL = L(M0) − L(M3)
  M0: 欠陥数                         M1: +幾何（方向・射影被覆・慣性）
  M2: +相関（最近接・クラスター・対相関）  M3a: +清浄系の局所電子応答（安価, G情報なし）
  M3b: +バイアス応答 δn 記述子（NEGF, 上限比較）
訓練: 配置族 {random, 異方性クラスター, 傾いた線分, 2クラスター}   N_def ∈ {6,9,12,15,18}
テスト: (i) 同族・別seed  (ii) 未使用族 {弧, 規則格子, 疎な壁+クラスター}
"""
import numpy as np, json, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from _paths import *   # noqa: cwd -> results/data, src on path
from negf_toy import Device

W, L, E, U_DEF = 20, 40, -1.1, 8.0
C0, C1 = 10, 30            # 欠陥を置く列範囲
def fresh(): return Device(W, L, E=E, eta=1e-6)
d0 = fresh(); T0 = d0.transmission(); N0 = d0.n_open_channels()
G0c, GL0, GR0 = d0.greens()
AL0 = np.real(np.diag(G0c @ GL0 @ G0c.conj().T)); AR0 = np.real(np.diag(G0c @ GR0 @ G0c.conj().T))
LDOS0 = (AL0 + AR0) / (2 * np.pi); dn0 = 0.5 * (AL0 - AR0)
Gfull0 = G0c  # 清浄系グリーン関数（欠陥間の伝播子に使う）

# ---------------- 配置生成族 ----------------
def uniq(coords, n, rng):
    s = {(int(c), int(r)) for c, r in coords}
    while len(s) < n:
        s.add((int(rng.integers(C0, C1)), int(rng.integers(0, W))))
    return sorted(s)[:n]

def fam_random(n, rng):
    return uniq(zip(rng.integers(C0, C1, n), rng.integers(0, W, n)), n, rng)
def fam_cluster(n, rng):
    th = rng.uniform(0, np.pi); a, b = rng.uniform(0.6, 4.0), rng.uniform(0.6, 4.0)
    cx, cy = rng.uniform(C0 + 4, C1 - 4), rng.uniform(4, W - 4)
    pts = rng.normal(size=(3 * n, 2)) * [a, b]
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); pts = pts @ R.T + [cx, cy]
    pts = np.round(pts).astype(int); pts = pts[(pts[:, 0] >= C0) & (pts[:, 0] < C1) & (pts[:, 1] >= 0) & (pts[:, 1] < W)]
    return uniq(pts, n, rng)
def fam_line(n, rng):
    th = rng.uniform(0, np.pi); cx, cy = rng.uniform(C0 + 5, C1 - 5), rng.uniform(5, W - 5)
    t = np.linspace(-n / 2, n / 2, n) * rng.uniform(0.8, 1.3)
    pts = np.stack([cx + t * np.cos(th), cy + t * np.sin(th)], 1) + rng.normal(0, 0.3, (n, 2))
    pts = np.round(pts).astype(int); pts = pts[(pts[:, 0] >= C0) & (pts[:, 0] < C1) & (pts[:, 1] >= 0) & (pts[:, 1] < W)]
    return uniq(pts, n, rng)
def fam_two(n, rng):
    return uniq(fam_cluster(n // 2, rng) + fam_cluster(n - n // 2, rng), n, rng)
# 未使用族
def fam_arc(n, rng):
    cx, cy, R = rng.uniform(C0 + 5, C1 - 5), rng.uniform(3, W - 3), rng.uniform(3, 7)
    ph0 = rng.uniform(0, 2 * np.pi); ph = ph0 + np.linspace(0, np.pi * rng.uniform(0.6, 1.4), n)
    pts = np.round(np.stack([cx + R * np.cos(ph), cy + R * np.sin(ph)], 1)).astype(int)
    pts = pts[(pts[:, 0] >= C0) & (pts[:, 0] < C1) & (pts[:, 1] >= 0) & (pts[:, 1] < W)]
    return uniq(pts, n, rng)
def fam_grid(n, rng):
    dx, dy = rng.integers(2, 5), rng.integers(2, 5); nx = int(np.ceil(np.sqrt(n))); ny = int(np.ceil(n / nx))
    ox, oy = rng.integers(C0, C1 - nx * dx + 1), rng.integers(0, max(1, W - ny * dy + 1))
    pts = [(ox + i * dx, oy + j * dy) for j in range(ny) for i in range(nx)]
    pts = [(c, r) for c, r in pts if C0 <= c < C1 and 0 <= r < W]
    return uniq(pts, n, rng)
def fam_wallplus(n, rng):
    c = rng.integers(C0 + 3, C1 - 3); k = n // 2
    wall = [(c, r) for r in rng.choice(W, k, replace=False)]
    return uniq(wall + fam_cluster(n - k, rng), n, rng)

TRAIN_FAMS = {"random": fam_random, "cluster": fam_cluster, "line": fam_line, "two": fam_two}
TEST_FAMS = {"arc": fam_arc, "grid": fam_grid, "wall+cluster": fam_wallplus}
NDEFS = [6, 9, 12, 15, 18]

# ---------------- 特徴量 ----------------
def features(coords, d=None):
    P = np.array(coords, float); n = len(P)
    f = {"M0_n": n}
    # --- M1: 幾何 ---
    cov = np.cov(P.T) if n > 2 else np.eye(2) * 1e-3
    ev, evec = np.linalg.eigh(cov); ev = np.clip(ev, 1e-6, None)
    major = evec[:, -1]; cos2 = major[0]**2           # 主軸が輸送方向(x)に沿う度合い
    f.update({"M1_lam_max": ev[-1], "M1_lam_min": ev[0], "M1_aniso": ev[-1] / ev[0], "M1_cos2_x": cos2,
              "M1_yproj": len(set(P[:, 1].astype(int))) / W,        # 横方向射影被覆（壁⊥の核心）
              "M1_xproj": len(set(P[:, 0].astype(int))) / (C1 - C0),
              "M1_maxcol": max(np.bincount(P[:, 0].astype(int)).max(), 0) / W,
              "M1_maxrow": np.bincount(P[:, 1].astype(int)).max() / (C1 - C0),
              "M1_yspan": np.ptp(P[:, 1]) / W, "M1_xspan": np.ptp(P[:, 0]) / (C1 - C0)})
    # --- M2: 相関 ---
    D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    adj = (D <= 1.01); comp = 0; seen = np.zeros(n, bool); sizes = []
    for i in range(n):
        if not seen[i]:
            comp += 1; stack = [i]; seen[i] = True; sz = 0
            while stack:
                j = stack.pop(); sz += 1
                for k in np.where(adj[j] & ~seen)[0]: seen[k] = True; stack.append(k)
            sizes.append(sz)
    Dy = np.abs(P[:, None, 1] - P[None, :, 1]); Dx = np.abs(P[:, None, 0] - P[None, :, 0]); iu = np.triu_indices(n, 1)
    f.update({"M2_nn_mean": nn.mean(), "M2_nn_std": nn.std(), "M2_pairs_adj": adj.sum() / 2 / n,
              "M2_nclust": comp / n, "M2_maxclust": max(sizes) / n,
              "M2_g_r2": (D[iu] <= 2.01).mean(), "M2_g_r4": (D[iu] <= 4.01).mean(),
              "M2_samecol_pairs": (Dx[iu] == 0).mean(), "M2_samerow_pairs": (Dy[iu] == 0).mean(),
              "M2_ydist_mean": Dy[iu].mean() / W, "M2_xdist_mean": Dx[iu].mean() / (C1 - C0)})
    # --- M3a: 清浄系の局所電子応答（G 情報を含まない, 安価） ---
    idx = [d0.idx(int(c), int(r)) for c, r in coords]
    Gsub = Gfull0[np.ix_(idx, idx)]
    f.update({"M3a_ldos_sum": LDOS0[idx].sum(), "M3a_ldos_std": LDOS0[idx].std(),
              "M3a_inj_asym": np.abs(AL0[idx] - AR0[idx]).mean(),
              "M3a_prop_abs": (np.abs(Gsub) - np.diag(np.diag(np.abs(Gsub)))).sum() / n,   # 欠陥間伝播子の大きさ
              "M3a_prop_re": (np.real(Gsub) - np.diag(np.diag(np.real(Gsub)))).sum() / n})
    # --- M3b: 構造込みのバイアス応答（NEGF, 上限比較） ---
    if d is not None:
        G, GL, GR = d.greens()
        AL = np.real(np.diag(G @ GL @ G.conj().T)); AR = np.real(np.diag(G @ GR @ G.conj().T))
        ddn = 0.5 * (AL - AR) - dn0; X = np.repeat(np.arange(L), W)
        f.update({"M3b_sum_abs_ddn": np.abs(ddn).sum(), "M3b_dipole": (ddn * (X - 19.5)).sum(),
                  "M3b_ddn_std": ddn.std(), "M3b_ldos_def": ((AL + AR)[idx] / (2 * np.pi)).sum()})
    return f

def sample(fam, n, rng):
    coords = fam(n, rng)
    d = fresh(); d.set_onsite([d.idx(c, r) for c, r in coords], U_DEF)
    T = d.transmission(); f = features(coords, d); f["G"] = T / T0; f["fam"] = fam.__name__[4:]
    return f

t0 = time.time(); rows = []
rng = np.random.default_rng(2026)
for name, fam in TRAIN_FAMS.items():
    for n in NDEFS:
        for _ in range(18): rows.append({**sample(fam, n, rng), "split": "train"})
    print(name, f"{time.time()-t0:.0f}s")
rng_t = np.random.default_rng(777)
for name, fam in TRAIN_FAMS.items():
    for n in NDEFS:
        for _ in range(5): rows.append({**sample(fam, n, rng_t), "split": "test_same"})
for name, fam in TEST_FAMS.items():
    for n in NDEFS:
        for _ in range(8): rows.append({**sample(fam, n, rng_t), "split": "test_novel"})
print("sampling done", f"{time.time()-t0:.0f}s", len(rows))

import pandas as pd
df = pd.DataFrame(rows); df.to_csv("dataset_D.csv", index=False)
y = np.log(np.clip(1 - df["G"], 1e-4, None))        # 目的: ln(1 − G/G0)  (抵抗側の対数)
sets = {
    "M0": ["M0_n"],
    "M1": [c for c in df if c.startswith(("M0", "M1"))],
    "M2": [c for c in df if c.startswith(("M0", "M1", "M2"))],
    "M3a": [c for c in df if c.startswith(("M0", "M1", "M2", "M3a"))],
    "M3b": [c for c in df if c.startswith("M")],
}
tr = df["split"] == "train"
res = {}
for mname, cols in sets.items():
    res[mname] = {}
    for reg_name, reg in [("ridge", make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))),
                          ("gbr", GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0))]:
        reg.fit(df.loc[tr, cols], y[tr])
        for sp in ["test_same", "test_novel"]:
            m = df["split"] == sp
            pred = reg.predict(df.loc[m, cols])
            rmse = float(np.sqrt(np.mean((pred - y[m]) ** 2)))
            r2 = float(1 - np.sum((pred - y[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2))
            res[mname][f"{reg_name}_{sp}"] = {"rmse": rmse, "r2": r2}
    print(mname, {k: (round(v["rmse"], 3), round(v["r2"], 3)) for k, v in res[mname].items()})
json.dump(res, open("results_D.json", "w"), indent=1)

# 図: ΔL 階段
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
names = list(sets)
for sp, c in [("test_same", "C0"), ("test_novel", "C3")]:
    ax[0].plot(names, [res[m][f"gbr_{sp}"]["rmse"] for m in names], "o-", color=c, label=f"GBR  {sp}")
    ax[0].plot(names, [res[m][f"ridge_{sp}"]["rmse"] for m in names], "s--", color=c, alpha=0.6, label=f"Ridge {sp}")
    ax[1].plot(names, [res[m][f"gbr_{sp}"]["r2"] for m in names], "o-", color=c, label=f"GBR  {sp}")
    ax[1].plot(names, [res[m][f"ridge_{sp}"]["r2"] for m in names], "s--", color=c, alpha=0.6, label=f"Ridge {sp}")
ax[0].set_ylabel("RMSE of ln(1−G/G0)  (lower = better)"); ax[1].set_ylabel("R² on held-out"); ax[1].set_ylim(-0.2, 1)
for a in ax: a.legend(fontsize=8); a.set_xlabel("feature set")
ax[0].set_title("ΔL = L(M0) − L(Mk): value of each information layer")
plt.tight_layout(); plt.savefig(FIGDIR + "figJ_D_deltaL.png", dpi=130); plt.close()

# 特徴重要度 (M3a GBR)
reg = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0).fit(df.loc[tr, sets["M3a"]], y[tr])
imp = sorted(zip(reg.feature_importances_, sets["M3a"]), reverse=True)[:12]
fig, ax = plt.subplots(figsize=(7, 4)); ax.barh([n for _, n in imp][::-1], [v for v, _ in imp][::-1]); ax.set_title("GBR feature importance (M3a)")
plt.tight_layout(); plt.savefig(FIGDIR + "figK_D_importance.png", dpi=130); plt.close()
print("importance", [(n, round(v, 3)) for v, n in imp])

# parity (novel, M0 vs M2 vs M3a)
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
m = df["split"] == "test_novel"
for a, mname in zip(ax, ["M0", "M2", "M3a"]):
    reg = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0).fit(df.loc[tr, sets[mname]], y[tr])
    pred = reg.predict(df.loc[m, sets[mname]])
    for fam, c in zip(TEST_FAMS, ["C0", "C1", "C2"]):
        mm = (df.loc[m, "fam"] == fam).values
        a.scatter(np.exp(y[m][mm]), np.exp(pred[mm]), s=14, c=c, label=fam)
    a.plot([0, 1], [0, 1], "k--", lw=0.8); a.set_xlabel("true 1−G/G0"); a.set_ylabel("predicted"); a.set_title(f"{mname}: novel families, R²={res[mname]['gbr_test_novel']['r2']:.2f}"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIGDIR + "figL_D_parity.png", dpi=130); plt.close()
print("done", f"{time.time()-t0:.0f}s")
