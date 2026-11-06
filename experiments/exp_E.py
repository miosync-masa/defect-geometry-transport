"""
第5報 Exp E: 逆空間の幾何 — 構造因子 S(q) を記述子に加える
  D1: M2 + S(q) で grid 族の残差が減るか（族別 R² 保存）
  D2: E_F を変え、S(q) を「正しい k_n(E_F)」と「別の E_F の k_n」で評価して性能比較
      → 残差が電子波数に追従するなら波動干渉が支配
  D3: 配置を固定し周期だけ変える → 特定周期で透過率が跳ぶか（縦周期・横周期）
座標は全て coords_E.json に保存。
"""
import numpy as np, json, time, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from _paths import *   # noqa: cwd -> results/data, src on path
from negf_toy import Device
import exp_D_fams as F   # 配置族 (exp_D から分離)

W, L, U_DEF = 20, 40, 8.0
C0, C1 = 10, 30
EFS = [-1.5, -1.1, -0.7]
t0 = time.time()

# ---------- 電子波数 ----------
def channels(E):
    n = np.arange(1, W + 1); eps = -2 * np.cos(n * np.pi / (W + 1))
    op = np.abs(E - eps) < 2 - 1e-9
    kx = np.arccos(np.clip(-(E - eps[op]) / 2, -1, 1))
    return n[op], kx          # 横モード番号, 縦波数

def structure_factor(P, qx, qy):
    ph = np.exp(1j * (np.outer(qx, P[:, 0]) + np.outer(qy, P[:, 1])))
    return np.abs(ph.sum(1)) ** 2 / len(P) ** 2        # ランダムなら ≈ 1/N_def

def sq_features(coords, E, tag):
    """E に対応する散乱波数で評価した S(q) 記述子。tag でプレフィックス。"""
    P = np.array(coords, float); n, kx = channels(E); Nn = len(n)
    ky = n * np.pi / (W + 1)
    # (1) チャネル内後方散乱 q = (2k_n, 0)
    S_back = structure_factor(P, 2 * kx, np.zeros(Nn)).mean()
    # (2) チャネル間反射 q = (k_n + k_m, ±(k_y,n ∓ k_y,m))  横は定在波なので差・和の両方
    qx = (kx[:, None] + kx[None, :]).ravel()
    qy1 = (ky[:, None] - ky[None, :]).ravel(); qy2 = (ky[:, None] + ky[None, :]).ravel()
    S_inter = 0.5 * (structure_factor(P, qx, qy1).mean() + structure_factor(P, qx, qy2).mean())
    # (3) 前方チャネル間散乱 q = (k_n − k_m, Δk_y)
    qxf = (kx[:, None] - kx[None, :]).ravel()
    S_fwd = 0.5 * (structure_factor(P, qxf, qy1).mean() + structure_factor(P, qxf, qy2).mean())
    # (4) 対照: 電子と無関係な波数（固定グリッドの最大値・平均）
    qg = np.linspace(0.3, np.pi, 24); QX, QY = np.meshgrid(qg, qg); Sg = structure_factor(P, QX.ravel(), QY.ravel())
    return {f"{tag}_S_back": S_back, f"{tag}_S_inter": S_inter, f"{tag}_S_fwd": S_fwd,
            f"{tag}_S_back_max": structure_factor(P, 2 * kx, np.zeros(Nn)).max(),
            f"{tag}_Sgrid_max": Sg.max(), f"{tag}_Sgrid_mean": Sg.mean()}

# ---------- データ生成（3 つの E_F）----------
NTR, NTS = 12, 6
data = {}
for E in EFS:
    d0 = Device(W, L, E=E, eta=1e-6); T0 = d0.transmission(); N0 = d0.n_open_channels()
    rows = []
    rng = np.random.default_rng(2026); rng_t = np.random.default_rng(777)
    def run(coords, split, fam):
        d = Device(W, L, E=E, eta=1e-6); d.set_onsite([d.idx(c, r) for c, r in coords], U_DEF)
        T = d.transmission()
        f = F.geom_features(coords); f.update({"G": T / T0, "split": split, "fam": fam, "coords": coords})
        for Ei in EFS: f.update(sq_features(coords, Ei, f"SqE{Ei}"))   # 全 E_F の k_n で評価しておく
        return f
    for name, fam in F.TRAIN_FAMS.items():
        for n in F.NDEFS:
            for _ in range(NTR): rows.append(run(fam(n, rng), "train", name))
    for name, fam in F.TRAIN_FAMS.items():
        for n in F.NDEFS:
            for _ in range(3): rows.append(run(fam(n, rng_t), "test_same", name))
    for name, fam in F.TEST_FAMS.items():
        for n in F.NDEFS:
            for _ in range(NTS): rows.append(run(fam(n, rng_t), "test_novel", name))
    data[E] = {"T0": T0, "N0": N0, "rows": rows}
    print(f"E={E}: N0={N0}, {len(rows)} structures ({time.time()-t0:.0f}s)", flush=True)
json.dump({str(E): [{k: (v if k != "coords" else [list(map(int, c)) for c in v]) for k, v in r.items()} for r in d["rows"]] for E, d in data.items()},
          open("coords_E.json", "w"))

import pandas as pd
def fit_eval(df, cols, label):
    tr = df["split"] == "train"; y = np.log(np.clip(1 - df["G"], 1e-4, None))
    reg = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0).fit(df.loc[tr, cols], y[tr])
    out = {}
    for sp in ["test_same", "test_novel"]:
        m = df["split"] == sp; pred = reg.predict(df.loc[m, cols]); yy = y[m].values
        out[sp] = {"r2": float(1 - ((pred - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()), "rmse": float(np.sqrt(((pred - yy) ** 2).mean()))}
        for fam in df.loc[m, "fam"].unique():
            mm = (df.loc[m, "fam"] == fam).values
            out[f"{sp}/{fam}"] = {"rmse": float(np.sqrt(((pred[mm] - yy[mm]) ** 2).mean())), "bias": float((pred[mm] - yy[mm]).mean())}
    return out, reg

results = {}
for E in EFS:
    df = pd.DataFrame(data[E]["rows"])
    geom = [c for c in df if c.startswith(("M0", "M1", "M2"))]
    sets = {"M2": geom}
    for Ei in EFS: sets[f"M2+S(q)@E{Ei}"] = geom + [c for c in df if c.startswith(f"SqE{Ei}_")]
    sets["M2+Sgrid only"] = geom + [c for c in df if c.startswith(f"SqE{E}_Sgrid")]
    results[E] = {}
    for name, cols in sets.items():
        r, reg = fit_eval(df, cols, name); results[E][name] = r
        print(f"E={E:5.1f} {name:18s} same R²={r['test_same']['r2']:.3f} novel R²={r['test_novel']['r2']:.3f}  grid rmse={r['test_novel/grid']['rmse']:.3f} bias={r['test_novel/grid']['bias']:+.3f}", flush=True)
    # 重要度（正しい E_F の S(q) 入り）
    r, reg = fit_eval(df, sets[f"M2+S(q)@E{E}"], "x")
    imp = sorted(zip(reg.feature_importances_, sets[f"M2+S(q)@E{E}"]), reverse=True)[:8]
    results[E]["importance"] = [(n, float(v)) for v, n in imp]
    print("   importance:", [(n, round(v, 3)) for v, n in imp], flush=True)
json.dump({str(k): v for k, v in results.items()}, open("results_E.json", "w"), indent=1)

# ---------- 図 D1/D2 ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
labels = ["M2", "M2+Sgrid only"] + [f"M2+S(q)@E{Ei}" for Ei in EFS]
xpos = np.arange(len(labels)); wdt = 0.26
for i, E in enumerate(EFS):
    ax[0].bar(xpos + (i - 1) * wdt, [results[E][l]["test_novel"]["r2"] for l in labels], wdt, label=f"data E_F={E}")
    ax[1].bar(xpos + (i - 1) * wdt, [results[E][l]["test_novel/grid"]["rmse"] for l in labels], wdt, label=f"data E_F={E}")
for a in ax: a.set_xticks(xpos); a.set_xticklabels(labels, rotation=20, fontsize=8); a.legend(fontsize=8)
ax[0].set_ylabel("R² (novel families)"); ax[0].set_title("D1/D2: adding S(q) evaluated at electron wavevectors of E_F")
ax[1].set_ylabel("RMSE on grid family"); ax[1].set_title("grid-family residual: does it follow the electron k?")
plt.tight_layout(); plt.savefig(FIGDIR + "figM_E_Sq_deltaL.png", dpi=130); plt.close()

# ---------- D3: 配置固定・周期だけ変える ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
D3 = {}
for E, c in zip(EFS, ["C0", "C1", "C3"]):
    d0 = Device(W, L, E=E, eta=1e-6); T0 = d0.transmission(); n_op, kx = channels(E)
    # 縦周期: 行 10 に N=8 個、周期 a (格子単位)
    per = np.arange(1, 7); Gx = []
    for a in per:
        coords = [(int(20 - 3.5 * a + j * a), 10) for j in range(8)]
        coords = [(cc, r) for cc, r in coords if 0 <= cc < L]
        d = Device(W, L, E=E, eta=1e-6); d.set_onsite([d.idx(cc, r) for cc, r in coords], U_DEF); Gx.append(d.transmission() / T0)
    # 横周期: 列 20 に周期 b で並べる（個数は W/b）
    perb = np.arange(1, 8); Gy = []; Ny = []
    for b in perb:
        rows_ = list(range(0, W, b)); coords = [(20, r) for r in rows_]
        d = Device(W, L, E=E, eta=1e-6); d.set_onsite([d.idx(cc, r) for cc, r in coords], U_DEF); Gy.append(d.transmission() / T0); Ny.append(len(rows_))
    D3[E] = {"period_x": per.tolist(), "G_x": Gx, "period_y": perb.tolist(), "G_y": Gy, "N_y": Ny, "kx": kx.tolist()}
    ax[0].plot(per, Gx, "o-", color=c, label=f"E_F={E}  (2k_x of channels: {np.round(2*kx[:3],2)}…)")
    # 期待される Bragg 周期 a* = π/k_x,n  → 縦に印
    for k in kx:
        a_star = np.pi / k
        if 1 <= a_star <= 6: ax[0].axvline(a_star, color=c, alpha=0.15, lw=1)
    ax[1].plot(perb, np.array(Gy), "s-", color=c, label=f"E_F={E}")
ax[0].set_xlabel("longitudinal period a (8 defects in one row)"); ax[0].set_ylabel("G/G0"); ax[0].set_title("D3a: longitudinal period scan (shaded lines: π/k_x,n)"); ax[0].legend(fontsize=7)
ax[1].set_xlabel("transverse period b (one column, N=⌈W/b⌉ defects)"); ax[1].set_ylabel("G/G0"); ax[1].set_title("D3b: transverse period scan"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIGDIR + "figN_E_D3_period.png", dpi=130); plt.close()
json.dump({str(k): v for k, v in D3.items()}, open("results_E_D3.json", "w"), indent=1)
print("done", f"{time.time()-t0:.0f}s")
