"""
Exp C-series: 「均一に悪くなるより、チャネルを選んで残す方が伝導が生き残る」を検証
  C1: 平均ボンド強度 ⟨t'⟩ を固定し、ばらつき s だけ振る  → G(s), N_eff(s), D_eff(s)
      (a) 算術平均固定 (Jensen 補正あり, 本命)   (b) 幾何平均固定 (前回条件)
      (c) ベルヌーイ型: 割合 p のボンドを切り、残りを強める（平均固定）＝パーコレーション的選択
  C2: 乱れが余剰体積と結合 s = κ·v  → G(v) にピークが出るか（SSOC ピーク型の骨格）
  C3: 透過固有値分布 (一様 vs 選択) の形
"""
import numpy as np, json, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _paths import *   # noqa: cwd -> results/data, src on path
from negf_toy import Device

W, L, E, BETA, GBCOL = 20, 40, -1.1, 3.0, 20
rng = np.random.default_rng(11)
def fresh(): return Device(W, L, E=E, eta=1e-6)
d0 = fresh(); T0, eig0 = d0.transmission(return_eigs=True); N0 = d0.n_open_channels()
def nd(eig):
    e = np.clip(eig, 0, None); S1, S2 = e.sum(), (e**2).sum(); return S1**2 / S2, S2 / S1

def gb_bonds(mean_t, mode, s, cap=1.0):
    """粒界 W 本のボンド強度を生成。mean_t = 目標平均 (算術)。"""
    if mode == "arith":      # ログノーマル, 算術平均 = mean_t
        t = mean_t * np.exp(s * rng.normal(size=W) - s**2 / 2)
    elif mode == "geom":     # ログノーマル, 幾何平均 = mean_t (Jensen で算術平均は上がる)
        t = mean_t * np.exp(s * rng.normal(size=W))
    elif mode == "bern":     # 割合 s=p を切断、残り = mean_t/(1-p)
        p = s; alive = rng.random(W) >= p
        t = np.where(alive, mean_t / max(1 - p, 1e-9), 0.0)
    return np.minimum(t, cap)   # ホッピングは基準 t=1 を超えない（余剰体積は縮まない）

def run(ts):
    d = fresh()
    for r in range(W): d.set_bond(d.idx(GBCOL, r), d.idx(GBCOL + 1, r), ts[r])
    T, eig = d.transmission(return_eigs=True); Ne, De = nd(eig)
    return T / T0, Ne / N0, De, eig[:N0]

out = {"T0": T0, "N0": N0}
NS = 10
t_start = time.time()

# ---------------- C1 ----------------
C1 = {}
svals = {"arith": np.linspace(0, 1.6, 9), "geom": np.linspace(0, 1.6, 9), "bern": np.linspace(0, 0.8, 9)}
for v in [0.3, 0.5, 0.7]:
    mean_t = np.exp(-BETA * v)
    for mode in ["arith", "geom", "bern"]:
        rows = []
        for s in svals[mode]:
            acc = np.array([run(gb_bonds(mean_t, mode, s))[:3] for _ in range(NS if s > 0 else 1)])
            rows.append([s, *acc.mean(0), acc[:, 0].std()])
        C1[f"{mode}_v{v}"] = np.array(rows).tolist()
        print(f"C1 {mode} v={v}: G(s)=", np.round(np.array(rows)[:, 1], 3), f"({time.time()-t_start:.0f}s)")
out["C1"] = C1

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
titles = {"arith": "(a) arithmetic mean ⟨t'⟩ fixed  (pure heterogeneity)",
          "geom": "(b) geometric mean fixed  (previous run; Jensen included)",
          "bern": "(c) Bernoulli: cut fraction p, strengthen rest (mean fixed)"}
for j, mode in enumerate(["arith", "geom", "bern"]):
    for v, c in zip([0.3, 0.5, 0.7], ["C0", "C1", "C3"]):
        R = np.array(C1[f"{mode}_v{v}"])
        axs[0, j].errorbar(R[:, 0], R[:, 1], yerr=R[:, 4], fmt="o-", color=c, label=f"v={v}")
        axs[0, j].axhline(R[0, 1], color=c, ls=":", lw=0.8)
        axs[1, j].plot(R[:, 0], R[:, 2], "o-", color=c, label=f"N_eff/N0 v={v}")
        axs[1, j].plot(R[:, 0], R[:, 3], "s--", color=c, alpha=0.6, label=f"D_eff v={v}")
    axs[0, j].set_title(titles[mode], fontsize=10); axs[0, j].set_ylabel("G/G0"); axs[0, j].legend(fontsize=8)
    axs[1, j].set_xlabel("heterogeneity s" if mode != "bern" else "cut fraction p"); axs[1, j].legend(fontsize=7)
axs[1, 0].set_ylabel("N_eff/N0 (circles), D_eff (squares)")
plt.suptitle("C1: does heterogeneity at fixed mean bond strength preserve conductance?")
plt.tight_layout(); plt.savefig(FIGDIR + "figG_C1_heterogeneity.png", dpi=130); plt.close()

# ---------------- C2: s = κ v 結合 → G(v) ----------------
vs = np.linspace(0, 1.0, 21); C2 = {}
for kappa in [0.0, 1.0, 2.0, 3.0]:
    rows = []
    for v in vs:
        mean_t = np.exp(-BETA * v); s = kappa * v
        acc = np.array([run(gb_bonds(mean_t, "arith", s))[:3] for _ in range(NS if s > 0 else 1)])
        rows.append([v, *acc.mean(0)])
    C2[kappa] = np.array(rows)
    print(f"C2 κ={kappa}: argmax v = {vs[np.argmax(C2[kappa][:,1])]:.2f}, Gmax={C2[kappa][:,1].max():.3f} ({time.time()-t_start:.0f}s)")
# ベルヌーイ結合 p = κ_p v
for kp in [0.5, 0.8]:
    rows = []
    for v in vs:
        mean_t = np.exp(-BETA * v); p = min(kp * v, 0.95)
        acc = np.array([run(gb_bonds(mean_t, "bern", p))[:3] for _ in range(NS if p > 0 else 1)])
        rows.append([v, *acc.mean(0)])
    C2[f"bern{kp}"] = np.array(rows)
    print(f"C2 bern κp={kp}: argmax v = {vs[np.argmax(C2[f'bern{kp}'][:,1])]:.2f} ({time.time()-t_start:.0f}s)")
out["C2"] = {str(k): v.tolist() for k, v in C2.items()}

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for k, R in C2.items():
    lab = f"lognormal s={k}·v" if not isinstance(k, str) else f"Bernoulli p={k[4:]}·v"
    ax[0].plot(R[:, 0], R[:, 1], "o-", ms=3, label=lab)
    ax[1].plot(R[:, 0], R[:, 2], "o-", ms=3, label=lab)
    ax[2].plot(R[:, 0], R[:, 3], "o-", ms=3, label=lab)
ax[0].set_ylabel("G/G0"); ax[1].set_ylabel("N_eff/N0"); ax[2].set_ylabel("D_eff")
for a in ax: a.set_xlabel("GB excess-volume proxy v (mean t' = e^{-3v}, heterogeneity grows with v)"); a.legend(fontsize=7)
ax[0].set_title("C2: G(v) when disorder is coupled to excess volume")
plt.tight_layout(); plt.savefig(FIGDIR + "figH_C2_coupled.png", dpi=130); plt.close()

# ---------------- C3: 透過固有値分布 ----------------
v = 0.5; mean_t = np.exp(-BETA * v)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for a, (mode, s, lab) in zip(ax, [("arith", 0.0, "uniform (s=0)"), ("arith", 1.2, "lognormal s=1.2"), ("bern", 0.5, "Bernoulli p=0.5")]):
    eigs = np.concatenate([run(gb_bonds(mean_t, mode, s))[3] for _ in range(20)])
    a.hist(eigs, bins=20, range=(0, 1), color="#9ecae1", edgecolor="k")
    G, Ne, De, _ = run(gb_bonds(mean_t, mode, s))
    a.set_title(f"{lab}: T_n distribution\n⟨G/G0⟩≈{np.mean([run(gb_bonds(mean_t, mode, s))[0] for _ in range(10)]):.3f}", fontsize=10)
    a.set_xlabel("T_n")
plt.suptitle(f"C3: channel transmission spectra at v={v} (same mean bond strength)")
plt.tight_layout(); plt.savefig(FIGDIR + "figI_C3_Tn_dist.png", dpi=130); plt.close()
json.dump(out, open("results_C.json", "w"), indent=1)
print("done", time.time() - t_start)
