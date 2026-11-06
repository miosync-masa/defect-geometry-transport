import numpy as np, json, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _paths import *   # noqa: cwd -> results/data, src on path
from negf_toy import Device, current_field

W, L, E = 20, 40, -1.1
U_DEF = 8.0          # 強散乱体（空孔に近い）
N_DEF = 12
rng = np.random.default_rng(7)
out = {}

def fresh():
    return Device(W, L, E=E, eta=1e-6)

d0 = fresh()
T0 = d0.transmission()
N_OPEN = d0.n_open_channels()
print("clean T0 =", T0, "open channels =", N_OPEN)
out["T0"] = T0; out["N_open"] = N_OPEN

# ============================================================
# Exp A (H1/H4): 同じ欠陥数 N_DEF=12、配置だけ変える
# ============================================================
def sites_from(coords, d):
    return [d.idx(c, r) for c, r in coords]

configs = {
    "wall ⊥ (粒界面に相当)": [(20, r) for r in range(4, 16)],
    "wall ∥ (輸送方向)":     [(c, 10) for c in range(14, 26)],
    "diagonal":             [(14 + k, 4 + k) for k in range(12)],
    "cluster 3x4":          [(19 + i, 9 + j) for i in range(3) for j in range(4)],
    "two half-walls (offset)": [(18, r) for r in range(2, 8)] + [(22, r) for r in range(12, 18)],
    "sparse ⊥ (every other row)": [(20, r) for r in range(0, 24, 2)][:12],
}
resA = {}
for name, coords in configs.items():
    d = fresh(); d.set_onsite(sites_from(coords, d), U_DEF)
    T, eig = d.transmission(return_eigs=True)
    resA[name] = {"T": T, "eig": eig[:N_OPEN].tolist()}
    print(f"{name:28s} T = {T:7.3f}   T/T0 = {T/T0:.3f}")

# ランダム配置の分布（中央領域 列12..27）
NS = 150
Trand = []
t0 = time.time()
for s in range(NS):
    d = fresh()
    cols = rng.integers(12, 28, N_DEF); rows = rng.integers(0, W, N_DEF)
    coords = list({(int(c), int(r)) for c, r in zip(cols, rows)})
    while len(coords) < N_DEF:
        coords.append((int(rng.integers(12, 28)), int(rng.integers(0, W))))
        coords = list(set(coords))
    d.set_onsite(sites_from(coords, d), U_DEF)
    Trand.append(d.transmission())
Trand = np.array(Trand)
print(f"random: mean {Trand.mean():.3f}  std {Trand.std():.3f}  min {Trand.min():.3f} max {Trand.max():.3f}  ({time.time()-t0:.1f}s)")
out["expA"] = {k: v["T"] for k, v in resA.items()}
out["expA_random"] = {"mean": float(Trand.mean()), "std": float(Trand.std()),
                      "min": float(Trand.min()), "max": float(Trand.max())}

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].hist(Trand / T0, bins=25, color="#9ecae1", edgecolor="k", label=f"random placement (n={NS})")
colors = plt.cm.tab10(np.arange(len(resA)))
for (name, r), c in zip(resA.items(), colors):
    ax[0].axvline(r["T"] / T0, color=c, lw=2.2, label=name)
ax[0].set_xlabel("T / T0  (same N_def = 12, U = 8t)"); ax[0].set_ylabel("count")
ax[0].set_title("H1/H4: same defect count, different geometry")
ax[0].legend(fontsize=7.5, loc="upper left")
# 透過固有値スペクトル
for (name, r), c in zip(resA.items(), colors):
    ax[1].plot(range(1, N_OPEN + 1), r["eig"], "o-", ms=4, color=c, label=name)
ax[1].axhline(1, color="gray", ls=":")
ax[1].set_xlabel("channel index n"); ax[1].set_ylabel("transmission eigenvalue T_n")
ax[1].set_title("channel-resolved response (D vs F の分離の芽)")
plt.tight_layout(); plt.savefig(FIGDIR + "figA_geometry.png", dpi=140); plt.close()

# ============================================================
# Exp B (H3): 2欠陥の非加法性 vs 距離・方向
# ============================================================
def loss_of(coords):
    d = fresh(); d.set_onsite(sites_from(coords, d), U_DEF)
    return T0 - d.transmission()

A = (18, 10)
dA = loss_of([A])
along, trans = [], []
for dx in range(1, 13):
    B = (18 + dx, 10)
    dB = loss_of([B]); dAB = loss_of([A, B])
    along.append((dx, dAB - (dA + dB)))
for dy in range(1, 10):
    B = (18, 10 - dy) if dy <= 10 else None
    dB = loss_of([B]); dAB = loss_of([A, B])
    trans.append((dy, dAB - (dA + dB)))
along = np.array(along); trans = np.array(trans)
out["expB"] = {"single_loss": dA, "along": along.tolist(), "trans": trans.tolist()}
print("single-defect loss:", dA)
print("non-additivity along transport:", np.round(along[:, 1], 3))
print("non-additivity transverse     :", np.round(trans[:, 1], 3))

# k_F 推定（輸送方向）: 支配チャネルの縦波数  E - eps_n = -2 cos k
n = np.arange(1, W + 1); eps_n = -2 * np.cos(n * np.pi / (W + 1))
kx = np.arccos(np.clip(-(E - eps_n) / 2, -1, 1))
kx = kx[np.abs(E - eps_n) < 2]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(along[:, 0], along[:, 1], "o-", label="separation along transport (Δx)")
ax.plot(trans[:, 0], trans[:, 1], "s--", label="separation transverse (Δy)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("defect separation (lattice units)")
ax.set_ylabel("non-additivity  ΔT_AB − (ΔT_A + ΔT_B)")
ax.set_title(f"H3: pair interaction (single-defect loss = {dA:.3f})")
ax.legend(); plt.tight_layout(); plt.savefig(FIGDIR + "figB_pair_nonadditivity.png", dpi=140); plt.close()

# ============================================================
# Exp C: 粒界「余剰体積」スキャン  t' = t·exp(−β v)   (SSOC の v_f と対比)
# ============================================================
BETA = 3.0
vs = np.linspace(0, 1.0, 26)
ordered, disordered, shifted, decomp = [], [], [], []
for v in vs:
    # (i) 秩序粒界: 全ボンド一様に弱化
    d = fresh(); d.scale_bonds_across_column(20, np.exp(-BETA * v))
    T, eig = d.transmission(return_eigs=True); ordered.append(T)
    eig = np.clip(eig[:N_OPEN], 0, 1)
    F = np.mean(eig > 0.1)                      # 「つながってる」チャネル割合（接続性, T_n>0.1）
    D = eig[eig > 0.1].mean() if F > 0 else 0.  # つながってるチャネルの平均透過（局所）
    decomp.append((F, D))
    # (ii) 乱れ粒界: ボンドごとに弱化率がばらつく（平均は同じ）
    Ts = []
    for s in range(8):
        d = fresh()
        for r in range(W):
            xi = rng.normal(0, 0.6)
            d.set_bond(d.idx(20, r), d.idx(21, r), np.exp(-BETA * v * max(0, 1 + xi)))
        Ts.append(d.transmission())
    disordered.append(np.mean(Ts))
    # (iii) 弱化 + 面内オンサイトシフト（粒界ポテンシャル）
    d = fresh(); d.scale_bonds_across_column(20, np.exp(-BETA * v))
    d.set_onsite([d.idx(20, r) for r in range(W)] + [d.idx(21, r) for r in range(W)], 1.5 * v)
    shifted.append(d.transmission())
ordered = np.array(ordered); disordered = np.array(disordered); shifted = np.array(shifted)
decomp = np.array(decomp)
out["expC"] = {"v": vs.tolist(), "ordered": ordered.tolist(), "disordered": disordered.tolist(),
               "shifted": shifted.tolist(), "F": decomp[:, 0].tolist(), "D": decomp[:, 1].tolist()}

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(vs, ordered / T0, "o-", label="ordered GB (uniform weakening)")
ax[0].plot(vs, disordered / T0, "s-", label="disordered GB (same mean exponent, bond-to-bond scatter)")
ax[0].plot(vs, shifted / T0, "^-", label="weakening + GB potential shift")
ax[0].set_xlabel("GB excess-volume proxy v  (t' = t·e^{-βv}, β=3)")
ax[0].set_ylabel("G / G0"); ax[0].set_title("Exp C: GB 'free volume' as control coordinate")
ax[0].legend(fontsize=8)
ax[1].plot(vs, decomp[:, 0], "o-", label="F: fraction of connected channels (T_n>0.1)")
ax[1].plot(vs, decomp[:, 1], "s-", label="D: mean T_n of connected channels")
ax[1].plot(vs, ordered / T0, "k--", lw=1, label="G/G0 (ordered)")
ax[1].plot(vs, decomp[:, 0] * decomp[:, 1], "r:", lw=1.5, label="F × D")
ax[1].set_xlabel("v"); ax[1].set_title("toy N×D×F decomposition (N = open channels)")
ax[1].legend(fontsize=8); plt.tight_layout(); plt.savefig(FIGDIR + "figC_gb_freevolume.png", dpi=140); plt.close()

# ============================================================
# Exp D: 観測窓 — 局所電流分布（同じ欠陥数、壁 vs ランダム）
# ============================================================
fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
maps = {
    "wall ⊥": configs["wall ⊥ (粒界面に相当)"],
    "wall ∥": configs["wall ∥ (輸送方向)"],
    "random (one sample)": [(int(c), int(r)) for c, r in zip(rng.integers(12, 28, N_DEF), rng.integers(0, W, N_DEF))],
}
for a, (name, coords) in zip(axs, maps.items()):
    d = fresh(); d.set_onsite(sites_from(coords, d), U_DEF)
    J = d.bond_currents(); vx, vy, mag = current_field(d, J)
    T = d.transmission()
    im = a.imshow(mag.T, origin="lower", cmap="magma", aspect="auto")
    a.quiver(np.arange(L)[::2], np.arange(W)[::2], vx[::2, ::2].T, vy[::2, ::2].T, color="w", scale=mag.max() * 40, width=0.003)
    cs = np.array(coords); a.scatter(cs[:, 0], cs[:, 1], s=14, c="cyan", marker="x")
    a.set_title(f"{name}:  T/T0 = {T/T0:.3f}"); a.set_xlabel("x (transport)"); a.set_ylabel("y")
plt.tight_layout(); plt.savefig(FIGDIR + "figD_current_maps.png", dpi=140); plt.close()

json.dump(out, open("results.json", "w"), indent=1, ensure_ascii=False)
print("done")
