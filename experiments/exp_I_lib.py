"""Feature library for Exp I/J: geometry + S(q) + A_mn + local-periodicity/Bloch features, exact T-matrix G."""
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
from _paths import *   # noqa
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

