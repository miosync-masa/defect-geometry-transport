import numpy as np
W, L = 20, 40
C0, C1 = 10, 30
NDEFS = [6, 9, 12, 15, 18]
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
def geom_features(coords):
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
    f['M1_aniso'] = float(np.log(f['M1_aniso']))
    return f
