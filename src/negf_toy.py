"""
Cu粒界研究・最初の一歩トイモデル
  「同じ格子 + 同じ欠陥数 + 異なる欠陥配置 → 異なる電気的応答？」
2D正方格子タイトバインディング + Landauer-NEGF (numpy のみ)

  H = Σ ε_i |i><i| - Σ t_ij |i><j|
  T(E) = Tr[Γ_L G^r Γ_R G^a]
  局所電流（観測窓）: J_ij ∝ Im[ H_ij (G^r Γ_L G^a)_ji ]
"""
import numpy as np
import numpy.linalg as la

T_HOP = 1.0  # 基準ホッピング


class Device:
    def __init__(self, W, L, E=-1.0, eta=1e-7):
        self.W, self.L, self.E, self.eta = W, L, E, eta
        self.N = W * L
        self.eps = np.zeros(self.N)                 # オンサイト
        # ホッピング辞書 (i,j)->t  (i<j)
        self.hop = {}
        for c in range(L):
            for r in range(W):
                i = self.idx(c, r)
                if r + 1 < W:
                    self.hop[(i, self.idx(c, r + 1))] = T_HOP
                if c + 1 < L:
                    self.hop[(i, self.idx(c + 1, r))] = T_HOP
        self._sigma_cache = None

    def idx(self, c, r):
        return c * self.W + r

    # ---------- 構造操作 ----------
    def set_onsite(self, sites, U):
        for s in sites:
            self.eps[s] = U

    def scale_bonds_across_column(self, c, factor):
        """列 c と c+1 の間の全ボンドを factor 倍（粒界面のモデル）"""
        for r in range(self.W):
            i, j = self.idx(c, r), self.idx(c + 1, r)
            self.hop[(i, j)] = T_HOP * factor

    def set_bond(self, i, j, t):
        key = (i, j) if i < j else (j, i)
        self.hop[key] = t

    # ---------- ハミルトニアン ----------
    def H(self):
        H = np.diag(self.eps.astype(complex))
        for (i, j), t in self.hop.items():
            H[i, j] = -t
            H[j, i] = -t
        return H

    # ---------- リード（半無限正方格子）の表面グリーン関数：Sancho-Rubio ----------
    def surface_gf(self):
        W, E, eta = self.W, self.E, self.eta
        H00 = np.zeros((W, W), complex)
        for r in range(W - 1):
            H00[r, r + 1] = H00[r + 1, r] = -T_HOP
        H01 = -T_HOP * np.eye(W, dtype=complex)
        z = (E + 1j * eta) * np.eye(W)
        eps_s, eps_b, a, b = H00.copy(), H00.copy(), H01.copy(), H01.conj().T.copy()
        for _ in range(200):
            g = la.inv(z - eps_b)
            agb = a @ g @ b
            bga = b @ g @ a
            eps_s = eps_s + agb
            eps_b = eps_b + agb + bga
            a = a @ g @ a
            b = b @ g @ b
            if la.norm(a) < 1e-12 and la.norm(b) < 1e-12:
                break
        gs = la.inv(z - eps_s)
        return gs, H01

    def self_energies(self):
        if self._sigma_cache is None:
            gs, H01 = self.surface_gf()
            # 左リード: Σ_L = H10 g H01, 右リード同形
            sig = H01.conj().T @ gs @ H01
            self._sigma_cache = sig
        W, N = self.W, self.N
        SL = np.zeros((N, N), complex)
        SR = np.zeros((N, N), complex)
        SL[:W, :W] = self._sigma_cache
        SR[N - W:, N - W:] = self._sigma_cache
        return SL, SR

    # ---------- 輸送 ----------
    def greens(self):
        SL, SR = self.self_energies()
        H = self.H()
        G = la.inv((self.E + 1j * self.eta) * np.eye(self.N) - H - SL - SR)
        GL = 1j * (SL - SL.conj().T)
        GR = 1j * (SR - SR.conj().T)
        return G, GL, GR

    def transmission(self, return_eigs=False):
        G, GL, GR = self.greens()
        M = GL @ G @ GR @ G.conj().T
        T = np.real(np.trace(M))
        if not return_eigs:
            return T
        # 透過固有値: t†t の固有値 = eigs of Γ_L^{1/2} G Γ_R G† Γ_L^{1/2}
        W = self.W
        gl = GL[:W, :W]
        gr = GR[-W:, -W:]
        wl, vl = la.eigh(gl)
        sql = (vl * np.sqrt(np.clip(wl, 0, None))) @ vl.conj().T
        Gdr = G[:W, -W:]
        A = sql @ Gdr @ gr @ Gdr.conj().T @ sql
        eig = np.sort(np.real(la.eigvalsh(A)))[::-1]
        return T, eig

    def n_open_channels(self):
        W, E = self.W, self.E
        n = np.arange(1, W + 1)
        eps_n = -2 * T_HOP * np.cos(n * np.pi / (W + 1))
        return int(np.sum(np.abs(E - eps_n) < 2 * T_HOP))

    def bond_currents(self):
        """左リードから注入した電子の局所ボンド電流（観測窓）"""
        G, GL, GR = self.greens()
        Gless = G @ GL @ G.conj().T  # ∝ G^<（左注入）
        H = self.H()
        J = {}
        for (i, j) in self.hop:
            J[(i, j)] = -np.imag(H[i, j] * Gless[j, i]) * 2.0
        return J

    def ldos(self):
        G, _, _ = self.greens()
        return -np.imag(np.diag(G)) / np.pi


def current_field(dev, J):
    """ボンド電流をサイト上のベクトル場に平均化（x: 輸送方向=列, y: 行）"""
    W, L = dev.W, dev.L
    vx = np.zeros((L, W)); vy = np.zeros((L, W)); mag = np.zeros((L, W))
    for (i, j), val in J.items():
        ci, ri = divmod(i, W); cj, rj = divmod(j, W)
        if ci != cj:   # x ボンド
            vx[ci, ri] += val / 2; vx[cj, rj] += val / 2
        else:          # y ボンド
            vy[ci, ri] += val / 2; vy[cj, rj] += val / 2
    mag = np.hypot(vx, vy)
    return vx, vy, mag
