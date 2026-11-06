"""
F3 (U scan): multiple-scattering hierarchy vs defect strength, using the exact on-site T-matrix.
  Born-1 (T=U), Born-2 (U + U G0 U), ISA (independent renormalised scatterers), pair (ISA + 2-body), full.
  RMSE of ln(1-G) against the exact result, per family. Output: results_F3_Uscan.json
"""
import numpy as np, json
from _paths import *          # noqa: sets cwd to results/data and adds src to sys.path
from negf_toy import Device

W, L, E = 20, 40, -1.1
d0 = Device(W, L, E=E, eta=1e-6); T0 = d0.transmission(); G0, GL, GR = d0.greens()
G0_LR = G0[:W, -W:]; gl = GL[:W, :W]; gr = GR[-W:, -W:]
def Tb(G_LR): return np.real(np.trace(gl @ G_LR @ gr @ G_LR.conj().T)) / T0
raw = json.load(open("coords_E.json"))[str(E)]
out = {}
for U_ in [0.5, 1.0, 2.0, 4.0, 8.0]:
    res = {k: [] for k in ["born1", "born2", "isa", "pair", "full"]}; fam = []
    for r in raw:
        idx = [d0.idx(c, rr) for c, rr in r["coords"]]; nd = len(idx)
        Gdd = G0[np.ix_(idx, idx)]; U = U_ * np.eye(nd); GLd = G0[:W, idx]; GdR = G0[idx, -W:]
        f = lambda T: Tb(G0_LR + GLd @ T @ GdR)
        tj = np.diag(U_ / (1 - np.diag(Gdd) * U_)); Goff = Gdd - np.diag(np.diag(Gdd))
        res["born1"].append(f(U)); res["born2"].append(f(U + U @ Gdd @ U)); res["isa"].append(f(tj)); res["pair"].append(f(tj + tj @ Goff @ tj))
        res["full"].append(f(np.linalg.solve(np.eye(nd) - Gdd @ U, U))); fam.append(r["fam"])
    fam = np.array(fam); full = np.array(res["full"]); y = np.log(np.clip(1 - full, 1e-4, None))
    line = {}
    for k in ["born1", "born2", "isa", "pair"]:
        p = np.array(res[k]); yp = np.log(np.clip(1 - p, 1e-4, None))
        line[k] = {"ALL": float(np.sqrt(np.mean((yp - y) ** 2))), "grid": float(np.sqrt(np.mean((yp - y)[fam == "grid"] ** 2))),
                   "random": float(np.sqrt(np.mean((yp - y)[fam == "random"] ** 2))), "frac>1": float((p > 1).mean())}
    out[U_] = line; print(f"U={U_}: mean 1-G_full={1-full.mean():.3f}", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in line.items()})
json.dump({str(k): v for k, v in out.items()}, open("results_F3_Uscan.json", "w"), indent=1)
