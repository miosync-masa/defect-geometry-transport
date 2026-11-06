"""
J: fixed-defect-count control. Same N_def, few (3-4) vs many (6-8) periods; random row subsets (no y-periodicity).
  (a) corr(G, Bloch fraction) per stratum, (b) leave-one-period-a-out delta R2 from adding H4c.
  Output: results_J_fixedN.json, dataset_J_fixedN.csv
"""
import numpy as np, pandas as pd, json, time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from _paths import *   # noqa
from exp_I_lib import all_features, W, L

rng = np.random.default_rng(2024); rows = []; t0 = time.time()
def make(rowsn, nper, a, rng):
    R = rng.choice(W, rowsn, replace=False); x0 = rng.integers(1, L - a * nper - 1)
    return [(x0 + i * a, int(y)) for i in range(nper) for y in R]
design = {24: [(8, 3), (6, 4), (4, 6), (3, 8)], 48: [(16, 3), (12, 4), (8, 6), (6, 8)]}
for N, combos in design.items():
    for rowsn, nper in combos:
        for a in [2, 3, 4]:
            for _ in range(10):
                f = all_features(make(rowsn, nper, a, rng), "fixedN", "x"); f.pop("coords")
                f.update({"N": N, "rows": rowsn, "nper": nper, "a": a, "group": "few" if nper <= 4 else "many"}); rows.append(f)
df = pd.DataFrame(rows); print(len(df), "structures", f"({time.time()-t0:.0f}s)"); print(df.groupby(["N", "group"]).G.agg(["mean", "std"]).round(3))
geom = [c for c in df if c.startswith(("M0", "M1", "M2"))]; sq = [c for c in df if c.startswith("SqE")]; amn = [c for c in df if c.startswith("A_")]
M2pp = geom + sq + amn; H4c = ["H4c_bloch_frac", "H4c_kmin", "H4c_bloch_frac_avg", "H3_peak"]
out = {"corr": {}, "loco": {}}
for (N, g), sub in df.groupby(["N", "group"]):
    out["corr"][f"{N}/{g}"] = {"bloch": float(np.corrcoef(sub.G, sub.H4c_bloch_frac)[0, 1]), "A_fwd_diag": float(np.corrcoef(sub.G, sub.A_fwd_diag)[0, 1])}
    print(f"  N={N} {g:4s}: Bloch {out['corr'][f'{N}/{g}']['bloch']:+.3f}   A_fwd_diag {out['corr'][f'{N}/{g}']['A_fwd_diag']:+.3f}")
def loco(sub, cols, mk):
    yy = np.log(np.clip(1 - sub.G, 1e-4, None)).values; p = np.zeros(len(sub))
    for a in [2, 3, 4]:
        te = (sub.a == a).values; p[te] = mk().fit(sub.loc[~te, cols], yy[~te]).predict(sub.loc[te, cols])
    return float(1 - ((p - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
gbr = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0)
rdg = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
for key, sub in list(df.groupby("group")) + list(df.groupby(["N", "group"])):
    sub = sub.reset_index(drop=True); r0 = loco(sub, M2pp, gbr); r1 = loco(sub, M2pp + H4c, gbr); rb = loco(sub, ["H4c_bloch_frac"], rdg)
    out["loco"][str(key)] = {"M2pp": r0, "M2pp+H4c": r1, "delta": r1 - r0, "bloch_only": rb}
    print(f"  {str(key):14s}: M2'' {r0:+.3f} → +H4c {r1:+.3f}  (Δ {r1-r0:+.3f})   Bloch-only Ridge {rb:+.3f}")
json.dump(out, open("results_J_fixedN.json", "w"), indent=1); df.to_csv("dataset_J_fixedN.csv", index=False)
