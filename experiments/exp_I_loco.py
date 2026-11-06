"""
I (LOCO): leave-one-(a,b)-out prediction of unseen periods within the multi-period regime.
  16 (a,b) combinations x 3 period counts x 5 samples = 220 longgrid structures.
  Compares M2'' vs M2''+H4c vs single Bloch-fraction feature. Output: results_I_loco.json, dataset_I_longgrid16.csv
"""
import numpy as np, pandas as pd, json, time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from _paths import *   # noqa
from exp_I_lib import all_features, W, L

t0 = time.time()
base = pd.read_csv("dataset_I.csv"); base = base[base.fam != "longgrid"]
rng = np.random.default_rng(7)
def longgrid(a, b, nper, rng):
    x0 = rng.integers(1, max(2, L - a * nper - 1)); y0 = rng.integers(0, b); return [(x0 + i * a, y) for i in range(nper) for y in range(y0, W, b)]
rows = []
for a in [2, 3, 4, 5]:
    for b in [2, 3, 4, 5]:
        for nper in [6, 7, 8]:
            if a * nper > L - 3: continue
            for _ in range(5):
                f = all_features(longgrid(a, b, nper, rng), "longgrid", "lg"); f.pop("coords"); f.update({"a": a, "b": b, "nper": nper}); rows.append(f)
lg = pd.DataFrame(rows); print("longgrid 16 combos:", len(lg), f"({time.time()-t0:.0f}s)")
geom = [c for c in lg if c.startswith(("M0", "M1", "M2"))]; sq = [c for c in lg if c.startswith("SqE")]; amn = [c for c in lg if c.startswith("A_")]
M2pp = geom + sq + amn; H4c = ["H4c_bloch_frac", "H4c_kmin", "H4c_bloch_frac_avg", "H4c_x_nper", "H_nperiods", "H3_peak"]
sets = {"M2''": M2pp, "M2''+H4c": M2pp + H4c, "H4c only": H4c, "Bloch frac only": ["H4c_bloch_frac"]}
y = np.log(np.clip(1 - lg.G, 1e-4, None)).values; combos = sorted(set(zip(lg.a, lg.b))); out = {}
for name, cols in sets.items():
    for mname, mk in [("GBR", lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0)),
                      ("Ridge", lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))))]:
        preds = np.zeros(len(lg))
        for (a, b) in combos:
            te = ((lg.a == a) & (lg.b == b)).values; preds[te] = mk().fit(lg.loc[~te, cols], y[~te]).predict(lg.loc[te, cols])
        r2 = 1 - ((preds - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(); rmse = np.sqrt(((preds - y) ** 2).mean())
        out[f"{name}/{mname}"] = {"r2": float(r2), "rmse": float(rmse)}; print(f"LOCO {name:16s} {mname:5s}: R² {r2:+.3f}  rmse {rmse:.3f}")
g = base[base.fam == "grid"].reset_index(drop=True); yg = np.log(np.clip(1 - g.G, 1e-4, None)).values
for name, cols in [("M2''", M2pp), ("M2''+H4c", M2pp + H4c), ("Bloch frac only", ["H4c_bloch_frac"])]:
    preds = np.zeros(len(g))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(g):
        preds[te] = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(g.loc[tr, cols], yg[tr]).predict(g.loc[te, cols])
    r2 = 1 - ((preds - yg) ** 2).sum() / ((yg - yg.mean()) ** 2).sum(); out[f"fewperiod_grid/{name}"] = {"r2": float(r2)}
    print(f"few-period grid 5-fold Ridge {name:16s}: R² {r2:+.3f}")
json.dump(out, open("results_I_loco.json", "w"), indent=1); lg.to_csv("dataset_I_longgrid16.csv", index=False)
