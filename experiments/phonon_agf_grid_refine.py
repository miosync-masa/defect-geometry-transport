"""
Stage 0e — σ-grid refinement to freeze the three-regime figure.
  σ ∈ {0.25,…,0.95} step 0.1; full shuffle only; M=24 multisets × P=40 permutations.
  Mountain: argmax ⟨log G⟩ ; Valley: argmin distance(⟨Ξ⟩, DMM) ; Crossing: order-share = 0.5.
  Uncertainty: bootstrap over multisets (B=1000); positions by local parabolic fit around the extremum.
  Decision rule fixed in advance: 90% intervals of σ_mountain and σ_valley overlap → "coincide"; else → "indicator-dependent band".
"""
import numpy as np, json, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _paths import *   # noqa
src = open(__file__.replace("phonon_agf_grid_refine.py", "phonon_agf_variance_decomp.py")).read()
ns = {"__file__": __file__}; exec(src[:src.index("rng = np.random.default_rng(11)")], ns)
spectrum, conductance, dist, place, shuffle = ns["spectrum"], ns["conductance"], ns["dist"], ns["place"], ns["shuffle"]
MAT, NI, W, G_dmm, Es = ns["MAT"], ns["NI"], ns["W"], ns["G_dmm"], ns["Es"]

rng = np.random.default_rng(23); sigmas = np.round(np.arange(0.25, 0.96, 0.1), 2); M, P, B = 24, 40, 1000
t0 = time.time(); raw = {}
for s in sigmas:
    per_ms = []   # per multiset: (mean logG, var logG within, mean spectrum)
    for _ in range(M):
        ms = MAT["I"]["m"] * np.exp(s * rng.normal(size=(NI, W)) - s ** 2 / 2); logs = []; Xs = np.zeros(len(Es))
        for _ in range(P):
            Xi = spectrum(place(shuffle(ms, "full", rng))); Xs += Xi; logs.append(np.log(max(conductance(Xi), 1e-12)))
        logs = np.array(logs); per_ms.append((logs.mean(), logs.var(), Xs / P))
    raw[float(s)] = per_ms
    mu = np.array([p[0] for p in per_ms]); vw = np.array([p[1] for p in per_ms]); Xbar = np.mean([p[2] for p in per_ms], axis=0)
    share = vw.mean() / (vw.mean() + mu.var())
    print(f"σ={s:.2f}: ⟨G⟩/G_DMM {np.exp(mu.mean())/G_dmm:.3f}  dist {dist(Xbar):.3f}  order share {share:.2f}  ({time.time()-t0:.0f}s)", flush=True)

# ---------- bootstrap over multisets ----------
def stats_from(idx_sets):
    """idx_sets: dict σ -> indices of multisets to use. returns arrays over σ: meanlogG, dist, share"""
    m, d, sh = [], [], []
    for s in sigmas:
        pm = [raw[float(s)][i] for i in idx_sets[float(s)]]
        mu = np.array([p[0] for p in pm]); vw = np.array([p[1] for p in pm]); Xbar = np.mean([p[2] for p in pm], axis=0)
        m.append(mu.mean()); d.append(dist(Xbar)); sh.append(vw.mean() / (vw.mean() + mu.var() + 1e-18))
    return np.array(m), np.array(d), np.array(sh)
def extremum(x, y, kind):
    i = int(np.argmax(y) if kind == "max" else np.argmin(y))
    if 0 < i < len(x) - 1:
        a, b, c = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
        if a != 0:
            xs = -b / (2 * a)
            if x[i - 1] <= xs <= x[i + 1]: return float(xs)
    return float(x[i])
def crossing(x, y, level=0.5):
    for i in range(len(x) - 1):
        if (y[i] - level) * (y[i + 1] - level) <= 0 and y[i] != y[i + 1]:
            return float(x[i] + (level - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return float("nan")
full = {float(s): list(range(M)) for s in sigmas}
m0, d0, s0 = stats_from(full)
boot = {"mountain": [], "valley": [], "crossing": [], "m": [], "d": [], "s": []}
brng = np.random.default_rng(99)
for _ in range(B):
    idx = {float(s): list(brng.integers(0, M, M)) for s in sigmas}
    m, d, sh = stats_from(idx); boot["m"].append(m); boot["d"].append(d); boot["s"].append(sh)
    boot["mountain"].append(extremum(sigmas, m, "max")); boot["valley"].append(extremum(sigmas, d, "min")); boot["crossing"].append(crossing(sigmas, sh))
def ci(a, lo=5, hi=95): a = np.array(a); a = a[np.isfinite(a)]; return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]
res = {"sigmas": sigmas.tolist(), "meanG_over_dmm": (np.exp(m0) / G_dmm).tolist(), "dist": d0.tolist(), "share": s0.tolist(),
       "mountain": {"point": extremum(sigmas, m0, "max"), "ci90": ci(boot["mountain"])},
       "valley": {"point": extremum(sigmas, d0, "min"), "ci90": ci(boot["valley"])},
       "crossing": {"point": crossing(sigmas, s0), "ci90": ci(boot["crossing"])},
       "band_meanG": np.percentile(np.exp(np.array(boot["m"])) / G_dmm, [5, 95], axis=0).tolist(),
       "band_dist": np.percentile(np.array(boot["d"]), [5, 95], axis=0).tolist(), "band_share": np.percentile(np.array(boot["s"]), [5, 95], axis=0).tolist(), "M": M, "P": P, "B": B}
ov = max(res["mountain"]["ci90"][0], res["valley"]["ci90"][0]) <= min(res["mountain"]["ci90"][1], res["valley"]["ci90"][1])
res["decision"] = "coincide (90% intervals overlap)" if ov else "indicator-dependent band (intervals disjoint)"
print(f"\nmountain σ* = {res['mountain']['point']:.3f}  90% CI {res['mountain']['ci90']}")
print(f"valley   σ* = {res['valley']['point']:.3f}  90% CI {res['valley']['ci90']}")
print(f"share=0.5 crossing σ = {res['crossing']['point']:.3f}  90% CI {res['crossing']['ci90']}")
print("DECISION:", res["decision"])
json.dump(res, open("stage0e_grid_refine.json", "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6)); S = sigmas
ax[0].plot(S, res["meanG_over_dmm"], "o-", color="C3"); ax[0].fill_between(S, *res["band_meanG"], color="C3", alpha=0.2); ax[0].axhline(1, color="gray", ls=":")
ax[0].axvspan(*res["mountain"]["ci90"], color="C3", alpha=0.15, label=f"σ_mountain 90% CI {res['mountain']['ci90'][0]:.2f}–{res['mountain']['ci90'][1]:.2f}"); ax[0].set_xlabel("σ"); ax[0].set_ylabel("⟨G⟩/G_DMM (geometric mean)"); ax[0].legend(fontsize=8); ax[0].set_title("mean response: mountain")
ax[1].plot(S, res["dist"], "^-", color="k"); ax[1].fill_between(S, *res["band_dist"], color="k", alpha=0.15)
ax[1].axvspan(*res["valley"]["ci90"], color="k", alpha=0.12, label=f"σ_valley 90% CI {res['valley']['ci90'][0]:.2f}–{res['valley']['ci90'][1]:.2f}"); ax[1].axvspan(*res["mountain"]["ci90"], color="C3", alpha=0.12, label="σ_mountain CI")
ax[1].set_xlabel("σ"); ax[1].set_ylabel("distance of ⟨Ξ⟩ to DMM"); ax[1].legend(fontsize=8); ax[1].set_title(f"shape: valley — {res['decision']}")
ax[2].plot(S, res["share"], "d-", color="C4"); ax[2].fill_between(S, *res["band_share"], color="C4", alpha=0.2); ax[2].axhline(0.5, color="gray", ls=":")
if np.isfinite(res["crossing"]["point"]): ax[2].axvspan(*res["crossing"]["ci90"], color="C4", alpha=0.15, label=f"share=0.5 crossing CI {res['crossing']['ci90'][0]:.2f}–{res['crossing']['ci90'][1]:.2f}")
ax[2].set_xlabel("σ"); ax[2].set_ylabel("order share of variance"); ax[2].legend(fontsize=8); ax[2].set_title("dominance: order ⇄ multiset")
plt.suptitle(f"Stage 0e: refined σ grid, M={M} multisets × P={P} permutations, bootstrap B={B}")
plt.tight_layout(); plt.savefig(FIGDIR + "figZ_phonon_grid_refine.png", dpi=130); plt.close(); print("done", f"{time.time()-t0:.0f}s")
