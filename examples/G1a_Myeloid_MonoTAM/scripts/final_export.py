#!/usr/bin/env python3
"""Final export: drivers + core CellRank2 figure set (PNG+CSV each) + final h5ad.

Model identical to run_final.py (fused w=0.5 soft kernel on the harmony k=30
graph, GPCCA K=10, terminal states = 8 mature macrostates via
set_terminal_states(names)). Figures follow make_figures.py @ a63627f
(native-first with manual fallback, 300dpi, same-name CSV per PNG). GAM gene
trends are omitted at 265k cells (pygam infeasible; recorded in report).
"""
import json, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import cellrank as cr
from cellrank.kernels import PseudotimeKernel
from cellrank.estimators import GPCCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/data/users/lianchong/workspace1/outputs/turn_20260919053513_88be624be961480b8b17e73e474572dd"
FIG = f"{OUT}/figs_final"
os.makedirs(FIG, exist_ok=True)
LOG = []
log = lambda m: (print(f"[figs] {m}", flush=True), LOG.append(str(m)))
fin = lambda n: os.path.join(FIG, n + ".png")
ok = lambda n: os.path.exists(fin(n))
t00 = time.time()

ad = sc.read_h5ad(f"{OUT}/working/working.h5ad")
ad.obs["ct2_time"] = 1.0 - ad.obs["CytoTRACE2_Score"].astype(float)
ROOT = "Mye_ClassicalMono"
CK = "final_subtype"

# ---- rebuild final model ----
kw = dict(threshold_scheme="soft", b=10.0, nu=0.5, check_irreducibility=False,
          show_progress_bar=False)
k_pal = PseudotimeKernel(ad, time_key="palantir_pseudotime", backward=False).compute_transition_matrix(**kw)
k_ct2 = PseudotimeKernel(ad, time_key="ct2_time", backward=False).compute_transition_matrix(**kw)
fused = 0.5 * k_pal + 0.5 * k_ct2
g = GPCCA(fused)
g.compute_eigendecomposition(k=20, which="LR")
g.compute_macrostates(n_states=10, cluster_key=CK)
ms = pd.Series(g.macrostates, index=ad.obs_names)
maj = pd.crosstab(ms.dropna().astype(str), ad.obs[CK].astype(str)).idxmax(axis=1).to_dict()
mature = [n for n, m in maj.items() if m != ROOT]
g.set_terminal_states(states=mature, n_cells=30, allow_overlap=False)
g.compute_fate_probabilities(solver="gmres", use_petsc=False, show_progress_bar=False)
log(f"model rebuilt; terminals={list(g.fate_probabilities.names)}")

fp = g.fate_probabilities
F = np.asarray(fp, dtype=float)
names = [str(n) for n in list(fp.names)]
umap = pd.DataFrame(ad.obsm["X_umap"], index=ad.obs_names, columns=["umap_1", "umap_2"])
umap[CK] = ad.obs[CK].astype(str).values
CATS = list(pd.unique(ad.obs[CK].astype(str)))

def save_close(n):
    plt.gcf().savefig(fin(n), dpi=300, bbox_inches="tight"); plt.close("all")

def native(call, name):
    try:
        call(fin(name))
        if ok(name):
            log(f"native ok {name}"); return True
        return False
    except Exception as e:
        log(f"native {name} failed: {type(e).__name__}: {str(e)[:120]}"); return False

def scatter_states(labels, name, title):
    df = pd.concat([umap, labels.rename("state")], axis=1)
    df.to_csv(os.path.join(FIG, name + ".csv"))
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.scatter(umap["umap_1"], umap["umap_2"], c="0.92", s=3, lw=0)
    cmap = plt.get_cmap("tab20")
    for i, s in enumerate(pd.unique(df["state"].dropna())):
        m = (df["state"] == s).values
        ax.scatter(umap["umap_1"][m], umap["umap_2"][m], c=[cmap(i % 20)], s=6, lw=0, label=str(s))
    ax.legend(markerscale=2, fontsize=6, loc="best", frameon=False); ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    save_close(name)

def scatter_cont(vals, name, prefix, ncols=None):
    vals.to_csv(os.path.join(FIG, name + ".csv"))
    cols = list(vals.columns)
    n = len(cols) if ncols is None else ncols
    fig, axes = plt.subplots(int(np.ceil(len(cols)/n)), n, figsize=(2.8*n, 2.8*int(np.ceil(len(cols)/n))))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        s_ = ax.scatter(umap["umap_1"], umap["umap_2"], c=vals[c].values, s=3, cmap="viridis", lw=0)
        plt.colorbar(s_, ax=ax); ax.set_title(f"{prefix}{c}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(cols):]:
        ax.axis("off")
    save_close(name)

# 1/2 macrostates + terminal states
scatter_states(ms, "fig_macrostates", "GPCCA macrostates (K=10)")
ts = pd.Series(g.terminal_states, index=ad.obs_names)
scatter_states(ts, "fig_terminal_states", "terminal states (mature macrostates)")

# 3 terminal membership (continuous)
tmem = g.terminal_states_memberships
tmm = pd.DataFrame(np.asarray(tmem.X if hasattr(tmem, "X") else tmem), index=ad.obs_names, columns=[str(c) for c in tmem.names])
tmm.round(6).to_csv(os.path.join(FIG, "fig_terminal_membership.csv"))
if not native(lambda p: g.plot_macrostates(which="terminal", discrete=False, same_plot=False, title="terminal membership", save=p), "fig_terminal_membership"):
    scatter_cont(tmm, "fig_terminal_membership", "m: ", ncols=4)

# 4 fate probabilities overview + panels
fpm = pd.DataFrame(F, index=ad.obs_names, columns=names)
fpm.round(6).to_csv(os.path.join(FIG, "fig_fate_probabilities.csv"))
if not native(lambda p: g.plot_fate_probabilities(same_plot=True, title="fate probabilities", save=p), "fig_fate_probabilities"):
    scatter_cont(fpm, "fig_fate_probabilities", "fate: ", ncols=4)
scatter_cont(fpm, "fig_fate_probabilities_panels", "fate: ", ncols=4)

# 5 coarse T
ct = g.coarse_T
ct.round(4).to_csv(os.path.join(FIG, "fig_coarse_T.csv"))
if not native(lambda p: g.plot_coarse_T(title="coarse-grained T", save=p), "fig_coarse_T"):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(ct.values, cmap="viridis")
    ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(ct.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels(ct.index, fontsize=7)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, f"{ct.values[i, j]:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if ct.values[i, j] < ct.values.max()*0.6 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8); ax.set_title("coarse-grained T (10 macrostates)")
    save_close("fig_coarse_T")

# 6 spectrum
eig = np.asarray(g.eigendecomposition["D"].real)[:20]
gap = int(g.eigendecomposition["eigengap"]) if "eigengap" in g.eigendecomposition else None
pd.DataFrame({"rank": np.arange(1, len(eig) + 1), "eigenvalue_real": eig}).to_csv(os.path.join(FIG, "fig_spectrum.csv"), index=False)
if not native(lambda p: g.plot_spectrum(n=len(eig), real_only=True, show_eigengap=True, title="GPCCA spectrum", save=p), "fig_spectrum"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(np.arange(1, len(eig) + 1), eig, s=24)
    ax.set_xlabel("eigenvalue rank"); ax.set_ylabel("Re(lambda)"); ax.set_title("GPCCA spectrum")
    save_close("fig_spectrum")

# 7 macrostate composition
ms_df = pd.concat([ms.rename("macrostate"), ad.obs[CK].astype(str).rename(CK)], axis=1)
comp = ms_df.dropna().groupby([CK, "macrostate"], observed=True).size().unstack(fill_value=0)
comp_pct = (comp.div(comp.sum(axis=0), axis=1) * 100).round(2)
comp_pct.to_csv(os.path.join(FIG, "fig_macrostate_composition.csv"))
fig, ax = plt.subplots(figsize=(8, 5))
bottom = np.zeros(comp_pct.shape[1]); x = np.arange(comp_pct.shape[1])
for clust in comp_pct.index:
    ax.bar(x, comp_pct.loc[clust].values, bottom=bottom, label=str(clust)); bottom += comp_pct.loc[clust].values
ax.set_xticks(x); ax.set_xticklabels(comp_pct.columns, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("% of macrostate cells"); ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6)
save_close("fig_macrostate_composition")

# 8 aggregate fate by subtype
agg = fpm.groupby(ad.obs[CK].astype(str).values, observed=True).mean().round(4)
agg.to_csv(os.path.join(FIG, "fig_aggregate_fate.csv"))
fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
for ax, ln in zip(axes.ravel(), names):
    ax.violinplot([fpm[ln].values[ad.obs[CK].astype(str).values == c] for c in CATS], showmedians=True)
    ax.set_xticks(range(1, len(CATS)+1)); ax.set_xticklabels([c.replace("Mye_", "") for c in CATS], rotation=90, fontsize=6)
    ax.set_title(ln.replace("Mye_", ""), fontsize=8)
fig.suptitle("aggregate fate probabilities by subtype", fontsize=10)
save_close("fig_aggregate_fate")

# 9 lineage drivers (fisher on log X)
drv = None
try:
    drv = g.compute_lineage_drivers(layer="X", method="fisher", cluster_key=CK)
    drv.assign(gene=drv.index).to_csv(os.path.join(FIG, "lineage_drivers_all.csv"))
    log(f"drivers computed: {drv.shape}")
except Exception as e:
    log(f"compute_lineage_drivers fisher failed ({str(e)[:120]}); falling back to corr")
    try:
        drv = g.compute_lineage_drivers(layer="X", method="corr", cluster_key=CK)
        drv.assign(gene=drv.index).to_csv(os.path.join(FIG, "lineage_drivers_all.csv"))
    except Exception as e2:
        log(f"corr drivers failed too: {str(e2)[:120]}")
corr_cols = [c for c in (drv.columns if drv is not None else []) if c.endswith("_corr")]
for lin in names[:4]:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(lin))
    col = f"{lin}_corr"
    if drv is None or col not in drv.columns:
        continue
    top = drv.sort_values(col, ascending=False).head(8)
    top[[col]].assign(gene=top.index).round(4).to_csv(os.path.join(FIG, f"fig_lineage_drivers_{safe}.csv"))
    fig, ax = plt.subplots(figsize=(3.6, 4.0))
    ax.barh(range(len(top))[::-1], top[col].values)
    ax.set_yticks(range(len(top))[::-1]); ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel("corr"); ax.set_title(f"top drivers: {lin}", fontsize=9)
    save_close(f"fig_lineage_drivers_{safe}")
if len(corr_cols) >= 2:
    cc = drv[corr_cols]; cc.columns = [c[:-5] for c in corr_cols]
    cc.round(4).to_csv(os.path.join(FIG, "fig_lineage_drivers_corr.csv"))
    tg = drv.sort_values(corr_cols[0], ascending=False).head(25).index
    cm = cc.loc[tg].corr()
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    im = ax.imshow(cm.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(cm.shape[1])); ax.set_xticklabels(cm.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(cm.shape[0])); ax.set_yticklabels(cm.index, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8); ax.set_title("driver correlation across lineages (top-25)")
    save_close("fig_lineage_drivers_corr")

# 11 circular projection (manual embedding; native at 265k unreliable)
K = fpm.shape[1]
ang = np.array([2*np.pi*k/K for k in range(K)])
w = np.clip(fpm.values, 0, None)
z = (w * np.exp(1j*ang)[None, :]).sum(axis=1)
prim = 1.0 + np.log(np.ptp(w, axis=1) + 1e-9) if False else np.linalg.norm(w, axis=1)/np.sqrt(K)
r = 0.15 + 0.85*(prim - prim.min())/(np.ptp(prim)+1e-9)
dom = fpm.idxmax(axis=1).values
circ = pd.DataFrame({"circ_x": r*np.cos(np.angle(z)), "circ_y": r*np.sin(np.angle(z)),
                     "dominant_fate": dom, "priming": prim}, index=ad.obs_names)
pd.concat([umap, circ], axis=1).to_csv(os.path.join(FIG, "fig_circular_projection.csv"))
fig, ax = plt.subplots(figsize=(6.8, 6.4))
cmap = plt.get_cmap("tab20")
for i, c in enumerate(pd.unique(dom)):
    m = dom == c
    ax.scatter(circ["circ_x"][m], circ["circ_y"][m], c=[cmap(i % 20)], s=4, lw=0, label=str(c))
th = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(th), np.sin(th), color="0.7", lw=1)
for k, nm in enumerate(fpm.columns):
    ax.annotate(str(nm), (1.09*np.cos(ang[k]), 1.09*np.sin(ang[k])), ha="center", va="center", fontsize=7)
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal")
ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
ax.set_title("circular fate projection (radius = lineage priming)")
save_close("fig_circular_projection")

# 12 lineage priming
prim_ser = pd.Series(prim, index=ad.obs_names, name="priming")
pd.concat([umap, prim_ser], axis=1).to_csv(os.path.join(FIG, "fig_lineage_priming.csv"))
fig, ax = plt.subplots(figsize=(4.8, 4.2))
s_ = ax.scatter(umap["umap_1"], umap["umap_2"], c=prim, s=3, cmap="magma", lw=0)
plt.colorbar(s_, ax=ax, label="lineage priming"); ax.set_title("lineage priming"); ax.set_xticks([]); ax.set_yticks([])
save_close("fig_lineage_priming")

# ---- final h5ad ----
ad.obsm["fate_probabilities"] = F
ad.uns["fate_probabilities_columns"] = names
ad.obs["dominant_fate"] = dom
ad.obs["dominant_fate_prob"] = fpm.max(axis=1).values
ad.obs["lineage_priming"] = prim
ad.obs["terminal_state_member"] = ts.astype(str).where(ts.notna(), "").values
ad.obs["macrostate"] = ms.astype(str).where(ms.notna(), "").values
# stringify uns tuples (pitfall #11)
def _str_tuples(d):
    for k, v in list(d.items()):
        if isinstance(v, tuple):
            d[k] = [str(x) for x in v]
        elif isinstance(v, dict):
            _str_tuples(v)
    return d
try:
    ad.uns = _str_tuples(dict(ad.uns))
except Exception:
    pass
fp_h5 = f"{OUT}/final/G1a_Myeloid_MonoTAM_cellrank2.h5ad"
os.makedirs(f"{OUT}/final", exist_ok=True)
ad.write_h5ad(fp_h5)
with open(os.path.join(FIG, "figure_log.txt"), "w") as f:
    f.write("\n".join(LOG))
pngs = sorted(f for f in os.listdir(FIG) if f.endswith(".png"))
log(f"DONE {time.time()-t00:.0f}s; {len(pngs)} figures; final h5ad -> {fp_h5}")
