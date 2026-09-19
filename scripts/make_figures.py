#!/usr/bin/env python3
"""Standard CellRank 2 publication figure set for the fused two-prior model.

Rebuilds the SAME deterministic model as run_cellrank2_fusion.py (the pipeline
is seed-deterministic: identical inputs/params give bit-identical fate
matrices — campaign evidence R* double runs, all Jaccard 1.0), then renders
the 16 standard figure types, each with a same-name CSV of the plotted data.

Native cellrank 2.0.7 plotting first (save=<absolute path>; do NOT pass
return_fig or s= — see references/pitfalls.md #17), matplotlib fallback from
the same model objects when the native call refuses. Gene trends use
GAM(gaussian/identity); the Gamma/log default diverges on zero-inflated
log-expression (pitfalls #16).

Usage:
  python make_figures.py --h5ad working.h5ad --cluster-key cell_type \
      --out-dir figs [--w-ct2 0.5] [--n-states 5] [--threshold soft]
"""
import os, sys, glob, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc, cellrank as cr


def build_model(ad, palantir_col, cyto_col, w_ct2, threshold, n_states, cluster_key):
    ad.obs["ct2_time"] = 1.0 - ad.obs[cyto_col]
    kw = dict(threshold_scheme=threshold)
    if threshold == "soft":
        kw.update(b=10.0, nu=0.5)
    k_pal = cr.kernels.PseudotimeKernel(ad, time_key=palantir_col).compute_transition_matrix(**kw)
    k_ct2 = cr.kernels.PseudotimeKernel(ad, time_key="ct2_time").compute_transition_matrix(**kw)
    g = cr.estimators.GPCCA((1 - w_ct2) * k_pal + w_ct2 * k_ct2)
    g.compute_eigendecomposition()
    g.compute_macrostates(n_states=n_states, cluster_key=cluster_key)
    g.predict_terminal_states()
    g.compute_fate_probabilities()
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--cluster-key", required=True)
    ap.add_argument("--palantir-col", default="palantir_pseudotime")
    ap.add_argument("--cyto-col", default="CytoTRACE2_Score")
    ap.add_argument("--w-ct2", type=float, default=0.5)
    ap.add_argument("--n-states", type=int, default=5)
    ap.add_argument("--threshold", choices=["soft", "hard"], default="soft")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT = os.path.abspath(a.out_dir)  # absolute: scvelo re-routes relative save= under settings.figdir (pitfalls #18)
    os.makedirs(OUT, exist_ok=True)
    LOG = []
    log = lambda m: (print(m, flush=True), LOG.append(str(m)))

    ad = sc.read_h5ad(a.h5ad)
    for col in (a.palantir_col, a.cyto_col, a.cluster_key):
        if col not in ad.obs:
            sys.exit(f"ERROR: missing obs column {col}")
    if ad.obs[[a.palantir_col, a.cyto_col]].isna().any().any():
        sys.exit("ERROR: prior columns contain NaN")
    if "X_umap" not in ad.obsm:
        sc.tl.umap(ad, random_state=42)
    ad.obs[a.cluster_key] = ad.obs[a.cluster_key].astype("category")
    g = build_model(ad, a.palantir_col, a.cyto_col, a.w_ct2, a.threshold, a.n_states, a.cluster_key)
    log(f"model rebuilt: macrostates={list(g.macrostates.cat.categories)}")

    umap = pd.DataFrame(ad.obsm["X_umap"], index=ad.obs_names, columns=["umap_1", "umap_2"])
    umap[a.cluster_key] = ad.obs[a.cluster_key].values
    CATS = list(ad.obs[a.cluster_key].cat.categories)
    fin = lambda n: os.path.join(OUT, n + ".png")
    ok = lambda n: os.path.exists(fin(n))
    fig_csv = lambda n, df: df.to_csv(os.path.join(OUT, n + ".csv"))
    save_close = lambda n: (plt.gcf().savefig(fin(n), dpi=300, bbox_inches="tight"), plt.close("all"))

    def native(call, name):
        try:
            call(fin(name))
            if ok(name):
                log(f"native ok {name}"); return True
            return False
        except Exception as e:
            log(f"native {name} failed: {type(e).__name__}: {str(e)[:140]}"); return False

    def scatter_states(labels, name, title):
        df = pd.concat([umap, labels.rename("state")], axis=1)
        fig_csv(name, df)
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.scatter(umap["umap_1"], umap["umap_2"], c="0.92", s=4, lw=0)
        cmap = plt.get_cmap("tab10")
        for i, s in enumerate(pd.unique(df["state"].dropna())):
            m = (df["state"] == s).values
            ax.scatter(umap["umap_1"][m], umap["umap_2"][m], c=[cmap(i % 10)], s=8, lw=0, label=str(s))
        ax.legend(markerscale=2, fontsize=7, loc="best"); ax.set_title(title)
        save_close(name)

    def scatter_cont(vals, name, prefix):
        fig_csv(name, pd.concat([umap, vals], axis=1))
        cols = list(vals.columns)
        fig, axes = plt.subplots(1, len(cols), figsize=(3.0 * len(cols), 3.0))
        if len(cols) == 1: axes = [axes]
        for ax, c in zip(axes, cols):
            s_ = ax.scatter(umap["umap_1"], umap["umap_2"], c=vals[c].values, s=4, cmap="viridis", lw=0)
            plt.colorbar(s_, ax=ax); ax.set_title(f"{prefix}{c}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        save_close(name)

    # 1/2 macrostates & terminal states
    ms = pd.Series(g.macrostates, index=ad.obs_names, name="macrostate")
    fig_csv("fig_macrostates", pd.concat([umap, ms.rename("state")], axis=1))
    if not native(lambda p: g.plot_macrostates(which="all", discrete=True, title="macrostates", save=p), "fig_macrostates"):
        scatter_states(ms, "fig_macrostates", "GPCCA macrostates")
    ts = pd.Series(g.terminal_states, index=ad.obs_names, name="terminal_state")
    fig_csv("fig_terminal_states", pd.concat([umap, ts.rename("state")], axis=1))
    if not native(lambda p: g.plot_macrostates(which="terminal", discrete=True, title="terminal states", save=p), "fig_terminal_states"):
        scatter_states(ts, "fig_terminal_states", "terminal states")

    # 3 terminal membership (continuous)
    tmem = g.terminal_states_memberships
    tmm = pd.DataFrame(np.asarray(tmem.X if hasattr(tmem, "X") else tmem), index=ad.obs_names, columns=list(tmem.names))
    fig_csv("fig_terminal_membership", pd.concat([umap, tmm], axis=1))
    if not native(lambda p: g.plot_macrostates(which="terminal", discrete=False, same_plot=False, title="terminal membership", save=p), "fig_terminal_membership"):
        scatter_cont(tmm, "fig_terminal_membership", "membership: ")

    # 4 fate probabilities
    fp = g.fate_probabilities
    fpm = pd.DataFrame(np.asarray(fp.X if hasattr(fp, "X") else fp), index=ad.obs_names, columns=list(fp.names))
    fig_csv("fig_fate_probabilities", pd.concat([umap, fpm], axis=1))
    if not native(lambda p: g.plot_fate_probabilities(same_plot=True, title="fate probabilities", save=p), "fig_fate_probabilities"):
        scatter_cont(fpm, "fig_fate_probabilities", "fate: ")
    fig_csv("fig_fate_probabilities_panels", pd.concat([umap, fpm], axis=1))
    if not native(lambda p: g.plot_fate_probabilities(same_plot=False, title="fate probabilities", save=p), "fig_fate_probabilities_panels"):
        scatter_cont(fpm, "fig_fate_probabilities_panels", "fate: ")

    # 5 coarse-grained T
    ct = g.coarse_T
    fig_csv("fig_coarse_T", ct.round(4))
    if not native(lambda p: g.plot_coarse_T(title="coarse-grained transition matrix", save=p), "fig_coarse_T"):
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        im = ax.imshow(ct.values, cmap="viridis")
        ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(ct.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels(ct.index, fontsize=8)
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                ax.text(j, i, f"{ct.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if ct.values[i, j] < ct.values.max() * 0.6 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8); ax.set_title("coarse-grained T (macrostates)")
        save_close("fig_coarse_T")

    # 6 spectrum + eigengap
    eig = np.asarray(g.eigendecomposition["D"].real)[:20]
    gap = int(g.eigendecomposition["eigengap"]) if "eigengap" in g.eigendecomposition else None
    fig_csv("fig_spectrum", pd.DataFrame({"rank": np.arange(1, len(eig) + 1), "eigenvalue_real": eig}))
    if not native(lambda p: g.plot_spectrum(n=20, real_only=True, show_eigengap=True, title="GPCCA spectrum", save=p), "fig_spectrum"):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(np.arange(1, len(eig) + 1), eig, s=28, c="tab:blue")
        if gap is not None:
            ax.axvline(gap + 0.5, color="tab:red", ls="--", lw=1, label=f"eigengap (n={gap + 1})"); ax.legend(fontsize=8)
        ax.set_xlabel("eigenvalue rank"); ax.set_ylabel("Re(lambda)"); ax.set_title("GPCCA spectrum")
        save_close("fig_spectrum")

    # 7 macrostate composition
    ms_df = pd.concat([ms, ad.obs[a.cluster_key]], axis=1)
    comp = ms_df.groupby([a.cluster_key, "macrostate"], observed=True).size().unstack(fill_value=0)
    comp_pct = comp.div(comp.sum(axis=0), axis=1) * 100.0
    fig_csv("fig_macrostate_composition", comp_pct.round(2))
    fig, ax = plt.subplots(figsize=(7, 5))
    bottom = np.zeros(comp_pct.shape[1]); x = np.arange(comp_pct.shape[1])
    for clust in comp_pct.index:
        ax.bar(x, comp_pct.loc[clust].values, bottom=bottom, label=str(clust)); bottom += comp_pct.loc[clust].values
    ax.set_xticks(x); ax.set_xticklabels(comp_pct.columns, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of macrostate cells"); ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    save_close("fig_macrostate_composition")

    # 8 aggregate fate probabilities (violin)
    fpm_cl = pd.concat([fpm, ad.obs[a.cluster_key].rename(a.cluster_key)], axis=1)
    fig_csv("fig_aggregate_fate", fpm_cl.groupby(a.cluster_key, observed=True).mean().round(4))
    try:
        cr.pl.aggregate_fate_probabilities(ad, cluster_key=a.cluster_key, mode="violin", lineages=list(fp.names), save=fin("fig_aggregate_fate"))
        assert ok("fig_aggregate_fate"); log("native ok fig_aggregate_fate")
    except Exception:
        fig, axes = plt.subplots(1, len(fp.names), figsize=(2.0 * len(fp.names), 4.2), sharey=True)
        if len(fp.names) == 1: axes = [axes]
        for ax, ln in zip(axes, list(fp.names)):
            ax.violinplot([fpm_cl[ln].values[fpm_cl[a.cluster_key] == c] for c in CATS], showmedians=True)
            ax.set_xticks(range(1, len(CATS) + 1)); ax.set_xticklabels(CATS, rotation=90, fontsize=6); ax.set_title(ln, fontsize=8)
        fig.suptitle("aggregate fate probabilities by cluster", fontsize=9)
        save_close("fig_aggregate_fate")

    # 9 lineage drivers
    drv = None
    try:
        drv = g.compute_lineage_drivers(layer="X", method="fisher")
        drv.assign(gene=drv.index).to_csv(os.path.join(OUT, "lineage_drivers_all.csv"))
    except Exception as e:
        log(f"compute_lineage_drivers failed: {str(e)[:160]}")
    lin_names = list(fp.names)
    if drv is not None:
        for lin in lin_names[:2]:
            safe = "".join(ch if ch.isalnum() else "_" for ch in str(lin))
            col = f"{lin}_corr"
            top = drv.sort_values(col, ascending=False).head(8) if col in drv.columns else None
            if top is not None:
                fig_csv(f"fig_lineage_drivers_{safe}", top[[col]].assign(gene=top.index).round(4))
            if not native(lambda p, _l=str(lin): g.plot_lineage_drivers(_l, n_genes=8, save=p), f"fig_lineage_drivers_{safe}"):
                if top is None:
                    continue
                fig, ax = plt.subplots(figsize=(3.4, 3.6))
                ax.barh(range(len(top))[::-1], top[col].values)
                ax.set_yticks(range(len(top))[::-1]); ax.set_yticklabels(top.index, fontsize=8)
                ax.set_xlabel("Spearman corr"); ax.set_title(f"top drivers: {lin}", fontsize=9)
                save_close(f"fig_lineage_drivers_{safe}")
        corr_cols = [c for c in drv.columns if c.endswith("_corr")]
        if len(corr_cols) >= 2:
            cc = drv[corr_cols]; cc.columns = [c[:-5] for c in corr_cols]
            top_genes = drv.sort_values(corr_cols[0], ascending=False).head(20).index
            cm_sub = cc.loc[top_genes].corr()
            fig_csv("fig_lineage_drivers_corr", cc.loc[top_genes].round(4))
            fig, ax = plt.subplots(figsize=(4.6, 4.2))
            im = ax.imshow(cm_sub.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(cm_sub.shape[1])); ax.set_xticklabels(cm_sub.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(cm_sub.shape[0])); ax.set_yticklabels(cm_sub.index, fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
            save_close("fig_lineage_drivers_corr")

    # 10 gene trends: GAM(gaussian/identity) first — Gamma/log diverges on zero-inflated log data
    def _fit_curve(model, gene, lineage):
        model.prepare(str(gene), str(lineage), data_key="X", time_key=a.palantir_col)
        model.fit()
        xg = np.linspace(0, 1, 100)[:, None]
        return xg.ravel(), np.asarray(model.predict(xg)).ravel()

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        plotted = []
        for lin in lin_names[:2]:
            safe = "".join(ch if ch.isalnum() else "_" for ch in str(lin))
            if drv is None or f"{lin}_corr" not in drv.columns:
                continue
            genes = [gn for gn in drv.sort_values(f"{lin}_corr", ascending=False).head(3).index if gn in ad.var_names]
            curves = []
            for gn in genes:
                for maker in (lambda: cr.models.GAM(ad, distribution="gaussian", link="identity"),
                              lambda: cr.models.SKLearnModel(ad, GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=0))):
                    try:
                        xg, yh = _fit_curve(maker(), gn, lin); break
                    except Exception as e:
                        log(f"trend {gn}/{lin}: {type(e).__name__}: {str(e)[:80]}"); xg = yh = None
                if yh is not None:
                    curves.append(pd.DataFrame({"pseudotime": xg, "predicted_expr": yh, "gene": gn, "lineage": str(lin)}))
                    plotted.append(f"{lin}:{gn}")
            if curves:
                cudf = pd.concat(curves, ignore_index=True)
                fig_csv(f"fig_gene_trends_{safe}", cudf.round(4))
                fig, ax = plt.subplots(figsize=(5.2, 3.8))
                for gn in genes:
                    sub = cudf[cudf["gene"] == gn]
                    if len(sub): ax.plot(sub["pseudotime"], sub["predicted_expr"], lw=1.6, label=gn)
                ax.set_xlabel(a.palantir_col); ax.set_ylabel("predicted expression")
                ax.set_title(f"gene trends toward {lin}", fontsize=9); ax.legend(fontsize=7)
                save_close(f"fig_gene_trends_{safe}")
        log(f"gene trends plotted: {plotted}")
    except Exception as e:
        log(f"FAIL gene trends: {type(e).__name__}: {str(e)[:160]}")

    # 11 circular projection (csv always; native first, manual circular embedding fallback)
    K = fpm.shape[1]
    ang = np.array([2 * np.pi * k / K for k in range(K)])
    w = np.clip(fpm.values, 0, None)
    z = (w * np.exp(1j * ang)[None, :]).sum(axis=1)
    g.compute_lineage_priming()
    pcols = [c for c in ad.obs.columns if "priming" in c]
    prim = ad.obs[pcols[0]].values if pcols else np.linalg.norm(w, axis=1) / np.sqrt(K)
    r = 0.15 + 0.85 * (prim - prim.min()) / (np.ptp(prim) + 1e-9)
    dom = fpm.idxmax(axis=1).values
    fig_csv("fig_circular_projection", pd.concat([umap, pd.DataFrame({"circ_x": r * np.cos(np.angle(z)), "circ_y": r * np.sin(np.angle(z)), "dominant_fate": dom, "priming": prim}, index=ad.obs_names)], axis=1))
    if not native(lambda p: cr.pl.circular_projection(ad, keys=[a.cluster_key], ncols=2, figsize=(6.4, 6.4), save=p), "fig_circular_projection"):
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        cmap = plt.get_cmap("tab10")
        for i, c in enumerate(pd.unique(dom)):
            m = dom == c
            ax.scatter(r[m] * np.cos(np.angle(z))[m], r[m] * np.sin(np.angle(z))[m], c=[cmap(i % 10)], s=6, lw=0, label=str(c))
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color="0.7", lw=1)
        for k, nm in enumerate(fpm.columns[:K]):
            ax.annotate(str(nm), (1.08 * np.cos(ang[k]), 1.08 * np.sin(ang[k])), ha="center", va="center", fontsize=8)
        ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.set_aspect("equal")
        ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax.set_title("circular fate projection (radius = lineage priming)")
        save_close("fig_circular_projection")

    # 12 lineage priming
    try:
        g.compute_lineage_priming()
        pcols = [c for c in ad.obs.columns if "priming" in c]
        if pcols:
            fig_csv("fig_lineage_priming", pd.concat([umap, ad.obs[pcols[0]].rename("priming")], axis=1))
            fig, ax = plt.subplots(figsize=(4.6, 4.0))
            s_ = ax.scatter(umap["umap_1"], umap["umap_2"], c=ad.obs[pcols[0]].values, s=4, cmap="magma", lw=0)
            plt.colorbar(s_, ax=ax, label="lineage priming"); ax.set_title("lineage priming")
            save_close("fig_lineage_priming")
    except Exception as e:
        log(f"FAIL priming: {str(e)[:120]}")

    pngs = sorted(f for f in os.listdir(OUT) if f.endswith(".png"))
    log(f"TOTAL figures: {len(pngs)}")
    with open(os.path.join(OUT, "figure_log.txt"), "w") as f:
        f.write("\n".join(LOG))
    print("DONE", OUT, f"({len(pngs)} figures)")


if __name__ == "__main__":
    main()
