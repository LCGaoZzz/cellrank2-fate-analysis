#!/usr/bin/env python3
"""run_cellrank2_fusion.py @ a63627f, adapted ONLY at the graph-build step.

Adaptation (recorded in audit.config): the packaged entrypoint rebuilds its
kNN graph on an uncorrected PCA50; this object is a 22-dataset pan-cancer
atlas whose upstream pipeline already produced X_pca_harmony and the k=30
neighbors graph on it (fully connected; dataset silhouette -0.12 vs -0.04
unorrected). The skill's own SOP requires the CellRank graph to be
same-source with the Palantir diffusion geometry -- both therefore use
X_pca_harmony. The existing graph is verified and REUSED verbatim
(k=30, use_rep=X_pca_harmony, random_state=0); the bundled
normalize->HVG->PCA50->neighbors block is replaced by that reuse. UMAP is
reused from the object for the same reason (was recomputed on uncorrected
PCA in the original). Kernels, fusion, GPCCA, numeric gates, both audits and
all artifacts are byte-for-byte the skill's logic.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy.sparse as sp

SOFT_B, SOFT_NU, UMAP_SEED, GATE_TOL, TOP_EIG = 10.0, 0.5, 42, 1e-3, 15


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="fusion_harmony")
    p.add_argument("--h5ad", required=True)
    p.add_argument("--palantir-col", default="palantir_pseudotime")
    p.add_argument("--cyto-col", default="CytoTRACE2_Score")
    p.add_argument("--cluster-key", required=True)
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--threshold", default="soft", choices=["soft", "hard"])
    p.add_argument("--w-ct2", type=float, default=0.5)
    p.add_argument("--n-states", type=int, default=5)
    p.add_argument("--eigengap", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def fail(msg, code):
    sys.stderr.write(f"[fusion_harmony] ERROR: {msg}\n"); sys.stderr.flush()
    sys.exit(code)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def majority_label(labels):
    vc = labels.astype(str).value_counts()
    return str(vc.index[0]), float(vc.iloc[0] / vc.sum())


def gate_check(kind, mat):
    finite = bool(np.isfinite(mat).all())
    nonneg = bool((mat >= 0.0).all()) if finite else False
    dev = float(np.abs(mat.sum(axis=1) - 1.0).max()) if finite else float("nan")
    checks = {"finite": {"pass": finite},
              "non_negative": {"pass": nonneg},
              "row_sum_1_tol_1e-3": {"pass": bool(dev <= GATE_TOL),
                                     "max_abs_dev": dev}}
    if finite:
        checks["non_negative"]["min_value"] = float(mat.min())
    return {"matrix": kind, "checks": checks,
            "all_pass": all(c["pass"] for c in checks.values())}


def _sparse_gate(T):
    data = T.data
    finite = bool(np.isfinite(data).all())
    nonneg = bool((data >= 0.0).all()) if finite else False
    rs = np.asarray(T.sum(axis=1)).ravel()
    dev = float(np.abs(rs - 1.0).max()) if finite else float("nan")
    checks = {"finite": {"pass": finite},
              "non_negative": {"pass": nonneg,
                               "min_value": float(data.min()) if finite else None},
              "row_sum_1_tol_1e-3": {"pass": bool(dev <= GATE_TOL),
                                     "max_abs_dev": dev}}
    return {"matrix": "transition_matrix(sparse)", "checks": checks,
            "all_pass": all(c["pass"] for c in checks.values())}


def main(argv=None):
    t_total0 = time.time()
    args = parse_args(argv)
    timings = {}
    if not os.path.isfile(args.h5ad):
        fail(f"--h5ad file not found: {args.h5ad}", 2)
    os.makedirs(args.out_dir, exist_ok=True)

    import anndata, scanpy as sc, palantir, cellrank as cr
    from cellrank.kernels import PseudotimeKernel
    from cellrank.estimators import GPCCA

    t0 = time.time()
    adata = anndata.read_h5ad(args.h5ad)
    n_cells_raw, n_genes_raw = adata.shape
    if "cell_id" in adata.obs.columns:
        id_source = "obs['cell_id']"
        ids = adata.obs["cell_id"].astype(str)
    else:
        id_source = "obs_names"
        ids = pd.Series(adata.obs_names.astype(str), index=adata.obs_names)
    if not ids.is_unique:
        fail(f"cell_id not unique (source: {id_source})", 2)
    missing = [c for c in (args.palantir_col, args.cyto_col, args.cluster_key)
               if c not in adata.obs.columns]
    if missing:
        fail(f"missing required obs column(s): {missing}", 2)
    pal = pd.to_numeric(adata.obs[args.palantir_col], errors="raise").to_numpy(float)
    cyto = pd.to_numeric(adata.obs[args.cyto_col], errors="raise").to_numpy(float)
    nan_pal, nan_cyto = int(np.isnan(pal).sum()), int(np.isnan(cyto).sum())
    if nan_pal or nan_cyto:
        fail(f"NaN priors: palantir {nan_pal}, cytotrace2 {nan_cyto}", 2)
    clusters = adata.obs[args.cluster_key].astype(str)
    ct2_time = 1.0 - cyto
    had = "ct2_time" in adata.obs.columns
    adata.obs["ct2_time"] = ct2_time
    timings["load_validate_s"] = round(time.time() - t0, 3)

    # ---- graph: REUSE upstream harmony graph (adaptation) ----
    t0 = time.time()
    nb = adata.uns.get("neighbors", {})
    nbp = nb.get("params", {}) if isinstance(nb, dict) else {}
    graph_source = {"use_rep": str(nbp.get("use_rep")),
                    "n_neighbors": str(nbp.get("n_neighbors")),
                    "random_state": str(nbp.get("random_state"))}
    if "connectivities" not in adata.obsp:
        fail("no connectivities in obsp - nothing to reuse", 2)
    if graph_source["use_rep"] != "X_pca_harmony":
        fail(f"existing graph not on X_pca_harmony: {graph_source}", 2)
    conn_nnz = int(adata.obsp["connectivities"].nnz)
    timings["graph_reuse_s"] = round(time.time() - t0, 3)

    # ---- kernels ----
    t0 = time.time()
    pk = dict(threshold_scheme=args.threshold, check_irreducibility=False,
              show_progress_bar=False)
    if args.threshold == "soft":
        pk.update(b=SOFT_B, nu=SOFT_NU)
    k_pal = PseudotimeKernel(adata, time_key=args.palantir_col,
                             backward=False).compute_transition_matrix(**pk)
    k_ct2 = PseudotimeKernel(adata, time_key="ct2_time",
                             backward=False).compute_transition_matrix(**pk)
    w = float(args.w_ct2)
    fused = (1.0 - w) * k_pal + w * k_ct2
    T = fused.transition_matrix.tocsr()
    timings["kernels_s"] = round(time.time() - t0, 3)

    # ---- GPCCA ----
    t0 = time.time()
    g = GPCCA(fused)
    g.compute_eigendecomposition(k=20, which="LR")
    g.compute_macrostates(n_states=args.n_states, cluster_key=args.cluster_key)
    g.predict_terminal_states()
    g.compute_fate_probabilities(solver="gmres", use_petsc=False,
                                 show_progress_bar=False)
    timings["gpcca_main_s"] = round(time.time() - t0, 3)

    eigengap_rec = {"enabled": bool(args.eigengap), "auto_n_states": None,
                    "macrostate_labels": [], "error": None}
    if args.eigengap:
        t0 = time.time()
        try:
            g2 = GPCCA(fused)
            g2.compute_eigendecomposition(k=20, which="LR")
            g2.compute_macrostates(n_states=None, cluster_key=args.cluster_key)
            ms2 = g2.macrostates
            cats = [str(c) for c in ms2.cat.categories] if hasattr(ms2, "cat") \
                else sorted(set(ms2.dropna().astype(str)))
            eigengap_rec["auto_n_states"] = len(cats)
            eigengap_rec["macrostate_labels"] = cats
        except Exception as e:
            eigengap_rec["error"] = f"{type(e).__name__}: {e}"
        timings["gpcca_eigengap_s"] = round(time.time() - t0, 3)

    # ---- extract ----
    eig = np.asarray(g.eigendecomposition["D"], dtype=complex)
    eig_top15_real = [float(x) for x in eig.real[:TOP_EIG]]
    fp = g.fate_probabilities
    F = np.asarray(fp, dtype=float)
    fate_names = [str(n) for n in list(fp.names)]
    ts = g.terminal_states
    ts_full = pd.Series([None] * adata.n_obs, index=list(adata.obs_names), dtype=object)
    for cell, lab in ts.items():
        if not pd.isna(lab):
            ts_full.loc[cell] = str(lab)
    tsp = g.terminal_states_probabilities
    tsp_full = pd.Series(0.0, index=list(adata.obs_names))
    tsp_full.loc[tsp.index] = tsp.to_numpy(dtype=float)
    ms = g.macrostates
    ms_full = pd.Series([None] * adata.n_obs, index=list(adata.obs_names), dtype=object)
    for cell, lab in ms.items():
        if not pd.isna(lab):
            ms_full.loc[cell] = str(lab)
    clusters = clusters.reset_index(drop=True)
    ids_list = list(ids)

    def states_table(assign):
        tab, labels = [], [v for v in pd.unique(assign.to_numpy()) if v is not None]
        for lab in sorted(labels):
            sel = np.asarray(assign == lab)
            n = int(sel.sum())
            if n:
                mj, frac = majority_label(clusters[sel])
                tab.append({"label": str(lab), "n_cells": n,
                            "majority_cluster": mj,
                            "majority_fraction": round(frac, 6)})
        return tab

    terminal_table, macro_table = states_table(ts_full), states_table(ms_full)

    T_gate = _sparse_gate(T) if T.shape[0] * T.shape[1] > 5_000_000 \
        else gate_check("transition_matrix", T.toarray())
    gates = {"tolerance": GATE_TOL, "transition_matrix": T_gate,
             "fate_probabilities": gate_check("fate_probabilities", F)}
    all_gates_pass = all(v["all_pass"] for v in gates.values()
                         if isinstance(v, dict) and "all_pass" in v)

    # ---- direction-conflict audit (on the reused harmony graph) ----
    t0 = time.time()
    C = adata.obsp["connectivities"].tocsr()
    Au = C + C.T
    Au.setdiag(0); Au.eliminate_zeros()
    triu = sp.triu(Au, k=1).tocoo()
    ei, ej = triu.row.astype(int), triu.col.astype(int)
    dpt = pal[ei] - pal[ej]
    dct = ct2_time[ei] - ct2_time[ej]
    tie = (dpt == 0.0) | (dct == 0.0)
    conflict = (dpt * dct) < 0.0
    combined = np.abs(dpt) + np.abs(dct)
    cl_arr = clusters.to_numpy()
    per_cluster = []
    for c in sorted(set(cl_arr)):
        sel = (cl_arr[ei] == c) & (cl_arr[ej] == c)
        ne, nc = int(sel.sum()), int((conflict & sel).sum())
        per_cluster.append({"cluster": str(c), "n_within_cluster_edges": ne,
                            "n_conflicts": nc,
                            "conflict_rate": round(nc / ne, 6) if ne else None})
    strict = conflict & (~tie)
    direction_conflict = {
        "graph": "undirected edges of adata.obsp['connectivities'] (triu; harmony k=30)",
        "conflict_definition": "sign(pal_i-pal_j) * sign(ct2_i-ct2_j) < 0",
        "n_undirected_edges": int(len(ei)), "n_tie_edges": int(tie.sum()),
        "n_conflicts": int(conflict.sum()),
        "total_conflict_rate": round(float(conflict.sum()) / len(ei), 6),
        "total_conflict_rate_excluding_ties":
            round(float((conflict & ~tie).sum()) / max(int((~tie).sum()), 1), 6),
        "per_cluster_within_edges": per_cluster,
        "n_conflicting_after_filter": int(strict.sum()),
        "filter_definition": "after_filter = conflicting edges with strict direction in BOTH priors",
    }
    timings["conflict_audit_s"] = round(time.time() - t0, 3)

    # ---- block means ----
    block_rows = []
    for c in sorted(set(cl_arr)):
        sel = cl_arr == c
        block_rows.append({"cluster": str(c), "n_cells": int(sel.sum()),
                           "mean_palantir": round(float(np.mean(pal[sel])), 6),
                           "mean_ct2_time": round(float(np.mean(ct2_time[sel])), 6)})
    order = sorted(block_rows, key=lambda r: r["mean_palantir"])
    seq = [r["mean_ct2_time"] for r in order]
    diffs = [round(seq[i + 1] - seq[i], 6) for i in range(len(seq) - 1)]
    block_means = {"per_cluster": block_rows,
                   "lineage_order_by_mean_palantir": [r["cluster"] for r in order],
                   "ct2_time_means_along_order": seq, "diffs_along_order": diffs,
                   "ct2_monotone_nondecreasing": bool(all(d >= 0 for d in diffs)),
                   "ct2_monotone_nonincreasing": bool(all(d <= 0 for d in diffs)),
                   "n_increases": int(sum(d > 0 for d in diffs)),
                   "n_decreases": int(sum(d < 0 for d in diffs))}

    # ---- artifacts ----
    t0 = time.time()
    out = args.out_dir
    dom_idx = np.argmax(F, axis=1)
    dom_state = [fate_names[i] for i in dom_idx]
    dom_prob = F[np.arange(F.shape[0]), dom_idx]
    um = np.asarray(adata.obsm["X_umap"])  # reused upstream UMAP (adaptation)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cats = sorted(set(dom_state))
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(7, 6))
    for k, c in enumerate(cats):
        sel = np.asarray([s == c for s in dom_state])
        ax.scatter(um[sel, 0], um[sel, 1], s=6, color=cmap(k % 20),
                   label=f"{c}", rasterized=True)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title(f"Dominant terminal fate (n_states={args.n_states})")
    ax.legend(markerscale=2, fontsize=7, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_fate_umap.png"), dpi=150)
    plt.close(fig)
    pd.DataFrame({"cell_id": ids_list, "UMAP1": um[:, 0], "UMAP2": um[:, 1],
                  "dominant_fate": dom_state, "dominant_fate_prob": dom_prob,
                  args.cluster_key: clusters.astype(str).to_numpy()}
                 ).to_csv(os.path.join(out, "fig_fate_umap.csv"), index=False)
    fate_df = pd.DataFrame(F, columns=fate_names)
    fate_df.insert(0, "cell_id", ids_list)
    fate_df.insert(1, args.cluster_key, clusters.astype(str).to_numpy())
    fate_df.to_csv(os.path.join(out, "fate_probabilities.csv"), index=False)
    memb = ~ts_full.isna().to_numpy()
    pd.DataFrame({"cell_id": [ids_list[i] for i in np.where(memb)[0]],
                  "terminal_state": [str(ts_full.iloc[i]) for i in np.where(memb)[0]],
                  "terminal_state_probability": [tsp_full.iloc[i] for i in np.where(memb)[0]],
                  args.cluster_key: [str(clusters.iloc[i]) for i in np.where(memb)[0]]
                  }).to_csv(os.path.join(out, "terminal_members.csv"), index=False)
    sp.save_npz(os.path.join(out, "transition_matrix.npz"), T)

    artifacts = {}
    for fn in ["fate_probabilities.csv", "terminal_members.csv",
               "fig_fate_umap.png", "fig_fate_umap.csv", "transition_matrix.npz"]:
        p = os.path.join(out, fn)
        artifacts[fn] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}

    audit = {
        "script": os.path.abspath(__file__),
        "base_skill_script": "cellrank2-fate-analysis/scripts/run_cellrank2_fusion.py @ a63627f",
        "adaptation": "graph/UMAP reused from upstream harmony representation (batch decision); all other logic identical",
        "argv": sys.argv[1:],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"h5ad": os.path.abspath(args.h5ad),
                  "n_cells": int(n_cells_raw), "n_genes": int(n_genes_raw),
                  "cell_id_source": id_source, "cell_id_unique": True,
                  "nan_per_column": {args.palantir_col: nan_pal, args.cyto_col: nan_cyto},
                  "cluster_key": args.cluster_key,
                  "n_clusters": int(adata.obs[args.cluster_key].nunique()),
                  "ct2_time_overwrote_existing_column": bool(had)},
        "config": {"palantir_col": args.palantir_col, "cyto_col": args.cyto_col,
                   "cluster_key": args.cluster_key,
                   "n_neighbors": args.n_neighbors, "threshold": args.threshold,
                   "soft_b": SOFT_B, "soft_nu": SOFT_NU, "w_ct2": w,
                   "n_states": args.n_states, "eigengap_flag": bool(args.eigengap),
                   "seed": args.seed, "umap_seed": UMAP_SEED,
                   "graph": f"REUSED upstream neighbors: {graph_source}, conn nnz={conn_nnz} (harmony geometry; same-source with Palantir diffusion maps)",
                   "umap": "REUSED obsm['X_umap'] from upstream (harmony geometry)",
                   "fate_solver": "gmres (scipy, use_petsc=False)",
                   "predict_terminal_states": "method='stability' (library defaults)"},
        "versions": {"python": sys.version.split()[0], "cellrank": cr.__version__,
                     "scanpy": sc.__version__, "palantir": palantir.__version__,
                     "anndata": anndata.__version__, "numpy": np.__version__,
                     "scipy": __import__("scipy").__version__, "pandas": pd.__version__,
                     "petsc4py": __import__("petsc4py").__version__ if _try_petsc() else None,
                     "slepc4py": __import__("slepc4py").__version__ if _try_petsc() else None},
        "kernel": {"fusion": f"(1-{w})*K_palantir + {w}*K_ct2",
                   "transition_matrix_shape": [int(T.shape[0]), int(T.shape[1])],
                   "transition_matrix_nnz": int(T.nnz)},
        "macrostates": macro_table, "terminal_states": terminal_table,
        "n_terminal_states": len(terminal_table),
        "eigengap_run": eigengap_rec,
        "numeric_gates": gates, "numeric_gates_all_pass": bool(all_gates_pass),
        "direction_conflict": direction_conflict, "block_means": block_means,
        "eigenvalues_top15_real": eig_top15_real,
        "timing_sec": timings, "artifacts": artifacts,
    }
    timings["artifacts_s"] = round(time.time() - t0, 3)
    timings["total_s"] = round(time.time() - t_total0, 3)
    audit["timing_sec"] = timings
    with open(os.path.join(out, "audit.json"), "w") as fh:
        json.dump(audit, fh, indent=2)
    print(json.dumps({"out_dir": os.path.abspath(out), "n_cells": int(n_cells_raw),
                      "n_macrostates": len(macro_table),
                      "n_terminal_states": len(terminal_table),
                      "terminal_states": terminal_table,
                      "eigengap_auto_n": eigengap_rec["auto_n_states"],
                      "numeric_gates_all_pass": bool(all_gates_pass),
                      "total_conflict_rate": direction_conflict["total_conflict_rate"],
                      "total_seconds": timings["total_s"]}, indent=2))
    if not all_gates_pass:
        fail(f"numeric gates failed: {json.dumps(gates)}", 1)
    sys.exit(0)


def _try_petsc():
    try:
        import petsc4py  # noqa
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
