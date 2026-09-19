#!/usr/bin/env python3
"""CellRank2 dual-prior (Palantir x CytoTRACE2) fused-kernel GPCCA fate analysis.

Pipeline (fixed order):
  1. load h5ad + input validation (unique cell ids, prior columns exist, no NaN)
  2. homogeneous preprocessing:
       normalize_total(target_sum=None i.e. per-cell median) ->
       palantir.preprocess.log_transform (log2(x+0.1)-log2(0.1)) ->
       HVG top 1500 (flavor=cell_ranger, subset) ->
       PCA 50 (arpack) ->
       kNN graph (n_neighbors, random_state=seed)
  3. ct2_time = 1 - CytoTRACE2_Score
  4. PseudotimeKernel(palantir, threshold) + PseudotimeKernel(ct2_time, threshold)
     fused as (1-w)*K_pal + w*K_ct2   (soft scheme: b=10, nu=0.5)
  5. GPCCA: compute_eigendecomposition -> compute_macrostates(n_states, cluster_key)
     -> predict_terminal_states() -> compute_fate_probabilities()
     (optional --eigengap: same kernel, n_states=None auto-selection, record chosen n)
  6. numeric gates on transition matrix and fate matrix (finite / non-negative /
     row-sums, tol 1e-3; any failure -> exit 1)
  7. audit.json + fate_probabilities.csv + terminal_members.csv +
     fig_fate_umap.png/.csv + transition_matrix.npz

Exit codes: 0 = success; 1 = numeric gate failure or runtime failure; 2 = input
validation failure (missing file/column, duplicated cell ids, NaN priors).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.sparse as sp

# fixed constants (spec)
HVG_TOP = 1500
HVG_FLAVOR = "cell_ranger"
N_PCS = 50
SOFT_B = 10.0
SOFT_NU = 0.5
UMAP_SEED = 42
GATE_TOL = 1e-3
TOP_EIG = 15


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="run_cellrank2_fusion.py",
        description=(
            "CellRank2 GPCCA fate analysis on a fused Palantir/CytoTRACE2 "
            "pseudotime kernel (PseudotimeKernel soft-scheme weighted sum)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--h5ad", required=True,
                   help="working AnnData .h5ad containing the prior columns")
    p.add_argument("--palantir-col", default="palantir_pseudotime",
                   help="obs column with Palantir pseudotime")
    p.add_argument("--cyto-col", default="CytoTRACE2_Score",
                   help="obs column with CytoTRACE2 score (ct2_time = 1 - score)")
    p.add_argument("--cluster-key", required=True,
                   help="obs column with cluster / cell-type annotation")
    p.add_argument("--n-neighbors", type=int, default=30,
                   help="k for the kNN graph")
    p.add_argument("--threshold", default="soft", choices=["soft", "hard"],
                   help="PseudotimeKernel thresholding scheme")
    p.add_argument("--w-ct2", type=float, default=0.5,
                   help="fusion weight w for the ct2 kernel: (1-w)*K_pal + w*K_ct2")
    p.add_argument("--n-states", type=int, default=5,
                   help="number of GPCCA macrostates (explicit)")
    p.add_argument("--eigengap", action="store_true",
                   help="additionally run macrostate auto-selection (n_states=None) "
                        "on the same fused kernel and record the chosen n")
    p.add_argument("--seed", type=int, default=0, help="global seed (PCA/kNN)")
    p.add_argument("--out-dir", required=True, help="output directory")
    return p.parse_args(argv)


def fail(msg: str, code: int) -> None:
    sys.stderr.write(f"[run_cellrank2_fusion] ERROR: {msg}\n")
    sys.stderr.flush()
    sys.exit(code)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def as_float(series: pd.Series, name: str) -> np.ndarray:
    try:
        return pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    except Exception as e:  # pragma: no cover - defensive
        fail(f"column '{name}' cannot be coerced to float ({e})", 2)


def majority_label(labels: pd.Series) -> tuple:
    """(majority label, fraction) of a non-empty categorical/string Series."""
    vc = labels.astype(str).value_counts()
    top = vc.index[0]
    return str(top), float(vc.iloc[0] / vc.sum())


def gate_check(matrix_kind: str, mat: np.ndarray) -> dict:
    finite = bool(np.isfinite(mat).all())
    nonneg = bool((mat >= 0.0).all()) if finite else False
    rowsum_dev = float(np.abs(mat.sum(axis=1) - 1.0).max()) if finite else float("nan")
    checks = {
        "finite": {"pass": finite},
        "non_negative": {"pass": nonneg},
        "row_sum_1_tol_1e-3": {"pass": bool(rowsum_dev <= GATE_TOL),
                                "max_abs_dev": rowsum_dev},
    }
    if finite:
        checks["non_negative"]["min_value"] = float(mat.min())
    return {"matrix": matrix_kind, "checks": checks,
            "all_pass": all(c["pass"] for c in checks.values())}


def main(argv=None) -> None:
    t_total0 = time.time()
    args = parse_args(argv)
    timings = {}

    if not os.path.isfile(args.h5ad):
        fail(f"--h5ad file not found: {args.h5ad}", 2)
    os.makedirs(args.out_dir, exist_ok=True)

    import anndata
    import scanpy as sc
    import palantir
    import cellrank as cr
    from cellrank.kernels import PseudotimeKernel
    from cellrank.estimators import GPCCA

    # ------------------------------------------------------------- 1. load
    t0 = time.time()
    adata = anndata.read_h5ad(args.h5ad)
    n_cells_raw, n_genes_raw = adata.shape

    # cell id source: obs['cell_id'] if present, else obs_names
    if "cell_id" in adata.obs.columns:
        id_source = "obs['cell_id']"
        ids = adata.obs["cell_id"].astype(str)
    else:
        id_source = "obs_names"
        ids = pd.Series(adata.obs_names.astype(str), index=adata.obs_names)
    if not ids.is_unique:
        n_dup = int(len(ids) - ids.nunique())
        fail(f"cell_id not unique (source: {id_source}): {n_dup} duplicated ids "
             f"across {len(ids)} cells", 2)

    # required columns
    missing_cols = [c for c in (args.palantir_col, args.cyto_col, args.cluster_key)
                    if c not in adata.obs.columns]
    if missing_cols:
        fail(f"missing required obs column(s): {missing_cols}; "
             f"available columns: {list(adata.obs.columns)}", 2)

    pal = as_float(adata.obs[args.palantir_col], args.palantir_col)
    cyto = as_float(adata.obs[args.cyto_col], args.cyto_col)
    nan_pal = int(np.isnan(pal).sum())
    nan_cyto = int(np.isnan(cyto).sum())
    n_missing_cells = int((np.isnan(pal) | np.isnan(cyto)).sum())
    if n_missing_cells > 0:
        fail(f"prior columns contain NaN (missing cells report): "
             f"{args.palantir_col}: {nan_pal} NaN, {args.cyto_col}: {nan_cyto} NaN, "
             f"total cells missing >=1 prior: {n_missing_cells} of {n_cells_raw}; "
             f"refusing to run on incomplete priors", 2)
    clusters = adata.obs[args.cluster_key].astype(str)
    n_nan_cluster = int(clusters.isna().sum())
    if n_nan_cluster > 0:
        fail(f"cluster column '{args.cluster_key}' contains {n_nan_cluster} NaN "
             f"labels", 2)
    n_clusters = int(clusters.nunique())
    timings["load_validate_s"] = round(time.time() - t0, 3)

    # ct2_time = 1 - cyto (overwrite pre-existing column if any)
    ct2_time = 1.0 - cyto
    had_ct2_col = "ct2_time" in adata.obs.columns
    adata.obs["ct2_time"] = ct2_time

    # ---------------------------------------------------- 2. preprocessing
    t0 = time.time()
    sc.pp.normalize_total(adata, target_sum=None)  # per-cell median total
    palantir.preprocess.log_transform(adata)
    sc.pp.highly_variable_genes(adata, flavor=HVG_FLAVOR, n_top_genes=HVG_TOP,
                                subset=True)
    sc.tl.pca(adata, n_comps=N_PCS, svd_solver="arpack", random_state=args.seed)
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, random_state=args.seed,
                    use_rep="X_pca", metric="euclidean")
    timings["preprocess_s"] = round(time.time() - t0, 3)
    n_hvg = int(adata.n_vars)

    # ------------------------------------------------------------ 3. kernels
    t0 = time.time()
    pk_kwargs = dict(threshold_scheme=args.threshold,
                     check_irreducibility=False, show_progress_bar=False)
    if args.threshold == "soft":
        pk_kwargs.update(b=SOFT_B, nu=SOFT_NU)
    k_pal = PseudotimeKernel(adata, time_key=args.palantir_col,
                             backward=False).compute_transition_matrix(**pk_kwargs)
    k_ct2 = PseudotimeKernel(adata, time_key="ct2_time",
                             backward=False).compute_transition_matrix(**pk_kwargs)
    w = float(args.w_ct2)
    fused = (1.0 - w) * k_pal + w * k_ct2
    T = fused.transition_matrix.tocsr()
    timings["kernels_s"] = round(time.time() - t0, 3)

    # -------------------------------------------------------------- 4. GPCCA
    t0 = time.time()
    g = GPCCA(fused)
    g.compute_eigendecomposition(k=20, which="LR")
    g.compute_macrostates(n_states=args.n_states, cluster_key=args.cluster_key)
    g.predict_terminal_states()  # method='stability' (cellrank defaults)
    g.compute_fate_probabilities(solver="gmres", use_petsc=False,
                                 show_progress_bar=False)
    timings["gpcca_main_s"] = round(time.time() - t0, 3)

    # eigengap auto-selection run on the SAME fused kernel
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

    # ------------------------------------------------------ 5. extract results
    eig = np.asarray(g.eigendecomposition["D"], dtype=complex)
    eig_top15_real = [float(x) for x in eig.real[:TOP_EIG]]

    fp = g.fate_probabilities
    F = np.asarray(fp, dtype=float)
    fate_names = [str(n) for n in list(fp.names)]
    ts = g.terminal_states
    ts_full = pd.Series([None] * adata.n_obs, index=list(adata.obs_names),
                        dtype=object)
    for cell, lab in ts.items():
        if pd.isna(lab):
            continue
        ts_full.loc[cell] = str(lab)
    tsp = g.terminal_states_probabilities
    tsp_full = pd.Series(0.0, index=list(adata.obs_names))
    tsp_full.loc[tsp.index] = tsp.to_numpy(dtype=float)
    ms = g.macrostates
    ms_full = pd.Series([None] * adata.n_obs, index=list(adata.obs_names),
                        dtype=object)
    for cell, lab in ms.items():
        if pd.isna(lab):
            continue
        ms_full.loc[cell] = str(lab)

    clusters = clusters.reset_index(drop=True)
    ids_list = list(ids)

    def states_table(assign: pd.Series) -> list:
        """assign: object Series, None for non-member cells."""
        tab = []
        labels = [v for v in pd.unique(assign.to_numpy()) if v is not None]
        for lab in sorted(labels):
            sel = np.asarray(assign == lab)
            n = int(sel.sum())
            if n == 0:
                continue
            mj, frac = majority_label(clusters[sel])
            tab.append({"label": str(lab), "n_cells": n,
                        "majority_cluster": mj, "majority_fraction": round(frac, 6)})
        return tab

    terminal_table = states_table(ts_full)
    macro_table = states_table(ms_full)

    # ------------------------------------------------------ 6. numeric gates
    T_gate = gate_check("transition_matrix", T.toarray()) \
        if T.shape[0] * T.shape[1] <= 5_000_000 else _sparse_gate(T)
    gates = {
        "tolerance": GATE_TOL,
        "transition_matrix": T_gate,
        "fate_probabilities": gate_check("fate_probabilities", F),
    }
    all_gates_pass = all(v["all_pass"] for k, v in gates.items()
                         if isinstance(v, dict) and "all_pass" in v)

    # ---------------------------------------------- 7. direction-conflict audit
    t0 = time.time()
    C = adata.obsp["connectivities"].tocsr()
    A = C + C.T
    A.setdiag(0)
    A.eliminate_zeros()
    triu = sp.triu(A, k=1).tocoo()
    ei, ej = triu.row.astype(int), triu.col.astype(int)
    n_edges = int(len(ei))
    dpt = pal[ei] - pal[ej]
    dct = ct2_time[ei] - ct2_time[ej]
    tie = (dpt == 0.0) | (dct == 0.0)
    conflict = (dpt * dct) < 0.0
    combined_delta = np.abs(dpt) + np.abs(dct)
    per_cluster = []
    cl_arr = clusters.to_numpy()
    for c in sorted(set(cl_arr)):
        sel = (cl_arr[ei] == c) & (cl_arr[ej] == c)
        ne = int(sel.sum())
        nc = int((conflict & sel).sum())
        per_cluster.append({
            "cluster": str(c), "n_within_cluster_edges": ne,
            "n_conflicts": nc,
            "conflict_rate": round(nc / ne, 6) if ne > 0 else None,
        })
    med_before = float(np.median(combined_delta[conflict])) \
        if conflict.any() else None
    strict = conflict & (~tie)
    med_after = float(np.median(combined_delta[strict])) \
        if strict.any() else None
    direction_conflict = {
        "graph": "undirected edges of adata.obsp['connectivities'] (triu)",
        "conflict_definition": "sign(pal_i-pal_j) * sign(ct2_i-ct2_j) < 0",
        "n_undirected_edges": n_edges,
        "n_tie_edges": int(tie.sum()),
        "n_conflicts": int(conflict.sum()),
        "total_conflict_rate": round(float(conflict.sum()) / n_edges, 6)
            if n_edges else None,
        "total_conflict_rate_excluding_ties": round(
            float((conflict & ~tie).sum()) / max(int((~tie).sum()), 1), 6),
        "per_cluster_within_edges": per_cluster,
        "median_abs_dpt_plus_abs_dct2_conflicting_edges_before_filter":
            round(med_before, 6) if med_before is not None else None,
        "n_conflicting_before_filter": int(conflict.sum()),
        "median_abs_dpt_plus_abs_dct2_conflicting_edges_after_filter":
            round(med_after, 6) if med_after is not None else None,
        "n_conflicting_after_filter": int(strict.sum()),
        "filter_definition": "after_filter = conflicting edges with a strict "
                             "direction in BOTH priors (ties removed)",
    }
    timings["conflict_audit_s"] = round(time.time() - t0, 3)

    # ------------------------------------------------- 8. block mean table
    block_rows = []
    for c in sorted(set(cl_arr)):
        sel = cl_arr == c
        block_rows.append({
            "cluster": str(c), "n_cells": int(sel.sum()),
            "mean_palantir": round(float(np.mean(pal[sel])), 6),
            "mean_ct2_time": round(float(np.mean(ct2_time[sel])), 6),
        })
    order = sorted(block_rows, key=lambda r: r["mean_palantir"])
    ct2_seq = [r["mean_ct2_time"] for r in order]
    diffs = [round(ct2_seq[i + 1] - ct2_seq[i], 6) for i in range(len(ct2_seq) - 1)]
    block_means = {
        "per_cluster": block_rows,
        "lineage_order_by_mean_palantir": [r["cluster"] for r in order],
        "ct2_time_means_along_order": ct2_seq,
        "diffs_along_order": diffs,
        "ct2_monotone_nondecreasing": bool(all(d >= 0 for d in diffs)),
        "ct2_monotone_nonincreasing": bool(all(d <= 0 for d in diffs)),
        "n_increases": int(sum(d > 0 for d in diffs)),
        "n_decreases": int(sum(d < 0 for d in diffs)),
    }

    # ----------------------------------------------------------- 9. artifacts
    t0 = time.time()
    out = args.out_dir
    dom_idx = np.argmax(F, axis=1)
    dom_state = [fate_names[i] for i in dom_idx]
    dom_prob = F[np.arange(F.shape[0]), dom_idx]

    # UMAP (seed fixed at 42 by spec)
    sc.tl.umap(adata, random_state=UMAP_SEED)
    um = np.asarray(adata.obsm["X_umap"])
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
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(f"Dominant terminal fate (n_states={args.n_states})")
    ax.legend(markerscale=2, fontsize=7, loc="best", frameon=False)
    fig.tight_layout()
    fig_path = os.path.join(out, "fig_fate_umap.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    pd.DataFrame({
        "cell_id": ids_list,
        "UMAP1": um[:, 0], "UMAP2": um[:, 1],
        "dominant_fate": dom_state, "dominant_fate_prob": dom_prob,
        args.cluster_key: clusters.astype(str).to_numpy(),
    }).to_csv(os.path.join(out, "fig_fate_umap.csv"), index=False)

    fate_df = pd.DataFrame(F, columns=fate_names)
    fate_df.insert(0, "cell_id", ids_list)
    fate_df.insert(1, args.cluster_key, clusters.astype(str).to_numpy())
    fate_df.to_csv(os.path.join(out, "fate_probabilities.csv"), index=False)

    memb = ~ts_full.isna().to_numpy()
    pd.DataFrame({
        "cell_id": [ids_list[i] for i in np.where(memb)[0]],
        "terminal_state": [str(ts_full.iloc[i]) for i in np.where(memb)[0]],
        "terminal_state_probability": [tsp_full.iloc[i] for i in np.where(memb)[0]],
        args.cluster_key: [str(clusters.iloc[i]) for i in np.where(memb)[0]],
    }).to_csv(os.path.join(out, "terminal_members.csv"), index=False)

    sp.save_npz(os.path.join(out, "transition_matrix.npz"), T)

    # ------------------------------------------------------------- audit
    artifacts = {}
    for fn in ["fate_probabilities.csv", "terminal_members.csv",
               "fig_fate_umap.png", "fig_fate_umap.csv", "transition_matrix.npz"]:
        p = os.path.join(out, fn)
        artifacts[fn] = {"sha256": sha256_file(p),
                         "bytes": os.path.getsize(p)}

    audit = {
        "script": os.path.abspath(__file__),
        "argv": sys.argv[1:],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "h5ad": os.path.abspath(args.h5ad),
            "n_cells": int(n_cells_raw), "n_genes": int(n_genes_raw),
            "cell_id_source": id_source, "cell_id_unique": True,
            "missing_prior_cells": n_missing_cells,
            "nan_per_column": {args.palantir_col: nan_pal, args.cyto_col: nan_cyto},
            "cluster_key": args.cluster_key, "n_clusters": n_clusters,
            "ct2_time_overwrote_existing_column": bool(had_ct2_col),
            "hvg_n_vars_after_subset": n_hvg,
        },
        "config": {
            "palantir_col": args.palantir_col, "cyto_col": args.cyto_col,
            "cluster_key": args.cluster_key, "n_neighbors": args.n_neighbors,
            "threshold": args.threshold, "soft_b": SOFT_B, "soft_nu": SOFT_NU,
            "w_ct2": w, "n_states": args.n_states,
            "eigengap_flag": bool(args.eigengap), "seed": args.seed,
            "umap_seed": UMAP_SEED,
            "preprocessing": ("normalize_total(target_sum=None/median) -> "
                              "palantir.preprocess.log_transform(pseudo_count=0.1) "
                              f"-> HVG top {HVG_TOP} flavor={HVG_FLAVOR} subset -> "
                              f"PCA {N_PCS} arpack -> kNN(n={args.n_neighbors})"),
            "fate_solver": "gmres (scipy, use_petsc=False)",
            "predict_terminal_states": "method='stability' (library defaults)",
        },
        "versions": {
            "python": sys.version.split()[0],
            "cellrank": cr.__version__, "scanpy": sc.__version__,
            "palantir": palantir.__version__, "anndata": anndata.__version__,
            "numpy": np.__version__, "scipy": sp.__version__ if hasattr(sp, "__version__") else "",
            "pandas": pd.__version__,
        },
        "kernel": {
            "fusion": f"(1-{w})*K_palantir + {w}*K_ct2",
            "transition_matrix_shape": [int(T.shape[0]), int(T.shape[1])],
            "transition_matrix_nnz": int(T.nnz),
        },
        "macrostates": macro_table,
        "terminal_states": terminal_table,
        "n_terminal_states": len(terminal_table),
        "eigengap_run": eigengap_rec,
        "numeric_gates": gates,
        "numeric_gates_all_pass": bool(all_gates_pass),
        "direction_conflict": direction_conflict,
        "block_means": block_means,
        "eigenvalues_top15_real": eig_top15_real,
        "timing_sec": timings,
        "artifacts": artifacts,
    }
    timings["artifacts_umap_s"] = round(time.time() - t0, 3)
    audit["timing_sec"] = timings
    timings["total_s"] = round(time.time() - t_total0, 3)
    audit["timing_sec"]["total_s"] = timings["total_s"]

    with open(os.path.join(out, "audit.json"), "w") as fh:
        json.dump(audit, fh, indent=2)

    # stdout summary (numbers only)
    print(json.dumps({
        "out_dir": os.path.abspath(out),
        "n_cells": int(n_cells_raw),
        "n_macrostates": len(macro_table),
        "n_terminal_states": len(terminal_table),
        "terminal_states": terminal_table,
        "eigengap_auto_n": eigengap_rec["auto_n_states"],
        "numeric_gates_all_pass": bool(all_gates_pass),
        "total_conflict_rate": direction_conflict["total_conflict_rate"],
        "total_seconds": timings["total_s"],
    }, indent=2))

    if not all_gates_pass:
        sys.stderr.write(f"[run_cellrank2_fusion] ERROR: numeric gates failed: "
                         f"{json.dumps(gates)}\n")
        sys.exit(1)
    sys.exit(0)


def _sparse_gate(T) -> dict:
    """Finite/non-negative/row-sum gate for a large sparse matrix without densify."""
    data = T.data
    finite = bool(np.isfinite(data).all())
    nonneg = bool((data >= 0.0).all()) if finite else False
    rs = np.asarray(T.sum(axis=1)).ravel()
    dev = float(np.abs(rs - 1.0).max()) if finite else float("nan")
    checks = {
        "finite": {"pass": finite},
        "non_negative": {"pass": nonneg, "min_value": float(data.min()) if finite else None},
        "row_sum_1_tol_1e-3": {"pass": bool(dev <= GATE_TOL), "max_abs_dev": dev},
    }
    return {"matrix": "transition_matrix(sparse)", "checks": checks,
            "all_pass": all(c["pass"] for c in checks.values())}


if __name__ == "__main__":
    main()
