#!/usr/bin/env python3
"""Merge both priors into ONE working AnnData by cell_id (SOP §1 red line)
and build the working file for the CellRank2 fusion run.

- alignment: identical barcode SET on both sides, order identity recorded
  before/after merge; NaN priors refused (entrypoint also re-checks).
- block-monotonicity PRE-CHECK (SOP §5.3) written before any weight choice.
- X becomes log-normalized (normalize_total median -> palantir log2) on ALL
  genes for lineage-driver computation; raw counts kept in layer 'counts'.
- trajectory geometry reused from upstream: obsm X_pca_harmony/X_umap +
  the k=30 neighbors graph on X_pca_harmony (batch decision, see batch/).
Output: working/working.h5ad, tables/block_monotonicity_precheck.csv,
working/merge_alignment.json
"""
import json, os
import numpy as np
import pandas as pd
import scanpy as sc
import palantir

OUT = "/data/users/lianchong/workspace1/outputs/turn_20260919053513_88be624be961480b8b17e73e474572dd"
INP = "/data/users/lianchong/workspace3/outputs/turn_20260918071647_68dfe17436d241028c6ceba1b54741c5/forfate/G1a_Myeloid_MonoTAM.h5ad"
os.makedirs(f"{OUT}/working", exist_ok=True)
os.makedirs(f"{OUT}/tables", exist_ok=True)

print("[merge] load", flush=True)
A = sc.read_h5ad(INP)
pal = pd.read_csv(f"{OUT}/palantir_out/palantir_priors.csv", index_col=0)
ct2 = pd.read_csv(f"{OUT}/ct2_out/cytotrace2_scores.csv", index_col=0)

# --- alignment red line ---
align = {
    "adata_cells": int(A.n_obs),
    "palantir_rows": int(pal.shape[0]), "cytotrace2_rows": int(ct2.shape[0]),
    "palantir_index_equals_adata_order": bool(pal.index.equals(A.obs_names)),
    "cytotrace2_index_equals_adata_order": bool(ct2.index.equals(A.obs_names)),
    "set_identical_palantir": bool(set(pal.index) == set(A.obs_names)),
    "set_identical_cytotrace2": bool(set(ct2.index) == set(A.obs_names)),
}
print("[merge] alignment:", align, flush=True)
assert align["set_identical_palantir"] and align["set_identical_cytotrace2"], \
    "prior cell sets differ from adata"
pal = pal.loc[A.obs_names]
ct2 = ct2.loc[A.obs_names]
align["reordered_to_adata_order"] = True

for c in pal.columns:
    A.obs[c] = pal[c].to_numpy()
for c in ct2.columns:
    A.obs[c] = ct2[c].to_numpy()
A.obs["ct2_time"] = 1.0 - A.obs["CytoTRACE2_Score"].astype(float)
nan_counts = {c: int(A.obs[c].isna().sum()) for c in
              ["palantir_pseudotime", "CytoTRACE2_Score", "ct2_time"]}
align["nan_after_merge"] = nan_counts
assert all(v == 0 for v in nan_counts.values()), "NaN priors after merge"
align["obs_names_unique"] = bool(A.obs_names.is_unique)

# --- block monotonicity pre-check (SOP §5.3, before choosing weights) ---
g = A.obs.groupby("final_subtype", observed=True)
bm = pd.DataFrame({
    "n_cells": g.size(),
    "mean_palantir_pseudotime": g["palantir_pseudotime"].mean(),
    "mean_CytoTRACE2_Score": g["CytoTRACE2_Score"].mean(),
    "mean_ct2_time": g["ct2_time"].mean(),
}).sort_values("mean_palantir_pseudotime")
bm["ct2_time_delta_along_palantir"] = bm["mean_ct2_time"].diff()
bm.round(6).to_csv(f"{OUT}/tables/block_monotonicity_precheck.csv")
align["block_order_by_palantir"] = list(bm.index)
print("[merge] block table:\n", bm.round(4), flush=True)

# --- log-normalized X on all genes (drivers layer), raw kept in counts ---
A.layers["counts"] = A.X.copy()
sc.pp.normalize_total(A, target_sum=None)
palantir.preprocess.log_transform(A)

# --- keep uns writable (pitfall #11: no tuples in uns) ---
def _str_tuples(d):
    for k, v in list(d.items()):
        if isinstance(v, tuple):
            d[k] = [str(x) for x in v]
        elif isinstance(v, dict):
            _str_tuples(v)
    return d
A.uns = _str_tuples(dict(A.uns)) if hasattr(A.uns, "items") else A.uns

wp = f"{OUT}/working/working.h5ad"
A.write_h5ad(wp)
with open(f"{OUT}/working/merge_alignment.json", "w") as f:
    json.dump(align, f, indent=2)
print("[merge] wrote", wp, A.shape, flush=True)
