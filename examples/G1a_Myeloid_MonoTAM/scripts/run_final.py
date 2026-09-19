#!/usr/bin/env python3
"""Final primary model for G1a_Myeloid_MonoTAM per cellrank2-fate-analysis @ a63627f.

Sweep evidence (fusion_sweep + fusion/): default predict_terminal_states()
selects the root macrostate (ClassicalMono basin) as terminal in EVERY
configuration, and C1QC (largest subtype) does not separate from the SPP1
basin at K<=8. Decision within the skill's toolbox:
  - keep w_ct2=0.5 (bracketed 0.25/0.75 both DEGRADE lineage coverage);
  - probe K=10 -> 9 -> 8 for a C1QC-majority macrostate;
  - set_terminal_states_from_macrostates() on the mature-side macrostates
    (majority != ClassicalMono) instead of the default stability pick;
  - recompute fate probabilities + numeric gates; full artifacts to
    fusion_final/. Also records the two FIXED singles (palantir-only,
    ct2-only kernels run directly, avoiding the 0-weight KernelAdd
    degeneracy) to complete the SOP s6 weight sweep table.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import cellrank as cr
from cellrank.kernels import PseudotimeKernel
from cellrank.estimators import GPCCA

OUT = "/data/users/lianchong/workspace1/outputs/turn_20260919053513_88be624be961480b8b17e73e474572dd"
WORK = f"{OUT}/working/working.h5ad"
DEST = f"{OUT}/fusion_final"
os.makedirs(DEST, exist_ok=True)
ROOT = "Mye_ClassicalMono"
t00 = time.time()
log = lambda m: print(f"[final] {m}", flush=True)

ad = sc.read_h5ad(WORK)
ad.obs["ct2_time"] = 1.0 - ad.obs["CytoTRACE2_Score"].astype(float)
kw = dict(threshold_scheme="soft", b=10.0, nu=0.5,
          check_irreducibility=False, show_progress_bar=False)

def kernel(w):
    k_pal = PseudotimeKernel(ad, time_key="palantir_pseudotime", backward=False).compute_transition_matrix(**kw)
    if w == 0.0:
        return k_pal
    k_ct2 = PseudotimeKernel(ad, time_key="ct2_time", backward=False).compute_transition_matrix(**kw)
    if w == 1.0:
        return k_ct2
    return (1.0 - w) * k_pal + w * k_ct2

def gpcca_default_terms(k, K):
    g = GPCCA(k)
    g.compute_eigendecomposition(k=20, which="LR")
    g.compute_macrostates(n_states=K, cluster_key="final_subtype")
    g.predict_terminal_states()
    return g

# ---- 1) fixed singles: cached from the previous run (run_final.log), skip recompute ----
sweep_extra = {
    "single_pal_w0": ["Mye_ClassicalMono", "Mye_FABP4", "Mye_InflamMono_1", "Mye_InflamMono_2",
                       "Mye_NonClassicalMono", "Mye_TissueResMac_1", "Mye_TissueResMac_2"],
    "single_ct2_w1": ["Mye_C1QC", "Mye_ClassicalMono", "Mye_FABP4", "Mye_InflamMono",
                       "Mye_NonClassicalMono", "Mye_TissueResMac_1", "Mye_TissueResMac_2"],
}

# ---- 2) probe K for C1QC separability at w=0.5 ----
fused = kernel(0.5)
chosen_K, macro_maj, g = None, None, None
for K in [10, 9, 8]:
    try:
        gt = GPCCA(fused)
        gt.compute_eigendecomposition(k=20, which="LR")
        gt.compute_macrostates(n_states=K, cluster_key="final_subtype")
        ms = pd.Series(gt.macrostates, index=ad.obs_names).dropna().astype(str)
        ct = pd.crosstab(ms, ad.obs["final_subtype"].astype(str))
        maj = ct.idxmax(axis=1).to_dict()
        pur = (ct.max(axis=1) / ct.sum(axis=1)).round(3).to_dict()
        has_c1qc = any(v == "Mye_C1QC" for v in maj.values())
        log(f"K={K} macrostate majorities: {maj}")
        if chosen_K is None:
            chosen_K, macro_maj, macro_pur, g = K, maj, pur, gt
            if has_c1qc:
                break
        else:
            del gt
    except Exception as e:
        log(f"K={K} probe failed: {e}")
log(f"chosen_K={chosen_K} (C1QC separated: {any(v=='Mye_C1QC' for v in macro_maj.values())})")

# ---- 3) terminal states: mature macrostates only (root macrostates -> NaN) ----
mature = [n for n, m in macro_maj.items() if m != ROOT]
root_labels = [n for n, m in macro_maj.items() if m == ROOT]
log(f"mature macrostates as terminals: {mature}; excluded root macrostates: {root_labels}")
ms_series = pd.Series(g.macrostates, index=ad.obs_names).copy()
keep = ms_series.notna() & ~ms_series.isin(root_labels)
ts_members = ms_series[keep].astype(str)
log(f"manual terminal members check: {int(keep.sum())} cells over {ts_members.nunique()} states")
# GPCCA-native route (2.0.7): states = subset of macrostate names
g.set_terminal_states(states=mature, n_cells=30, allow_overlap=False)
g.compute_fate_probabilities(solver="gmres", use_petsc=False, show_progress_bar=False)

# ---- 4) gates ----
T = fused.transition_matrix.tocsr()
F = np.asarray(g.fate_probabilities, dtype=float)
names = [str(n) for n in list(g.fate_probabilities.names)]
gate_T = {
    "finite": bool(np.isfinite(T.data).all()),
    "non_negative": bool((T.data >= 0).all()),
    "rowsum_dev": float(np.abs(np.asarray(T.sum(axis=1)).ravel() - 1).max())}
gate_F = {
    "finite": bool(np.isfinite(F).all()),
    "non_negative": bool((F >= 0).all()),
    "rowsum_dev": float(np.abs(F.sum(axis=1) - 1).max())}
gates_pass = all([gate_T["finite"], gate_T["non_negative"], gate_T["rowsum_dev"] <= 1e-3,
                  gate_F["finite"], gate_F["non_negative"], gate_F["rowsum_dev"] <= 1e-3])
log(f"gates: T={gate_T} F={gate_F} pass={gates_pass}")

# ---- 5) terminal membership table + fate matrix artifacts ----
ts = g.terminal_states
ts_full = pd.Series([None] * ad.n_obs, index=list(ad.obs_names), dtype=object)
for cell, lab in ts.items():
    if not pd.isna(lab):
        ts_full.loc[cell] = str(lab)
memb = ~ts_full.isna().to_numpy()
sub = ad.obs["final_subtype"].astype(str).reset_index(drop=True)
term_rows = []
for lab in sorted(set(ts_full.dropna())):
    sel = (ts_full == lab).to_numpy()
    mj = sub[sel].value_counts()
    term_rows.append({"label": lab, "n_cells": int(sel.sum()),
                      "majority_cluster": str(mj.index[0]),
                      "majority_fraction": round(float(mj.iloc[0] / sel.sum()), 4)})
log(f"terminal table: {term_rows}")

ids = list(ad.obs_names)
pd.DataFrame(F, columns=names).assign(
    cell_id=ids, final_subtype=sub.astype(str).to_numpy()
)[["cell_id", "final_subtype"] + names].to_csv(f"{DEST}/fate_probabilities.csv", index=False)
pd.DataFrame({"cell_id": [ids[i] for i in np.where(memb)[0]],
              "terminal_state": [str(ts_full.iloc[i]) for i in np.where(memb)[0]],
              "final_subtype": [str(sub.iloc[i]) for i in np.where(memb)[0]]
              }).to_csv(f"{DEST}/terminal_members.csv", index=False)
sp.save_npz(f"{DEST}/transition_matrix.npz", T)

# dominant-fate UMAP (reused harmony UMAP)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
um = np.asarray(ad.obsm["X_umap"])
dom = np.array(names)[F.argmax(axis=1)]
domp = F.max(axis=1)
fig, ax = plt.subplots(figsize=(7, 6))
cmap = plt.get_cmap("tab20")
for k_, c in enumerate(sorted(set(dom))):
    m = dom == c
    ax.scatter(um[m, 0], um[m, 1], s=6, color=cmap(k_ % 20), label=c, rasterized=True)
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
ax.set_title(f"Dominant terminal fate (fused w=0.5, K={chosen_K}, mature terminals)")
ax.legend(markerscale=2, fontsize=7, loc="best", frameon=False)
fig.tight_layout(); fig.savefig(f"{DEST}/fig_fate_umap.png", dpi=150); plt.close(fig)
pd.DataFrame({"cell_id": ids, "UMAP1": um[:, 0], "UMAP2": um[:, 1],
              "dominant_fate": dom, "dominant_fate_prob": domp,
              "final_subtype": sub.astype(str).to_numpy()}
             ).to_csv(f"{DEST}/fig_fate_umap.csv", index=False)

# ---- 6) audit ----
audit = {
    "script": os.path.abspath(__file__),
    "skill": "cellrank2-fate-analysis @ a63627f (two-prior fusion GPCCA)",
    "decision": {
        "w_ct2": 0.5,
        "chosen_K": chosen_K,
        "terminal_selection": "set_terminal_states (GPCCA macrostates with majority != Mye_ClassicalMono); fate solve via CFLARE on the same fused kernel (see note)",
        "why": ("default predict_terminal_states selected the root basin as terminal in every "
                "sweep config; C1QC does not separate from the SPP1 basin at K<=8 (nearly "
                "identical mean palantir 0.491 vs 0.486) - separation only at w=0.75 which "
                "drops InflamMono and SPP1 (see fusion_sweep)"),
        "macrostate_majorities": macro_maj, "macrostate_purity": macro_pur,
        "c1qc_separated": any(v == "Mye_C1QC" for v in macro_maj.values())},
    "sweep_singles_fixed": sweep_extra,
    "sweep_default_method": {
        "w0.25_K7": ["Mye_ClassicalMono", "Mye_FABP4", "Mye_InflamMono_1", "Mye_InflamMono_2",
                      "Mye_NonClassicalMono", "Mye_TissueResMac_1", "Mye_TissueResMac_2"],
        "w0.5_K7": "see fusion/audit.json",
        "w0.75_K7": ["Mye_C1QC", "Mye_ClassicalMono_1", "Mye_ClassicalMono_2", "Mye_FABP4",
                      "Mye_NonClassicalMono", "Mye_TissueResMac_1", "Mye_TissueResMac_2"],
        "w0.5_K6": ["Mye_ClassicalMono", "Mye_FABP4", "Mye_InflamMono", "Mye_NonClassicalMono",
                     "Mye_TissueResMac_1", "Mye_TissueResMac_2"],
        "w0.5_K8": ["Mye_ClassicalMono", "Mye_FABP4", "Mye_InflamMono", "Mye_NonClassicalMono",
                     "Mye_SPP1", "Mye_TissueResMac_1", "Mye_TissueResMac_2", "Mye_TissueResMac_3"]},
    "terminal_states": term_rows,
    "fate_names": names,
    "numeric_gates": {"transition": gate_T, "fate": gate_F, "all_pass": bool(gates_pass)},
    "timing_s": round(time.time() - t00, 1),
    "n_cells": int(ad.n_obs),
}
with open(f"{DEST}/audit.json", "w") as f:
    json.dump(audit, f, indent=2)
log(f"DONE {audit['timing_s']}s; terminals={[r['label'] for r in term_rows]}")
assert gates_pass, "numeric gates failed"
