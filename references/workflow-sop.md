# Workflow SOP: two-prior direction fusion in CellRank 2 (campaign-verified)

Campaign: `camp_1bcd6858e44d4d978a04a8da1040925c` (2026-09-18). Evidence root
for all trace paths below: the campaign output directory
`outputs/turn_20260918175839_7811a521c3b7463bbb36a49d0bde01cf/`.

## 0. What this workflow answers

Per-cell fate probabilities over terminal states, from a KNN graph plus TWO
direction priors instead of one: Palantir pseudotime (diffusion-geometry
clock) and `1 - CytoTRACE2_Score` (developmental-potential axis). The fusion
is a weighted kernel sum, not a score average, unless you deliberately choose
score-level fusion (see §6).

## 1. Produce the priors (prerequisite skills)

- `palantir-fate-analysis` (https://github.com/LCGaoZzz/myskills/tree/main/palantir-fate-analysis):
  run its pipeline first; you need `obs["palantir_pseudotime"]` and its
  `palantir_audit.json` (terminal self-absorption > 0.7 on both validation
  datasets). Pseudotime must INCREASE along differentiation.
- `cytotrace2-fast` (https://github.com/LCGaoZzz/myskills/tree/main/cytotrace2-fast):
  five score columns; use `CytoTRACE2_Score` (higher = less differentiated).
  Same input/seed/batch regime as any parity comparison.

Alignment red line: merge priors into ONE working adata by cell_id; record
row counts and order identity on both sides before and after (campaign
evidence: `rounds/R3_fusion/D{1,2}/alignment_check.json`). Never merge on
position without checking order; never proceed with NaN priors.

## 2. Graph build (same-source as Palantir)

`normalize_total(target_sum=None → median) → palantir.preprocess.log_transform
(= log2(x+0.1) − log2(0.1)) → HVG 1500 (cell_ranger) → PCA 50 (HVG) →
sc.pp.neighbors(n_neighbors=30, use_rep="X_pca", random_state=<seed>)`.

This replicates the geometry Palantir's diffusion maps were built on (knn=30
on the same PCA). Do not feed log values to CytoTRACE2, and do not feed
counts to Palantir steps. On very wide matrices cell_ranger HVG can throw
"Bin edges must be unique" — filter genes (min_cells=3) on a copy and record
it (trace: `rounds/R1_palantir/error_log.md` E2).

## 3. Kernels

```python
ad.obs["ct2_time"] = 1.0 - ad.obs["CytoTRACE2_Score"]
k_pal = cr.kernels.PseudotimeKernel(ad, time_key="palantir_pseudotime").compute_transition_matrix(threshold_scheme="soft", b=10.0, nu=0.5)
k_ct2 = cr.kernels.PseudotimeKernel(ad, time_key="ct2_time").compute_transition_matrix(threshold_scheme="soft", b=10.0, nu=0.5)
k = (1 - w) * k_pal + w * k_ct2          # KernelAdd, constants auto-normalize
```

Semantics verified from cellrank 2.0.7 source: `time_key` values increase
along differentiation; soft scheme is a generalized logistic
`2/(1+exp(b*Δt))^(1/ν)` on future-vs-past neighbors; `backward=True` equals
`max(pt) − pt` bitwise. The built-in `CytoTRACEKernel` subclasses
PseudotimeKernel with `ct_pseudotime = 1 − score` — kernel arithmetic is
IDENTICAL, so adding it to a fused model double-counts the prior and is
forbidden here (campaign probe: `rounds/R2_intel/probe_kernel_equivalence.json`,
max|ΔT|=0.0 for equivalent inputs; 0.372 when scores differ).

## 4. GPCCA

```python
g = cr.estimators.GPCCA(k)
g.compute_eigendecomposition()                       # required before n_states=None
g.compute_macrostates(n_states=K, cluster_key=<annotation>)   # K explicit, always
g.predict_terminal_states()
g.compute_fate_probabilities()
```

- v2.0.x names are `predict_terminal_states` / `compute_fate_probabilities`
  (1.5.x names do not exist; see pitfalls #1).
- Choose K from biology + spectrum, NOT from the eigengap auto heuristic
  alone: auto uses `_eigengap` (`J = gap − α·eps`, α=1.0, on leading real
  eigenvalues of T) and collapses to n=1 on ct2-weighted kernels on the
  pancreas data while the fixed-K model is healthy (8/8 auto choices
  reproduced from source in `rounds/R5_generalize/eigengap_reproduction.csv`).
- Macrostate labels are annotation majority votes — they are not clusters and
  not proof of terminal identity; read `terminal_members.csv`.

## 5. Numeric gate + audits (every run)

1. Four checks: T and fate matrices finite, non-negative, row sums within
   1e-3 of 1. Any failure → trace reachability/graph connectivity first; do
   not renormalize to mask it.
2. Direction-conflict audit: on the shared kNN graph's undirected edges,
   compare `pt_j > pt_i` vs `score_j < score_i`. Report overall rate,
   per-cluster table, and a high-|Δ| (above-median |Δpt|+|Δct2_time|)
   filtered version. Expect ~0.47–0.49 with near-zero rank correlation on
   messy data — that number is structural (block-order disagreement), not
   noise and not reversibility (both interpretations were tested and
   refuted; campaign replay §3).
3. Block monotonicity: cluster-mean `ct2_time` along each lineage's cluster
   sequence. A DECREASING mature end (paul15 13Baso: 0.777→0.605; 8Mk:
   0.874→0.698) means that lineage cannot become absorbing under
   ct2-weighted kernels — lower the ct2 weight or drop ct2 for that analysis
   (campaign: 13Baso only matched at w_ct2 ≤ 0.4).

## 6. Fusion weight policy

- Sweep w_ct2 over at least {0.25, 0.5, 0.75} plus the two singles; do not
  default to 0.5. On paul15 the terminal set switches regime between 0.4 and
  0.6 (5→3 terminals, fate-vs-reference ρ from +0.89 to −0.10;
  `rounds/R5_generalize/figures/fig_weight_sweep.png`).
- Score-level fusion (single time_key = weighted rank average) is a different
  model, not a shortcut: it collapsed to n=1 under auto selection on the
  pancreas data like every other ct2-containing kernel.

## 7. Stability matrix (report-grade runs)

Re-run the primary model under: graph seeds {1,42}, neighbors {15,50},
hard vs soft, weights ±0.25, K ±1, and 80% cell holdouts ×3. Compare to the
reference with terminal-label Jaccard, matched terminal cell-set Jaccard,
and matched-pair fate Spearman. Campaign expectations (both datasets):
seed axis exactly deterministic (Jaccard 1.0); neighbors/hard-soft move cell
membership more than labels; holdouts land at cellJ 0.66–0.78; the one axis
that can destroy the result outright is auto state selection.

## 8. Reproducibility triple

Versions (contract lock file), seeds (graph/PCA seed + GPCCA path),
and the output directory contents (parameters, graph, terminals, full
probability matrix). Same seeds reproduced bit-identical fate matrices in
campaign (R* double runs, Jaccard 1.0 on both datasets).
