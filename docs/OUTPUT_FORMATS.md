# Output data formats

Every artifact this repository's runner (`scripts/run_cellrank2_fusion.py`),
figure set (`scripts/make_figures.py`) and prerequisite skills write, with
its exact schema. Concrete filled examples for every file live in
[`examples/G1a_Myeloid_MonoTAM/data/`](../examples/G1a_Myeloid_MonoTAM/data/)
(production-scale case, 265,480 cells) and
[`examples/paul15/`](../examples/paul15/) (2,730 cells).

Two conventions hold everywhere:

- **Numeric gates.** Any probability-like matrix is finite, non-negative,
  and row-stochastic within 1e-3. `audit.json` records the measured
  deviation for both the transition matrix and the fate matrix; the runner
  exits non-zero if either fails.
- **Figure-data pairing.** Every figure PNG ships a same-name CSV of the
  plotted values. A figure without its CSV is a bug.

## 1. `fate_probabilities.csv` — the primary per-cell output

One row per cell; the model's central deliverable.

| column | type | meaning |
|---|---|---|
| `cell_id` | string | cell barcode / obs name (unique; contract-checked) |
| `<cluster_key>` | string | the annotation column passed as `--cluster-key` (here `final_subtype`) |
| one column **per terminal state** | float ∈ [0,1] | probability of absorption into that terminal state |

- Row sums = 1 ± 1e-3 (absorption probabilities over all terminals).
- Terminal-state column names are macrostate labels (annotation majority
  votes), e.g. `Mye_C1QC`, `Mye_TissueResMac_3`.
- **Reading rule:** these are model-conditional probabilities. Use the full
  row (e.g. mean per cluster), not only `argmax` — at scale one deep
  attractor can hold most argmax mass while the matrix structure stays
  informative.
- At production scale the file is gzip-compressed
  (`fate_probabilities.csv.gz`, 265,480 × 10 → 21 MB);
  `pd.read_csv(path)` reads it transparently.

Load and aggregate:

```python
import pandas as pd
fate = pd.read_csv("examples/G1a_Myeloid_MonoTAM/data/fate_probabilities.csv.gz")
by_subtype = fate.groupby("final_subtype")[fate.columns[2:]].mean()
```

## 2. `terminal_members.csv` / `terminal_states.csv` / `macrostates.csv`

- `terminal_members.csv` — `cell_id, terminal_state, final_subtype`; the
  member cells of each terminal macrostate (GPCCA membership, 30 cells per
  state by default). This is the file to read when checking what a
  terminal state *is*, since labels are majority votes, not clusters.
- `terminal_states.csv` (case-level summary) — `label, n_cells,
  majority_cluster, majority_fraction` per terminal state. In the G1a case
  all eight states have `majority_fraction = 1.0`.
- `macrostates.csv` (case-level summary) — `macrostate, majority_cluster,
  majority_purity` for every GPCCA macrostate, terminal or not (root
  macrostates included — that is how the excluded root basins stay
  auditable).

## 3. `audit.json` — provenance, config, and every audit

Single JSON, one block per concern. Field-by-field:

| key | contents |
|---|---|
| `input` | h5ad, n_cells, n_genes, cluster_key, n_clusters, NaN counts per prior column |
| `config` | palantir/cyto column names, w_ct2, n_states, threshold (b, ν), seeds, graph source, fate solver |
| `versions` | python, cellrank, scanpy, palantir, anndata, numpy, scipy, pandas, petsc4py/slepc4py |
| `kernel` | fusion formula, transition-matrix shape and nnz |
| `macrostates` / `terminal_states` | per-state `{label, n_cells, majority_cluster, majority_fraction}` |
| `eigengap_run` | the contrast run with auto `n_states=None`: chosen n, labels, error (on ct2-weighted kernels expect the documented collapse to n=1) |
| `numeric_gates` | finite / non-negative / row-sum≈1 for transition and fate matrices, with measured deviations; `all_pass` |
| `direction_conflict` | edge-level disagreement between the two priors on the kNN graph: overall rate, per-cluster table, tie handling |
| `block_means` | cluster-mean `palantir` and `ct2_time`, lineage order, monotonicity flags |
| `eigenvalues_top15_real` | GPCCA spectrum |
| `timing_sec` / `artifacts` | wall time per phase; sha256 + bytes per output file |

In published copies, absolute local paths are rewritten to repo-relative
names (`_publication_note` marks this); numeric fields are untouched run
output.

## 4. `transition_matrix.npz` — the fused Markov chain

`scipy.sparse.save_npz` of the **row-stochastic CSR** transition matrix of
the fused kernel, shape n_cells × n_cells (here 265,480², nnz ≈ 15.9M,
115.4 MB). Load with:

```python
from scipy.sparse import load_npz
T = load_npz("transition_matrix.npz")   # T[i].sum() == 1 ± 1e-3
```

Beyond GitHub's 100 MB file limit it is distributed as a Release asset
(sha256 recorded in the release notes and in `audit.json` → `artifacts`).

## 5. Figure set (16 types, PNG + same-name CSV)

Rendered by `make_figures.py`; native cellrank plotting first, manual
fallback per panel, `figure_log.txt` records which path served each file.

| file | CSV columns |
|---|---|
| `fig_macrostates` / `fig_terminal_states` | umap_1, umap_2, cluster, state |
| `fig_terminal_membership` | umap_1, umap_2, one membership column per terminal |
| `fig_fate_probabilities`(`_panels`) | umap_1, umap_2, one probability column per terminal |
| `fig_coarse_T` | the macrostate × macrostate coarse-grained matrix |
| `fig_spectrum` | rank, eigenvalue_real |
| `fig_macrostate_composition` | macrostate × annotation composition (%) |
| `fig_aggregate_fate` | annotation × terminal mean probabilities (＝ `data/fate_by_subtype.csv`) |
| `fig_lineage_drivers_<state>` | gene, corr |
| `fig_lineage_drivers_corr` | top-gene driver correlation across lineages |
| `fig_gene_trends_<state>` | pseudotime, predicted expression (GAM gaussian/identity; omitted at 265k scale) |
| `fig_circular_projection` | circ_x, circ_y, dominant_fate, priming, umap coords |
| `fig_lineage_priming` | umap coords + priming (1 − fate entropy) |

## 6. `lineage_drivers_all.csv` — per-lineage driver genes

Index = gene symbol (all genes in X); per terminal state two columns,
`<state>_corr` (correlation with fate probability) and `<state>_qval`
(false-discovery-adjusted), from a Fisher-z test on the log-normalized
layer. Compact browsing copy: `lineage_drivers_top50_per_lineage.csv`
(`lineage, gene, <state>_corr, <state>_qval`).

## 7. Prior artifacts (prerequisite skills)

- `palantir_priors.csv` — `cell_id, palantir_pseudotime ∈ [0,1]`
  (increases along differentiation), `palantir_entropy`, and one
  `fate_<state>` column per Palantir terminal; plus `palantir_audit.json`
  (terminal self-absorption — require > 0.7 — branch-mask sizes,
  fate × annotation crosstab).
- `cytotrace2_scores.csv` — `cell_id, CytoTRACE2_Score ∈ [0,1]`
  (higher = less differentiated), `CytoTRACE2_Potency`
  (Differentiated…Totipotent), `CytoTRACE2_Relative`,
  `preKNN_CytoTRACE2_Score`, `preKNN_CytoTRACE2_Potency`, and where
  applicable `ct2_imputed` (bool) marking scores rescued from the
  documented multi-batch NaN edge — see the case's `ct2_rescue.json` for
  the per-cell fill provenance.

## 8. Final h5ad (case workspace; not redistributed)

The working object after the full run carries: `obs` — both prior columns
plus `ct2_time = 1 − CytoTRACE2_Score`, `dominant_fate`,
`dominant_fate_prob`, `lineage_priming`, `macrostate`,
`terminal_state_member`; `obsm['fate_probabilities']` with column names in
`uns['fate_probabilities_columns']`; `X` = palantir-log-normalized
expression with raw counts in `layers['counts']`; the trajectory graph in
`obsp` (`connectivities`, k=30). Every numeric column above is byte-identical
to the corresponding CSV in `data/`.

## 9. Integrity

`audit.json → artifacts` carries sha256 per output file at run time;
case-level sha256 for every checked-in file (and for Release assets) are in
the case README's manifest. `REPRODUCIBILITY_SOURCE.sha256` at the repo
root covers the skill source files themselves (`sha256sum -c` from the
repo root).
