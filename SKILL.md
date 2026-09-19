---
id: cellrank2-fate-analysis
name: cellrank2-fate-analysis
description: Fuse Palantir pseudotime and 1-CytoTRACE2 scores as direction priors into CellRank 2 GPCCA fate analysis, with audited graph/direction/terminal checks, block-monotonicity pre-checks, numeric gates and stability diagnostics.
tier: community
category: general_omics_analysis
summary: Campaign-verified CellRank 2 direction-fusion workflow from prior-bearing h5ad to audited fate probabilities.
execution_mode: packaged_python
runtime_entrypoint: scripts/run_cellrank2_fusion.py
---

# cellrank2-fate-analysis

CellRank 2 answers "what will each cell become?" by turning a KNN graph plus a
direction prior into an absorbing Markov chain. This skill runs the specific
two-prior fusion studied in the campaign: Palantir pseudotime (from
`palantir-fate-analysis`) plus `1 - CytoTRACE2_Score` (from `cytotrace2-fast`
scores), combined as a weighted sum of two soft-threshold PseudotimeKernels,
then GPCCA for macrostates / terminal states / fate probabilities — with the
audits that the campaign showed actually catch failures.

**Preparation-stage references (prerequisite skills, use them to produce the
prior columns):**

- https://github.com/LCGaoZzz/myskills/tree/main/palantir-fate-analysis —
  produces `palantir_pseudotime` and annotation-driven terminal-state evidence
  (`palantir_audit.json`).
- https://github.com/LCGaoZzz/myskills/tree/main/cytotrace2-fast — produces
  the five CytoTRACE2 score columns (`CytoTRACE2_Score`, ...).

Do not run this skill on log-transformed counts or on priors that were not
aligned by cell_id against the same cell set (the entrypoint refuses NaN or
missing priors). Do NOT also add CellRank's built-in `CytoTRACEKernel` on top
of the fused model: it re-implements CytoTRACE v1 internally, double-counts
the direction prior, and fails outright on log data or restricted gene panels
(see pitfalls #3/#4).

## Portable entrypoint

```bash
<python> scripts/run_cellrank2_fusion.py \
    --h5ad working.h5ad --cluster-key cell_type \
    --palantir-col palantir_pseudotime --cyto-col CytoTRACE2_Score \
    --n-neighbors 30 --threshold soft --w-ct2 0.5 --n-states 5 \
    --eigengap --seed 0 --out-dir cr_out
```

Exit codes: 0 = numeric gates passed; 1 = gate or run failure; 2 = input
contract violation (missing columns, NaN priors). Outputs land in `--out-dir`:
`audit.json` (terminals, macrostates, numeric gates, direction-conflict audit,
block monotonicity, spectrum, effective config + versions), `fate_probabilities.csv`,
`terminal_members.csv`, `transition_matrix.npz`, `fig_fate_umap.png` (+ csv).

The standard publication figure set is a second entrypoint (rebuilds the
same deterministic model, then renders 16 CellRank 2 figure types with a
same-name CSV each — see [figures](references/figures.md)):

```bash
<python> scripts/make_figures.py \
    --h5ad working.h5ad --cluster-key cell_type \
    --w-ct2 0.5 --n-states 5 --threshold soft --out-dir figs
```

## What the audits are for

- **Numeric gate (four checks)** — transition and fate matrices finite,
  non-negative, row sums ≈ 1 (1e-3). In the campaign every instability traced
  to state selection or weights, never to the solver; the gate is cheap and
  must stay green before any interpretation.
- **Direction-conflict audit** — edge-level disagreement between the two
  priors (overall and per-cluster, raw and high-|delta| filtered). Near-0.5
  conflict with near-0 rank correlation is normal and is NOT reversibility;
  read the block table before blaming the graph.
- **Block monotonicity table** — cluster-mean `ct2_time` along each lineage.
  If it decreases toward the mature end (as on paul15 short branches), that
  lineage is unreachable under ct2-weighted kernels at any weight: check this
  BEFORE choosing fusion weights.
- **Eigengap report** — `n_states=None` (auto) uses cellrank's `_eigengap`
  criterion (`J = gap - alpha*eps`, alpha=1.0, on the leading real eigenvalues
  of T). On ct2-containing kernels this can collapse to n=1 while the fixed-K
  model is perfectly healthy. Always pass an explicit `--n-states` for the
  primary model and record the auto choice for contrast.

## Defaults and why

`--n-states 5` explicit (auto degeneracy, pitfall #5); `--w-ct2 0.5` is a
starting point only — on paul15 the terminal set switches regime between
0.4 and 0.6, so sweep weights when a lineage is missing (pitfall #6);
`--threshold soft` (b=10, nu=0.5) because hard thresholding changed terminal
cell membership (Jaccard 0.58 vs reference) without changing labels.

Fate probabilities are "given this model" probabilities — not real transition
rates, not model correctness. Keep the parameters, graph, terminals and full
probability matrix from `--out-dir` for any replay.

## References on demand

- [Workflow SOP](references/workflow-sop.md): the audited path end to end,
  alignment rules, graph/kernel/GPCCA settings, stability-matrix template.
- [Pitfalls](references/pitfalls.md): 17 symptom → cause → fix entries with
  real traces (API renames, eigengap collapse, CytoTRACEKernel failures,
  NaN phantom terminals, parallel-kernel races, GAM divergence, plotting
  kwargs).
- [Figures](references/figures.md): the 16-type standard publication set,
  what each panel licenses you to conclude, native plotting conventions,
  and the figure-data (PNG+CSV) policy.
- [Runtime](references/runtime.md): pinned version matrix, install notes,
  solver fallbacks (brandts / scipy gmres), cytotrace2-fast binary build.
- [Campaign replay](references/campaign-replay.md): the 5-round harness loop,
  what was measured, what was refuted, where the evidence lives.
- [Boundaries](references/boundaries.md): applicability limits and open items
  (per-sample conflict, permutation test, third-dataset generalization).

Reproducibility contract: [contracts/reproducibility.v1.json](contracts/reproducibility.v1.json).
