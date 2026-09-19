# Worked example: paul15 mouse myeloid (2,730 cells × 3,451 genes)

Everything in `paul15/` was produced by this repository's own entrypoints,
zero code changes, from a working h5ad carrying the two direction-prior
columns. The h5ad itself is not redistributed; regenerate it with the two
prerequisite skills (the verified path, ~2 min):

| step | tool | writes |
|---|---|---|
| 1. paul15 raw counts | `sc.datasets.paul15()` (scanpy) | AnnData, raw counts |
| 2. Palantir pseudotime | [palantir-fate-analysis](https://github.com/LCGaoZzz/myskills/tree/main/palantir-fate-analysis) `run_palantir_pipeline.py` | `obs['palantir_pseudotime']` (+ `palantir_audit.json`) |
| 3. CytoTRACE2 scores | [cytotrace2-fast](https://github.com/LCGaoZzz/cytotrace2-fast) binary via the [cytotrace2-fast skill](https://github.com/LCGaoZzz/myskills/tree/main/cytotrace2-fast) recipe (species `mouse`, seed 14, single-batch regime) | `obs['CytoTRACE2_Score']` (+4 more columns) |
| 4. align priors by cell id on the same cell set | any pandas merge | `working.h5ad` |

Then the exact commands that produced the checked-in outputs:

```bash
python scripts/run_cellrank2_fusion.py \
    --h5ad working.h5ad --cluster-key paul15_clusters \
    --eigengap --seed 0 --out-dir examples/paul15/cr_out

python scripts/make_figures.py \
    --h5ad working.h5ad --cluster-key paul15_clusters \
    --w-ct2 0.5 --n-states 5 --threshold soft \
    --out-dir examples/paul15/figures
```

Environment: the `requirements-reproducibility.txt` pin (cellrank 2.0.7,
palantir 1.4.5, scanpy 1.11.5, Python 3.11). Absolute local paths inside
`cr_out/audit.json` were rewritten to repo-relative names for publication
(`_publication_note` marks this); all numeric fields are untouched run
output.

## `cr_out/` — the audited run (exit 0, ~25 s)

- `audit.json` — 5 macrostates / 5 terminal states ({3Ery, 7MEP, 11DC,
  15Mo, 19Lymph}, 30 cells each); numeric gates all green; edge conflict
  rate 0.465; `eigengap_run.auto_n_states = 4` recorded as the contrast
  column (explicit K=5 is the contract — see README);
  `ct2_monotone_nondecreasing = false` — the paul15 block-order inversion
  that makes 6Ery/13Baso/8Mk unreachable under ct2-heavy weights.
- `fate_probabilities.csv` (2,730 × 5 + id/cluster),
  `terminal_members.csv`, `transition_matrix.npz` (sparse, 2,730²),
  `fig_fate_umap.png` + `.csv`.

## `figures/` — the 16-type publication set (300 dpi, PNG+CSV each)

`figure_log.txt` records native-vs-fallback per figure: 12 rendered by
native cellrank 2.0.7 plotting, gene trends by the GAM(gaussian/identity)
path, macrostate composition and lineage priming by the manual renderers.
Driver sanity anchors reproduce: 11DC → *H2-Eb1 / H2-Aa / Cd74* (MHC-II),
19Lymph → *Ccl5 / Ctsw / Cd2*; `lineage_drivers_all.csv` carries the full
per-lineage driver table.
