# Figures: the standard CellRank 2 publication set

Rendered by `scripts/make_figures.py` (evidence: campaign round R6,
`rounds/R6_figures/{D1,D2}/`, 16 figure types × 2 datasets, every PNG with a
same-name CSV). The script rebuilds the model with the exact same code path
as `run_cellrank2_fusion.py` — the pipeline is seed-deterministic (campaign
R* double runs: label/cell-set/fate Jaccard all 1.0), so the figures ARE the
saved run's figures; no estimator state needs to be serialized.

## The 16 figure types and what to read from them

| file | shows | interpretation guardrail |
|---|---|---|
| fig_macrostates | GPCCA macrostates (discrete) on UMAP | labels are annotation majority votes, not clusters |
| fig_terminal_states | terminal states (discrete) | check priors at mature extreme before trusting direction |
| fig_terminal_membership | continuous membership per terminal | soft membership ≠ hard assignment |
| fig_fate_probabilities(.png / _panels) | fate probability overview + one panel per terminal | model-conditional probabilities, not real rates |
| fig_coarse_T | coarse-grained transition matrix between macrostates (+ stationary dist) | the quantitative backbone of any UMAP-arrow claim |
| fig_spectrum | top-20 Re(λ) + eigengap marker | the eigengap criterion collapsed to n=1 on ct2-weighted kernels — read with references/workflow-sop.md §4 |
| fig_macrostate_composition | % annotation composition per macrostate | catches progenitor contamination of "terminal" states |
| fig_aggregate_fate | fate probability distributions per annotation cluster (violin + mean table csv) | lineages the model cannot reach show flat distributions — see block-monotonicity audit |
| fig_lineage_drivers_<lin> | top-8 driver genes (corr + qval) per lineage | computed on the working (log) layer via Fisher test |
| fig_lineage_drivers_corr | driver-gene correlation across lineages (top-20) | shared drivers hint at fused/primitive states |
| fig_gene_trends_<lin> | GAM trend of top-3 drivers along the pseudotime | GAM(gaussian/identity); Gamma/log diverges (pitfalls #16) |
| fig_circular_projection | cells inside the fate simplex, terminals on the circle | radius = lineage priming; classic CellRank hematopoiesis view |
| fig_lineage_priming | 1 − fate entropy on UMAP | high priming = committed, not "better" |
| lineage_drivers_all.csv | full driver table, all lineages | the plot is a view; the table is the evidence |

Sanity anchors from the campaign: D1 Beta-terminal drivers include Neurod1,
Alpha includes Cpe; D2 11DC drivers are H2-Eb1/H2-Aa/Cd74 (MHC-II), 19Lymph
are Ccl5/Ctsw/Cd2. If your terminal drivers do not match known lineage
markers, suspect the terminal, not the marker.

## Driving the script

```bash
<python> scripts/make_figures.py \
    --h5ad working.h5ad --cluster-key cell_type \
    --w-ct2 0.5 --n-states 5 --threshold soft --out-dir figs
```

Same argument names as the runner; 300 dpi PNGs land in `--out-dir` with
`figure_log.txt` recording native-vs-fallback per figure and every failure
verbatim.

## Native plotting conventions (cellrank 2.0.7, scvelo backend)

- Save with `save=<absolute path ending in .png>`; the plotting methods
  return None (they call `plt.show()`-equivalents), so there is nothing to
  hand to `savefig` — do not pass `return_fig` (AttributeError) or `s=`
  (keyword collision inside scvelo scatter).
- `plot_spectrum` sizes with `n=`, not `show_n_first`.
- `cr.pl.circular_projection` takes `keys=` + `lineages=`, not a lineage
  matrix; the script passes the cluster key and lets it consume the estimator
  state on the AnnData, with a manual circular embedding (weighted angle by
  fate probabilities, radius by priming) as fallback.
- Every native call is wrapped: on refusal the script re-renders the same
  panel from the model objects (memberships, fate probabilities, coarse_T,
  eigenvalues, driver table), so the figure set never silently degrades.

## Figure-data policy

Every `<name>.png` ships `<name>.csv`: scatter panels carry UMAP coordinates
plus the plotted color/membership columns; heatmaps carry the matrix; trends
carry the predicted curves; drivers carry the ranked table. A figure without
its CSV is a bug.
