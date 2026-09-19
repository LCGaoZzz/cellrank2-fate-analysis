# Pitfalls: symptom → cause → fix (all traces real, campaign 2026-09-18)

Trace root: `outputs/turn_20260918175839_7811a521c3b7463bbb36a49d0bde01cf/`.

1. **AttributeError / missing method on GPCCA (1.5.x names).** Symptom:
   `set_terminal_states_from_macrostates` / `compute_absorption_probabilities`
   do not exist. Cause: renamed in 2.0.x to `predict_terminal_states` /
   `compute_fate_probabilities` (source-verified, `rounds/R2_intel/intel_src.md`).
   Fix: use 2.0.x names; pin cellrank and record the version.

2. **"Compute eigendecomposition first as '.compute_eigendecomposition()' or
   supply 'n_states != None'."** Cause: calling `compute_macrostates(n_states=None)`
   without a prior eigendecomposition. Fix: `compute_eigendecomposition()`
   then `compute_macrostates(...)`. Trace: R3 attempt-1 errors, both datasets.

3. **CytoTRACEKernel on log-transformed X: "Top '200' positively correlated
   genes contain '200' NaN values."** Cause: v1 CytoTRACE score needs raw
   counts — on palantir-log data every gene is expressed in every cell, the
   gene-count variance is 0 and correlations are all NaN. Fix: put raw counts
   in `.raw` and call `compute_cytotrace(layer="X", use_raw=True)`. Better:
   don't use CytoTRACEKernel at all — fuse external CytoTRACE2 scores via
   PseudotimeKernel (arithmetically identical). Trace:
   `rounds/R3_fusion/D1/models/G5_ctk_v1/error_attempt3.txt`.

4. **CytoTRACEKernel on restricted gene panels: "No genes have been
   selected."** Cause: the internal top-200 gene selection fails on a
   3,451-gene panel (paul15) even with the raw-count fix. Fix: none within
   the built-in kernel — use cytotrace2-fast scores + PseudotimeKernel.
   Trace: `rounds/R4_stability/D2/runs/g5_ctk_soft_rawfix/error.txt`.

5. **Auto eigengap state selection collapses to n=1 on ct2-weighted
   kernels.** Symptom: single terminal state, fate Spearman vs reference
   flips negative (−0.74), while the same kernel with explicit K=5 is
   healthy (ρ 0.88–0.98). Cause: cellrank's `_eigengap` criterion
   (`J = gap − α·eps`, α=1.0; `cellrank/_utils/_utils.py:595`) rewards the
   gap right after λ1 on these spectra; reproduces 8/8 from source
   (`rounds/R5_generalize/eigengap_reproduction.csv`). Not basin collapse,
   not block ordering — both were tested with interventions and refuted.
   Fix: explicit `--n-states` for the primary model, keep the auto run as a
   contrast column.

6. **Terminal set switches regime inside the weight sweep.** Symptom
   (paul15, K=8): w_ct2 ≤ 0.4 → 5–6 terminals incl. 13Baso; w_ct2 ≥ 0.6 →
   3 terminals {19Lymph, 7MEP, 11DC}, fate-vs-reference ρ negative. Cause:
   ct2 block-order inversion on short branches (see #7) crosses a mixture
   threshold. Fix: sweep weights, report the boundary; never ship a single
   0.5 run. Trace: `rounds/R4_stability/D2/stability_matrix.csv` wscan rows,
   `rounds/R5_generalize/figures/fig_weight_sweep.png`.

7. **CytoTRACE2 block means inverted on short branches.** Symptom: 6Ery /
   13Baso / 8Mk never become terminal under any ct2-heavy kernel. Cause:
   cluster-mean `ct2_time` DEcreases toward the mature end (13Baso
   0.777→0.605, 8Mk 0.874→0.698; `rounds/R4_stability/D2/blocks_monotonicity.csv`)
   — on a shallow 3,451-gene dataset the score axis is locally backwards.
   Fix: pre-check block monotonicity (the entrypoint prints the table);
   lower ct2 weight or rely on the Palantir prior for those lineages.

8. **Edge direction conflicts ≈ 0.47–0.49 with near-zero rank correlation.**
   Symptom: the two priors look orthogonal at edge level. Cause: structural
   (block-order disagreement + plateau zones), NOT noise and NOT
   reversibility — the noise-floor reading was refuted by high-|Δ| filtering
   (mature-cluster conflicts rose 0.57→0.67 / 0.58→0.65 after filtering;
   `rounds/R4_stability/D{1,2}/direction_conflict_filtered.json`). Fix:
   report it, gate on the block table, don't narrate UMAP arrows.

9. **NaN phantom terminal state.** Symptom: a 2,700-cell `'nan'` terminal
   appears in members CSV. Cause: cellrank 2.0.7 `terminal_states` is a
   full-length Series (NaN for non-members); naive string conversion turns
   NaN into a label. Fix: `pd.isna` guard when extracting members (fixed in
   the shipped entrypoint during the R5 zero-change run).

10. **"Bin edges must be unique" from cell_ranger HVG.** Cause: ~28k-gene
    matrix with massive mean ties. Fix: `filter_genes(min_cells=3)` on a
    copy, record cell-set invariance. Trace: `rounds/R1_palantir/error_log.md` E2.

11. **anndata write fails with `IORegistryError: No method registered for
    writing <class 'tuple'>`.** Cause: tuples stored in `uns` (check
    reports). Fix: stringify uns entries before `write_h5ad`. Trace:
    `rounds/R1_cytotrace2/D2_run.log`.

12. **Palantir auto dimensionality silently drops terminals.** Symptom: 4
    UserWarnings "No valid component found ... will be skipped", 3/7
    terminals returned, audit still `passed=true` (it only audits anchors
    actually used). Fix: enumerate expected terminals vs found; raise n_dc /
    n_eigs (paul15 needed 20/20) and re-run. Trace:
    `rounds/R1_palantir/D2/attempt1_partial3terminals/`.

13. **Marker-argmax start cell lands in a terminal cluster.** Symptom:
    Hoxa9 argmax cell sits in 15Mo; four pancreas markers' argmax all
    outside the progenitor compartment. Cause: global argmax ignores
    compartment. Fix: restrict argmax to progenitor clusters (or use
    min-max-normalized marker sum within the compartment). Trace:
    `rounds/R1_palantir/D{1,2}/run_record.json`.

14. **`'Lineage' object has no attribute 'var_names'`.** Cause: cellrank
    lineage containers renamed accessors in 2.0.x. Fix: read column names
    via `.names`. Trace: R4 execution notes (`rounds/R4_stability/D2/`).

15. **Parallel workers sharing one kernel corrupt each other's outputs.**
    Symptom: error files written into another worker's round directory
    (global `OUT` variable overwritten mid-run). Fix: run heavy grids as
    standalone scripts via subprocess with literal paths, no shared
    kernel state. Trace: R3 incident report in `rounds/R3_fusion/D1/`.

16. **pygam GAM diverges on zero-inflated log-expression.** Symptom:
    `RuntimeError: Fatal model failure <FailedModel[origin=GAM[...,
    model=GammaGAM(...]]>` during gene-trend fitting. Cause: the GAM default
    Gamma family + log link requires strictly positive y; palantir log2
    (`log2(x+0.1)-log2(0.1)`) contains exact zeros. Fix:
    `cr.models.GAM(ad, distribution="gaussian", link="identity")`, with
    `SKLearnModel(GradientBoostingRegressor)` as the last resort. Trace:
    R6 first pass, `rounds/R6_figures/D1/figure_log.txt`.

17. **cellrank 2.0.7 plotting kwargs fight the scvelo backend.** Symptom:
    `PathCollection.set() got an unexpected keyword argument 'return_fig'`,
    `scatter() got multiple values for keyword argument 's'`, and
    `plot_spectrum` ignores `show_n_first`. Fix: the plotting methods return
    None — save via `save=<absolute path ending .png>`, never pass
    `return_fig` or `s=`, size the spectrum with `n=`. Wrap every call and
    re-render the same panel from the model objects on refusal
    (`scripts/make_figures.py` does exactly this). Trace: R6 first pass
    figure_log.txt; native-vs-fallback is logged per figure on every run.

18. **Relative `save=` paths silently re-route for `cr.pl.*` but not
    `g.plot_*`.** Symptom: `cr.pl.circular_projection` completes without
    error yet no file appears at the given path; the PNG lands under
    `./figures/<your-relative-path>.png` (scvelo prepends `settings.figdir`
    to non-absolute `save`), while estimator methods (`g.plot_macrostates`,
    `g.plot_spectrum`, ...) honor relative paths. Cause: two different save
    code paths behind the same keyword. Fix: resolve `--out-dir` to an
    absolute path once at startup (`make_figures.py` now does
    `os.path.abspath(out_dir)`), and treat "native call returned but the
    file is missing" as a hard fallback trigger. Trace: public-repo smoke
    run 2026-09-19, `figure_log.txt` + stray `./figures/` tree.
