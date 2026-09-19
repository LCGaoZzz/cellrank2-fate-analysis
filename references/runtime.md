# Runtime: install, versions, solver fallbacks (campaign 2026-09-18)

## Verified version matrix (reproducibility pin)

| package | version |
|---|---|
| Python | 3.11.15 |
| cellrank | 2.0.7 (PyPI 2025-04-07; latest 2.3.3 untested here) |
| palantir | 1.4.5 |
| mellon / jax / jaxopt | 1.7.1 / 0.7.1 / 0.8.5 |
| scanpy / anndata | 1.11.5 / 0.11.4 |
| numpy / scipy / pandas | 2.2.6 / 1.16.3 / 2.3.3 |
| scikit-learn / igraph / umap-learn | 1.7.2 / 1.0.0 / 0.5.12 |

Notes: the palantir-fate-analysis skill's baseline pins jax 0.10.2; this
environment ran 0.7.1 with no numerical incident on either dataset — record
whichever you run. Install: `pip install palantir cellrank` (host index;
no conflicts observed).

## Solver fallbacks

- Without petsc4py/slepc4py, GPCCA eigendecomposition falls back to
  `method='brandts'` (cellrank WARNING about densifying the sparse matrix —
  fine at ≤3k cells; budget memory beyond that).
- Fate probabilities solve `(I−Q)X = S`; the entrypoint forces scipy `gmres`
  (`use_petsc=False`, tol 1e-6) so runs do not depend on PETSc availability.
  Negative values or row-sum deviation > 1e-3 are hard errors in cellrank,
  surfaced by the numeric gate.

## CytoTRACE2 side (prerequisite skill runtime, summarized)

`cytotrace2-fast` v1.2.0 @ d334c0eb builds from
https://github.com/LCGaoZzz/cytotrace2-fast.git with a stable Rust toolchain
(`cargo build --release`; campaign build log:
`env/cytotrace2-fast-src/`, 39.6 s). Assets resolve from the checked-in
`assets/MANIFEST.json`. Parity vs the official Python vignette output:
Spearman 1.0, max|Δ| 1.7e-9, potency agreement 100% (same input, seed 14,
single-batch regime). The official R-package CSV differs (ρ≈0.964) — never
mix regimes in a parity claim. That single-batch regime is a vignette-scale
convention only (≤30k cells): diffusion smoothing takes ~n² memory, and a
265k-cell single-batch run OOM'd at 483 GB (2026-09-19, killed the backend
service with it). On production-scale data produce the score column with
bounded batches (`batch_size=50000` or defaults) — see the cytotrace2-fast
skill's pitfalls #11; the recipe refuses single-batch beyond 30k cells.

## Determinism

With pinned versions and fixed seeds, the full fusion → GPCCA → fate path is
deterministic: reference double-runs reproduced terminal sets and fate
matrices exactly (Jaccard 1.0, both datasets). Graph-build seeds (PCA /
neighbors random_state) changed nothing on either dataset at n=3k.

## Compute cost (validation hardware)

- Full entrypoint run: ~23–27 s per dataset (2.7–2.9k cells, 50 PCs).
- R3 nine-model grid: ~8 s/model. R4 stability matrix: 20–26 runs in
  125–154 s total.
