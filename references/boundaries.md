# Boundaries and open items

Validated scope (and nothing beyond it):

- **Datasets:** mouse pancreas epithelium, 2,850 cells, 8 annotations
  (downsampled 10x, cytotrace2 vignette source) and paul15 mouse myeloid,
  2,730 cells × 3,451 genes (scanpy). Both raw counts. Other lineages,
  species, and panel-restricted assays: run the audits before believing the
  biology. No third-dataset gate was run (per the task brief, optional).
- **Versions:** behavior verified on cellrank 2.0.7 / palantir 1.4.5 /
  scanpy 1.11.5. 1.5.x cellrank has different method names; ≥2.1 untested.
  Fate probabilities can shift materially across palantir versions (RNG
  changed to PCG64 at 1.4.5).
- **Interpretation limits:** fate probabilities are model-conditional — they
  are neither measured transition rates nor model accuracy. Macrostate
  labels are annotation majority votes. On D1, single-prior Palantir models
  placed progenitor clusters (MPP/EP) among terminal states under auto
  selection — a direction/selection symptom the audits flag, not biology.
- **No per-sample auditing:** the validation data carry no sample/donor
  columns (D1 barcode suffixes undocumented, paul15 is one experiment), so
  conflict rates are per-cluster only. With multi-sample data, per-sample
  conflict and per-sample score rescaling (forbidden here) become checkable.
- **Built-in CytoTRACEKernel:** excluded by design (double direction +
  double inversion risk). It additionally fails on log data (all-NaN gene
  correlations) and on restricted panels ("No genes have been selected") —
  the external CytoTRACE2 route is the only one that worked on paul15.
- **Auto state selection:** the eigengap criterion is documented and
  reproduced (8/8) but not "fixed"; it remains wrong-prone on ct2-weighted
  kernels. Explicit K is the contract.
- **Open scientific item:** within-cluster edge-delta permutation (preserve
  cluster means) was pre-registered as a discriminator and never executed;
  the campaign closed with the mechanism question it targets (coarse-block
  vs fine-edge carrying of absorption structure) answered only indirectly.
- **Residual from logistics:** a temporary private GitHub repo
  `LCGaoZzz/d1-drive-fetch-tmp` (used to fetch D1 from Google Drive via a
  runner) could not be deleted with the available token (403) — delete
  manually.
