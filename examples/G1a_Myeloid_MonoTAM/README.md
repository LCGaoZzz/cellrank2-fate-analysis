# Production-scale case: pan-cancer Mono→TAM atlas (265,480 cells × 19,697 genes)

Everything in this directory is real run output from one execution of this
repository's methodology on a 97×-larger, batch-structured dataset than the
paul15 worked example: a pan-cancer myeloid atlas (7 cancer types, 22
datasets, 19 cohorts, 392 donors, 648 samples, 10 platforms; Tumor 53% /
Blood 23% / Metastasis 11% / Adjacent 11% / PreLesion 1.5% / PleuralFluids
0.5%), restricted to the classical-monocyte → tumor-associated-macrophage
compartment. Raw integer counts; **no spliced/unspliced layers** — the
two-prior pseudotime fusion route is the only applicable CellRank route
here, which is exactly what this repo is for.

The final result tables are checked in under `data/` (schemas:
[`docs/OUTPUT_FORMATS.md`](../../docs/OUTPUT_FORMATS.md)); the 265,480²
sparse transition matrix exceeds GitHub's 100 MB file limit and ships as a
[Release asset](https://github.com/LCGaoZzz/cellrank2-fate-analysis/releases/tag/case-g1a-monotam)
(sha256 in the release notes).

| | paul15 example | this case |
|---|---|---|
| cells × genes | 2,730 × 3,451 | 265,480 × 19,697 |
| batch structure | none | 22 datasets / 648 samples |
| graph | built by entrypoint (PCA50) | reused upstream Harmony k=30 (see §1) |
| eigendecomposition | `brandts` fallback | PETSc/SLEPc krylov (mandatory ≥100k cells) |
| CytoTRACE2 regime | single-batch (~min) | bounded batches `--batch-size 50000`, 7.9 h |
| terminal selection | default `predict_terminal_states` | explicit mature-macrostate selection (§4) |
| runner wall time | ~25 s | 252 s (K=7) / 368 s (final K=10) |

## 1. Input and the batch decision

The object ships a precomputed `X_pca_harmony` (23 components) and a k=30
neighbors graph built on it. Both were **verified, then reused**:

| check (20k-cell subsample, seed 0) | uncorrected `X_pca` | `X_pca_harmony` |
|---|---:|---:|
| silhouette by `datasetID` (22 levels) | −0.0429 | **−0.1236** (better mixing) |
| silhouette by `final_subtype` (7 levels) | 0.1887 | **0.2108** (better separation) |
| graph connectivity | — | 1 component, 100% of cells |

Harmony improves batch mixing and biological separation simultaneously, so
the trajectory geometry reuses it: the Palantir diffusion maps and the
CellRank kernel graph are built on the **same** representation (the SOP's
same-source rule), CytoTRACE2 runs on raw counts per its own contract.
Evidence: `data/batch_check.json`, `figures/fig_batch_mixing.png`.

## 2. The two direction priors

- **Palantir** (`palantir-fate-analysis`, public-API mode on the Harmony
  diffusion space; n_dc=15, knn=30, seed=20). Root = S100A8-argmax cell
  restricted to the atlas' own `traj_role == root_anchor` compartment
  (classical monocytes, 49,614 cells); terminals annotation-driven.
  Audit **passed**: terminal self-absorption 0.879–1.000 (threshold 0.7),
  no degenerate branches (`data/palantir_audit.json`).
- **CytoTRACE2** (`cytotrace2-fast` v1.2.0 binary; species `human`,
  seed 14, `--batch-size 50000`; 7.9 h). Input check: 100% integer counts,
  feature hit rate 0.7081, 0 duplicate cells. 20/265,480 scores came back
  NaN — the documented multi-batch × RNG edge (`data/ct2_rescue.json`):
  14 filled from `preKNN_CytoTRACE2_Score` (same scale, the skill's
  score-preserving view), 6 by Harmony-graph neighbor mean (the same
  smoothing operator), every filled cell flagged `ct2_imputed`, cell set
  unchanged.

## 3. Fusion and GPCCA at 265k cells

`0.5·K_palantir + 0.5·K_ct2` (soft scheme b=10, ν=0.5; `ct2_time =
1 − CytoTRACE2_Score`) on the reused Harmony k=30 graph; GPCCA with
petsc4py/slepc4py 3.22.4/3.22.2 krylov eigendecomposition; fate solve
scipy `gmres`. Numeric gates green (transition row-sum dev 6.7e-16, fate
row-sum dev 1.6e-5 vs tol 1e-3). Direction-conflict rate **0.482**
(per-cluster 0.459–0.501) — the structural block-order band, not
reversibility. Block monotonicity: `ct2_time` rises along the
pseudotime-ordered lineage sequence
`ClassicalMono → NonClassicalMono → InflamMono → SPP1 → C1QC → FABP4 →
TissueResMac` with a single small decrease at FABP4 (−0.0277) → `w_ct2`
kept at 0.5 (`data/block_monotonicity.csv`). Eigengap auto-selection
collapsed to n=1 (pitfall #5 at production scale) — explicit K throughout.

## 4. Model selection (weight × K sweep, 8 measured configurations)

Coverage = how many of the 6 mature annotated subtypes appear as a
terminal-state label (root excluded):

| config | terminals found | coverage |
|---|---|---:|
| palantir only (w=0), K=7 | ClassicalMono + 4 mature | 4/6 |
| ct2 only (w=1), K=7 | ClassicalMono + 5 mature (no SPP1) | 5/6 |
| fused w=0.25, K=7 | ClassicalMono + 4 mature (no SPP1, no C1QC) | 4/6 |
| fused **w=0.5**, K=7, default selection | root basin selected terminal; SPP1 impure (0.567), no C1QC | 5/6 |
| fused w=0.75, K=7 | C1QC separates; InflamMono and SPP1 lost | 4/6 |
| fused w=0.5, K=6 | no SPP1, no C1QC | 4/6 |
| fused w=0.5, K=8 | SPP1 present but C1QC-mixed; no C1QC | 5/6 |
| fused **w=0.5, K=10 + explicit mature terminals** (final) | 8 mature macrostates, all purity 1.0, root excluded | 5/6 |

Two degeneracies drove the final choice:

1. **The default `predict_terminal_states` selected the classical-monocyte
   root basin as a terminal state in every single configuration** (a
   61k-cell strongly metastable basin). Fix: GPCCA-native
   `set_terminal_states(names=<mature macrostates>)` — the two
   ClassicalMono-majority macrostates are excluded from the terminal set.
2. **C1QC (the largest subtype, 101,857 cells) never separates from the
   SPP1 basin at K ≤ 8** (their mean Palantir times are nearly identical,
   0.491 vs 0.486); at K=10 it separates at purity 1.0. Cost recorded: at
   K=10 SPP1 is not a standalone macrostate — its fate mass splits toward
   C1QC / FABP4 / TissueResMac_3 (0.234 / 0.197 / 0.333). No configuration
   reached 6/6; the SPP1↔C1QC exclusivity is the honest resolution ceiling
   of this dataset at K ≤ 10.

Full sweep machine output: `data/sweep_summary.json`; final model audit:
`data/audit.json`.

## 5. Results

**Terminal states** (30 member cells each, all majority purity 1.0 —
`data/terminal_states.csv`, `data/terminal_members.csv`):
Mye_C1QC · Mye_FABP4 · Mye_InflamMono_1 · Mye_InflamMono_2 ·
Mye_NonClassicalMono · Mye_TissueResMac_1/2/3.

**Fate mass by subtype** (`data/fate_by_subtype.csv`, row = annotated
subtype, mean probability):

| subtype | C1QC | FABP4 | Inflam_1 | Inflam_2 | NonCl | TRM_3 |
|---|---:|---:|---:|---:|---:|---:|
| ClassicalMono (root) | .184 | .134 | .113 | .017 | .117 | .266 |
| NonClassicalMono | .106 | .070 | .212 | .008 | **.354** | .153 |
| InflamMono | .218 | .154 | .081 | .021 | .024 | .309 |
| SPP1 | .234 | .197 | .020 | .007 | .006 | .333 |
| C1QC | **.255** | .124 | .016 | .004 | .005 | .368 |
| FABP4 | .098 | **.655** | .009 | .002 | .003 | .145 |
| TissueResMac | .064 | .028 | .004 | .001 | .001 | **.607** |

Direction: root = ClassicalMono (Palantir 0.117, highest CytoTRACE2
potency) → mature end = TissueResMac (Palantir 0.748). Driver genes
reproduce every lineage's canonical markers (`data/top_drivers_digest.json`,
full table `data/lineage_drivers_all.csv.gz`, compact
`data/lineage_drivers_top50_per_lineage.csv`): C1QC → *GPR183/A2M/RNASE1/
LGMN*; FABP4 → *FABP4/GPD1/MCEMP1*; InflamMono_2 → *S100A8/S100A12/S100A9/
CSF3R*; NonClassicalMono → *CX3CR1/FCN1/CORO1A*; TissueResMac → *FOLR2/
CD5L/SLC40A1/C1QC/CETP/CXCL12*.

**Robustness spot-checks** (all from the shipped object):
- The deep-macrophage attractor TRM_3 holds 84.9% of dominant-fate argmax
  — but its per-cohort mean fate is uniform across all 19 cohorts
  (0.254–0.388, `data/fate_TissueResMac3_by_cohort.csv`): a systematic
  property of the TAM-continuum mass, not a batch artifact. Read the
  probability matrix, not the argmax.
- The three TissueResMac macrostates are 100% / 87% / 100% Adjacent-tissue
  members (Kupffer-like niche), so the split is program- or donor-structure
  within one niche, not tissue stratification.
- Root cells' fate mass spreads across all eight terminals (top: TRM_3
  .266, C1QC .184, FABP4 .134) — no collapse onto a single lineage.

## 6. Downstream input contract — the per-cell interface

For downstream methods that need per-cell pseudotime + fate + sample
metadata in one place (e.g. sample-level fate-weighted sub-densities
`f_hat_slr(u) = (1/n_sl) * sum_i q_ir * kappa_h(u | tau_i)`), `data/`
carries the complete interface — nothing else needs recomputing:

- **`per_cell_interface.csv.gz`** (265,480 × 19, keyed by `cell_id`, joins
  1:1 to `fate_probabilities.csv.gz`): `sample_id, donor_id, dataset_id,
  cohort_id, cancer_type, sample_type, tissue` (verbatim from obs, mapped
  `sampleID→sample_id, donorID→donor_id, datasetID→dataset_id,
  cohortID→cohort_id, cancerType→cancer_type, sampleType→sample_type` — no
  field inferred from barcode format), `lineage_id = G1_Myeloid`,
  `state_domain_id = G1a_Myeloid_MonoTAM`, **`palantir_pseudotime`** and
  `palantir_entropy` (the per-cell tau), `CytoTRACE2_Score` +
  `ct2_imputed` (20 rescued cells; per-cell provenance in
  `ct2_rescue.json`), `final_subtype`, `macrostate`, `dominant_fate`,
  `model_id`, `reference_id`.
- **`sample_lineage_coverage.csv`** — planned-universe coverage over the
  atlas' **653** samples: 648 observed in this state domain with cell
  counts, **5 absent** (recorded `observed=false` + `exclusion_reason`;
  missing ≠ zero). The sampling unit is `sample_id` (653 samples / 392
  donors — same donor, different tissues stay separate observations).
- **`consistency_check.json`** — the fate CSV (runner build) and the final
  h5ad (rebuilt for the figure set) are two deterministic builds of the
  same model: same cell order and fate columns, **max abs fate difference
  1.1e-16**, terminal member sets identical. The single-source concern is
  closed by measurement, not by parameter identity.
- Reading rule for Q: the 8 declared terminal states are the complete
  probability space. No standalone SPP1 terminal is modeled (its mass
  splits toward C1QC/FABP4/TRM_3) — downstream outputs must not be
  narrated as containing an SPP1 fate.

## 7. What is claimed and what is not

Fate probabilities are model-conditional quantities over a cross-patient,
Harmony-integrated atlas — transcriptional state-transition propensity, not
measured within-patient cell flow, and not real transition rates. Root and
terminal anchors were annotation-driven, so the axis quantifies the
monocyte→TAM hypothesis rather than discovering it. InflamMono and the TRM
trio are metastable model states, not proven self-maintaining fates. The
SOP §7 stability matrix (graph seeds / neighbors / holdouts) was not run —
budget went to the 7.9 h CytoTRACE2 prior. GAM gene-trend figures are
omitted at this scale (pygam infeasible at 265k; the paul15 example carries
them).

## 8. Directory contents

- `data/` — final result tables + audits, one file per artifact; every
  schema in [`docs/OUTPUT_FORMATS.md`](../../docs/OUTPUT_FORMATS.md). The
  two large tables are gzip-compressed; the 115.4 MB
  `transition_matrix.npz` is a Release asset. The downstream
  per-cell interface (§6): `per_cell_interface.csv.gz`,
  `sample_lineage_coverage.csv`, `consistency_check.json`.
- `figures/` — six representative PNGs (fate UMAP, terminal states,
  macrostate composition, aggregate fate, fate panels, batch mixing); the
  full 16-type set with per-figure CSVs lives in the run workspace.
- `scripts/` — the exact code that ran, including the two documented
  adaptations of the stock entrypoint: `fusion_harmony.py` (graph/UMAP
  reused from the upstream Harmony representation instead of rebuilding an
  uncorrected PCA50 graph — every other line identical to
  `scripts/run_cellrank2_fusion.py`) and `run_final.py` (K=10 probe +
  explicit mature-terminal selection). `make_working.py` (prior merge +
  alignment red line) and `final_export.py` (figures + final h5ad) are
  included for completeness.

Environment: cellrank 2.0.7, palantir 1.4.5, scanpy 1.11.5, anndata 0.11.4,
petsc4py/slepc4py 3.22.4/3.22.2, Python 3.11, 224-core / 503 GB host.
Timings: Palantir 244 s · CytoTRACE2 7.9 h · K=7 run 252 s · final K=10
model 368 s · figure set 516 s.
