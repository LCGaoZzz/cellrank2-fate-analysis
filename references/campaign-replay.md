# Campaign replay: how this skill was learned (2026-09-18)

Campaign `camp_1bcd6858e44d4d978a04a8da1040925c`, run as a strategy-layer
loop over role agents (theory / hypothesis / experiment, evaluator blind to
expectations), with a pre-registered prediction file before each execution
round. Evidence root: `outputs/turn_20260918175839_7811a521c3b7463bbb36a49d0bde01cf/`.

## Rounds

1. **Prerequisites (R1).** cytotrace2-fast built from source, run on mouse
   pancreas D1 (2850×27998) and paul15 D2 (2730×3451); parity vs the official
   Python vignette output Spearman 1.0 / max|Δ| 1.7e-9 / potency 100%
   (`rounds/R1_cytotrace2/parity_report.json`). Palantir pipeline run on both
   datasets, annotation-driven anchors, audits passed (min self-absorption
   0.855) (`rounds/R1_palantir/`). Two silent failures caught: auto-eigengap
   terminal dropping (3/7 found, audit still passed), and marker-argmax start
   cells landing in terminal compartments.
2. **Intel (R2).** Paper + docs + installed-source + v2.0.7 tests. Two
   corrections to the task brief: the CellRank 2 citation is Weiler et al.,
   Nature Methods 21(7):1196–1205, 2024, doi:10.1038/s41592-024-02303-9
   (NOT btae426); 2.0.x renamed the GPCCA methods. A read-only probe proved
   `backward=True` ≡ manual `1−timekey` bitwise
   (`rounds/R2_intel/probe_kernel_equivalence.json`).
3. **Direction & fusion grid (R3).** 9 models × 2 datasets (singles ×2
   thresholds, DM-graph variant, built-in CytoTRACEKernel, kernel-add joints
   at 0.25/0.5/0.75, score-level fusion). Numeric gates 17/17 pass. Findings:
   edge conflict rates 0.474/0.491 with rank correlation −0.089/−0.006; on
   D1 every ct2-containing kernel collapsed under auto eigengap (n=1) while
   fixed-K=5 was healthy; ct2 single prior matched Palantir's 15Mo at ρ=0.92.
4. **Stability matrix + adjudication (R4).** 46 runs across both datasets:
   seeds/neighbors/threshold/weights/K/holdout + uniform-kernel baseline,
   cluster-mean ε-reorder intervention, weight sweep, high-|Δ| conflict
   filtering, kernel spectra. Reference config reproduces bit-identically.
   Adjudications: graph-sparsity and block-reorder interventions did NOT
   rescue the D1 degeneracy; the noise-floor reading of conflicts was
   refuted on both datasets; paul15 short-branch ct2 block inversion
   quantified (`blocks_monotonicity.csv`).
5. **Generalization + criterion verification (R5).** This skill's entrypoint,
   zero parameter changes, run on both datasets: exit 0, all gates green,
   the D1 auto-eigengap degeneracy reproduced as documented. The auto
   selection criterion was located in source (`cellrank/_utils/_utils.py:595`,
   `J = gap − α·eps`) and reproduced all 8 recorded state counts
   (`rounds/R5_generalize/eigengap_reproduction.csv`).

## Theory ledger outcome (what the evidence decided)

- **Confirmed:** selection-layer dominance (fixed-K restores stability;
  eigengap criterion mechanically explains the n=1 collapses); kernel
  arithmetic equivalence (built-in CytoTRACEKernel ≡ PseudotimeKernel on
  1−score, differences are score source + input scale); block-order
  inversion governs which lineages are reachable under ct2-weighted kernels
  and fusion weights act through a regime boundary (D2: between 0.4 and 0.6).
- **Refuted by forward tests:** graph-source/compatibility dominance;
  single-prior sufficiency; conflicts-as-reversibility; conflicts-as-noise-
  floor; basin-collapse and block-reorder accounts of the D1 degeneracy.
- **Open:** the permutation invariance test (shuffle within-cluster edge
  deltas, preserve cluster means) was never run — it is the cleanest remaining
  discriminator for how much of the absorption structure the coarse (block)
  level carries; per-sample conflict auditing awaits data with sample columns.

## Audit rubric

100-point per-round scores (prerequisite 92, intel 95, fusion 84/84,
stability 89, generalization 93; delivery scored at close). Prediction
calibration on record: R3 conflict-rate prediction interval (0.10–0.35) was
wrong on both datasets — registered before measurement, which is why the
structural interpretation survived review.
