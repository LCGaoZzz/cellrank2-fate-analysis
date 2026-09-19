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

## PETSc/SLEPc build recipe (sparse krylov path, battle-tested 2026-09-19)

Why: at ≥100k cells the `brandts` fallback above is fatal (densify = n²
memory; 265k ≈ 564 GB). PyPI ships petsc4py/slepc4py as source-only
distributions — `pip download --only-binary :all:` finds nothing — and
building the bindings requires compiled PETSc/SLEPc libraries, so on a
pip/uv host the stack is built once from source (~20 min at `-j48`;
evidence: campaign logs `petsc_build*.log`, smoke `smoke_krylov*.py`):

```bash
BUILD=<empty build dir>; VENV=<target python venv>
cd "$BUILD"
curl -sL -o p.tgz https://github.com/petsc/petsc/archive/refs/tags/v3.22.4.tar.gz && tar xf p.tgz
cd petsc-3.22.4
./configure --with-cc=gcc --with-cxx=g++ --with-fc=0 \
  --download-mpich=1 --download-f2cblaslapack=1 \
  --with-debugging=0 --with-shared-libraries=1 --with-make-np=48 \
  COPTFLAGS='-O2' CXXOPTFLAGS='-O2'
make -j48
export PETSC_DIR="$BUILD/petsc-3.22.4"
export PETSC_ARCH="arch-linux-c-opt"        # bare arch dir name ONLY — pitfall (a)
cd "$BUILD"
curl -sL -o s.tgz https://github.com/slepc/slepc/archive/refs/tags/v3.22.2.tar.gz && tar xf s.tgz
export SLEPC_DIR="$BUILD/slepc-3.22.2"      # BEFORE configure AND make — pitfall (b)
cd slepc-3.22.2
./configure
make -j48 SLEPC_DIR="$SLEPC_DIR" PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH"
"$VENV/bin/python3" -m pip install "Cython<3.1"                     # pitfall (c)
"$VENV/bin/python3" -m pip install --no-build-isolation "petsc4py==3.22.4" "slepc4py==3.22.2"
"$VENV/bin/python3" -c "import petsc4py, slepc4py; print('OK', petsc4py.__version__, slepc4py.__version__)"
```

Pitfalls (all hit on 2026-09-19; symptoms verbatim from the build logs):

- (a) `PETSC_ARCH` must be the bare architecture directory name (e.g.
  `arch-linux-c-opt`), never an absolute path — SLEPc configure rejects
  absolute values; `PETSC_DIR` already carries the prefix.
- (b) `SLEPC_DIR` must reach the SLEPc `make` step — export it (or pass it
  on the make command line as above) before invoking make; exporting only
  after configure leaves make resolving an empty prefix (mid-build path
  errors, "SLEPC_DIR 为空").
- (c) petsc4py 3.22.4 does not build under Cython ≥ 3.1
  (`AttributeError: 'ExpressionWriter' object has no attribute
  'emit_string'`). Pin `Cython<3.1` (3.0.11 verified) and install the
  bindings with `--no-build-isolation` so the build sees the pinned Cython.
  Build-time only; the installed bindings do not depend on it.
- Long silent builds/smokes under the omicos kernel: keep output flowing
  (tee to a log, verbose make); a 300 s no-output window can get the
  process killed even when healthy (the first smoke run died exactly this
  way; the rerun with streamed logs passed).

Verification before trusting the stack: GPCCA on a ~20k-cell synthetic
chain — cellrank takes the petsc/krylov path, eigendecomposition completes
(~0.8 s at 20k), macro/terminal states match the chain topology, and the
gmres fate solve converges.

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
