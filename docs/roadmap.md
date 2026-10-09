# Status and planned work

The current code provides sparse grids, numerical kernels, surface reconstruction, and offline rendering.
The longer-term goal is an adaptive phase-field FLIP simulator.

## Implemented

| Component | Source | Tests |
| --- | --- | --- |
| Aligned pools and sparse maps | `src/blockpool.rs`, `src/blockmap.rs` | Layout, allocation, concurrency, lookup |
| Grids and typed channels | `src/sparse_grid.rs`, `src/channel.rs` | Edge cases, conversions, C++ storage hash |
| Sampling | `src/math/` | Analytic fields, boundaries, C++ goldens |
| Halos and SIMD stencils | `src/multires/halo.rs`, `src/math/` | Halo boundaries, C++ stencil goldens |
| Multiresolution grids | `src/multires/` | Refinement, metadata, gathering, downsampling |
| Eight-color smoothing | `src/solver.rs` | Serial reference, C++ goldens |
| Surface reconstruction | `src/particles/` | Sparse/dense equivalence, C++ goldens |
| Slices and raymarching | `msbg-render/` | Geometry, traversal, boundaries |
| MAC-grid transfer | `src/fluid/` | Transfer, blend, divergence |

See [benchmarks.md](benchmarks.md) for the large reconstruction results.

## Maintenance

1. Audit public API safety: shared mutation, pool reset, raw pointers, unchecked conversions, and scratch-buffer ownership.
2. Retain matching C++ harness revisions, run configurations, correctness results, and raw timings.
3. Keep benchmark units and memory estimates explicit.
4. Run the added CI workflow on hosted runners.
5. Choose a distribution license before publishing a Rust package.

## Pressure solver

Implement a matrix-free, variable-coefficient MAC-grid Poisson operator and relaxation.
Define coefficient normalization, signs, inverse diagonals, boundaries, and coarse/fine transitions.
Test analytic identities, scalar-reference results, and conservation.

Then add coefficient coarsening, adaptive relaxation, and an outer Krylov iteration.
Use the paper and relevant C++ code as references.
Measure convergence for each test scenario.

## Fluid simulation

Connect transfer, phase-field processing, pressure projection, gathering, and advection.
Start with a single-resolution case and check physical invariants.
Then add adaptive grids, two-phase interfaces, cut cells, and larger examples.

## Acceptance criteria

Document the interface. Test numerical and boundary behavior. Measure the implemented operation.
For C++ comparisons, match inputs, parameters, precision, thread counts, and hardware.
Record the reference revision and confirm equivalent output before comparing speed.
