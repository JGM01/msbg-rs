# Architecture

This document describes the current code.
See [roadmap.md](roadmap.md) for planned work and [benchmarks.md](benchmarks.md) for measurements.

## Block storage

`Block<D, BSX, N>` contains a voxel array with 64-byte alignment.
`BSX` must be a power of two. `N` must equal `BSX³`.
Const parameters let the compiler specialize indexing and stencil loops.
The vector sampler also uses the block's tail storage for padded loads.
Review those loads when changing the layout.

`Pool<T>` allocates contiguous segments. An atomic cursor assigns raw pointers.
Allocation within a segment avoids a mutex. Segment growth uses a mutex and
release/acquire ordering to publish the new pointer.
Pool reset does not track outstanding references or drop live items.

`SparseGrid` has absent, constant-full, and allocated blocks.
Shared dummy blocks supply empty and full values.
`BlockMap` stores allocated and full blocks in separate key and value arrays.
It uses linear probing and a multiplicative hash. Absent keys represent empty space.

Map storage scales with active blocks.
For comparison, a dense eight-byte pointer table needs 68.7 GB for a 32,768³ domain with 16³ blocks.
This is a storage estimate. Actual overhead depends on map capacity and occupancy.
Some multiresolution topology arrays remain dense.

Block IDs use `usize` on the supported 64-bit targets.
Morton keys use 21 bits per axis to order blocks by location.

Sources: [`blockpool.rs`](../src/blockpool.rs), [`blockmap.rs`](../src/blockmap.rs),
[`sparse_grid.rs`](../src/sparse_grid.rs), [`multires/sort.rs`](../src/multires/sort.rs).

## Channels and sampling

Channel types distinguish density, pressure, velocity, flags, and solver values.
`Density` stores a normalized scalar as `u16`.
`Density8` stores sqrt-compressed density as `u8`.
Sampling converts stored values to floating point.
`Density8` interpolation uses sqrt space; rendering converts the result to physical density.

Trilinear and cubic B-spline sampling provide values, gradients, and Hessians.
A window inside one block uses a direct gather.
Other windows use boundary-aware sampling.
The caller selects boundary conditions and corner or cell-centered alignment.

Sources: [`channel.rs`](../src/channel.rs), [`math/sample.rs`](../src/math/sample.rs),
[`math/gather.rs`](../src/math/gather.rs).

## Halos and SIMD

Halo gathering resolves neighbor pointers before copying rows.
A one-cell halo uses a 3×3×3 block neighborhood.
Stored values are converted to `f32` in a scratch buffer.
Faces-only gathering serves the seven-point Laplacian.
Full halos provide edges and corners for mean-curvature and bi-Laplacian kernels.

Kernels use 4, 8, or 16 float lanes, based on target features.
The x86 `f32` path uses streaming stores and a store fence.
ARM64 uses ordinary stores. Quantized channels have separate writeback paths.
The supplied Rayon pool enables FTZ/DAZ on x86 workers.

Sources: [`multires/halo.rs`](../src/multires/halo.rs), [`math/simd.rs`](../src/math/simd.rs),
[`math/stencil.rs`](../src/math/stencil.rs), [`thread_pool.rs`](../src/thread_pool.rs).

## Smoothing

`Sweeper` groups active blocks into eight spatial colors and Morton-sorts each group.
Colors run in sequence. Blocks within a color run in parallel, with per-worker halo buffers.
Under the halo constraints, blocks of the same color do not read each other's updated values.

The smoother supports Laplacian, mean-curvature, and bi-Laplacian updates.
It writes the field in place and needs no full output channel.
The eight-color method comes from MSBG.
This smoother operates on a single-resolution `SparseGrid`.

Source: [`solver.rs`](../src/solver.rs).

## Surface reconstruction

The pipeline loads PLY positions, places instances, finds active blocks, and groups particles by center block.
It then splats a distance field, converts it to density, and optionally smooths it.

Bucketing uses sparse histograms and ranges indexed by occupied-block rank.
Explicit input chunks limit the number of local histograms.
A presized map merges counts. A parallel scatter builds the block-major permutation.
Tests compare this path with dense reference implementations.

Splatting accumulates values in a private worker buffer, then commits them to nearby blocks.
Eight colors schedule block contributions without write conflicts.
This requires limits on particle radius, buffer extent, and block size.

The large example supports 16³ and 32³ blocks.
Larger blocks reduced halo reads in the recorded large smoothing run.
The best size depends on the workload.

Sources: [`particles/sort.rs`](../src/particles/sort.rs),
[`particles/splat.rs`](../src/particles/splat.rs), [`particles/mod.rs`](../src/particles/mod.rs).

## Fluid transfer and rendering

`fluid` supplies MAC face channels, mass/momentum splatting, FLIP/PIC gathering, and divergence.
These components still need a timestep driver and pressure solver.
Surface reconstruction uses a minimum-distance reduction. Fluid transfer uses weighted sums.

`msbg-render` uses the core crate's public API.
Slicing samples each output pixel.
The raymarcher traverses blocks with DDA, skips empty blocks, and samples inside allocated blocks.
Tests cover block-boundary origins, grazing rays, and skipping on/off equivalence.

Sources: [`fluid`](../src/fluid), [`msbg-render`](../msbg-render/src).

## Safety and numerical limits

The code uses raw allocation, shared pointers, unchecked operations, and parallel writes.
The public safety boundary needs review, including:

- Pool reset while grids still hold allocations.
- Writes through shared grid references.
- Scratch buffers used by nested calls or multiple thread pools.
- Duplicate active block IDs.
- Pointer initialization, padding, and arbitrary sample coordinates.

Debug assertions do not enforce release-build preconditions.

Numerical tests allow operation-specific floating-point differences.
The live surface comparison allows two `u16` code steps and at most 0.1% of field entries to differ.
See [development.md](development.md) for each test's scope.
