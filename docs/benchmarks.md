# Benchmarks

## Scope

Benchmarks measure storage, sampling, stencils, reconstruction, and rendering.
Kernel time, reconstruction time, and fluid-timestep time measure different work.
A complete fluid timestep is not implemented.

Each Rust/C++ pair must run on the same hardware.
Cross-machine runs measure portability and scale.
The paper's published workstation results are separate from the EC2 runs.

## Commands

Some workloads need tens or hundreds of gigabytes. Start with the small configuration:

```sh
MSBG_BENCH_SCALE=small cargo bench --bench allocator_benches -- halo_gather
MSBG_BENCH_SCALE=small cargo bench --bench interp_bench
MSBG_BENCH_SCALE=small cargo bench --bench multires_bench
MSBG_BENCH_SCALE=small cargo bench --bench surface_bench
```

The surface benchmark needs the bunny data. See [development.md](development.md).
In PowerShell, set `$env:MSBG_BENCH_SCALE = 'small'` before running Cargo.

| Target | Operation | Configuration |
| --- | --- | --- |
| `allocator_benches` | Allocation, access, halos, stencils, smoothing | `MSBG_BENCH_SCALE=small/big/xbig`; machine presets |
| `blockmap_bench` | Dense index, custom sparse map, hashbrown | Fixed key distributions |
| `density_bench` | Quantization and dequantization | Block-sized and larger buffers |
| `interp_bench` | Values, gradients, Hessians | Small/big grids |
| `multires_bench` | Refinement, mixed-resolution halos | `MSBG_BENCH_SCALE=small` |
| `surface_bench` | Reconstruction phases and six smoothing sweeps | Small/big/xbig; surface overrides |
| `velocity_bench` | MAC P2G, staggered G2P, vector gather | `MSBG_VEL_RES`, `MSBG_VEL_NSAMPLES`, `MSBG_VEL_NPARTICLES`, `MSBG_VEL_ITERS` |
| `multires_scale_bench` | Multiresolution grids and transfers | `MSBG_STRESS_GB`; default ~28 GB, small ~0.5 GB target |
| `scale_stress` | Quantized gyroid, halos, smoothing | Fixed 4096³ domain; large-memory workload |
| `msbg-render/render_bench` | Slices, raymarching, skipping on/off | Small/big; `MSBG_RENDER_W`, `MSBG_RENDER_H` |

Machine presets select workload sizes.
`allocator_benches` supports `dell`, `macbook`, `windows`, and `aws`.
`surface_bench` supports `dell`, `macbook`, and `aws`.
Set `MSBG_BENCH_MACHINE` to override the default.
Only the allocator benchmark also accepts `--scale` and `--machine` arguments.

Run the rendering benchmark from its crate directory to resolve the default data path:

```sh
cd msbg-render
MSBG_BENCH_SCALE=small cargo bench --bench render_bench
```

## Recorded results

These values come from development notes. The raw logs were not retained.
Collect new raw results before using these values to assess a regression.

### Laptop reconstruction

Ryzen 5500U; 512³ domain; 64 instances of the 8,171-point bunny;
522,944 particles; 16³ blocks; six mean-curvature iterations.

| Phase | C++ | Rust | C++ time / Rust time |
| --- | --- | --- | --- |
| Placement + particle organization | 5.96 ms | 20.0 ms | 0.30× |
| Splat | 379.6 ms | 110.6 ms | 3.43× |
| Reconstruction + smoothing | 394.2 ms | 149.4 ms | 2.64× |

Reconstruction took 62.1% less time. Rust placement was slower; splatting accounted for most of the improvement.
Rust timing includes in-memory PLY parsing through six smoothing sweeps.
File loading and rendering are excluded.

### EC2 reconstruction

The recorded 32,768³ run used 32³ blocks, 1.29 billion particles,
107.7 billion active voxels, and approximately 223 GiB peak RSS.
The virtual domain contains 35.2 trillion voxels.

| Phase | Time |
| --- | --- |
| Placement | 35.6 s |
| Bucketing | 3.3 s |
| Allocation + fill | 16.3 s |
| Splat | 22.6 s |
| Finalize | 3.7 s |
| Six mean-curvature sweeps | 51.1 s |
| Rendering | 10.6 s |

A separate summary recorded 12.9 billion smoothing updates/s.
The active count and time above give 12.65 billion updates/s.
The notes do not establish whether these were separate runs.
Use **over 12 billion updates/s** when summarizing the result.
One update processes one active voxel in one iteration.

Other recorded Rust experiments:

- Bucketing: 375 s to 6.5 s after limiting local histograms and presizing the merge map.
- Allocation + initialization: 110.7 s to 16.3 s after parallel initialization of a ~215 GB field.
- Smoothing: 7.85B updates/s with 16³ blocks and 12.9B with 32³ blocks on the 32-core machine.
- Smoothing: 16.5B updates/s with AVX-512 and 16³ blocks on a separate 64-core Genoa machine.

The first two compare Rust versions. The 375 s baseline was paired with 6.5 s, not the 3.3 s summary.

### Other C++ comparisons

- Multiresolution halos on the 5500U: 1.35 to 3.16 Gvox/s at 608 fine blocks;
  2.02 to 3.63 Gvox/s at 5,240 fine blocks.
- Raymarching at 512³ and 960×540: 1.26 to 9.52 million rays/s.
  Shading interpolation and ray precision differ between implementations.
- Three-plane slicing at 512³: 21.5 to 200.9 Mpixels/s.
  The C++ helper scans the volume. Rust samples output pixels and uses different interpolation.

Rust empty-space skipping on/off gave an 18.5× raymarching ratio for that workload.

## Reference harnesses

The current local C++ checkout has allocator, halo, density, interpolation, and smoothing benchmarks.
It lacks later surface, multiresolution, rendering, and velocity harness additions.
Its smoothing benchmark also uses a different schedule from the later eight-color comparison.
Recover matching harnesses before reproducing those ratios.

## Recording a run

Check numerical output before timing. Record:

1. Rust and C++ revisions, including harness changes.
2. CPU, OS, memory, core counts, and thread limits.
3. Compiler versions, flags, target features, and arithmetic mode.
4. Data revision, dimensions, precision, and active counts.
5. Parameters, iterations, boundaries, and error limits.
6. Warmup, repeated samples, raw times, and summary statistics.

Store raw output and a configuration manifest in `target/benchmark-results/<run-name>/`.
Git ignores that directory. Add reviewed result sets to the documentation separately.
For targets without Criterion, collect repeated independent runs.

## Memory and units

Payload bytes, allocated capacity, current RSS, and peak RSS are different quantities.
The gyroid benchmark estimates scalar payload only.
It excludes maps, unused pool capacity, scratch buffers, and process overhead.
The surface benchmark reads RSS from Linux `/proc`.
A zero result on other systems means the measurement is unavailable.
Linux `VmHWM` is the process-wide peak across all earlier phases.

The paper's OpenVDB comparisons, bandwidth figures, and solver iteration counts belong to the authors' experiments.
Keep them separate from this implementation's measurements.
