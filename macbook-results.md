```
jacob@Trollbook-Pro msbg-rs % sysctl -n machdep.cpu.brand_string
Apple M3 Pro
jacob@Trollbook-Pro msbg-rs % sysctl -n hw.memsize hw.ncpu hw.physicalcpu hw.logicalcpu
38654705664
12
12
12
jacob@Trollbook-Pro msbg-rs % sysctl -n hw.perflevel0.count hw.perflevel1.count
sysctl: unknown oid 'hw.perflevel0.count'
sysctl: unknown oid 'hw.perflevel1.count'
jacob@Trollbook-Pro msbg-rs % sysctl -a 2>/dev/null | grep -i cachesize
hw.perflevel1.l1icachesize: 131072
hw.perflevel1.l1dcachesize: 65536
hw.perflevel1.l2cachesize: 4194304
hw.perflevel0.l1icachesize: 196608
hw.perflevel0.l1dcachesize: 131072
hw.perflevel0.l2cachesize: 16777216
hw.cachesize: 3621273600 65536 4194304 0 0 0 0 0 0 0
hw.l1icachesize: 131072
hw.l1dcachesize: 65536
hw.l2cachesize: 4194304
jacob@Trollbook-Pro msbg-rs % rustup show
Default host: aarch64-apple-darwin
rustup home:  /Users/jacob/.rustup

installed toolchains
--------------------
stable-aarch64-apple-darwin
nightly-aarch64-apple-darwin (active, default)
1.90.0-aarch64-apple-darwin

active toolchain
----------------
name: nightly-aarch64-apple-darwin
active because: overridden by '/Users/jacob/Programming/msbg_workspace/msbg-rs/rust-toolchain.toml'
installed targets:
  aarch64-apple-darwin
  thumbv7em-none-eabihf
  wasm32-unknown-unknown
  x86_64-unknown-linux-gnu
  x86_64-unknown-linux-musl
  x86_64-unknown-none
```

```
[msbg-rs bench] machine=Macbook arch=aarch64 os=macos logical_cores=12 rayon_threads=12
Gnuplot not found, using plotters backend
blockpool_hot_path/single_thread/1000
                        time:   [1.7822 µs 1.7875 µs 1.7996 µs]
blockpool_hot_path/single_thread/10000
                        time:   [45.345 µs 45.367 µs 45.388 µs]
                        change: [−1.5566% −1.1502% −0.7907%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high severe
blockpool_hot_path/single_thread/50000
                        time:   [241.88 µs 243.19 µs 245.81 µs]
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high severe

blockpool_contention/12_threads_49152_blocks
                        time:   [3.3959 ms 3.3989 ms 3.4029 ms]
                        change: [−2.9024% −2.3345% −1.7626%] (p = 0.00 < 0.05)
                        Performance has improved.

blockpool_cold_alloc/cold_alloc_1000_blocks
                        time:   [2.7825 µs 2.7891 µs 2.7926 µs]
Found 3 outliers among 10 measurements (30.00%)
  1 (10.00%) low severe
  2 (20.00%) high severe

voxel_access/get_sequential
                        time:   [188.87 µs 188.93 µs 188.97 µs]
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high mild
voxel_access/set_allocated
                        time:   [216.24 µs 216.32 µs 216.37 µs]
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high mild

laplacian_compute_only/mocked_fill/200
                        time:   [60.677 µs 61.076 µs 61.558 µs]
                        thrpt:  [13.308 Gelem/s 13.413 Gelem/s 13.501 Gelem/s]
Found 2 outliers among 10 measurements (20.00%)
  1 (10.00%) low mild
  1 (10.00%) high severe
laplacian_compute_only/mocked_fill/1000
                        time:   [236.40 µs 236.99 µs 237.68 µs]
                        thrpt:  [17.233 Gelem/s 17.284 Gelem/s 17.327 Gelem/s]
laplacian_compute_only/mocked_fill/5000
                        time:   [1.5079 ms 1.5176 ms 1.5354 ms]
                        thrpt:  [13.338 Gelem/s 13.495 Gelem/s 13.582 Gelem/s]
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high severe

halo_gather/shell_fill_full/1000
                        time:   [181.72 µs 183.40 µs 187.34 µs]
                        thrpt:  [24.312 Gelem/s 24.834 Gelem/s 25.065 Gelem/s]
halo_gather/shell_fill_faces/1000
                        time:   [194.29 µs 197.67 µs 204.46 µs]
                        thrpt:  [22.277 Gelem/s 23.042 Gelem/s 23.443 Gelem/s]
halo_gather/shell_fill_full/5000
                        time:   [1.0609 ms 1.0634 ms 1.0659 ms]
                        thrpt:  [20.136 Gelem/s 20.184 Gelem/s 20.231 Gelem/s]
Found 2 outliers among 10 measurements (20.00%)
  1 (10.00%) low mild
  1 (10.00%) high mild
halo_gather/shell_fill_faces/5000
                        time:   [1.0576 ms 1.0610 ms 1.0646 ms]
                        thrpt:  [20.161 Gelem/s 20.229 Gelem/s 20.295 Gelem/s]
Found 2 outliers among 10 measurements (20.00%)
  1 (10.00%) low severe
  1 (10.00%) high severe
halo_gather/shell_fill_full/10000
                        time:   [2.2183 ms 2.2301 ms 2.2515 ms]
                        thrpt:  [19.182 Gelem/s 19.366 Gelem/s 19.469 Gelem/s]
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high severe
halo_gather/shell_fill_faces/10000
                        time:   [2.2624 ms 2.2671 ms 2.2762 ms]
                        thrpt:  [18.974 Gelem/s 19.050 Gelem/s 19.090 Gelem/s]
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high mild

laplacian_smoothing_e2e/shell_sweep/1000
                        time:   [466.71 µs 469.19 µs 474.19 µs]
                        thrpt:  [9.6053 Gelem/s 9.7077 Gelem/s 9.7592 Gelem/s]
laplacian_smoothing_e2e/shell_sweep/5000
                        time:   [2.5348 ms 2.5420 ms 2.5491 ms]
                        thrpt:  [8.4197 Gelem/s 8.4433 Gelem/s 8.4673 Gelem/s]
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high severe
laplacian_smoothing_e2e/shell_sweep/10000
                        time:   [5.2183 ms 5.2269 ms 5.2411 ms]
                        thrpt:  [8.2403 Gelem/s 8.2627 Gelem/s 8.2764 Gelem/s]
```
```
jacob@Trollbook-Pro msbg-rs % cargo bench --bench interp_bench
   Compiling msbg-rs v0.1.0 (/Users/jacob/Programming/msbg_workspace/msbg-rs)
    Finished `bench` profile [optimized] target(s) in 0.93s
     Running benches/interp_bench.rs (target/release/deps/interp_bench-c13caf5f6894a501)
Gnuplot not found, using plotters backend
interp_linear_value/sample
                        time:   [620.87 µs 620.95 µs 621.04 µs]
                        thrpt:  [161.02 Melem/s 161.04 Melem/s 161.06 Melem/s]
Found 7 outliers among 100 measurements (7.00%)
  2 (2.00%) low mild
  3 (3.00%) high mild
  2 (2.00%) high severe

interp_linear_grad/gradient
                        time:   [961.37 µs 961.67 µs 962.01 µs]
                        thrpt:  [103.95 Melem/s 103.99 Melem/s 104.02 Melem/s]
Found 7 outliers among 100 measurements (7.00%)
  1 (1.00%) low severe
  2 (2.00%) low mild
  2 (2.00%) high mild
  2 (2.00%) high severe

interp_cubic_grad/gradient
                        time:   [5.9304 ms 5.9318 ms 5.9333 ms]
                        thrpt:  [16.854 Melem/s 16.858 Melem/s 16.862 Melem/s]
Found 2 outliers among 100 measurements (2.00%)
  1 (1.00%) high mild
  1 (1.00%) high severe

interp_cubic_hess/hessian
                        time:   [7.5505 ms 7.5524 ms 7.5544 ms]
                        thrpt:  [13.237 Melem/s 13.241 Melem/s 13.244 Melem/s]
Found 2 outliers among 100 measurements (2.00%)
  2 (2.00%) high severe
```

```
jacob@Trollbook-Pro msbg-rs % cargo bench --bench density_bench
   Compiling msbg-rs v0.1.0 (/Users/jacob/Programming/msbg_workspace/msbg-rs)
    Finished `bench` profile [optimized] target(s) in 0.73s
     Running benches/density_bench.rs (target/release/deps/density_bench-b2bee16c1f079a06)
Gnuplot not found, using plotters backend
density_dequant/4096    time:   [231.65 ns 233.98 ns 236.45 ns]
                        thrpt:  [17.323 Gelem/s 17.505 Gelem/s 17.682 Gelem/s]
Found 8 outliers among 100 measurements (8.00%)
  5 (5.00%) high mild
  3 (3.00%) high severe
density_dequant/4194304 time:   [219.34 µs 219.59 µs 219.86 µs]
                        thrpt:  [19.077 Gelem/s 19.100 Gelem/s 19.122 Gelem/s]
Found 18 outliers among 100 measurements (18.00%)
  8 (8.00%) high mild
  10 (10.00%) high severe

density_quantize/4096   time:   [242.55 ns 242.62 ns 242.69 ns]
                        thrpt:  [16.877 Gelem/s 16.882 Gelem/s 16.887 Gelem/s]
Found 14 outliers among 100 measurements (14.00%)
  1 (1.00%) low mild
  7 (7.00%) high mild
  6 (6.00%) high severe
density_quantize/4194304
                        time:   [246.75 µs 246.90 µs 247.08 µs]
                        thrpt:  [16.976 Gelem/s 16.988 Gelem/s 16.998 Gelem/s]
Found 15 outliers among 100 measurements (15.00%)
  10 (10.00%) high mild
  5 (5.00%) high severe
```
```
jacob@Trollbook-Pro msbg-rs % cargo bench --bench allocator_benches
    Finished `bench` profile [optimized] target(s) in 0.10s
     Running benches/allocator_benches.rs (target/release/deps/allocator_benches-d7f5c136449799c5)
[msbg-rs bench] machine=Macbook arch=aarch64 os=macos logical_cores=12 rayon_threads=12
Gnuplot not found, using plotters backend
blockpool_hot_path/single_thread/100000
                        time:   [719.43 µs 755.69 µs 795.75 µs]
                        change: [+31.585% +36.921% +41.731%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 2 outliers among 100 measurements (2.00%)
  2 (2.00%) high mild
blockpool_hot_path/single_thread/500000
                        time:   [7.7029 ms 8.1807 ms 8.6665 ms]
Benchmarking blockpool_hot_path/single_thread/1000000: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 92.0s, or reduce sample count to 10.
blockpool_hot_path/single_thread/1000000
                        time:   [876.08 ms 881.42 ms 887.02 ms]
Found 30 outliers among 100 measurements (30.00%)
  1 (1.00%) low severe
  10 (10.00%) low mild
  6 (6.00%) high mild
  13 (13.00%) high severe

blockpool_contention/12_threads_49152_blocks
                        time:   [3.3836 ms 3.3875 ms 3.3913 ms]
                        change: [−0.4685% −0.3213% −0.1677%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 5 outliers among 100 measurements (5.00%)
  4 (4.00%) low mild
  1 (1.00%) high mild

blockpool_cold_alloc/cold_alloc_10000_blocks
                        time:   [44.860 µs 44.903 µs 44.959 µs]

Benchmarking voxel_access/get_sequential: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 7.8s, enable flat sampling, or reduce sample count to 50.
voxel_access/get_sequential
                        time:   [1.5448 ms 1.5452 ms 1.5456 ms]
                        change: [+717.74% +718.17% +718.57%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 10 outliers among 100 measurements (10.00%)
  1 (1.00%) low mild
  6 (6.00%) high mild
  3 (3.00%) high severe
Benchmarking voxel_access/set_allocated: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 9.1s, enable flat sampling, or reduce sample count to 50.
voxel_access/set_allocated
                        time:   [1.7965 ms 1.7978 ms 1.7992 ms]
                        change: [+730.69% +731.62% +732.59%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 2 outliers among 100 measurements (2.00%)
  1 (1.00%) high mild
  1 (1.00%) high severe

laplacian_compute_only/mocked_fill/1000
                        time:   [235.61 µs 236.12 µs 236.64 µs]
                        thrpt:  [17.309 Gelem/s 17.347 Gelem/s 17.384 Gelem/s]
                 change:
                        time:   [−0.5378% −0.1914% +0.1454%] (p = 0.26 > 0.05)
                        thrpt:  [−0.1452% +0.1918% +0.5407%]
                        No change in performance detected.
Found 5 outliers among 100 measurements (5.00%)
  1 (1.00%) low mild
  3 (3.00%) high mild
  1 (1.00%) high severe
laplacian_compute_only/mocked_fill/10000
                        time:   [3.0825 ms 3.0905 ms 3.0992 ms]
                        thrpt:  [13.216 Gelem/s 13.253 Gelem/s 13.288 Gelem/s]
Found 6 outliers among 100 measurements (6.00%)
  4 (4.00%) high mild
  2 (2.00%) high severe
laplacian_compute_only/mocked_fill/50000
                        time:   [15.271 ms 15.307 ms 15.347 ms]
                        thrpt:  [13.345 Gelem/s 13.379 Gelem/s 13.411 Gelem/s]
Found 6 outliers among 100 measurements (6.00%)
  5 (5.00%) high mild
  1 (1.00%) high severe

halo_gather/shell_fill_full/50000
                        time:   [11.036 ms 11.069 ms 11.104 ms]
                        thrpt:  [18.849 Gelem/s 18.909 Gelem/s 18.966 Gelem/s]
Found 5 outliers among 100 measurements (5.00%)
  5 (5.00%) high mild
halo_gather/shell_fill_faces/50000
                        time:   [11.602 ms 11.670 ms 11.749 ms]
                        thrpt:  [17.814 Gelem/s 17.935 Gelem/s 18.041 Gelem/s]
Found 11 outliers among 100 measurements (11.00%)
  7 (7.00%) high mild
  4 (4.00%) high severe
Benchmarking halo_gather/shell_fill_full/250000: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 5.8s, or reduce sample count to 80.
halo_gather/shell_fill_full/250000
                        time:   [58.537 ms 59.244 ms 59.987 ms]
                        thrpt:  [17.592 Gelem/s 17.813 Gelem/s 18.028 Gelem/s]
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high mild
Benchmarking halo_gather/shell_fill_faces/250000: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 6.0s, or reduce sample count to 80.
halo_gather/shell_fill_faces/250000
                        time:   [59.879 ms 60.434 ms 61.064 ms]
                        thrpt:  [17.282 Gelem/s 17.462 Gelem/s 17.624 Gelem/s]
Found 11 outliers among 100 measurements (11.00%)
  2 (2.00%) high mild
  9 (9.00%) high severe
Benchmarking halo_gather/shell_fill_full/500000: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 11.2s, or reduce sample count to 40.
halo_gather/shell_fill_full/500000
                        time:   [116.45 ms 118.43 ms 120.52 ms]
                        thrpt:  [17.296 Gelem/s 17.601 Gelem/s 17.900 Gelem/s]
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high mild
Benchmarking halo_gather/shell_fill_faces/500000: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 13.4s, or reduce sample count to 30.
halo_gather/shell_fill_faces/500000
                        time:   [131.16 ms 133.20 ms 135.24 ms]
                        thrpt:  [15.412 Gelem/s 15.649 Gelem/s 15.892 Gelem/s]

laplacian_smoothing_e2e/shell_sweep/50000
                        time:   [26.256 ms 26.579 ms 26.919 ms]
                        thrpt:  [7.7753 Gelem/s 7.8748 Gelem/s 7.9719 Gelem/s]
Benchmarking laplacian_smoothing_e2e/shell_sweep/250000: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 13.2s, or reduce sample count to 30.
laplacian_smoothing_e2e/shell_sweep/250000
                        time:   [130.21 ms 131.00 ms 131.96 ms]
                        thrpt:  [7.9971 Gelem/s 8.0556 Gelem/s 8.1043 Gelem/s]
Found 12 outliers among 100 measurements (12.00%)
  7 (7.00%) high mild
  5 (5.00%) high severe
Benchmarking laplacian_smoothing_e2e/shell_sweep/500000: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 139.9s, or reduce sample count to 10.
laplacian_smoothing_e2e/shell_sweep/500000
                        time:   [610.66 ms 656.30 ms 705.19 ms]
                        thrpt:  [2.9558 Gelem/s 3.1760 Gelem/s 3.4134 Gelem/s]
Found 15 outliers among 100 measurements (15.00%)
  11 (11.00%) high mild
  4 (4.00%) high severe
```

```
jacob@Trollbook-Pro msbg-rs % MSBG_BENCH_SCALE=xbig cargo bench --bench allocator_benches
    Finished `bench` profile [optimized] target(s) in 0.07s
     Running benches/allocator_benches.rs (target/release/deps/allocator_benches-d7f5c136449799c5)
[msbg-rs bench] machine=Macbook arch=aarch64 os=macos logical_cores=12 rayon_threads=12
Gnuplot not found, using plotters backend
blockpool_hot_path/single_thread/100000
                        time:   [611.94 µs 643.91 µs 674.14 µs]
                        change: [−24.837% −21.294% −17.562%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 19 outliers among 100 measurements (19.00%)
  4 (4.00%) high mild
  15 (15.00%) high severe
blockpool_hot_path/single_thread/1000000
                        time:   [16.790 ms 17.737 ms 18.723 ms]
                        change: [−98.085% −97.988% −97.872%] (p = 0.00 < 0.05)
                        Performance has improved.
Benchmarking blockpool_hot_path/single_thread/1500000: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 108.8s, or reduce sample count to 10.
blockpool_hot_path/single_thread/1500000
                        time:   [962.94 ms 966.29 ms 970.06 ms]
Found 7 outliers among 100 measurements (7.00%)
  3 (3.00%) high mild
  4 (4.00%) high severe

blockpool_contention/12_threads_49152_blocks
                        time:   [3.3802 ms 3.3864 ms 3.3918 ms]
                        change: [−0.2715% −0.0332% +0.1582%] (p = 0.76 > 0.05)
                        No change in performance detected.
Found 8 outliers among 100 measurements (8.00%)
  5 (5.00%) low severe
  3 (3.00%) low mild

blockpool_cold_alloc/cold_alloc_10000_blocks
                        time:   [44.791 µs 44.807 µs 44.824 µs]
                        change: [−0.4090% −0.1844% +0.0624%] (p = 0.13 > 0.05)
                        No change in performance detected.
Found 18 outliers among 100 measurements (18.00%)
  2 (2.00%) high mild
  16 (16.00%) high severe

Benchmarking voxel_access/get_sequential: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 7.8s, enable flat sampling, or reduce sample count to 50.
voxel_access/get_sequential
                        time:   [1.5441 ms 1.5446 ms 1.5451 ms]
                        change: [−0.1009% −0.0235% +0.0740%] (p = 0.64 > 0.05)
                        No change in performance detected.                                                                                   Found 3 outliers among 100 measurements (3.00%)
  3 (3.00%) high severe
Benchmarking voxel_access/set_allocated: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 9.1s, enable flat sampling, or reduce sample count to 50.
voxel_access/set_allocated
                        time:   [1.7968 ms 1.7984 ms 1.7999 ms]
                        change: [−0.2988% −0.1394% +0.0194%] (p = 0.10 > 0.05)
                        No change in performance detected.
Found 2 outliers among 100 measurements (2.00%)
  1 (1.00%) high mild
  1 (1.00%) high severe

laplacian_compute_only/mocked_fill/10000
                        time:   [3.0775 ms 3.0828 ms 3.0884 ms]
                        thrpt:  [13.262 Gelem/s 13.287 Gelem/s 13.310 Gelem/s]
                 change:
                        time:   [−0.5798% −0.2494% +0.0760%] (p = 0.13 > 0.05)
                        thrpt:  [−0.0759% +0.2500% +0.5831%]
                        No change in performance detected.
Found 8 outliers among 100 measurements (8.00%)
  1 (1.00%) low mild
  7 (7.00%) high mild
laplacian_compute_only/mocked_fill/50000
                        time:   [15.251 ms 15.292 ms 15.337 ms]
                        thrpt:  [13.353 Gelem/s 13.393 Gelem/s 13.429 Gelem/s]
                 change:
                        time:   [−0.4779% −0.1002% +0.2479%] (p = 0.60 > 0.05)
                        thrpt:  [−0.2473% +0.1003% +0.4801%]
                        No change in performance detected.
Found 6 outliers among 100 measurements (6.00%)
  6 (6.00%) high mild
laplacian_compute_only/mocked_fill/100000
                        time:   [30.390 ms 30.506 ms 30.638 ms]
                        thrpt:  [13.369 Gelem/s 13.427 Gelem/s 13.478 Gelem/s]
Found 14 outliers among 100 measurements (14.00%)
  5 (5.00%) high mild
  9 (9.00%) high severe

halo_gather/shell_fill_full/50000
                        time:   [11.016 ms 11.044 ms 11.074 ms]
                        thrpt:  [18.900 Gelem/s 18.951 Gelem/s 19.000 Gelem/s]
                 change:
                        time:   [−0.6274% −0.2232% +0.1834%] (p = 0.28 > 0.05)
                        thrpt:  [−0.1830% +0.2237% +0.6314%]
                        No change in performance detected.
Found 2 outliers among 100 measurements (2.00%)
  2 (2.00%) high mild
halo_gather/shell_fill_faces/50000
                        time:   [11.613 ms 11.667 ms 11.723 ms]
                        thrpt:  [17.854 Gelem/s 17.940 Gelem/s 18.023 Gelem/s]
                 change:
                        time:   [−0.8593% −0.0310% +0.7502%] (p = 0.94 > 0.05)
                        thrpt:  [−0.7446% +0.0310% +0.8667%]
                        No change in performance detected.
Found 6 outliers among 100 measurements (6.00%)
  6 (6.00%) high mild
Benchmarking halo_gather/shell_fill_full/250000: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 5.7s, or reduce sample count to 80.
halo_gather/shell_fill_full/250000
                        time:   [56.693 ms 57.334 ms 58.072 ms]
                        thrpt:  [18.172 Gelem/s 18.406 Gelem/s 18.614 Gelem/s]
                 change:
                        time:   [−4.8363% −3.2240% −1.4805%] (p = 0.00 < 0.05)
                        thrpt:  [+1.5027% +3.3314% +5.0820%]
                        Performance has improved.
Found 12 outliers among 100 measurements (12.00%)
  1 (1.00%) high mild
  11 (11.00%) high severe
Benchmarking halo_gather/shell_fill_faces/250000: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 6.0s, or reduce sample count to 80.
halo_gather/shell_fill_faces/250000
                        time:   [59.907 ms 60.427 ms 61.044 ms]
                        thrpt:  [17.288 Gelem/s 17.464 Gelem/s 17.616 Gelem/s]
                 change:
                        time:   [−1.3857% −0.0118% +1.3338%] (p = 0.99 > 0.05)
                        thrpt:  [−1.3163% +0.0118% +1.4052%]
                        No change in performance detected.
Found 5 outliers among 100 measurements (5.00%)
  5 (5.00%) high severe
Benchmarking halo_gather/shell_fill_full/750000: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 17.5s, or reduce sample count to 20.
halo_gather/shell_fill_full/750000
                        time:   [169.95 ms 171.76 ms 173.78 ms]
                        thrpt:  [17.923 Gelem/s 18.134 Gelem/s 18.328 Gelem/s]
Found 16 outliers among 100 measurements (16.00%)
  3 (3.00%) high mild
  13 (13.00%) high severe
Benchmarking halo_gather/shell_fill_faces/750000: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 17.9s, or reduce sample count to 20.                halo_gather/shell_fill_faces/750000
                        time:   [177.81 ms 179.11 ms 180.68 ms]
                        thrpt:  [17.239 Gelem/s 17.390 Gelem/s 17.518 Gelem/s]
Found 15 outliers among 100 measurements (15.00%)
  5 (5.00%) high mild
  10 (10.00%) high severe

laplacian_smoothing_e2e/shell_sweep/50000
                        time:   [25.597 ms 25.730 ms 25.874 ms]
                        thrpt:  [8.0894 Gelem/s 8.1348 Gelem/s 8.1769 Gelem/s]
                 change:
                        time:   [−4.4878% −3.1959% −1.8742%] (p = 0.00 < 0.05)
                        thrpt:  [+1.9100% +3.3014% +4.6987%]
                        Performance has improved.
Found 12 outliers among 100 measurements (12.00%)
  2 (2.00%) high mild
  10 (10.00%) high severe
Benchmarking laplacian_smoothing_e2e/shell_sweep/250000: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 13.2s, or reduce sample count to 30.
laplacian_smoothing_e2e/shell_sweep/250000
                        time:   [130.75 ms 131.99 ms 133.41 ms]
                        thrpt:  [7.9099 Gelem/s 7.9955 Gelem/s 8.0709 Gelem/s]
                 change:
                        time:   [−0.4754% +0.7516% +1.8787%] (p = 0.23 > 0.05)
                        thrpt:  [−1.8441% −0.7460% +0.4777%]
                        No change in performance detected.
Found 16 outliers among 100 measurements (16.00%)
  3 (3.00%) high mild
  13 (13.00%) high severe
Benchmarking laplacian_smoothing_e2e/shell_sweep/750000: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 183.3s, or reduce sample count to 10.
laplacian_smoothing_e2e/shell_sweep/750000
                        time:   [1.7815 s 1.7947 s 1.8076 s]
                        thrpt:  [1.7232 Gelem/s 1.7355 Gelem/s 1.7484 Gelem/s]
Found 2 outliers among 100 measurements (2.00%)
  2 (2.00%) low mild
```
