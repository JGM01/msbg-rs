use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use msbg_rs::multires::{MultiresGrid, RefinementMap};
use msbg_rs::pressure::{
    CoefficientFormat, FaceCoefficients, PressureBoundary, PressureOperator, PressurePattern,
    PressurePatternBuilder, RelaxParams,
};
use std::hint::black_box;
use std::sync::Arc;

const CLOSED: [PressureBoundary; 6] = [PressureBoundary::ClosedNeumann; 6];

/// Thread count: MSBG_PRESSURE_THREADS > RAYON_NUM_THREADS > logical cores.
/// A 1-thread run is the serial kernel leg (the C++ side is run with 1 TBB
/// thread for the same comparison).
fn bench_threads() -> usize {
    for var in ["MSBG_PRESSURE_THREADS", "RAYON_NUM_THREADS"] {
        if let Ok(v) = std::env::var(var)
            && let Ok(n) = v.parse::<usize>()
            && n >= 1
        {
            return n;
        }
    }
    rayon::current_num_threads()
}

fn thread_pool() -> &'static msbg_rs::thread_pool::Pool {
    static POOL: std::sync::OnceLock<msbg_rs::thread_pool::Pool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| msbg_rs::thread_pool::Pool::new(bench_threads()))
}

fn pattern_at(block_size: usize, resolution: usize, adaptive: bool) -> Arc<PressurePattern> {
    let mut grid = MultiresGrid::create(
        "pressure_bench",
        resolution,
        resolution,
        resolution,
        block_size,
        2,
        2,
    );
    let active: Vec<usize> = (0..grid.dims.n_blocks).collect();
    let levels = if adaptive {
        active
            .iter()
            .map(|&bid| {
                let (x, y, z) = grid.dims.coords(bid);
                usize::from((x + 2 * y + 3 * z) % 7 == 0) as u8
            })
            .collect()
    } else {
        vec![0; active.len()]
    };
    let mut map = RefinementMap::new(levels);
    grid.set_refinement_map(&mut map);
    PressurePatternBuilder::new(&grid, &active, CLOSED)
        .build()
        .unwrap()
}

fn pattern(block_size: usize, adaptive: bool) -> Arc<PressurePattern> {
    let blocks_per_axis = if block_size == 32 { 3 } else { 4 };
    pattern_at(block_size, block_size * blocks_per_axis, adaptive)
}

fn coefficients(len: usize, contrast: f32) -> FaceCoefficients {
    FaceCoefficients {
        right: std::array::from_fn(|axis| {
            (0..len)
                .map(|i| {
                    let phase =
                        ((i.wrapping_mul(2_654_435_761usize) + axis * 17) & 255) as f32 / 255.0;
                    1.0 / (1.0 + phase * (contrast - 1.0))
                })
                .collect()
        }),
    }
}

fn bench_ops(
    c: &mut Criterion,
    group_name: &str,
    patterns: &[(Arc<PressurePattern>, String, String)],
    contrasts: &[(f32, String)],
    ops: &[(String, u32)],
) {
    let pool = thread_pool();
    let mut group = c.benchmark_group(group_name);
    for (pattern, bs, topo) in patterns {
        for (contrast, cstr) in contrasts {
            let coeff = coefficients(pattern.len(), *contrast);
            let operator =
                PressureOperator::with_format(pattern.clone(), coeff, CoefficientFormat::F32)
                    .unwrap();
            let mut x = pattern.vector_with_value(0.25);
            for (i, value) in x.as_mut_slice().iter_mut().enumerate() {
                *value += (i & 31) as f32 * (1.0 / 32.0);
            }
            let mut y = pattern.vector();
            let rhs = pattern.vector_with_value(1.0);
            let id = format!("{bs}/{topo}/{cstr}");
            group.throughput(Throughput::Elements(pattern.len() as u64));
            for (op, local_sweeps) in ops {
                let name = match op.as_str() {
                    "matvec" => "matvec",
                    "matvec_dot" => "matvec_dot",
                    "residual" => "residual",
                    "relax_1" => "relax_1",
                    "relax_4_local" => "relax_4_local",
                    _ => unreachable!(),
                };
                let params = RelaxParams::new(6.0 / 7.0, (*local_sweeps) as usize).unwrap();
                match name {
                    "matvec" => group.bench_function(
                        BenchmarkId::new("matvec", &id),
                        |bench| {
                            bench.iter(|| {
                                pool.install(|| {
                                    operator.apply(black_box(&x), black_box(&mut y)).unwrap()
                                })
                            })
                        },
                    ),
                    "matvec_dot" => group.bench_function(
                        BenchmarkId::new("matvec_dot", &id),
                        |bench| {
                            bench.iter(|| {
                                pool.install(|| {
                                    black_box(
                                        operator
                                            .apply_and_dot(black_box(&x), black_box(&mut y))
                                            .unwrap(),
                                    )
                                })
                            })
                        },
                    ),
                    "residual" => group.bench_function(
                        BenchmarkId::new("residual", &id),
                        |bench| {
                            bench.iter(|| {
                                pool.install(|| {
                                    black_box(
                                        operator
                                            .residual(black_box(&rhs), black_box(&x), black_box(&mut y))
                                            .unwrap(),
                                    )
                                })
                            })
                        },
                    ),
                    "relax_1" => group.bench_function(
                        BenchmarkId::new("relax_1", &id),
                        |bench| {
                            bench.iter_batched(
                                || pattern.vector(),
                                |mut pressure| {
                                    pool.install(|| {
                                        operator
                                            .relax(&mut pressure, &rhs, params, None)
                                            .unwrap()
                                    })
                                },
                                BatchSize::SmallInput,
                            )
                        },
                    ),
                    _ => group.bench_function(
                        BenchmarkId::new("relax_4_local", &id),
                        |bench| {
                            bench.iter_batched(
                                || pattern.vector(),
                                |mut pressure| {
                                    pool.install(|| {
                                        operator.relax(&mut pressure, &rhs, params, None).unwrap()
                                    })
                                },
                                BatchSize::SmallInput,
                            )
                        },
                    ),
                };
            }
        }
    }
    group.finish();
}

fn bench_pressure(c: &mut Criterion) {
    let _ = thread_pool(); // eager build so warmup isn't charged
    let small = [
        (pattern(16, false), "b16".to_string(), "uniform".to_string()),
        (pattern(16, true), "b16".to_string(), "adaptive".to_string()),
        (pattern(32, false), "b32".to_string(), "uniform".to_string()),
        (pattern(32, true), "b32".to_string(), "adaptive".to_string()),
    ];
    let contrasts = [(1.0, "c1".to_string()), (1000.0, "c1000".to_string())];
    let ops = [
        ("matvec".to_string(), 1u32),
        ("matvec_dot".to_string(), 1u32),
        ("residual".to_string(), 1u32),
        ("relax_1".to_string(), 1u32),
        ("relax_4_local".to_string(), 4u32),
    ];
    bench_ops(c, "pressure", &small, &contrasts, &ops);
}

fn bench_pressure_large(c: &mut Criterion) {
    let _ = thread_pool(); // eager build so warmup isn't charged
    let res = std::env::var("MSBG_PRESSURE_RES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(192);
    let large = [
        (
            pattern_at(16, res, false),
            format!("b16"),
            "uniform".to_string(),
        ),
    ];
    let contrasts = [(1.0, "c1".to_string()), (1000.0, "c1000".to_string())];
    let ops = [("matvec".to_string(), 1u32), ("relax_1".to_string(), 1u32)];
    bench_ops(c, "pressure_large", &large, &contrasts, &ops);
}

fn bench_construction(c: &mut Criterion) {
    let pattern_obj = pattern(16, true);
    let coeff = coefficients(pattern_obj.len(), 1000.0);
    c.bench_function("pressure/coefficient_construction", |bench| {
        bench
            .iter(|| PressureOperator::new(pattern_obj.clone(), black_box(coeff.clone())).unwrap());
    });
    c.bench_function("pressure/topology_preparation", |bench| {
        bench.iter(|| black_box(pattern(16, true)));
    });
}

criterion_group!(benches, bench_pressure, bench_pressure_large, bench_construction);
criterion_main!(benches);
