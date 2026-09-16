// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Criterion benchmarks for spatial6's core per-operation cost, run
//! identically across every enabled backend so the cost of the
//! backend-agnostic abstraction itself is visible per backend.
//!
//! `cargo bench --bench primitives` covers `builtin` only (the default
//! feature); add `--features nalgebra,glam` (or `--all-features`) to include
//! the other backends in the same report groups.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use spatial6::{
    ArticulatedBodyInertia, ForceVector, MotionSubspace, MotionVector, RigidBodyInertia,
    SpatialRepresentation, SpatialTransform,
};

fn sample_transform<R: SpatialRepresentation<f64>>() -> SpatialTransform<f64, R> {
    let (sin, cos) = 0.4_f64.sin_cos();
    SpatialTransform::<f64, R>::new(
        R::rotation3_from_array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]),
        R::vector3_from_array([1.0, 0.0, 0.0]),
    )
}

fn sample_motion<R: SpatialRepresentation<f64>>() -> MotionVector<f64, R> {
    MotionVector::<f64, R>::new(
        R::vector3_from_array([0.2, -0.1, 0.4]),
        R::vector3_from_array([0.1, 0.3, -0.2]),
    )
}

fn sample_force<R: SpatialRepresentation<f64>>() -> ForceVector<f64, R> {
    ForceVector::<f64, R>::new(
        R::vector3_from_array([0.3, -0.2, 0.5]),
        R::vector3_from_array([0.4, 0.1, -0.6]),
    )
}

fn sample_motion_subspace<const N: usize, R: SpatialRepresentation<f64>>()
-> MotionSubspace<N, f64, R> {
    MotionSubspace::from_columns(std::array::from_fn(|index| {
        MotionVector::<f64, R>::from_array(std::array::from_fn(|component| {
            if component == index { 1.0 } else { 0.0 }
        }))
    }))
}

fn sample_inertia<R: SpatialRepresentation<f64>>() -> RigidBodyInertia<f64, R> {
    RigidBodyInertia::<f64, R>::try_new(
        2.0,
        R::vector3_from_array([0.5, 0.0, 0.0]),
        R::matrix3_from_array([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]]),
    )
    .unwrap()
}

fn bench_transform_compose<R: SpatialRepresentation<f64>>(c: &mut Criterion, backend: &str) {
    let a = sample_transform::<R>();
    let b = sample_transform::<R>();
    c.benchmark_group("transform_compose").bench_function(
        BenchmarkId::from_parameter(backend),
        |bencher| {
            bencher.iter(|| black_box(&a).then(black_box(&b)));
        },
    );
}

fn bench_transform_motion<R: SpatialRepresentation<f64>>(c: &mut Criterion, backend: &str) {
    let transform = sample_transform::<R>();
    let motion = sample_motion::<R>();
    c.benchmark_group("transform_motion").bench_function(
        BenchmarkId::from_parameter(backend),
        |bencher| {
            bencher.iter(|| black_box(&transform).transform_motion(black_box(&motion)));
        },
    );
}

fn bench_inertia_apply<R: SpatialRepresentation<f64>>(c: &mut Criterion, backend: &str) {
    let inertia = sample_inertia::<R>();
    let motion = sample_motion::<R>();
    c.benchmark_group("inertia_apply").bench_function(
        BenchmarkId::from_parameter(backend),
        |bencher| {
            bencher.iter(|| black_box(&inertia).apply(black_box(&motion)));
        },
    );
}

fn bench_inertia_inverse_dynamics<R: SpatialRepresentation<f64>>(c: &mut Criterion, backend: &str) {
    let inertia = sample_inertia::<R>();
    let velocity = sample_motion::<R>();
    let acceleration = MotionVector::<f64, R>::new(
        R::vector3_from_array([0.05, 0.1, -0.05]),
        R::vector3_from_array([1.0, 0.0, -0.5]),
    );
    c.benchmark_group("inertia_inverse_dynamics")
        .bench_function(BenchmarkId::from_parameter(backend), |bencher| {
            bencher.iter(|| {
                black_box(&inertia).inverse_dynamics(black_box(&velocity), black_box(&acceleration))
            });
        });
}

fn bench_inertia_try_combined<R: SpatialRepresentation<f64>>(c: &mut Criterion, backend: &str) {
    let a = sample_inertia::<R>();
    let b = sample_inertia::<R>().transformed(&sample_transform::<R>());
    c.benchmark_group("inertia_try_combined").bench_function(
        BenchmarkId::from_parameter(backend),
        |bencher| {
            bencher.iter(|| black_box(&a).try_combined(black_box(&b)));
        },
    );
}

fn bench_motion_subspace<const N: usize, R: SpatialRepresentation<f64>>(
    c: &mut Criterion,
    backend: &str,
) {
    let subspace = sample_motion_subspace::<N, R>();
    let coefficients: [f64; N] = std::array::from_fn(|index| 0.2 * (index as f64 + 1.0));
    let force = sample_force::<R>();
    let inertia = sample_inertia::<R>();
    let articulated = ArticulatedBodyInertia::try_from(&inertia).unwrap();
    let id = || BenchmarkId::new(backend, N);

    c.benchmark_group("motion_subspace_apply")
        .bench_function(id(), |bencher| {
            bencher.iter(|| black_box(&subspace).apply(black_box(&coefficients)));
        });

    c.benchmark_group("motion_subspace_generalized_force")
        .bench_function(id(), |bencher| {
            bencher.iter(|| black_box(&subspace).generalized_force(black_box(&force)));
        });

    c.benchmark_group("motion_subspace_apply_subspace")
        .bench_function(id(), |bencher| {
            bencher.iter(|| black_box(&inertia).apply_subspace(black_box(&subspace)));
        });

    c.benchmark_group("articulated_inertia_apply_subspace")
        .bench_function(id(), |bencher| {
            bencher.iter(|| black_box(&articulated).apply_subspace(black_box(&subspace)));
        });

    c.benchmark_group("motion_subspace_generalized_inertia")
        .bench_function(id(), |bencher| {
            bencher.iter(|| {
                let force_columns = black_box(&inertia).apply_subspace(black_box(&subspace));
                black_box(&subspace).generalized_forces(black_box(&force_columns))
            });
        });
}

fn bench_motion_subspaces<R: SpatialRepresentation<f64>>(c: &mut Criterion, backend: &str) {
    bench_motion_subspace::<1, R>(c, backend);
    bench_motion_subspace::<3, R>(c, backend);
    bench_motion_subspace::<6, R>(c, backend);
}

fn backends(c: &mut Criterion) {
    #[cfg(feature = "builtin")]
    {
        bench_transform_compose::<spatial6::Builtin>(c, "builtin");
        bench_transform_motion::<spatial6::Builtin>(c, "builtin");
        bench_inertia_apply::<spatial6::Builtin>(c, "builtin");
        bench_inertia_inverse_dynamics::<spatial6::Builtin>(c, "builtin");
        bench_inertia_try_combined::<spatial6::Builtin>(c, "builtin");
        bench_motion_subspaces::<spatial6::Builtin>(c, "builtin");
    }
    #[cfg(feature = "nalgebra")]
    {
        bench_transform_compose::<spatial6::Nalgebra>(c, "nalgebra");
        bench_transform_motion::<spatial6::Nalgebra>(c, "nalgebra");
        bench_inertia_apply::<spatial6::Nalgebra>(c, "nalgebra");
        bench_inertia_inverse_dynamics::<spatial6::Nalgebra>(c, "nalgebra");
        bench_inertia_try_combined::<spatial6::Nalgebra>(c, "nalgebra");
        bench_motion_subspaces::<spatial6::Nalgebra>(c, "nalgebra");
    }
    #[cfg(feature = "glam")]
    {
        bench_transform_compose::<spatial6::Glam>(c, "glam");
        bench_transform_motion::<spatial6::Glam>(c, "glam");
        bench_inertia_apply::<spatial6::Glam>(c, "glam");
        bench_inertia_inverse_dynamics::<spatial6::Glam>(c, "glam");
        bench_inertia_try_combined::<spatial6::Glam>(c, "glam");
        bench_motion_subspaces::<spatial6::Glam>(c, "glam");
    }
}

criterion_group!(primitives, backends);
criterion_main!(primitives);
