// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Comparison benchmark: the same shape of core spatial-algebra operation
//! (transform a motion vector, apply an inertia to a motion vector) run
//! against spatial6's `Builtin` backend and against `spatial-math`'s
//! `PluckerTransform`/`RigidBodyInertia`, so the two show up side by side
//! under `target/criterion/{transform_motion,inertia_apply}_vs_spatial_math`.
//!
//! Per-iteration wall-clock comparison only, not a numerical-agreement check:
//! the two crates use unrelated rotation and inertia storage layouts.
//!
//! `spatial-math` also exposes `ArticulatedBodyInertia`, which isn't compared
//! here: spatial6 only models rigid-body inertia (`RigidBodyInertia`), not
//! articulated-body inertia (see `examples/articulated_inertia_builtin.rs`),
//! so there is no equivalent spatial6 operation to put beside it.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use spatial6::{
    MotionVector as S6Motion, RigidBodyInertia as S6Inertia, SpatialTransform as S6Transform,
};

use spatial_math::{
    PluckerTransform, PlukerRotation, RigidBodyInertia as SmInertia,
    SpatialMotionVector as SmMotion, SymmetricMat3, unit_vec3, vec3,
};

fn sample_s6_transform() -> S6Transform<f64> {
    let (sin, cos) = 0.4_f64.sin_cos();
    S6Transform::<f64>::new(
        [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
        [1.0, 0.0, 0.0],
    )
}

fn sample_s6_motion() -> S6Motion<f64> {
    S6Motion::<f64>::new([0.2, -0.1, 0.4], [0.1, 0.3, -0.2])
}

fn sample_s6_inertia() -> S6Inertia<f64> {
    S6Inertia::<f64>::try_new(
        2.0,
        [0.5, 0.0, 0.0],
        [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
    )
    .unwrap()
}

fn sample_sm_transform() -> PluckerTransform {
    PluckerTransform {
        rotation: PlukerRotation::from_axis_angle(unit_vec3(0.0, 0.0, 1.0), 0.4),
        translation: vec3(1.0, 0.0, 0.0),
    }
}

fn sample_sm_motion() -> SmMotion {
    SmMotion::from_array([0.2, -0.1, 0.4, 0.1, 0.3, -0.2])
}

fn sample_sm_inertia() -> SmInertia {
    SmInertia::new(
        2.0,
        vec3(0.5, 0.0, 0.0),
        SymmetricMat3::from_array([0.2, 0.0, 0.0, 0.2, 0.0, 0.2]),
    )
}

fn transform_motion_vs_spatial_math(c: &mut Criterion) {
    let mut group = c.benchmark_group("transform_motion_vs_spatial_math");

    let s6_transform = sample_s6_transform();
    let s6_motion = sample_s6_motion();
    group.bench_function(BenchmarkId::new("spatial6", "single"), |b| {
        b.iter(|| black_box(&s6_transform).transform_motion(black_box(&s6_motion)));
    });

    let sm_transform = sample_sm_transform();
    let sm_motion = sample_sm_motion();
    group.bench_function(BenchmarkId::new("spatial_math", "single"), |b| {
        b.iter(|| black_box(&sm_transform).transform_motion_vec(black_box(sm_motion)));
    });

    group.finish();
}

fn inertia_apply_vs_spatial_math(c: &mut Criterion) {
    let mut group = c.benchmark_group("inertia_apply_vs_spatial_math");

    let s6_inertia = sample_s6_inertia();
    let s6_motion = sample_s6_motion();
    group.bench_function(BenchmarkId::new("spatial6", "single"), |b| {
        b.iter(|| black_box(&s6_inertia).apply(black_box(&s6_motion)));
    });

    let sm_inertia = sample_sm_inertia();
    let sm_motion = sample_sm_motion();
    group.bench_function(BenchmarkId::new("spatial_math", "single"), |b| {
        b.iter(|| black_box(&sm_inertia).mul_vec(black_box(sm_motion)));
    });

    group.finish();
}

criterion_group!(
    vs_spatial_math,
    transform_motion_vs_spatial_math,
    inertia_apply_vs_spatial_math
);
criterion_main!(vs_spatial_math);
