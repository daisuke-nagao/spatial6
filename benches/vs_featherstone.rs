// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Comparison benchmark: spatial6's hand-rolled RNEA (see
//! `benches/rnea_chain.rs`, generalizing `examples/rnea_builtin.rs`) against
//! `featherstone`'s `rnea_inverse_dynamics`, over the same serial revolute
//! chain topology -- the one `featherstone`'s own
//! `benches/solver_bench.rs::make_revolute_chain` builds -- and the same
//! chain lengths, so the two show up side by side under
//! `target/criterion/rnea_vs_featherstone`.
//!
//! Only RNEA is compared. spatial6 exports articulated-body inertia primitives,
//! but no complete chain-scaling forward-dynamics or mass-matrix algorithm is
//! exported for comparison with `featherstone`'s ABA/CRBA implementations.
//!
//! This is a per-iteration wall-clock comparison, not a numerical-agreement
//! check: the two use unrelated implementations (and `featherstone` runs in
//! `f32`, spatial6 here in `f64`), so results are not expected to match bit
//! for bit. Each joint's transform is recomputed from its stored angle on
//! every call on both sides (rather than cached in the chain), matching
//! `featherstone`'s own `rnea_inverse_dynamics`, which derives every joint's
//! transform from its current position `q` internally -- so neither side
//! amortizes that cost away into one-time setup.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use featherstone::prelude::*;
use nalgebra_featherstone::{Matrix3, Vector3};

use spatial6::{
    ForceVector, MotionVector, SpatialInertia as S6Inertia, SpatialTransform as S6Transform,
};

const CHAIN_LENGTHS: [usize; 5] = [2, 4, 8, 16, 32];
const LINK_LENGTH: f64 = 0.1;

// ---------------------------------------------------------------------------
// spatial6 side
// ---------------------------------------------------------------------------

struct Spatial6Chain {
    joint_axis: MotionVector<f64>,
    joint_angle: Vec<f64>,
    inertias: Vec<S6Inertia<f64>>,
    joint_rate: Vec<f64>,
    joint_accel: Vec<f64>,
}

fn make_spatial6_chain(n: usize) -> Spatial6Chain {
    let mass = 1.0_f64;
    let rotational_inertia = [[0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, 0.001]];
    let com = [LINK_LENGTH / 2.0, 0.0, 0.0];

    let joint_axis = MotionVector::<f64>::new([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);
    let joint_angle = (0..n).map(|i| 0.2 * (i as f64 + 1.0)).collect();
    let inertias = (0..n)
        .map(|_| S6Inertia::<f64>::try_new(mass, com, rotational_inertia).unwrap())
        .collect();
    let joint_rate = (0..n).map(|i| 0.5 * (i as f64 + 1.0)).collect();
    let joint_accel = (0..n).map(|i| 0.3 * (i as f64 + 1.0)).collect();

    Spatial6Chain {
        joint_axis,
        joint_angle,
        inertias,
        joint_rate,
        joint_accel,
    }
}

fn spatial6_rnea(chain: &Spatial6Chain) -> Vec<f64> {
    let n = chain.joint_angle.len();

    // Recomputed every call, rather than cached on `chain`, so this side pays
    // the same per-call "transform from current joint position" cost that
    // `featherstone::rnea_inverse_dynamics` pays internally on the other side.
    let transforms: Vec<S6Transform<f64>> = chain
        .joint_angle
        .iter()
        .map(|&angle| {
            let (sin, cos) = angle.sin_cos();
            S6Transform::<f64>::new(
                [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
                [LINK_LENGTH, 0.0, 0.0],
            )
        })
        .collect();

    let mut velocity = vec![MotionVector::<f64>::zeros(); n + 1];
    let mut acceleration = vec![MotionVector::<f64>::zeros(); n + 1];
    acceleration[0] = MotionVector::<f64>::new([0.0, 0.0, 0.0], [0.0, 0.0, 9.81]);
    let mut force = vec![ForceVector::<f64>::zeros(); n + 1];

    for i in 1..=n {
        let joint_velocity = chain.joint_axis * chain.joint_rate[i - 1];
        let joint_acceleration = chain.joint_axis * chain.joint_accel[i - 1];

        velocity[i] = transforms[i - 1].transform_motion(&velocity[i - 1]) + joint_velocity;
        acceleration[i] = transforms[i - 1].transform_motion(&acceleration[i - 1])
            + joint_acceleration
            + velocity[i].cross_motion(&joint_velocity);
        force[i] = chain.inertias[i - 1].inverse_dynamics(&velocity[i], &acceleration[i]);
    }

    let mut torque = vec![0.0; n];
    for i in (1..=n).rev() {
        torque[i - 1] = chain.joint_axis.dot(&force[i]);
        if i > 1 {
            let transmitted = transforms[i - 1].inverse().transform_force(&force[i]);
            force[i - 1] += transmitted;
        }
    }
    torque
}

// ---------------------------------------------------------------------------
// featherstone side -- same topology as `featherstone`'s own
// `benches/solver_bench.rs::make_revolute_chain`.
// ---------------------------------------------------------------------------

fn make_featherstone_chain(n: usize, link_mass: f32) -> ArticulatedBody {
    let mut body = ArticulatedBody::new();
    body.set_gravity(Vector3::new(0.0, -9.81, 0.0));

    let inertia = SpatialInertia::from_mass_inertia(
        link_mass,
        Vector3::new(0.05, 0.0, 0.0),
        Matrix3::from_diagonal(&Vector3::new(0.001, 0.001, 0.001)) * link_mass,
    );

    for i in 0..n {
        let parent = if i == 0 { -1 } else { (i - 1) as i32 };
        body.add_body(
            format!("link{i}"),
            parent,
            GenJoint::Revolute { axis: Vector3::z() },
            inertia.clone(),
            SpatialTransform::from_translation(Vector3::new(0.1, 0.0, 0.0)),
        );
    }

    for i in 0..body.body_count() {
        body.set_joint_q(i, &[0.2 * (i as f32 + 1.0)]);
        body.set_joint_qd(i, &[0.5 * (i as f32 + 1.0)]);
    }
    for i in 0..body.dof_count() {
        body.qdd[i] = 0.3 * (i as f32 + 1.0);
    }

    body
}

fn rnea_vs_featherstone(c: &mut Criterion) {
    let mut group = c.benchmark_group("rnea_vs_featherstone");
    for &n in &CHAIN_LENGTHS {
        let chain = make_spatial6_chain(n);
        group.bench_with_input(BenchmarkId::new("spatial6", n), &n, |b, _| {
            b.iter(|| spatial6_rnea(black_box(&chain)));
        });

        let body = make_featherstone_chain(n, 1.0);
        group.bench_with_input(BenchmarkId::new("featherstone", n), &n, |b, _| {
            b.iter(|| rnea_inverse_dynamics(black_box(&body)));
        });
    }
    group.finish();
}

criterion_group!(vs_featherstone, rnea_vs_featherstone);
criterion_main!(vs_featherstone);
