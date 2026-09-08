// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Criterion benchmark for the Recursive Newton-Euler Algorithm (RNEA) --
//! see `examples/rnea_builtin.rs` for the algorithm this generalizes to an
//! N-link chain -- run across a range of chain lengths to show how the cost
//! of one full inverse-dynamics pass scales with the number of links.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use spatial6::{ForceVector, MotionVector, SpatialInertia, SpatialTransform};

const CHAIN_LENGTHS: [usize; 5] = [2, 4, 8, 16, 32];

/// An N-body serial revolute chain: each link is a 1m rod of mass 1kg,
/// offset from its parent along the parent's local x-axis, rotating about
/// its own local z-axis.
struct Chain {
    joint_axis: MotionVector<f64>,
    transforms: Vec<SpatialTransform<f64>>,
    inertias: Vec<SpatialInertia<f64>>,
    joint_rate: Vec<f64>,
    joint_accel: Vec<f64>,
}

fn make_chain(n: usize) -> Chain {
    let link_length = 1.0;
    let mass = 1.0;
    let rotational_inertia = [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]];
    let com = [link_length / 2.0, 0.0, 0.0];

    let joint_axis = MotionVector::<f64>::new([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);
    let transforms = (0..n)
        .map(|i| {
            let angle = 0.2 * (i as f64 + 1.0);
            let (sin, cos) = angle.sin_cos();
            SpatialTransform::<f64>::new(
                [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
                [link_length, 0.0, 0.0],
            )
        })
        .collect();
    let inertias = (0..n)
        .map(|_| SpatialInertia::<f64>::try_new(mass, com, rotational_inertia).unwrap())
        .collect();
    let joint_rate = (0..n).map(|i| 0.5 * (i as f64 + 1.0)).collect();
    let joint_accel = (0..n).map(|i| 0.3 * (i as f64 + 1.0)).collect();

    Chain {
        joint_axis,
        transforms,
        inertias,
        joint_rate,
        joint_accel,
    }
}

/// One full inverse-dynamics pass: outward velocity/acceleration/force sweep,
/// then an inward pass reading off each joint's torque.
fn rnea(chain: &Chain) -> Vec<f64> {
    let n = chain.transforms.len();
    let mut velocity = vec![MotionVector::<f64>::zeros(); n + 1];
    let mut acceleration = vec![MotionVector::<f64>::zeros(); n + 1];
    // Base "accelerates" upward by -g, folding gravity into the outward pass.
    acceleration[0] = MotionVector::<f64>::new([0.0, 0.0, 0.0], [0.0, 0.0, 9.81]);
    let mut force = vec![ForceVector::<f64>::zeros(); n + 1];

    for i in 1..=n {
        let joint_velocity = chain.joint_axis * chain.joint_rate[i - 1];
        let joint_acceleration = chain.joint_axis * chain.joint_accel[i - 1];

        velocity[i] = chain.transforms[i - 1].transform_motion(&velocity[i - 1]) + joint_velocity;
        acceleration[i] = chain.transforms[i - 1].transform_motion(&acceleration[i - 1])
            + joint_acceleration
            + velocity[i].cross_motion(&joint_velocity);
        force[i] = chain.inertias[i - 1].inverse_dynamics(&velocity[i], &acceleration[i]);
    }

    let mut torque = vec![0.0; n];
    for i in (1..=n).rev() {
        torque[i - 1] = chain.joint_axis.dot(&force[i]);
        if i > 1 {
            let transmitted = chain.transforms[i - 1].inverse().transform_force(&force[i]);
            force[i - 1] += transmitted;
        }
    }
    torque
}

fn rnea_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("rnea_scaling");
    for &n in &CHAIN_LENGTHS {
        let chain = make_chain(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| rnea(black_box(&chain)));
        });
    }
    group.finish();
}

criterion_group!(rnea_benches, rnea_scaling);
criterion_main!(rnea_benches);
