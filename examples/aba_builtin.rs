// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Featherstone's three-pass Articulated-Body Algorithm (ABA) for forward
//! dynamics of a fixed-base serial chain. Given `q`, `qd`, and `tau`, it
//! computes `qdd`; RNEA instead maps `q`, `qd`, and `qdd` to `tau`.
//!
//! The backward pass uses `ArticulatedBodyInertia` for both the current
//! articulated inertia and the reduced operator obtained after eliminating
//! each joint acceleration.

use spatial6::{
    ArticulatedBodyInertia, Builtin, ForceVector, MotionSubspace, MotionVector, SpatialInertia,
    SpatialTransform,
};

const NUM_LINKS: usize = 3;

fn main() {
    let q: [f64; NUM_LINKS] = [0.3, -0.2, 0.5];
    let qd = [0.4, -0.3, 0.2];
    let tau = [1.0, -0.5, 0.25];
    let gravity = [0.0, -9.81, 0.0];

    let s = MotionSubspace::<1>::from(MotionVector::<f64>::new([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]));
    let rigid_inertia: [SpatialInertia<f64>; NUM_LINKS] = std::array::from_fn(|_| {
        SpatialInertia::<f64>::try_new(
            1.0,
            [0.5, 0.1, 0.0],
            [[0.08, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.12]],
        )
        .unwrap()
    });

    // x_up[i] maps motion from the parent frame into child-link i's frame.
    let x_up: [SpatialTransform<f64>; NUM_LINKS] = q.map(|angle| {
        let (sin, cos) = angle.sin_cos();
        SpatialTransform::<f64>::new(
            [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
            [1.0, 0.0, 0.0],
        )
    });

    let mut velocity: [MotionVector<f64>; NUM_LINKS] =
        std::array::from_fn(|_| MotionVector::zeros());
    let mut bias_acceleration: [MotionVector<f64>; NUM_LINKS] =
        std::array::from_fn(|_| MotionVector::zeros());
    let mut acceleration: [MotionVector<f64>; NUM_LINKS] =
        std::array::from_fn(|_| MotionVector::zeros());
    let mut articulated_inertia: [ArticulatedBodyInertia<f64, Builtin>; NUM_LINKS] =
        std::array::from_fn(|_| ArticulatedBodyInertia::zeros());
    let mut bias_force: [ForceVector<f64>; NUM_LINKS] =
        std::array::from_fn(|_| ForceVector::zeros());
    let mut u_force: [ForceVector<f64>; NUM_LINKS] = std::array::from_fn(|_| ForceVector::zeros());
    let mut d = [0.0; NUM_LINKS];
    let mut u_scalar = [0.0; NUM_LINKS];

    // Pass 1: kinematics and velocity-dependent bias forces.
    for i in 0..NUM_LINKS {
        let joint_velocity = s.apply(&[qd[i]]);
        let parent_velocity = if i == 0 {
            MotionVector::zeros()
        } else {
            velocity[i - 1]
        };
        velocity[i] = x_up[i].transform_motion(&parent_velocity) + joint_velocity;
        bias_acceleration[i] = velocity[i].cross_motion(&joint_velocity);
        articulated_inertia[i] = ArticulatedBodyInertia::try_from(&rigid_inertia[i]).unwrap();
        bias_force[i] = velocity[i].cross_force(&rigid_inertia[i].apply(&velocity[i]));
    }

    // Pass 2: articulated-body inertia and bias-force propagation.
    for i in (0..NUM_LINKS).rev() {
        u_force[i] = articulated_inertia[i].apply_subspace(&s)[0];
        d[i] = s.generalized_force(&u_force[i])[0];
        u_scalar[i] = tau[i] - s.generalized_force(&bias_force[i])[0];
        assert!(d[i].is_finite() && d[i] > 0.0);

        if i > 0 {
            let reduced = articulated_inertia[i]
                .try_rank_one_updated(-1.0 / d[i], &u_force[i])
                .unwrap();
            let reduced_bias = bias_force[i]
                + reduced.apply(&bias_acceleration[i])
                + u_force[i] * (u_scalar[i] / d[i]);
            let child_to_parent = x_up[i].inverse();
            let propagated_inertia = reduced.try_transformed(&child_to_parent).unwrap();
            articulated_inertia[i - 1] = articulated_inertia[i - 1]
                .try_combined(&propagated_inertia)
                .unwrap();
            bias_force[i - 1] += child_to_parent.transform_force(&reduced_bias);
        }
    }

    let base_acceleration = MotionVector::<f64>::new([0.0; 3], gravity.map(|value| -value));
    let mut qdd = [0.0; NUM_LINKS];

    // Pass 3: forward acceleration solve.
    for i in 0..NUM_LINKS {
        let parent_acceleration = if i == 0 {
            base_acceleration
        } else {
            acceleration[i - 1]
        };
        let acceleration_before_joint =
            x_up[i].transform_motion(&parent_acceleration) + bias_acceleration[i];
        qdd[i] = (u_scalar[i] - acceleration_before_joint.dot(&u_force[i])) / d[i];
        acceleration[i] = acceleration_before_joint + s.apply(&[qdd[i]]);
    }

    assert!(qdd.iter().all(|value| value.is_finite()));
    // Independently calculated with an f64 RNEA mass-matrix solve.
    let expected_qdd = [
        -6.821_078_053_116_178,
        0.815_013_495_082_429_5,
        11.376_680_805_339_552,
    ];
    for (actual, expected) in qdd.iter().zip(expected_qdd) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "{actual} != {expected}"
        );
    }

    for (i, value) in qdd.iter().enumerate() {
        println!("joint {} acceleration: {value:.6}", i + 1);
    }
}
