// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A compact motion-subspace example using native `glam` types.
//!
//! The same calculations appear in the `builtin` and `nalgebra` examples.

use glam::{DMat3, DVec3};
use spatial6::{
    ArticulatedBodyInertia, ForceVector, Glam, MotionSubspace, MotionVector, SpatialInertia,
};

fn assert_matrix_close(actual: [[f64; 3]; 3], expected: [[f64; 3]; 3]) {
    for (actual_row, expected_row) in actual.into_iter().zip(expected) {
        for (actual, expected) in actual_row.into_iter().zip(expected_row) {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "{actual} != {expected}"
            );
        }
    }
}

fn main() {
    // A one-DoF revolute joint remains directly constructible from its motion
    // vector.
    let revolute_column = MotionVector::<f64, Glam>::new(DVec3::Z, DVec3::ZERO);
    let revolute = MotionSubspace::<1, f64, Glam>::from(revolute_column);
    assert_eq!(revolute.columns(), &[revolute_column]);
    assert_eq!(revolute.apply(&[2.0]), revolute_column * 2.0);

    // A spherical joint spans the three pure rotations at one joint origin:
    // S x, with x the three generalized velocity coefficients.
    let spherical = MotionSubspace::from_columns([
        MotionVector::new(DVec3::X, DVec3::ZERO),
        MotionVector::new(DVec3::Y, DVec3::ZERO),
        MotionVector::new(DVec3::Z, DVec3::ZERO),
    ]);
    assert!(spherical.is_finite());
    assert_eq!(
        spherical.apply(&[0.2, -0.3, 0.4]).to_array(),
        [0.2, -0.3, 0.4, 0.0, 0.0, 0.0]
    );

    let force = ForceVector::new(DVec3::new(1.0, -2.0, 3.0), DVec3::new(0.7, -0.8, 0.9));
    // S^T f is the generalized force (joint torque for this spherical joint).
    assert_eq!(spherical.generalized_force(&force), [1.0, -2.0, 3.0]);

    let rigid = SpatialInertia::<f64, Glam>::try_new(
        2.0,
        DVec3::new(0.5, -0.25, 0.75),
        DMat3::from_cols(
            DVec3::new(0.4, 0.0, 0.0),
            DVec3::new(0.0, 0.5, 0.0),
            DVec3::new(0.0, 0.0, 0.6),
        ),
    )
    .unwrap();
    let expected_joint_inertia = [
        [1.65, 0.25, -0.75],
        [0.25, 2.125, 0.375],
        [-0.75, 0.375, 1.225],
    ];
    let rigid_force_columns = rigid.apply_subspace(&spherical);
    // H = S^T I S, assembled from I applied to every subspace column.
    let joint_inertia = spherical.generalized_forces(&rigid_force_columns);
    assert_matrix_close(joint_inertia, expected_joint_inertia);

    let articulated = ArticulatedBodyInertia::try_from(&rigid).unwrap();
    // Multi-DoF ABA primitives, without implementing the full ABA:
    // U = I_A S, D = S^T U, u = tau - S^T p_A.
    let u_force_columns = articulated.apply_subspace(&spherical);
    let d = spherical.generalized_forces(&u_force_columns);
    assert_matrix_close(d, expected_joint_inertia);
    let p_a = ForceVector::new(DVec3::new(0.5, -1.5, 2.5), DVec3::new(0.2, -0.4, 0.6));
    let tau = [2.0, -1.0, 3.0];
    let generalized_bias = spherical.generalized_force(&p_a);
    let u = std::array::from_fn(|index| tau[index] - generalized_bias[index]);
    assert_eq!(u, [1.5, 0.5, 0.5]);
}
