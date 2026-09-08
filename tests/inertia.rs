#![cfg(feature = "builtin")]

// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use spatial6::{
    ForceVector, InertiaError, MotionVector, RigidBodyInertia, SpatialInertia, SpatialTransform,
};

type Matrix3 = [[f64; 3]; 3];
type Matrix6 = [[f64; 6]; 6];

fn fixture() -> RigidBodyInertia {
    RigidBodyInertia::<f64>::try_new(
        2.0,
        [1.0, 2.0, 3.0],
        [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
    )
    .unwrap()
}

fn second_fixture() -> RigidBodyInertia {
    RigidBodyInertia::<f64>::try_new(
        3.0,
        [-2.0, 1.0, 0.5],
        [[7.0, 0.2, -0.1], [0.2, 8.0, 0.3], [-0.1, 0.3, 9.0]],
    )
    .unwrap()
}

fn transform_fixture() -> SpatialTransform {
    SpatialTransform::<f64>::new(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [3.0, 5.0, 7.0],
    )
}

fn assert_scalar_close(actual: f64, expected: f64) {
    let tolerance = 1.0e-10 + 1.0e-10 * actual.abs().max(expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} differs from {expected} by more than {tolerance}"
    );
}

fn assert_scalar_close_f32(actual: f32, expected: f32) {
    let tolerance = 1.0e-5 + 1.0e-5 * actual.abs().max(expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} differs from {expected} by more than {tolerance}"
    );
}

fn assert_matrix3_close(actual: Matrix3, expected: Matrix3) {
    for (actual_row, expected_row) in actual.into_iter().zip(expected) {
        for (actual, expected) in actual_row.into_iter().zip(expected_row) {
            assert_scalar_close(actual, expected);
        }
    }
}

fn assert_matrix_close(actual: Matrix6, expected: Matrix6) {
    for (actual_row, expected_row) in actual.into_iter().zip(expected) {
        for (actual, expected) in actual_row.into_iter().zip(expected_row) {
            assert_scalar_close(actual, expected);
        }
    }
}

fn assert_motion_close(actual: MotionVector, expected: MotionVector) {
    for (actual, expected) in actual.to_vector().into_iter().zip(expected.to_vector()) {
        assert_scalar_close(actual, expected);
    }
}

fn assert_force_close(actual: ForceVector, expected: ForceVector) {
    for (actual, expected) in actual.to_vector().into_iter().zip(expected.to_vector()) {
        assert_scalar_close(actual, expected);
    }
}

fn assert_rigid_body_inertia_close(actual: &RigidBodyInertia, expected: &RigidBodyInertia) {
    assert_scalar_close(actual.mass(), expected.mass());
    for (actual, expected) in (*actual.center_of_mass())
        .into_iter()
        .zip(*expected.center_of_mass())
    {
        assert_scalar_close(actual, expected);
    }
    assert_matrix3_close(
        actual.inertia_at_center_of_mass(),
        expected.inertia_at_center_of_mass(),
    );
    assert_matrix_close(actual.matrix(), expected.matrix());
}

#[test]
fn rigid_body_inertia_validates_physical_mass_properties() {
    let center = [0.0, 0.0, 0.0];
    let valid = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];

    assert_eq!(
        RigidBodyInertia::<f64>::try_new(0.0, center, valid),
        Err(InertiaError::NonPositiveMass)
    );
    assert_eq!(
        RigidBodyInertia::<f64>::try_new(-1.0, center, valid),
        Err(InertiaError::NonPositiveMass)
    );
    assert_eq!(
        RigidBodyInertia::<f64>::try_new(1.0, [f64::NAN, 0.0, 0.0], valid),
        Err(InertiaError::NonFinite)
    );
    assert_eq!(
        RigidBodyInertia::<f64>::try_new(f64::INFINITY, center, valid),
        Err(InertiaError::NonFinite)
    );

    let non_symmetric = [[1.0, 0.5, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
    assert_eq!(
        RigidBodyInertia::<f64>::try_new(1.0, center, non_symmetric),
        Err(InertiaError::NonSymmetric)
    );
    assert_eq!(
        RigidBodyInertia::<f64>::try_new(
            1.0,
            center,
            [[1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 3.0]],
        ),
        Err(InertiaError::NotPositiveDefinite)
    );

    // Inputs violating more than one condition at once resolve to the
    // documented check order: finite, then positive mass, then symmetric,
    // then positive definite.
    assert_eq!(
        RigidBodyInertia::<f64>::try_new(-1.0, [f64::NAN, 0.0, 0.0], non_symmetric),
        Err(InertiaError::NonFinite)
    );
    assert_eq!(
        RigidBodyInertia::<f64>::try_new(0.0, center, non_symmetric),
        Err(InertiaError::NonPositiveMass)
    );
    assert_eq!(
        RigidBodyInertia::<f64>::try_new(
            1.0,
            center,
            [[1.0, 0.5, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 3.0]],
        ),
        Err(InertiaError::NonSymmetric)
    );
}

#[test]
fn rigid_body_inertia_matches_the_ten_parameter_block_matrix() {
    let inertia = fixture();
    let expected = [
        [30.0, -4.0, -6.0, 0.0, -6.0, 4.0],
        [-4.0, 25.0, -12.0, 6.0, 0.0, -2.0],
        [-6.0, -12.0, 16.0, -4.0, 2.0, 0.0],
        [0.0, 6.0, -4.0, 2.0, 0.0, 0.0],
        [-6.0, 0.0, 2.0, 0.0, 2.0, 0.0],
        [4.0, -2.0, 0.0, 0.0, 0.0, 2.0],
    ];

    assert_matrix_close(inertia.matrix(), expected);
    assert_eq!(inertia.mass(), 2.0);
    assert_eq!(inertia.center_of_mass(), &[1.0, 2.0, 3.0]);
    assert_eq!(
        inertia.inertia_at_center_of_mass(),
        [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]
    );
    let _: SpatialInertia = inertia;
    assert!(size_of::<RigidBodyInertia<f64>>() <= 10 * size_of::<f64>());
}

#[test]
fn inertia_maps_motion_to_force_and_is_positive_definite() {
    let inertia = fixture();
    let acceleration = MotionVector::<f64>::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let expected = ForceVector::<f64>::new([24.0, -4.0, -4.0], [0.0, -4.0, 4.0]);

    assert_eq!(inertia.apply(&acceleration), expected);
    assert_eq!(inertia * acceleration, expected);
    assert!((acceleration.dot(&expected) - 20.0).abs() <= 1.0e-12);
    assert_eq!(
        inertia.apply(&MotionVector::<f64>::zeros()),
        ForceVector::<f64>::zeros()
    );
}

#[test]
fn rigid_body_inertia_transforms_by_congruence() {
    let inertia = fixture();
    let transform = transform_fixture();
    let expected = [
        [45.0, 12.0, 24.0, 0.0, 8.0, -4.0],
        [12.0, 54.0, -16.0, -8.0, 0.0, -6.0],
        [24.0, -16.0, 32.0, 4.0, 6.0, 0.0],
        [0.0, -8.0, 4.0, 2.0, 0.0, 0.0],
        [8.0, 0.0, 6.0, 0.0, 2.0, 0.0],
        [-4.0, -6.0, 0.0, 0.0, 0.0, 2.0],
    ];

    assert_matrix_close(inertia.transformed(&transform).matrix(), expected);
}

#[test]
fn inertia_time_derivative_matches_the_spatial_cross_identity() {
    let inertia = fixture();
    let velocity = MotionVector::<f64>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let expected = [
        [112.0, -29.0, -32.0, 0.0, -12.0, 10.0],
        [-29.0, 88.0, -55.0, 12.0, 0.0, -8.0],
        [-32.0, -55.0, 56.0, -10.0, 8.0, 0.0],
        [0.0, 12.0, -10.0, 0.0, 0.0, 0.0],
        [-12.0, 0.0, 8.0, 0.0, 0.0, 0.0],
        [10.0, -8.0, 0.0, 0.0, 0.0, 0.0],
    ];

    assert_matrix_close(inertia.time_derivative(&velocity), expected);
}

#[test]
fn single_body_dynamics_include_bias_force_and_round_trip() {
    let inertia = fixture();
    let velocity = MotionVector::<f64>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let acceleration = MotionVector::<f64>::new([0.5, -1.0, 2.0], [-3.0, 4.0, 1.0]);
    let expected_bias = ForceVector::<f64>::new([-42.0, -18.0, 26.0], [-6.0, 12.0, -6.0]);

    assert_eq!(inertia.bias_force(&velocity), expected_bias);
    assert_eq!(
        inertia.inverse_dynamics(&velocity, &MotionVector::<f64>::zeros()),
        expected_bias
    );
    let force = inertia.inverse_dynamics(&velocity, &acceleration);
    assert_motion_close(
        inertia.forward_dynamics(&velocity, &force).unwrap(),
        acceleration,
    );

    let pure_translation = MotionVector::<f64>::new([0.0, 0.0, 0.0], [4.0, 5.0, 6.0]);
    assert_eq!(
        inertia.bias_force(&pure_translation),
        ForceVector::<f64>::zeros()
    );
    assert!(
        inertia
            .forward_dynamics(
                &velocity,
                &ForceVector::<f64>::new([f64::NAN, 0.0, 0.0], [0.0, 0.0, 0.0]),
            )
            .is_none()
    );
    assert!(
        inertia
            .forward_dynamics(
                &MotionVector::<f64>::new([f64::NAN, 0.0, 0.0], [0.0, 0.0, 0.0]),
                &force,
            )
            .is_none()
    );
}

#[test]
fn try_combined_matches_rigid_aggregate_invariants() {
    let left = fixture();
    let right = second_fixture();
    let combined = left.try_combined(&right).unwrap();

    assert_scalar_close(combined.mass(), 5.0);
    for (actual, expected) in (*combined.center_of_mass())
        .into_iter()
        .zip([-0.8, 1.4, 1.5])
    {
        assert_scalar_close(actual, expected);
    }
    assert_matrix3_close(
        combined.inertia_at_center_of_mass(),
        [[19.7, -3.4, -9.1], [-3.4, 31.3, -2.7], [-9.1, -2.7, 27.0]],
    );

    let left_matrix = left.matrix();
    let right_matrix = right.matrix();
    let expected_matrix = std::array::from_fn(|row| {
        std::array::from_fn(|column| left_matrix[row][column] + right_matrix[row][column])
    });
    assert_matrix_close(combined.matrix(), expected_matrix);

    let motion = MotionVector::new([0.3, -0.7, 1.1], [2.0, -1.0, 0.5]);
    assert_force_close(
        combined.apply(&motion),
        left.apply(&motion) + right.apply(&motion),
    );

    let swapped = right.try_combined(&left).unwrap();
    assert_rigid_body_inertia_close(&swapped, &combined);
}

#[test]
fn try_combined_handles_shared_centers_grouping_and_frame_changes() {
    let shared_center = [1.0, -2.0, 0.5];
    let first = RigidBodyInertia::<f64>::try_new(
        1.0,
        shared_center,
        [[2.0, 0.1, 0.0], [0.1, 3.0, 0.2], [0.0, 0.2, 4.0]],
    )
    .unwrap();
    let second = RigidBodyInertia::<f64>::try_new(
        2.0,
        shared_center,
        [[5.0, 0.3, 0.1], [0.3, 6.0, 0.0], [0.1, 0.0, 7.0]],
    )
    .unwrap();
    let shared = first.try_combined(&second).unwrap();
    for (actual, expected) in (*shared.center_of_mass()).into_iter().zip(shared_center) {
        assert_scalar_close(actual, expected);
    }
    assert_matrix3_close(
        shared.inertia_at_center_of_mass(),
        [[7.0, 0.4, 0.1], [0.4, 9.0, 0.2], [0.1, 0.2, 11.0]],
    );

    let left = fixture();
    let middle = second_fixture();
    let right = RigidBodyInertia::<f64>::try_new(
        1.5,
        [0.25, -1.5, 2.0],
        [[3.0, 0.1, 0.0], [0.1, 4.0, 0.2], [0.0, 0.2, 5.0]],
    )
    .unwrap();
    let left_middle = left.try_combined(&middle).unwrap();
    let middle_right = middle.try_combined(&right).unwrap();
    let left_grouped = left_middle.try_combined(&right).unwrap();
    let right_grouped = left.try_combined(&middle_right).unwrap();
    let reordered = right
        .try_combined(&left)
        .unwrap()
        .try_combined(&middle)
        .unwrap();
    assert_rigid_body_inertia_close(&left_grouped, &right_grouped);
    assert_rigid_body_inertia_close(&left_grouped, &reordered);

    let transform = transform_fixture();
    let transformed_once = left_middle.transformed(&transform);
    let transformed_parts = left
        .transformed(&transform)
        .try_combined(&middle.transformed(&transform))
        .unwrap();
    assert_rigid_body_inertia_close(&transformed_once, &transformed_parts);
}

#[test]
fn try_combined_supports_f32_and_reports_overflow() {
    let left = RigidBodyInertia::<f32>::try_new(
        2.0,
        [1.0, 0.0, 0.0],
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
    )
    .unwrap();
    let right = RigidBodyInertia::<f32>::try_new(
        3.0,
        [-1.0, 2.0, 0.5],
        [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
    )
    .unwrap();
    let combined = left.try_combined(&right).unwrap();
    assert_scalar_close_f32(combined.mass(), 5.0);
    for (actual, expected) in (*combined.center_of_mass())
        .into_iter()
        .zip([-0.2, 1.2, 0.3])
    {
        assert_scalar_close_f32(actual, expected);
    }
    let motion = MotionVector::<f32>::new([0.5, -0.25, 1.0], [1.0, 2.0, -1.0]);
    for (actual, expected) in combined
        .apply(&motion)
        .to_vector()
        .into_iter()
        .zip((left.apply(&motion) + right.apply(&motion)).to_vector())
    {
        assert_scalar_close_f32(actual, expected);
    }

    let huge = RigidBodyInertia::<f32>::try_new(
        f32::MAX,
        [0.0; 3],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    .unwrap();
    assert_eq!(huge.try_combined(&huge), Err(InertiaError::NonFinite));
}

#[test]
fn inertia_types_are_generic_send_and_sync_values() {
    fn assert_send_sync<T: Send + Sync>() {}

    let inertia = RigidBodyInertia::<f32>::try_new(
        2.0,
        [0.0, 0.0, 0.0],
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
    )
    .unwrap();
    let motion = MotionVector::<f32>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);

    assert!(
        inertia
            .apply(&motion)
            .to_vector()
            .iter()
            .all(|value| value.is_finite())
    );
    assert_send_sync::<RigidBodyInertia<f32>>();
}
