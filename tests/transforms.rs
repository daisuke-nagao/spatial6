#![cfg(feature = "builtin")]

// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use spatial6::{ForceVector, MotionVector, SpatialTransform};

fn assert_coordinates_close(actual: [f64; 6], expected: [f64; 6]) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
}

fn assert_motion_close(actual: MotionVector, expected: MotionVector) {
    assert_coordinates_close(actual.to_vector(), expected.to_vector());
}

fn assert_force_close(actual: ForceVector, expected: ForceVector) {
    assert_coordinates_close(actual.to_vector(), expected.to_vector());
}

fn assert_matrix_close(actual: [[f64; 6]; 6], expected: [[f64; 6]; 6]) {
    for (actual_row, expected_row) in actual.into_iter().zip(expected) {
        assert_coordinates_close(actual_row, expected_row);
    }
}

fn matrix_vector_mul(matrix: [[f64; 6]; 6], vector: [f64; 6]) -> [f64; 6] {
    std::array::from_fn(|row| {
        (0..6).fold(0.0, |sum, column| {
            sum + matrix[row][column] * vector[column]
        })
    })
}

fn fixture() -> SpatialTransform {
    SpatialTransform::<f64>::new(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [3.0, 5.0, 7.0],
    )
}

#[test]
fn compact_transform_matches_the_documented_motion_and_force_matrices() {
    let transform = fixture();
    let motion = MotionVector::<f64>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let force = ForceVector::<f64>::new([10.0, 11.0, 12.0], [13.0, 14.0, 15.0]);

    assert_motion_close(
        transform.transform_motion(&motion),
        MotionVector::<f64>::new([-2.0, 1.0, 3.0], [-7.0, 3.0, 5.0]),
    );
    assert_force_close(
        transform.transform_force(&force),
        ForceVector::<f64>::new([35.0, 33.0, 35.0], [-14.0, 13.0, 15.0]),
    );

    assert_coordinates_close(
        matrix_vector_mul(transform.motion_matrix(), motion.to_vector()),
        transform.transform_motion(&motion).to_vector(),
    );
    assert_coordinates_close(
        matrix_vector_mul(transform.force_matrix(), force.to_vector()),
        transform.transform_force(&force).to_vector(),
    );
    let inverse_motion = transform.inverse().motion_matrix();
    assert_matrix_close(
        transform.force_matrix(),
        std::array::from_fn(|row| std::array::from_fn(|column| inverse_motion[column][row])),
    );
}

#[test]
fn translation_sign_is_unambiguous() {
    let transform = SpatialTransform::<f64>::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [3.0, 5.0, 7.0],
    );
    let motion = MotionVector::<f64>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let force = ForceVector::<f64>::new([10.0, 11.0, 12.0], [13.0, 14.0, 15.0]);

    assert_eq!(
        transform.transform_motion(&motion),
        MotionVector::<f64>::new([1.0, 2.0, 3.0], [3.0, 7.0, 5.0])
    );
    assert_eq!(
        transform.transform_force(&force),
        ForceVector::<f64>::new([33.0, -35.0, 35.0], [13.0, 14.0, 15.0])
    );
}

#[test]
fn transform_preserves_duality_and_spatial_cross_products() {
    let transform = fixture();
    let velocity = MotionVector::<f64>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let motion = MotionVector::<f64>::new([7.0, 8.0, 9.0], [10.0, 11.0, 12.0]);
    let force = ForceVector::<f64>::new([10.0, 11.0, 12.0], [13.0, 14.0, 15.0]);

    let transformed_velocity = transform.transform_motion(&velocity);
    let transformed_motion = transform.transform_motion(&motion);
    let transformed_force = transform.transform_force(&force);

    assert!((velocity.dot(&force) - transformed_velocity.dot(&transformed_force)).abs() < 1.0e-12);
    assert_motion_close(
        transform.transform_motion(&velocity.cross_motion(&motion)),
        transformed_velocity.cross_motion(&transformed_motion),
    );
    assert_force_close(
        transform.transform_force(&velocity.cross_force(&force)),
        transformed_velocity.cross_force(&transformed_force),
    );
}

#[test]
fn inverse_and_composition_match_sequential_application() {
    let first = fixture();
    let angle: f64 = 0.37;
    let second = SpatialTransform::<f64>::new(
        [
            [1.0, 0.0, 0.0],
            [0.0, angle.cos(), -angle.sin()],
            [0.0, angle.sin(), angle.cos()],
        ],
        [-2.0, 1.0, 0.5],
    );
    let motion = MotionVector::<f64>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let force = ForceVector::<f64>::new([10.0, 11.0, 12.0], [13.0, 14.0, 15.0]);

    assert_motion_close(
        first
            .inverse()
            .transform_motion(&first.transform_motion(&motion)),
        motion,
    );
    assert_force_close(
        first
            .inverse()
            .transform_force(&first.transform_force(&force)),
        force,
    );
    assert_motion_close(
        first.then(&second).transform_motion(&motion),
        second.transform_motion(&first.transform_motion(&motion)),
    );
    assert_force_close(
        first.then(&second).transform_force(&force),
        second.transform_force(&first.transform_force(&force)),
    );
    assert_motion_close(
        SpatialTransform::<f64>::identity().transform_motion(&motion),
        motion,
    );
}

#[test]
fn transform_is_compact_and_generic() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert!(size_of::<SpatialTransform<f64>>() <= 12 * size_of::<f64>());
    let transform = SpatialTransform::<f32>::identity();
    let motion = MotionVector::<f32>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    assert_eq!(transform.transform_motion(&motion), motion);
    assert_send_sync::<SpatialTransform<f64>>();
}
