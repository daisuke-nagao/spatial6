#![cfg(feature = "builtin")]

// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use spatial6::{ForceVector, MotionVector};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-12,
        "{actual} != {expected}"
    );
}

fn matrix_vector_mul(matrix: [[f64; 6]; 6], vector: [f64; 6]) -> [f64; 6] {
    std::array::from_fn(|row| {
        (0..6).fold(0.0, |sum, column| {
            sum + matrix[row][column] * vector[column]
        })
    })
}

#[test]
fn stores_angular_components_before_linear_components_and_pairs_dually() {
    let motion = MotionVector::<f64>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let force = ForceVector::<f64>::new([10.0, 11.0, 12.0], [13.0, 14.0, 15.0]);

    assert_eq!(motion.to_vector(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(force.to_vector(), [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    assert_eq!(MotionVector::<f64>::from_vector(motion.to_vector()), motion);
    assert_eq!(ForceVector::<f64>::from_vector(force.to_vector()), force);
    assert_close(motion.dot(&force), 280.0);
    assert_close(force.dot(&motion), 280.0);
    assert_close(motion * force, 280.0);
    assert_close(force * motion, 280.0);
}

#[test]
fn motion_and_force_arithmetic_stays_in_its_own_space() {
    let first_motion = MotionVector::<f64>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let second_motion = MotionVector::<f64>::new([7.0, 8.0, 9.0], [10.0, 11.0, 12.0]);
    let first_force = ForceVector::<f64>::new([1.0, 3.0, 5.0], [7.0, 9.0, 11.0]);
    let second_force = ForceVector::<f64>::new([2.0, 4.0, 6.0], [8.0, 10.0, 12.0]);

    assert_eq!(
        second_motion - first_motion,
        MotionVector::<f64>::new([6.0, 6.0, 6.0], [6.0, 6.0, 6.0])
    );
    assert_eq!(
        first_motion + second_motion,
        MotionVector::<f64>::new([8.0, 10.0, 12.0], [14.0, 16.0, 18.0])
    );
    assert_eq!(
        second_force - first_force,
        ForceVector::<f64>::new([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    );
    assert_eq!(
        first_force + second_force,
        ForceVector::<f64>::new([3.0, 7.0, 11.0], [15.0, 19.0, 23.0])
    );
}

#[test]
fn spatial_cross_products_match_the_plucker_formulas() {
    let velocity = MotionVector::<f64>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let motion = MotionVector::<f64>::new([7.0, 8.0, 9.0], [10.0, 11.0, 12.0]);
    let force = ForceVector::<f64>::new([10.0, 11.0, 12.0], [13.0, 14.0, 15.0]);

    assert_eq!(
        velocity.cross_motion(&motion),
        MotionVector::<f64>::new([-6.0, 12.0, -6.0], [-12.0, 24.0, -12.0])
    );
    assert_eq!(
        velocity.cross_force(&force),
        ForceVector::<f64>::new([-18.0, 36.0, -18.0], [-12.0, 24.0, -12.0])
    );
    assert_eq!(
        velocity.cross_motion(&velocity),
        MotionVector::<f64>::zeros()
    );
    assert_eq!(
        MotionVector::<f64>::zeros().cross_motion(&motion),
        MotionVector::<f64>::zeros()
    );
    assert_eq!(
        velocity.cross_motion(&motion),
        -motion.cross_motion(&velocity)
    );
    assert_close(
        motion.dot(&velocity.cross_force(&force)),
        -velocity.cross_motion(&motion).dot(&force),
    );
}

#[test]
fn cross_matrices_match_compact_cross_products_and_are_dual() {
    let velocity = MotionVector::<f64>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let motion = MotionVector::<f64>::new([7.0, 8.0, 9.0], [10.0, 11.0, 12.0]);
    let force = ForceVector::<f64>::new([10.0, 11.0, 12.0], [13.0, 14.0, 15.0]);

    assert_eq!(
        matrix_vector_mul(velocity.cross_matrix(), motion.to_vector()),
        velocity.cross_motion(&motion).to_vector()
    );
    assert_eq!(
        matrix_vector_mul(velocity.cross_dual_matrix(), force.to_vector()),
        velocity.cross_force(&force).to_vector()
    );
    assert_eq!(
        velocity.cross_dual_matrix(),
        std::array::from_fn(|row| {
            std::array::from_fn(|column| -velocity.cross_matrix()[column][row])
        })
    );
}

#[test]
fn public_values_support_f32_and_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    let motion = MotionVector::<f32>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let force = ForceVector::<f32>::new([7.0, 8.0, 9.0], [10.0, 11.0, 12.0]);

    assert!((motion.dot(&force) - 217.0).abs() < 1.0e-5);
    assert_send_sync::<MotionVector<f32>>();
    assert_send_sync::<ForceVector<f64>>();
}
