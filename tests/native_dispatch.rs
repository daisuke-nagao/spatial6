// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};

use spatial6::{
    ForceVector, MotionVector, RigidBodyInertia, SpatialRepresentation, SpatialTransform,
};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Probe;

static ARRAY_CONVERSION_CALLS: AtomicUsize = AtomicUsize::new(0);
static MATRIX3_ADD_CALLS: AtomicUsize = AtomicUsize::new(0);

fn record_array_conversion() {
    ARRAY_CONVERSION_CALLS.fetch_add(1, Ordering::SeqCst);
}

fn vector3_add(left: &[f64; 3], right: &[f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn vector3_sub(left: &[f64; 3], right: &[f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn vector3_neg(value: &[f64; 3]) -> [f64; 3] {
    [-value[0], -value[1], -value[2]]
}

fn vector3_scale(value: &[f64; 3], scale: f64) -> [f64; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

fn vector3_dot(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn vector3_cross(left: &[f64; 3], right: &[f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn matrix3_vector_mul(matrix: &[[f64; 3]; 3], vector: &[f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

fn matrix3_mul(left: &[[f64; 3]; 3], right: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            left[row][0] * right[0][column]
                + left[row][1] * right[1][column]
                + left[row][2] * right[2][column]
        })
    })
}

fn matrix3_add(left: &[[f64; 3]; 3], right: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    MATRIX3_ADD_CALLS.fetch_add(1, Ordering::SeqCst);
    std::array::from_fn(|row| std::array::from_fn(|column| left[row][column] + right[row][column]))
}

fn matrix3_transpose(value: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| value[column][row]))
}

impl SpatialRepresentation<f64> for Probe {
    type Vector3 = [f64; 3];
    type Vector6 = [f64; 6];
    type Matrix3 = [[f64; 3]; 3];
    type Matrix6 = [[f64; 6]; 6];
    type Rotation3 = [[f64; 3]; 3];

    fn vector3_from_array(value: [f64; 3]) -> Self::Vector3 {
        record_array_conversion();
        value
    }

    fn vector3_to_array(value: &Self::Vector3) -> [f64; 3] {
        record_array_conversion();
        *value
    }

    fn vector6_from_array(value: [f64; 6]) -> Self::Vector6 {
        record_array_conversion();
        value
    }

    fn vector6_to_array(value: &Self::Vector6) -> [f64; 6] {
        record_array_conversion();
        *value
    }

    fn matrix3_from_array(value: [[f64; 3]; 3]) -> Self::Matrix3 {
        record_array_conversion();
        value
    }

    fn matrix3_to_array(value: &Self::Matrix3) -> [[f64; 3]; 3] {
        record_array_conversion();
        *value
    }

    fn matrix6_from_array(value: [[f64; 6]; 6]) -> Self::Matrix6 {
        record_array_conversion();
        value
    }

    fn matrix6_to_array(value: &Self::Matrix6) -> [[f64; 6]; 6] {
        record_array_conversion();
        *value
    }

    fn rotation3_from_array(value: [[f64; 3]; 3]) -> Self::Rotation3 {
        record_array_conversion();
        value
    }

    fn rotation3_to_array(value: &Self::Rotation3) -> [[f64; 3]; 3] {
        record_array_conversion();
        *value
    }

    fn vector3_zero() -> Self::Vector3 {
        [0.0; 3]
    }

    fn vector3_add(left: &Self::Vector3, right: &Self::Vector3) -> Self::Vector3 {
        vector3_add(left, right)
    }

    fn vector3_sub(left: &Self::Vector3, right: &Self::Vector3) -> Self::Vector3 {
        vector3_sub(left, right)
    }

    fn vector3_neg(value: &Self::Vector3) -> Self::Vector3 {
        vector3_neg(value)
    }

    fn vector3_scale(value: &Self::Vector3, scale: f64) -> Self::Vector3 {
        vector3_scale(value, scale)
    }

    fn vector3_dot(left: &Self::Vector3, right: &Self::Vector3) -> f64 {
        vector3_dot(left, right)
    }

    fn vector3_cross(left: &Self::Vector3, right: &Self::Vector3) -> Self::Vector3 {
        vector3_cross(left, right)
    }

    fn vector3_is_finite(value: &Self::Vector3) -> bool {
        value.iter().all(|component| component.is_finite())
    }

    fn matrix3_add(left: &Self::Matrix3, right: &Self::Matrix3) -> Self::Matrix3 {
        matrix3_add(left, right)
    }

    fn rotation3_identity() -> Self::Rotation3 {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    fn rotation3_inverse(value: &Self::Rotation3) -> Self::Rotation3 {
        matrix3_transpose(value)
    }

    fn rotation3_mul(left: &Self::Rotation3, right: &Self::Rotation3) -> Self::Rotation3 {
        matrix3_mul(left, right)
    }

    fn rotation3_transform_vector(
        rotation: &Self::Rotation3,
        vector: &Self::Vector3,
    ) -> Self::Vector3 {
        matrix3_vector_mul(rotation, vector)
    }
}

#[test]
fn native_operations_do_not_round_trip_through_arrays() {
    let velocity = MotionVector::<f64, Probe>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let motion = MotionVector::<f64, Probe>::new([7.0, 8.0, 9.0], [10.0, 11.0, 12.0]);
    let force = ForceVector::<f64, Probe>::new([10.0, 11.0, 12.0], [13.0, 14.0, 15.0]);
    let transform = SpatialTransform::<f64, Probe>::new(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [3.0, 5.0, 7.0],
    );
    let next = SpatialTransform::<f64, Probe>::new(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [-2.0, 1.0, 0.5],
    );
    let left_inertia = RigidBodyInertia::<f64, Probe>::try_new(
        2.0,
        [1.0, 2.0, 3.0],
        [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
    )
    .unwrap();
    let right_inertia = RigidBodyInertia::<f64, Probe>::try_new(
        3.0,
        [-2.0, 1.0, 0.5],
        [[7.0, 0.2, -0.1], [0.2, 8.0, 0.3], [-0.1, 0.3, 9.0]],
    )
    .unwrap();

    ARRAY_CONVERSION_CALLS.store(0, Ordering::SeqCst);
    MATRIX3_ADD_CALLS.store(0, Ordering::SeqCst);

    let matrix_sum = <Probe as SpatialRepresentation<f64>>::matrix3_add(
        &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        &[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
    );
    assert_eq!(matrix_sum, [[10.0; 3]; 3]);
    assert_eq!(MATRIX3_ADD_CALLS.load(Ordering::SeqCst), 1);

    assert_eq!(MotionVector::<f64, Probe>::zeros().angular, [0.0; 3]);
    assert_eq!(MotionVector::<f64, Probe>::zeros().linear, [0.0; 3]);
    assert!(velocity.is_finite());
    assert!(!MotionVector::<f64, Probe>::new([f64::NAN; 3], [0.0; 3]).is_finite());

    let sum = velocity + motion;
    assert_eq!(sum.angular, [8.0, 10.0, 12.0]);
    assert_eq!(sum.linear, [14.0, 16.0, 18.0]);

    let difference = velocity - motion;
    assert_eq!(difference.angular, [-6.0, -6.0, -6.0]);
    assert_eq!(difference.linear, [-6.0, -6.0, -6.0]);

    let negated = -velocity;
    assert_eq!(negated.angular, [-1.0, -2.0, -3.0]);
    assert_eq!(negated.linear, [-4.0, -5.0, -6.0]);

    let scaled = 2.0 * velocity;
    assert_eq!(scaled.angular, [2.0, 4.0, 6.0]);
    assert_eq!(scaled.linear, [8.0, 10.0, 12.0]);

    assert_eq!(velocity.dot(&force), 280.0);

    let crossed_motion = velocity.cross_motion(&motion);
    assert_eq!(crossed_motion.angular, [-6.0, 12.0, -6.0]);
    assert_eq!(crossed_motion.linear, [-12.0, 24.0, -12.0]);

    let crossed_force = velocity.cross_force(&force);
    assert_eq!(crossed_force.moment, [-18.0, 36.0, -18.0]);
    assert_eq!(crossed_force.force, [-12.0, 24.0, -12.0]);

    let transformed_motion = transform.transform_motion(&velocity);
    assert_eq!(transformed_motion.angular, [-2.0, 1.0, 3.0]);
    assert_eq!(transformed_motion.linear, [-7.0, 3.0, 5.0]);

    let transformed_force = transform.transform_force(&force);
    assert_eq!(transformed_force.moment, [35.0, 33.0, 35.0]);
    assert_eq!(transformed_force.force, [-14.0, 13.0, 15.0]);

    let inverse = transform.inverse();
    assert_eq!(
        *inverse.rotation(),
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    );
    assert_eq!(*inverse.translation(), [5.0, -3.0, -7.0]);

    let round_trip = inverse.transform_motion(&transformed_motion);
    assert_eq!(round_trip.angular, velocity.angular);
    assert_eq!(round_trip.linear, velocity.linear);

    let composed = transform.then(&next);
    let sequential = next.transform_motion(&transform.transform_motion(&velocity));
    let composed_motion = composed.transform_motion(&velocity);
    assert_eq!(composed_motion.angular, sequential.angular);
    assert_eq!(composed_motion.linear, sequential.linear);

    assert_eq!(ARRAY_CONVERSION_CALLS.load(Ordering::SeqCst), 0);

    MATRIX3_ADD_CALLS.store(0, Ordering::SeqCst);
    let combined = left_inertia.try_combined(&right_inertia).unwrap();
    assert_eq!(combined.mass(), 5.0);
    assert_eq!(MATRIX3_ADD_CALLS.load(Ordering::SeqCst), 1);
}
