// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use spatial6::{
    ForceVector, MotionVector, RigidBodyInertia, SpatialRepresentation, SpatialScalar,
    SpatialTransform,
};

fn assert_num_traits_float<T: num_traits::Float>() {}

fn assert_spatial_scalar_is_float<T: SpatialScalar>() {
    assert_num_traits_float::<T>();
}

fn assert_close(actual: f64, expected: f64) {
    let tolerance = 1.0e-10 + 1.0e-10 * actual.abs().max(expected.abs());
    assert!((actual - expected).abs() <= tolerance);
}

#[test]
fn spatial_scalar_uses_num_traits_float() {
    assert_spatial_scalar_is_float::<f32>();
    assert_spatial_scalar_is_float::<f64>();
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Custom;

impl SpatialRepresentation<f64> for Custom {
    type Vector3 = [f64; 3];
    type Vector6 = [f64; 6];
    type Matrix3 = [[f64; 3]; 3];
    type Matrix6 = [[f64; 6]; 6];
    type Rotation3 = [[f64; 3]; 3];

    fn vector3_from_array(value: [f64; 3]) -> Self::Vector3 {
        value
    }

    fn vector3_to_array(value: &Self::Vector3) -> [f64; 3] {
        *value
    }

    fn vector6_from_array(value: [f64; 6]) -> Self::Vector6 {
        value
    }

    fn vector6_to_array(value: &Self::Vector6) -> [f64; 6] {
        *value
    }

    fn matrix3_from_array(value: [[f64; 3]; 3]) -> Self::Matrix3 {
        value
    }

    fn matrix3_to_array(value: &Self::Matrix3) -> [[f64; 3]; 3] {
        *value
    }

    fn matrix6_from_array(value: [[f64; 6]; 6]) -> Self::Matrix6 {
        value
    }

    fn matrix6_to_array(value: &Self::Matrix6) -> [[f64; 6]; 6] {
        *value
    }

    fn rotation3_from_array(value: [[f64; 3]; 3]) -> Self::Rotation3 {
        value
    }

    fn rotation3_to_array(value: &Self::Rotation3) -> [[f64; 3]; 3] {
        *value
    }
}

#[test]
fn default_matrix3_add_preserves_row_major_order() {
    let left = [[1.0, 2.0, 4.0], [8.0, 16.0, 32.0], [64.0, 128.0, 256.0]];
    let right = [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5], [6.5, 7.5, 8.5]];

    assert_eq!(
        <Custom as SpatialRepresentation<f64>>::matrix3_add(&left, &right),
        [[1.5, 3.5, 6.5], [11.5, 20.5, 37.5], [70.5, 135.5, 264.5]]
    );
}

#[test]
fn user_defined_representation_drives_the_complete_api() {
    let velocity = MotionVector::<f64, Custom>::from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let force = ForceVector::<f64, Custom>::from_vector([10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);

    assert_eq!(velocity.to_vector(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(velocity.dot(&force), 280.0);
    assert_eq!(
        velocity.cross_force(&force).to_vector(),
        [-18.0, 36.0, -18.0, -12.0, 24.0, -12.0]
    );

    let transform = SpatialTransform::<f64, Custom>::new(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [3.0, 5.0, 7.0],
    );
    assert_eq!(
        transform.transform_motion(&velocity).to_vector(),
        [-2.0, 1.0, 3.0, -7.0, 3.0, 5.0]
    );

    let inertia = RigidBodyInertia::<f64, Custom>::try_new(
        2.0,
        [1.0, 2.0, 3.0],
        [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
    )
    .unwrap();
    let other = RigidBodyInertia::<f64, Custom>::try_new(
        3.0,
        [-2.0, 1.0, 0.5],
        [[7.0, 0.2, -0.1], [0.2, 8.0, 0.3], [-0.1, 0.3, 9.0]],
    )
    .unwrap();
    let combined = inertia.try_combined(&other).unwrap();
    assert_close(combined.mass(), 5.0);
    for (actual, expected) in (*combined.center_of_mass())
        .into_iter()
        .zip([-0.8, 1.4, 1.5])
    {
        assert_close(actual, expected);
    }
    let acceleration = MotionVector::<f64, Custom>::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    assert_eq!(
        inertia.apply(&acceleration).to_vector(),
        [24.0, -4.0, -4.0, 0.0, -4.0, 4.0]
    );
    for (actual, expected) in combined
        .apply(&acceleration)
        .to_vector()
        .into_iter()
        .zip((inertia.apply(&acceleration) + other.apply(&acceleration)).to_vector())
    {
        assert_close(actual, expected);
    }
}

#[cfg(feature = "builtin")]
#[test]
fn builtin_representation_is_the_default() {
    use spatial6::Builtin;

    let default: MotionVector = MotionVector::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let explicit = MotionVector::<f64, Builtin>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);

    assert_eq!(default.to_vector(), explicit.to_vector());
}
