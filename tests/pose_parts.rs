#![cfg(any(feature = "builtin", feature = "nalgebra", feature = "glam"))]

// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use spatial6::{SpatialRepresentation, SpatialScalar, SpatialTransform};

#[cfg(feature = "builtin")]
use spatial6::Builtin;

type Matrix3 = [[f64; 3]; 3];
type Matrix4 = [[f64; 4]; 4];
type Vector3 = [f64; 3];

fn scalar<T: SpatialScalar>(value: f64) -> T {
    T::from(value).expect("test value must be representable")
}

fn rotation<T, R>(value: Matrix3) -> R::Rotation3
where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
{
    R::rotation3_from_array(value.map(|row| row.map(scalar::<T>)))
}

fn vector<T, R>(value: Vector3) -> R::Vector3
where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
{
    R::vector3_from_array(value.map(scalar::<T>))
}

fn transform<T, R>(rotation_value: Matrix3, translation: Vector3) -> SpatialTransform<T, R>
where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
{
    SpatialTransform::new(
        rotation::<T, R>(rotation_value),
        vector::<T, R>(translation),
    )
}

fn transpose(value: Matrix3) -> Matrix3 {
    std::array::from_fn(|row| std::array::from_fn(|column| value[column][row]))
}

fn homogeneous_pose(rotation: Matrix3, position: Vector3) -> Matrix4 {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| match (row, column) {
            (0..3, 0..3) => rotation[row][column],
            (0..3, 3) => position[row],
            (3, 3) => 1.0,
            _ => 0.0,
        })
    })
}

fn matrix4_mul(left: Matrix4, right: Matrix4) -> Matrix4 {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..4)
                .map(|index| left[row][index] * right[index][column])
                .sum()
        })
    })
}

fn compose_pose(left: (Matrix3, Vector3), right: (Matrix3, Vector3)) -> (Matrix3, Vector3) {
    let product = matrix4_mul(
        homogeneous_pose(left.0, left.1),
        homogeneous_pose(right.0, right.1),
    );
    let rotation = std::array::from_fn(|row| std::array::from_fn(|column| product[row][column]));
    let position = std::array::from_fn(|row| product[row][3]);
    (rotation, position)
}

fn assert_close<T: SpatialScalar>(actual: T, expected: T) {
    let scale = T::one().max(actual.abs()).max(expected.abs());
    let tolerance = scalar::<T>(64.0) * T::epsilon() * scale;
    assert!((actual - expected).abs() <= tolerance);
}

fn assert_matrix_close<T: SpatialScalar>(actual: [[T; 3]; 3], expected: Matrix3) {
    for (actual_row, expected_row) in actual.into_iter().zip(expected) {
        for (actual, expected) in actual_row.into_iter().zip(expected_row) {
            assert_close(actual, scalar::<T>(expected));
        }
    }
}

fn assert_vector_close<T: SpatialScalar>(actual: [T; 3], expected: Vector3) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_close(actual, scalar::<T>(expected));
    }
}

fn assert_point_mapping<T: SpatialScalar>(
    rotation: [[T; 3]; 3],
    position: [T; 3],
    point: Vector3,
    expected: Vector3,
) {
    for row in 0..3 {
        let actual = (0..3)
            .map(|column| rotation[row][column] * scalar::<T>(point[column]))
            .fold(position[row], |sum, value| sum + value);
        assert_close(actual, scalar::<T>(expected[row]));
    }
}

fn assert_parts<T, R>(
    parts: &(R::Rotation3, R::Vector3),
    expected_rotation: Matrix3,
    expected_position: Vector3,
) where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
{
    assert_matrix_close::<T>(R::rotation3_to_array(&parts.0), expected_rotation);
    assert_vector_close::<T>(R::vector3_to_array(&parts.1), expected_position);
}

fn assert_round_trip<T, R>(
    original: &SpatialTransform<T, R>,
    expected_rotation: Matrix3,
    expected_position: Vector3,
) where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
{
    let parts = original.to_pose_parts();
    assert_parts::<T, R>(&parts, expected_rotation, expected_position);
    let restored = SpatialTransform::<T, R>::from_pose_parts(parts.0, parts.1);
    assert_eq!(restored, *original);
}

fn rz(angle: f64) -> Matrix3 {
    let (sin, cos) = angle.sin_cos();
    [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]
}

fn rx(angle: f64) -> Matrix3 {
    let (sin, cos) = angle.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]]
}

fn ry(angle: f64) -> Matrix3 {
    let (sin, cos) = angle.sin_cos();
    [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]]
}

fn pose_parts_contract<T, R>()
where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
{
    let e_ab = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let r_ab = [1.0, 2.0, 3.0];
    let q_ab = transpose(e_ab);

    // P2-01, P2-02: both conversion directions and their independent conventions.
    let original = transform::<T, R>(e_ab, r_ab);
    assert_round_trip::<T, R>(&original, q_ab, r_ab);
    let supplied_parts =
        SpatialTransform::<T, R>::from_pose_parts(rotation::<T, R>(q_ab), vector::<T, R>(r_ab));
    assert_parts::<T, R>(&supplied_parts.to_pose_parts(), q_ab, r_ab);

    // P2-03, P2-04: fixed non-commuting chain and point equation.
    let e_bc = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
    let r_bc = [4.0, 5.0, 6.0];
    let chain = original.then(&transform::<T, R>(e_bc, r_bc));
    let chain_parts = chain.to_pose_parts();
    let expected_chain = compose_pose((q_ab, r_ab), (transpose(e_bc), r_bc));
    assert_parts::<T, R>(&chain_parts, expected_chain.0, expected_chain.1);
    assert_point_mapping::<T>(
        R::rotation3_to_array(&chain_parts.0),
        R::vector3_to_array(&chain_parts.1),
        [2.0, -1.0, 4.0],
        [10.0, -4.0, 10.0],
    );

    // P2-04: general-angle prefixes, with expected values composed as raw 4x4
    // homogeneous pose matrices.
    let e_cd = ry(0.22);
    let r_cd = [2.0, -1.25, 4.5];
    let e_ab_angle = rz(0.37);
    let r_ab_angle = [1.25, -2.5, 3.75];
    let e_bc_angle = rx(-0.61);
    let r_bc_angle = [-4.5, 5.25, 0.75];
    let x_ab = transform::<T, R>(e_ab_angle, r_ab_angle);
    let x_bc = transform::<T, R>(e_bc_angle, r_bc_angle);
    let x_cd = transform::<T, R>(e_cd, r_cd);
    let x_abc = x_ab.then(&x_bc);
    let x_abcd = x_abc.then(&x_cd);
    let expected_ab = (transpose(e_ab_angle), r_ab_angle);
    let expected_abc = compose_pose(expected_ab, (transpose(e_bc_angle), r_bc_angle));
    let expected_abcd = compose_pose(expected_abc, (transpose(e_cd), r_cd));
    assert_parts::<T, R>(&x_ab.to_pose_parts(), expected_ab.0, expected_ab.1);
    assert_parts::<T, R>(&x_abc.to_pose_parts(), expected_abc.0, expected_abc.1);
    assert_parts::<T, R>(&x_abcd.to_pose_parts(), expected_abcd.0, expected_abcd.1);

    // P2-05: stored fields and inverse pose have distinct translation semantics.
    assert_matrix_close::<T>(R::rotation3_to_array(original.rotation()), e_ab);
    assert_vector_close::<T>(R::vector3_to_array(original.translation()), r_ab);
    let inverse = original.inverse();
    assert_matrix_close::<T>(R::rotation3_to_array(inverse.rotation()), transpose(e_ab));
    assert_vector_close::<T>(
        R::vector3_to_array(inverse.translation()),
        [2.0, -1.0, -3.0],
    );
    let inverse_parts = inverse.to_pose_parts();
    assert_parts::<T, R>(&inverse_parts, e_ab, [2.0, -1.0, -3.0]);

    // P2-06: identity, pure translation, pure rotation, negative translation,
    // and a non-identity transform used as the composition start.
    let identity = SpatialTransform::identity();
    assert_round_trip::<T, R>(
        &identity,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0, 0.0, 0.0],
    );
    assert_round_trip::<T, R>(
        &transform::<T, R>(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [4.0, 5.0, 6.0],
        ),
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [4.0, 5.0, 6.0],
    );
    assert_round_trip::<T, R>(&transform::<T, R>(e_ab, [0.0, 0.0, 0.0]), q_ab, [0.0; 3]);
    assert_round_trip::<T, R>(
        &transform::<T, R>(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [-4.0, -5.0, -6.0],
        ),
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [-4.0, -5.0, -6.0],
    );
    let start = transform::<T, R>(e_ab, r_ab);
    let relative = transform::<T, R>(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [4.0, 5.0, 6.0],
    );
    let started_non_identity = start.then(&relative);
    let expected_started_non_identity = compose_pose(
        (q_ab, r_ab),
        (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [4.0, 5.0, 6.0],
        ),
    );
    assert_parts::<T, R>(
        &started_non_identity.to_pose_parts(),
        expected_started_non_identity.0,
        expected_started_non_identity.1,
    );

    // P2-11: returned values are owned, Copy, and reusable without consuming the source.
    let parts = original.to_pose_parts();
    let copied_parts = parts;
    let restored = SpatialTransform::<T, R>::from_pose_parts(parts.0, parts.1);
    let restored_again = SpatialTransform::<T, R>::from_pose_parts(copied_parts.0, copied_parts.1);
    assert_eq!(restored, restored_again);
    assert_eq!(original.to_pose_parts(), parts);
}

#[cfg(feature = "builtin")]
#[test]
fn builtin_f64_pose_parts_round_trip() {
    pose_parts_contract::<f64, Builtin>();
}

#[cfg(any(feature = "nalgebra", feature = "glam"))]
macro_rules! backend_tests {
    ($module:ident, $backend:ty) => {
        mod $module {
            #[test]
            fn pose_parts_f64() {
                super::pose_parts_contract::<f64, $backend>();
            }

            #[test]
            fn pose_parts_f32() {
                super::pose_parts_contract::<f32, $backend>();
            }
        }
    };
}

#[cfg(feature = "builtin")]
mod builtin_f32 {
    #[test]
    fn pose_parts_f32() {
        super::pose_parts_contract::<f32, spatial6::Builtin>();
    }
}

#[cfg(feature = "nalgebra")]
backend_tests!(nalgebra_backend, spatial6::Nalgebra);

#[cfg(feature = "glam")]
backend_tests!(glam_backend, spatial6::Glam);

#[cfg(feature = "builtin")]
#[test]
fn builtin_f64_pose_parts_accept_invalid_and_nonfinite_values() {
    let invalid_rotation = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
    let position = [f64::NAN, f64::INFINITY, -0.0];
    let transform = SpatialTransform::<f64>::from_pose_parts(invalid_rotation, position);
    let (rotation, position) = transform.to_pose_parts();

    assert_eq!(rotation, invalid_rotation);
    assert!(position[0].is_nan());
    assert!(position[1].is_infinite());
    assert_eq!(position[2].to_bits(), (-0.0f64).to_bits());

    let nan_rotation = [[f64::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let (rotation, _) =
        SpatialTransform::<f64>::from_pose_parts(nan_rotation, [0.0, 0.0, 0.0]).to_pose_parts();
    assert!(rotation[0][0].is_nan());

    let large_position = [f64::MAX, f64::NEG_INFINITY, 0.0];
    let (_, returned_position) =
        SpatialTransform::<f64>::from_pose_parts(invalid_rotation, large_position).to_pose_parts();
    assert_eq!(returned_position[0].to_bits(), f64::MAX.to_bits());
    assert!(returned_position[1].is_infinite());
    assert!(returned_position[1].is_sign_negative());
}
