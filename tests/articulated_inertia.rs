#![allow(clippy::needless_range_loop)]

// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "builtin")]
mod builtin {
    use std::mem::size_of;

    use spatial6::{
        ArticulatedBodyInertia, ForceVector, InertiaError, MotionVector, RigidBodyInertia,
        SpatialRepresentation, SpatialTransform,
    };

    type Matrix6 = [[f64; 6]; 6];

    fn matrix_fixture() -> Matrix6 {
        [
            [8.0, 0.2, -0.3, 1.0, 0.5, -0.2],
            [0.2, 9.0, 0.4, -0.1, 0.7, 0.3],
            [-0.3, 0.4, 10.0, 0.6, -0.4, 0.8],
            [1.0, -0.1, 0.6, 3.0, 0.2, -0.5],
            [0.5, 0.7, -0.4, 0.2, 4.0, 0.9],
            [-0.2, 0.3, 0.8, -0.5, 0.9, 5.0],
        ]
    }

    fn assert_matrix_close(actual: Matrix6, expected: Matrix6, tolerance: f64) {
        for (actual_row, expected_row) in actual.into_iter().zip(expected) {
            for (actual, expected) in actual_row.into_iter().zip(expected_row) {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn validates_finite_then_symmetry_and_canonicalizes_upper_triangle() {
        let mut matrix = matrix_fixture();
        let delta = 32.0 * f64::EPSILON;
        matrix[0][1] += delta;
        let inertia = ArticulatedBodyInertia::<f64>::try_from_matrix(matrix).unwrap();
        let expected = (0.2 + (0.2 + delta)) / 2.0;
        assert_eq!(inertia.matrix()[0][1], expected);
        assert_eq!(inertia.matrix()[1][0], expected);

        matrix[0][1] = 0.2 + 65.0 * f64::EPSILON;
        assert_eq!(
            ArticulatedBodyInertia::<f64>::try_from_matrix(matrix),
            Err(InertiaError::NonSymmetric)
        );

        matrix[0][1] = f64::NAN;
        assert_eq!(
            ArticulatedBodyInertia::<f64>::try_from_matrix(matrix),
            Err(InertiaError::NonFinite)
        );
    }

    #[test]
    fn finite_extreme_pairs_use_an_overflow_safe_average() {
        let mut matrix = [[0.0; 6]; 6];
        for diagonal in 0..6 {
            matrix[diagonal][diagonal] = 1.0;
        }
        matrix[0][1] = f64::MAX;
        matrix[1][0] = f64::MAX;
        let inertia = ArticulatedBodyInertia::<f64>::try_from_matrix(matrix).unwrap();
        assert_eq!(inertia.matrix()[0][1], f64::MAX);
    }

    #[test]
    fn zeros_apply_and_have_the_fixed_packed_size() {
        let inertia = ArticulatedBodyInertia::<f64>::zeros();
        assert_eq!(inertia.matrix(), [[0.0; 6]; 6]);
        assert_eq!(
            inertia.apply(&MotionVector::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            ForceVector::zeros()
        );
        assert_eq!(
            size_of::<ArticulatedBodyInertia<f64>>(),
            21 * size_of::<f64>()
        );
    }

    #[test]
    fn f32_uses_the_same_packed_representation_and_operations() {
        let mut matrix = [[0.0_f32; 6]; 6];
        for diagonal in 0..6 {
            matrix[diagonal][diagonal] = 1.0;
        }
        matrix[0][1] = 0.25;
        matrix[1][0] = 0.25;
        let inertia = ArticulatedBodyInertia::<f32>::try_from_matrix(matrix).unwrap();
        let motion = MotionVector::<f32>::from_array([1.0, -2.0, 0.5, 3.0, -1.0, 2.0]);
        assert_eq!(
            inertia.apply(&motion).to_array(),
            spatial6::Builtin::matrix6_vector_mul(&inertia.matrix(), &motion.to_vector())
        );
        assert_eq!(inertia.try_combined(&inertia).unwrap().matrix()[0][0], 2.0);
    }

    #[test]
    fn apply_matches_the_backend_matrix_multiply() {
        let inertia = ArticulatedBodyInertia::<f64>::try_from_matrix(matrix_fixture()).unwrap();
        let motion = MotionVector::from_array([0.3, -0.8, 1.1, 2.0, -1.5, 0.4]);
        let expected =
            spatial6::Builtin::matrix6_vector_mul(&inertia.matrix(), &motion.to_vector());
        assert_eq!(inertia.apply(&motion).to_vector(), expected);
    }

    #[test]
    fn combines_and_reports_derived_non_finite_values() {
        let left = ArticulatedBodyInertia::<f64>::try_from_matrix(matrix_fixture()).unwrap();
        let right = ArticulatedBodyInertia::<f64>::try_from_matrix({
            let mut matrix = [[0.0; 6]; 6];
            for diagonal in 0..6 {
                matrix[diagonal][diagonal] = 2.0;
            }
            matrix
        })
        .unwrap();
        let combined = left.try_combined(&right).unwrap();
        assert_eq!(combined.matrix()[0][0], 10.0);
        assert_eq!(combined.matrix()[0][1], 0.2);

        let huge = ArticulatedBodyInertia::<f64>::try_from_matrix({
            let mut matrix = [[0.0; 6]; 6];
            for diagonal in 0..6 {
                matrix[diagonal][diagonal] = f64::MAX;
            }
            matrix
        })
        .unwrap();
        assert_eq!(huge.try_combined(&huge), Err(InertiaError::NonFinite));
    }

    #[test]
    fn rank_one_update_handles_sign_zero_and_overflow() {
        let inertia = ArticulatedBodyInertia::<f64>::try_from_matrix(matrix_fixture()).unwrap();
        let u = ForceVector::from_array([1.0, -2.0, 0.5, 3.0, -1.0, 2.0]);
        let updated = inertia.try_rank_one_updated(-0.5, &u).unwrap();
        let mut expected = matrix_fixture();
        let coordinates = u.to_array();
        for row in 0..6 {
            for column in 0..6 {
                expected[row][column] += -0.5 * coordinates[row] * coordinates[column];
            }
        }
        assert_matrix_close(updated.matrix(), expected, 1.0e-12);
        assert_eq!(inertia.try_rank_one_updated(0.0, &u).unwrap(), inertia);

        let nan_u = ForceVector::from_array([f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            inertia.try_rank_one_updated(0.0, &nan_u),
            Err(InertiaError::NonFinite)
        );

        let huge_u = ForceVector::from_array([f64::MAX, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            inertia.try_rank_one_updated(2.0, &huge_u),
            Err(InertiaError::NonFinite)
        );
    }

    #[test]
    fn transforms_by_force_congruence() {
        let inertia = ArticulatedBodyInertia::<f64>::try_from_matrix(matrix_fixture()).unwrap();
        let transform = SpatialTransform::<f64>::new(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [0.5, -1.0, 2.0],
        );
        let transformed = inertia.try_transformed(&transform).unwrap();
        let force = transform.force_matrix();
        let expected = spatial6::Builtin::matrix6_mul(
            &spatial6::Builtin::matrix6_mul(&force, &inertia.matrix()),
            &spatial6::Builtin::matrix6_transpose(&force),
        );
        assert_matrix_close(transformed.matrix(), expected, 1.0e-10);

        let bad = SpatialTransform::<f64>::new(
            [[f64::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0, 0.0, 0.0],
        );
        assert_eq!(inertia.try_transformed(&bad), Err(InertiaError::NonFinite));
    }

    #[test]
    fn converts_rigid_body_inertia_through_matrix_validation() {
        let rigid = RigidBodyInertia::<f64>::try_new(
            2.0,
            [1.0, -2.0, 0.5],
            [[3.0, 0.2, -0.1], [0.2, 4.0, 0.3], [-0.1, 0.3, 5.0]],
        )
        .unwrap();
        let articulated = ArticulatedBodyInertia::try_from(&rigid).unwrap();
        assert_eq!(articulated.matrix(), rigid.matrix());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_uses_the_fixed_packed_shape_and_rejects_bad_inputs() {
        let inertia = ArticulatedBodyInertia::<f64>::try_from_matrix(matrix_fixture()).unwrap();
        let json = serde_json::to_string(&inertia).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let packed = value.get("packed").unwrap().as_array().unwrap();
        assert_eq!(packed.len(), 21);
        assert_eq!(
            serde_json::from_str::<ArticulatedBodyInertia<f64>>(&json).unwrap(),
            inertia
        );

        let too_short = serde_json::json!({ "packed": vec![0.0; 20] });
        assert!(serde_json::from_value::<ArticulatedBodyInertia<f64>>(too_short).is_err());
        assert!(
            serde_json::from_str::<ArticulatedBodyInertia<f64>>(
                r#"{"packed":[NaN,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}"#
            )
            .is_err()
        );
    }

    #[cfg(feature = "nalgebra")]
    #[test]
    fn nalgebra_representation_uses_the_same_contract() {
        use spatial6::Nalgebra;

        let matrix = Nalgebra::matrix6_from_array(matrix_fixture());
        let inertia = ArticulatedBodyInertia::<f64, Nalgebra>::try_from_matrix(matrix).unwrap();
        let motion = MotionVector::<f64, Nalgebra>::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let expected = Nalgebra::matrix6_vector_mul(&inertia.matrix(), &motion.to_vector());
        assert_eq!(inertia.apply(&motion).to_vector(), expected);
    }

    #[cfg(feature = "glam")]
    #[test]
    fn glam_representation_uses_the_same_contract() {
        use spatial6::Glam;

        let matrix = Glam::matrix6_from_array(matrix_fixture());
        let inertia = ArticulatedBodyInertia::<f64, Glam>::try_from_matrix(matrix).unwrap();
        let motion = MotionVector::<f64, Glam>::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let expected = <Glam as SpatialRepresentation<f64>>::matrix6_vector_mul(
            &inertia.matrix(),
            &motion.to_vector(),
        );
        assert_eq!(inertia.apply(&motion).to_vector(), expected);
    }

    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn articulated_inertia_is_send_sync() {
        assert_send_sync::<ArticulatedBodyInertia<f64>>();
    }
}

#[cfg(any(feature = "builtin", feature = "nalgebra", feature = "glam"))]
mod backend_matrix {
    use std::mem::size_of;

    use spatial6::{
        ArticulatedBodyInertia, ForceVector, InertiaError, MotionVector, RigidBodyInertia,
        SpatialRepresentation, SpatialScalar, SpatialTransform,
    };

    fn scalar<T: SpatialScalar>(value: f64) -> T {
        T::from(value).expect("test value must be representable")
    }

    fn matrix<T, R>(value: [[f64; 6]; 6]) -> R::Matrix6
    where
        T: SpatialScalar,
        R: SpatialRepresentation<T>,
    {
        R::matrix6_from_array(value.map(|row| row.map(scalar::<T>)))
    }

    fn matrix3<T, R>(value: [[f64; 3]; 3]) -> R::Matrix3
    where
        T: SpatialScalar,
        R: SpatialRepresentation<T>,
    {
        R::matrix3_from_array(value.map(|row| row.map(scalar::<T>)))
    }

    fn rotation<T, R>(value: [[f64; 3]; 3]) -> R::Rotation3
    where
        T: SpatialScalar,
        R: SpatialRepresentation<T>,
    {
        R::rotation3_from_array(value.map(|row| row.map(scalar::<T>)))
    }

    fn vector<T, R>(value: [f64; 6]) -> MotionVector<T, R>
    where
        T: SpatialScalar,
        R: SpatialRepresentation<T>,
    {
        MotionVector::from_array(value.map(scalar::<T>))
    }

    fn force<T, R>(value: [f64; 6]) -> ForceVector<T, R>
    where
        T: SpatialScalar,
        R: SpatialRepresentation<T>,
    {
        ForceVector::from_array(value.map(scalar::<T>))
    }

    fn matrix_array<T, R>(value: &R::Matrix6) -> [[T; 6]; 6]
    where
        T: SpatialScalar,
        R: SpatialRepresentation<T>,
    {
        R::matrix6_to_array(value)
    }

    fn abi_matrix<T, R>(value: &ArticulatedBodyInertia<T, R>) -> [[T; 6]; 6]
    where
        T: SpatialScalar,
        R: SpatialRepresentation<T>,
    {
        matrix_array::<T, R>(&value.matrix())
    }

    fn fixture() -> [[f64; 6]; 6] {
        [
            [8.0, 0.2, -0.3, 1.0, 0.5, -0.2],
            [0.2, 9.0, 0.4, -0.1, 0.7, 0.3],
            [-0.3, 0.4, 10.0, 0.6, -0.4, 0.8],
            [1.0, -0.1, 0.6, 3.0, 0.2, -0.5],
            [0.5, 0.7, -0.4, 0.2, 4.0, 0.9],
            [-0.2, 0.3, 0.8, -0.5, 0.9, 5.0],
        ]
    }

    fn second_fixture() -> [[f64; 6]; 6] {
        [
            [1.5, -0.6, 0.7, 0.8, -0.9, 1.1],
            [-0.6, 2.5, -1.2, 1.3, 0.4, -0.5],
            [0.7, -1.2, 3.5, -0.2, 1.4, 0.6],
            [0.8, 1.3, -0.2, 4.5, -0.7, -1.5],
            [-0.9, 0.4, 1.4, -0.7, 5.5, 0.9],
            [1.1, -0.5, 0.6, -1.5, 0.9, 6.5],
        ]
    }

    #[cfg(feature = "serde")]
    fn distinctive_fixture() -> [[f64; 6]; 6] {
        let mut value = [[0.0; 6]; 6];
        for row in 0..6 {
            for column in row..6 {
                let entry = 100.0 * row as f64 + column as f64 + 0.25;
                value[row][column] = entry;
                value[column][row] = entry;
            }
        }
        value
    }

    fn matrix_add<T: SpatialScalar>(left: [[T; 6]; 6], right: [[T; 6]; 6]) -> [[T; 6]; 6] {
        std::array::from_fn(|row| {
            std::array::from_fn(|column| left[row][column] + right[row][column])
        })
    }

    fn matrix_mul<T: SpatialScalar>(left: [[T; 6]; 6], right: [[T; 6]; 6]) -> [[T; 6]; 6] {
        std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                (0..6).fold(T::zero(), |sum, index| {
                    sum + left[row][index] * right[index][column]
                })
            })
        })
    }

    fn matrix_vector_mul<T: SpatialScalar>(matrix: [[T; 6]; 6], vector: [T; 6]) -> [T; 6] {
        std::array::from_fn(|row| {
            (0..6).fold(T::zero(), |sum, column| {
                sum + matrix[row][column] * vector[column]
            })
        })
    }

    fn transpose<T: SpatialScalar>(value: [[T; 6]; 6]) -> [[T; 6]; 6] {
        std::array::from_fn(|row| std::array::from_fn(|column| value[column][row]))
    }

    fn assert_scalar_close<T: SpatialScalar>(actual: T, expected: T, absolute: f64, relative: f64) {
        let absolute = scalar::<T>(absolute);
        let relative = scalar::<T>(relative);
        let tolerance = absolute + relative * actual.abs().max(expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual:?} differs from {expected:?} by more than {tolerance:?}"
        );
    }

    fn assert_matrix_close<T: SpatialScalar, R: SpatialRepresentation<T>>(
        actual: &ArticulatedBodyInertia<T, R>,
        expected: [[T; 6]; 6],
        absolute: f64,
        relative: f64,
    ) {
        for (actual_row, expected_row) in abi_matrix(actual).into_iter().zip(expected) {
            for (actual, expected) in actual_row.into_iter().zip(expected_row) {
                assert_scalar_close(actual, expected, absolute, relative);
            }
        }
    }

    fn assert_vector_close<T: SpatialScalar>(
        actual: [T; 6],
        expected: [T; 6],
        absolute: f64,
        relative: f64,
    ) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_scalar_close(actual, expected, absolute, relative);
        }
    }

    fn assert_strictly_symmetric<T: SpatialScalar, R: SpatialRepresentation<T>>(
        value: &ArticulatedBodyInertia<T, R>,
    ) {
        let value = abi_matrix(value);
        for row in 0..6 {
            for column in (row + 1)..6 {
                assert_eq!(value[row][column], value[column][row]);
            }
        }
    }

    fn construction_and_normalization<T, R>()
    where
        T: SpatialScalar,
        R: SpatialRepresentation<T>,
    {
        let zero =
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>([[0.0; 6]; 6])).unwrap();
        assert_eq!(zero, ArticulatedBodyInertia::zeros());

        let mut spd = [[0.0; 6]; 6];
        for index in 0..6 {
            spd[index][index] = 1.0 + index as f64;
        }
        let singular = [[0.0; 6]; 6];
        let mut indefinite = [[0.0; 6]; 6];
        indefinite[0][0] = -1.0;
        indefinite[1][1] = 2.0;
        assert!(ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(spd)).is_ok());
        assert!(ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(singular)).is_ok());
        assert!(
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(indefinite)).is_ok()
        );

        for nonfinite in [T::nan(), T::infinity(), T::neg_infinity()] {
            let mut invalid = fixture();
            invalid[0][0] = nonfinite.to_f64().unwrap();
            assert_eq!(
                ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(invalid)),
                Err(InertiaError::NonFinite)
            );
        }
        let mut priority = fixture();
        priority[0][1] = f64::NAN;
        priority[1][0] = 100.0;
        assert_eq!(
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(priority)),
            Err(InertiaError::NonFinite)
        );

        let epsilon = T::epsilon();
        let base = T::one();
        let inside = base + epsilon * scalar::<T>(64.0);
        let mut boundary = [[0.0; 6]; 6];
        boundary[0][1] = inside.to_f64().unwrap();
        boundary[1][0] = 1.0;
        let boundary_value =
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(boundary)).unwrap();
        assert_strictly_symmetric(&boundary_value);

        let outside = base + epsilon * scalar::<T>(65.0);
        boundary[0][1] = outside.to_f64().unwrap();
        assert_eq!(
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(boundary)),
            Err(InertiaError::NonSymmetric)
        );

        let large_base = scalar::<T>(1.0e8);
        let relative_inside = large_base + epsilon * large_base * scalar::<T>(32.0);
        boundary[0][1] = relative_inside.to_f64().unwrap();
        boundary[1][0] = large_base.to_f64().unwrap();
        assert!(ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(boundary)).is_ok());
        let relative_outside = large_base + epsilon * large_base * scalar::<T>(128.0);
        boundary[0][1] = relative_outside.to_f64().unwrap();
        assert_eq!(
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(boundary)),
            Err(InertiaError::NonSymmetric)
        );

        let maximum = T::max_value();
        let mut extremes = [[0.0; 6]; 6];
        extremes[0][1] = maximum.to_f64().unwrap();
        extremes[1][0] = maximum.to_f64().unwrap();
        let extremes =
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(extremes)).unwrap();
        assert_eq!(abi_matrix(&extremes)[0][1], maximum);

        let tiny = T::min_positive_value() / scalar::<T>(2.0);
        let mut tiny_matrix = [[0.0; 6]; 6];
        tiny_matrix[0][1] = tiny.to_f64().unwrap();
        tiny_matrix[1][0] = tiny.to_f64().unwrap();
        let tiny_value =
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(tiny_matrix)).unwrap();
        assert_eq!(abi_matrix(&tiny_value)[0][1], tiny);
    }

    fn public_traits_and_layout<T, R>()
    where
        T: SpatialScalar,
        R: SpatialRepresentation<T>,
        ArticulatedBodyInertia<T, R>: Send + Sync + Clone + Copy + std::fmt::Debug + PartialEq,
    {
        assert_eq!(
            size_of::<ArticulatedBodyInertia<T, R>>(),
            21 * size_of::<T>()
        );
        let value = ArticulatedBodyInertia::<T, R>::zeros();
        let copy = value;
        assert_eq!(value, copy);
    }

    fn rigid_conversion<T, R>(absolute: f64, relative: f64)
    where
        T: SpatialScalar,
        R: SpatialRepresentation<T>,
    {
        let rigid = RigidBodyInertia::<T, R>::try_new(
            scalar(2.0),
            R::vector3_from_array([scalar(1.0), scalar(-2.0), scalar(0.5)]),
            matrix3::<T, R>([[3.0, 0.2, -0.1], [0.2, 4.0, 0.3], [-0.1, 0.3, 5.0]]),
        )
        .unwrap();
        let articulated = ArticulatedBodyInertia::try_from(&rigid).unwrap();
        let rigid_matrix = R::matrix6_to_array(&rigid.matrix());
        assert_matrix_close(&articulated, rigid_matrix, absolute, relative);

        let motion = vector::<T, R>([0.3, -0.8, 1.1, 2.0, -1.5, 0.4]);
        let articulated_force = articulated.apply(&motion).to_array();
        let rigid_force = rigid.apply(&motion).to_array();
        assert_vector_close(articulated_force, rigid_force, absolute, relative);

        let transform = SpatialTransform::<T, R>::new(
            rotation::<T, R>([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            R::vector3_from_array([scalar(0.5), scalar(-1.0), scalar(2.0)]),
        );
        let transformed = articulated.try_transformed(&transform).unwrap();
        let rigid_transformed = rigid.transformed(&transform);
        assert_matrix_close(
            &transformed,
            R::matrix6_to_array(&rigid_transformed.matrix()),
            absolute,
            relative,
        );

        let maximum = T::max_value();
        let overflow_rigid = RigidBodyInertia::<T, R>::try_new(
            maximum,
            R::vector3_from_array([maximum, T::zero(), T::zero()]),
            matrix3::<T, R>([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        )
        .unwrap();
        assert!(
            R::matrix6_to_array(&overflow_rigid.matrix())
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
        );
        assert_eq!(
            ArticulatedBodyInertia::try_from(&overflow_rigid),
            Err(InertiaError::NonFinite)
        );
    }

    fn combined<T, R>(absolute: f64, relative: f64)
    where
        T: SpatialScalar,
        R: SpatialRepresentation<T>,
    {
        let left =
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(fixture())).unwrap();
        let right =
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(second_fixture()))
                .unwrap();
        let left_before = left;
        let right_before = right;
        let actual = left.try_combined(&right).unwrap();
        let expected = matrix_add(
            fixture().map(|row| row.map(scalar::<T>)),
            second_fixture().map(|row| row.map(scalar::<T>)),
        );
        assert_matrix_close(&actual, expected, absolute, relative);
        assert_eq!(left, left_before);
        assert_eq!(right, right_before);

        let zero = ArticulatedBodyInertia::<T, R>::zeros();
        assert_eq!(left.try_combined(&zero).unwrap(), left);
        assert_eq!(zero.try_combined(&left).unwrap(), left);

        let mut cancellation_left = [[0.0; 6]; 6];
        let mut cancellation_right = [[0.0; 6]; 6];
        for index in 0..6 {
            cancellation_left[index][index] = if index == 0 { -3.0 } else { 2.0 };
            cancellation_right[index][index] = -cancellation_left[index][index];
        }
        let cancellation_left =
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(cancellation_left))
                .unwrap();
        let cancellation_right =
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(cancellation_right))
                .unwrap();
        assert_eq!(
            cancellation_left
                .try_combined(&cancellation_right)
                .unwrap()
                .matrix(),
            R::matrix6_from_array([[T::zero(); 6]; 6])
        );

        let maximum = T::max_value().to_f64().unwrap();
        let mut overflow_matrix = [[0.0; 6]; 6];
        overflow_matrix[0][0] = maximum;
        let overflow =
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(overflow_matrix))
                .unwrap();
        assert_eq!(
            overflow.try_combined(&overflow),
            Err(InertiaError::NonFinite)
        );
        let overflow_before = overflow;
        assert_eq!(overflow, overflow_before);
    }

    fn rank_updates<T, R>(absolute: f64, relative: f64)
    where
        T: SpatialScalar,
        R: SpatialRepresentation<T>,
    {
        let base =
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(fixture())).unwrap();
        let before = base;
        let coordinates = [1.0, -2.0, 0.5, 3.0, -1.0, 2.0];
        let update = force::<T, R>(coordinates);

        for alpha in [0.25, -0.5] {
            let alpha_t = scalar::<T>(alpha);
            let actual = base.try_rank_one_updated(alpha_t, &update).unwrap();
            let mut expected = fixture().map(|row| row.map(scalar::<T>));
            for row in 0..6 {
                for column in 0..6 {
                    expected[row][column] = expected[row][column]
                        + (alpha_t * scalar::<T>(coordinates[row]))
                            * scalar::<T>(coordinates[column]);
                }
            }
            assert_matrix_close(&actual, expected, absolute, relative);
            assert_strictly_symmetric(&actual);
        }

        assert_eq!(base.try_rank_one_updated(T::zero(), &update).unwrap(), base);
        let zero_update = force::<T, R>([0.0; 6]);
        assert_eq!(
            base.try_rank_one_updated(scalar(3.0), &zero_update)
                .unwrap(),
            base
        );
        let repeated = base
            .try_rank_one_updated(scalar(0.25), &update)
            .unwrap()
            .try_rank_one_updated(scalar(0.25), &update)
            .unwrap();
        let mut repeated_expected = fixture().map(|row| row.map(scalar::<T>));
        for row in 0..6 {
            for column in 0..6 {
                let contribution = (scalar::<T>(0.25) * scalar::<T>(coordinates[row]))
                    * scalar::<T>(coordinates[column]);
                repeated_expected[row][column] =
                    repeated_expected[row][column] + contribution + contribution;
            }
        }
        assert_matrix_close(&repeated, repeated_expected, absolute, relative);
        assert_eq!(base, before);

        let finite_update = force::<T, R>([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            base.try_rank_one_updated(T::nan(), &finite_update),
            Err(InertiaError::NonFinite)
        );
        let mut nonfinite_vector = [0.0; 6];
        nonfinite_vector[0] = f64::INFINITY;
        assert_eq!(
            base.try_rank_one_updated(scalar(0.0), &force::<T, R>(nonfinite_vector)),
            Err(InertiaError::NonFinite)
        );

        let maximum = T::max_value();
        let mut alpha_overflow_vector = [0.0; 6];
        alpha_overflow_vector[0] = 2.0;
        assert_eq!(
            base.try_rank_one_updated(maximum, &force::<T, R>(alpha_overflow_vector)),
            Err(InertiaError::NonFinite)
        );

        let mut contribution_overflow_vector = [0.0; 6];
        contribution_overflow_vector[0] = maximum.to_f64().unwrap();
        assert_eq!(
            base.try_rank_one_updated(scalar(0.5), &force::<T, R>(contribution_overflow_vector)),
            Err(InertiaError::NonFinite)
        );

        let mut result_matrix = [[0.0; 6]; 6];
        result_matrix[0][1] = maximum.to_f64().unwrap();
        result_matrix[1][0] = maximum.to_f64().unwrap();
        let result_overflow =
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(result_matrix)).unwrap();
        let maximum_vector = [
            maximum.to_f64().unwrap(),
            maximum.to_f64().unwrap(),
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        assert_eq!(
            result_overflow
                .try_rank_one_updated(T::one() / maximum, &force::<T, R>(maximum_vector)),
            Err(InertiaError::NonFinite)
        );
        assert_eq!(base, before);
    }

    fn transforms<T, R>(absolute: f64, relative: f64)
    where
        T: SpatialScalar,
        R: SpatialRepresentation<T>,
    {
        let inertia =
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(fixture())).unwrap();
        let transform = SpatialTransform::<T, R>::new(
            rotation::<T, R>([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            R::vector3_from_array([scalar(0.5), scalar(-1.0), scalar(2.0)]),
        );
        let before = inertia;
        let transformed = inertia.try_transformed(&transform).unwrap();
        let force_array = R::matrix6_to_array(&transform.force_matrix());
        let expected = matrix_mul(
            matrix_mul(force_array, abi_matrix(&inertia)),
            transpose(force_array),
        );
        assert_matrix_close(&transformed, expected, absolute, relative);

        let inverse_round_trip = transformed.try_transformed(&transform.inverse()).unwrap();
        assert_matrix_close(
            &inverse_round_trip,
            abi_matrix(&inertia),
            absolute,
            relative,
        );

        let next = SpatialTransform::<T, R>::new(
            rotation::<T, R>([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
            R::vector3_from_array([scalar(-0.25), scalar(0.75), scalar(1.5)]),
        );
        let sequential = inertia
            .try_transformed(&transform)
            .unwrap()
            .try_transformed(&next)
            .unwrap();
        let composed = inertia.try_transformed(&transform.then(&next)).unwrap();
        assert_matrix_close(&sequential, abi_matrix(&composed), absolute, relative);

        let motion = vector::<T, R>([0.3, -0.8, 1.1, 2.0, -1.5, 0.4]);
        let transformed_motion = MotionVector::from_vector(R::matrix6_vector_mul(
            &transform.motion_matrix(),
            &motion.to_vector(),
        ));
        let left = transformed.apply(&transformed_motion).to_array();
        let right = ForceVector::<T, R>::from_vector(R::matrix6_vector_mul(
            &transform.force_matrix(),
            &inertia.apply(&motion).to_vector(),
        ))
        .to_array();
        assert_vector_close(left, right, absolute, relative);

        let bad_rotation = SpatialTransform::<T, R>::new(
            rotation::<T, R>([[f64::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            R::vector3_from_array([T::zero(), T::zero(), T::zero()]),
        );
        assert_eq!(
            inertia.try_transformed(&bad_rotation),
            Err(InertiaError::NonFinite)
        );
        let bad_translation = SpatialTransform::<T, R>::new(
            rotation::<T, R>([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            R::vector3_from_array([T::infinity(), T::zero(), T::zero()]),
        );
        assert_eq!(
            inertia.try_transformed(&bad_translation),
            Err(InertiaError::NonFinite)
        );

        let scale_two = SpatialTransform::<T, R>::new(
            rotation::<T, R>([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),
            R::vector3_from_array([T::zero(), T::zero(), T::zero()]),
        );
        let maximum = T::max_value();
        let mut first_product_matrix = [[0.0; 6]; 6];
        first_product_matrix[0][0] = maximum.to_f64().unwrap();
        let first_product =
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(first_product_matrix))
                .unwrap();
        let first_product_reference = matrix_mul(
            R::matrix6_to_array(&scale_two.force_matrix()),
            abi_matrix(&first_product),
        );
        assert!(
            first_product_reference
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
        );
        assert_eq!(
            first_product.try_transformed(&scale_two),
            Err(InertiaError::NonFinite)
        );

        let mut final_product_matrix = [[0.0; 6]; 6];
        final_product_matrix[0][0] = (maximum / scalar::<T>(2.0)).to_f64().unwrap();
        let final_product =
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(final_product_matrix))
                .unwrap();
        let scale_force = R::matrix6_to_array(&scale_two.force_matrix());
        let final_left_reference = matrix_mul(scale_force, abi_matrix(&final_product));
        assert!(
            final_left_reference
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
        let final_product_reference = matrix_mul(final_left_reference, transpose(scale_force));
        assert!(
            final_product_reference
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
        );
        assert_eq!(
            final_product.try_transformed(&scale_two),
            Err(InertiaError::NonFinite)
        );
        assert_eq!(inertia, before);
    }

    fn apply_behavior<T, R>(absolute: f64, relative: f64)
    where
        T: SpatialScalar,
        R: SpatialRepresentation<T>,
    {
        let inertia =
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(fixture())).unwrap();
        let motion = vector::<T, R>([0.3, -0.8, 1.1, 2.0, -1.5, 0.4]);
        let expected = matrix_vector_mul(abi_matrix(&inertia), motion.to_array());
        assert_vector_close(
            inertia.apply(&motion).to_array(),
            expected,
            absolute,
            relative,
        );

        let mut nan_motion = [0.0; 6];
        nan_motion[0] = f64::NAN;
        assert!(
            inertia
                .apply(&vector::<T, R>(nan_motion))
                .to_array()
                .iter()
                .any(|value| !value.is_finite())
        );
        let mut infinite_motion = [0.0; 6];
        infinite_motion[1] = f64::INFINITY;
        assert!(
            inertia
                .apply(&vector::<T, R>(infinite_motion))
                .to_array()
                .iter()
                .any(|value| !value.is_finite())
        );

        let maximum = T::max_value();
        let mut overflow_matrix = [[0.0; 6]; 6];
        overflow_matrix[0][0] = maximum.to_f64().unwrap();
        let overflow =
            ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>(overflow_matrix))
                .unwrap();
        let mut overflow_motion = [0.0; 6];
        overflow_motion[0] = 2.0;
        assert!(
            overflow
                .apply(&vector::<T, R>(overflow_motion))
                .to_array()
                .iter()
                .any(|value| !value.is_finite())
        );
    }

    fn articulated_elimination<T, R>(absolute: f64, relative: f64)
    where
        T: SpatialScalar,
        R: SpatialRepresentation<T>,
    {
        let full = ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>([
            [4.0, 0.1, 0.0, 0.2, 0.0, 0.0],
            [0.1, 5.0, 0.2, 0.0, 0.1, 0.0],
            [0.0, 0.2, 6.0, 0.0, 0.0, 0.1],
            [0.2, 0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.1, 0.0, 0.0, 2.5],
        ]))
        .unwrap();
        let full_before = full;
        let subspace = vector::<T, R>([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let u = full.apply(&subspace);
        let d = subspace.dot(&u);
        assert!(d.is_finite() && d > T::zero());
        let reduced = full.try_rank_one_updated(-T::one() / d, &u).unwrap();
        assert_vector_close(
            reduced.apply(&subspace).to_array(),
            [T::zero(); 6],
            absolute,
            relative,
        );
        assert_eq!(full, full_before);

        let parent = ArticulatedBodyInertia::<T, R>::try_from_matrix(matrix::<T, R>([
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]))
        .unwrap();
        let child_to_parent = SpatialTransform::<T, R>::new(
            rotation::<T, R>([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            R::vector3_from_array([scalar(0.5), scalar(-1.0), scalar(0.25)]),
        );
        let child_in_parent = reduced.try_transformed(&child_to_parent).unwrap();
        let accumulated = parent.try_combined(&child_in_parent).unwrap();
        assert_strictly_symmetric(&accumulated);
        assert!(
            abi_matrix(&accumulated)
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
    }

    #[cfg(feature = "serde")]
    fn deserialize_packed_f64<R>(
        values: [f64; 21],
    ) -> Result<ArticulatedBodyInertia<f64, R>, serde::de::value::Error>
    where
        R: SpatialRepresentation<f64>,
    {
        use serde::de::value::{Error, MapDeserializer, SeqDeserializer};

        let packed = SeqDeserializer::<_, Error>::new(values.into_iter());
        let map = MapDeserializer::<_, Error>::new(std::iter::once(("packed", packed)));
        <ArticulatedBodyInertia<f64, R> as serde::Deserialize>::deserialize(map)
    }

    #[cfg(feature = "serde")]
    fn serde_contract<R>()
    where
        R: SpatialRepresentation<f64>,
    {
        let source = distinctive_fixture();
        let inertia =
            ArticulatedBodyInertia::<f64, R>::try_from_matrix(matrix::<f64, R>(source)).unwrap();
        let serialized = serde_json::to_string(&inertia).unwrap();
        let value: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        let packed = value.get("packed").unwrap().as_array().unwrap();
        let expected: Vec<f64> = (0..6)
            .flat_map(|row| (row..6).map(move |column| source[row][column]))
            .collect();
        assert_eq!(packed.len(), 21);
        for (actual, expected) in packed.iter().zip(expected) {
            assert_eq!(actual.as_f64().unwrap(), expected);
        }
        assert_eq!(
            serde_json::from_str::<ArticulatedBodyInertia<f64, R>>(&serialized).unwrap(),
            inertia
        );

        for malformed in [
            serde_json::json!({ "packed": vec![0.0; 20] }),
            serde_json::json!({ "packed": vec![0.0; 22] }),
            serde_json::json!({}),
            serde_json::json!({ "packed": "invalid" }),
        ] {
            assert!(serde_json::from_value::<ArticulatedBodyInertia<f64, R>>(malformed).is_err());
        }
        let mut nan = [0.0; 21];
        nan[3] = f64::NAN;
        let mut infinite = [0.0; 21];
        infinite[7] = f64::INFINITY;
        assert!(deserialize_packed_f64::<R>(nan).is_err());
        assert!(deserialize_packed_f64::<R>(infinite).is_err());
    }

    #[cfg(feature = "serde")]
    #[cfg(all(feature = "builtin", feature = "nalgebra", feature = "glam"))]
    #[test]
    fn serde_wire_form_is_backend_identical() {
        use spatial6::{Builtin, Glam, Nalgebra};

        let source = distinctive_fixture();
        let builtin =
            ArticulatedBodyInertia::<f64, Builtin>::try_from_matrix(matrix::<f64, Builtin>(source))
                .unwrap();
        let nalgebra = ArticulatedBodyInertia::<f64, Nalgebra>::try_from_matrix(matrix::<
            f64,
            Nalgebra,
        >(source))
        .unwrap();
        let glam =
            ArticulatedBodyInertia::<f64, Glam>::try_from_matrix(matrix::<f64, Glam>(source))
                .unwrap();
        assert_eq!(
            serde_json::to_string(&builtin).unwrap(),
            serde_json::to_string(&nalgebra).unwrap()
        );
        assert_eq!(
            serde_json::to_string(&builtin).unwrap(),
            serde_json::to_string(&glam).unwrap()
        );
    }

    macro_rules! backend_tests {
        ($module:ident, $backend:ty) => {
            mod $module {
                #[test]
                fn construction_and_normalization_f64() {
                    super::construction_and_normalization::<f64, $backend>();
                }

                #[test]
                fn public_traits_and_layout_f64() {
                    super::public_traits_and_layout::<f64, $backend>();
                }

                #[test]
                fn construction_and_normalization_f32() {
                    super::construction_and_normalization::<f32, $backend>();
                }

                #[test]
                fn rigid_conversion_f64() {
                    super::rigid_conversion::<f64, $backend>(1.0e-10, 1.0e-10);
                }

                #[test]
                fn rigid_conversion_f32() {
                    super::rigid_conversion::<f32, $backend>(1.0e-5, 1.0e-5);
                }

                #[test]
                fn combined_f64() {
                    super::combined::<f64, $backend>(1.0e-10, 1.0e-10);
                }

                #[test]
                fn combined_f32() {
                    super::combined::<f32, $backend>(1.0e-5, 1.0e-5);
                }

                #[test]
                fn rank_updates_f64() {
                    super::rank_updates::<f64, $backend>(1.0e-10, 1.0e-10);
                }

                #[test]
                fn rank_updates_f32() {
                    super::rank_updates::<f32, $backend>(1.0e-5, 1.0e-5);
                }

                #[test]
                fn transforms_f64() {
                    super::transforms::<f64, $backend>(1.0e-10, 1.0e-10);
                }

                #[test]
                fn transforms_f32() {
                    super::transforms::<f32, $backend>(1.0e-5, 1.0e-5);
                }

                #[test]
                fn apply_f64() {
                    super::apply_behavior::<f64, $backend>(1.0e-10, 1.0e-10);
                }

                #[test]
                fn apply_f32() {
                    super::apply_behavior::<f32, $backend>(1.0e-5, 1.0e-5);
                }

                #[test]
                fn articulated_elimination_f64() {
                    super::articulated_elimination::<f64, $backend>(1.0e-10, 1.0e-10);
                }

                #[test]
                fn articulated_elimination_f32() {
                    super::articulated_elimination::<f32, $backend>(1.0e-5, 1.0e-5);
                }

                #[cfg(feature = "serde")]
                #[test]
                fn serde_contract_f64() {
                    super::serde_contract::<$backend>();
                }
            }
        };
    }

    #[cfg(feature = "builtin")]
    backend_tests!(builtin_backend, spatial6::Builtin);
    #[cfg(feature = "nalgebra")]
    backend_tests!(nalgebra_backend, spatial6::Nalgebra);
    #[cfg(feature = "glam")]
    backend_tests!(glam_backend, spatial6::Glam);
}
