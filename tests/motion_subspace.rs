// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(any(feature = "builtin", feature = "nalgebra", feature = "glam"))]
use spatial6::{
    ForceVector, MotionSubspace, MotionVector, SpatialRepresentation, SpatialTransform,
};

#[cfg(any(feature = "builtin", feature = "nalgebra", feature = "glam"))]
fn assert_motion_close<R>(actual: &MotionVector<f64, R>, expected: &MotionVector<f64, R>)
where
    R: SpatialRepresentation<f64>,
{
    for (actual, expected) in actual.to_array().into_iter().zip(expected.to_array()) {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "{actual} != {expected}"
        );
    }
}

#[cfg(any(feature = "builtin", feature = "nalgebra", feature = "glam"))]
fn exercise_core<R>()
where
    R: SpatialRepresentation<f64>,
{
    fn assert_send_sync<T: Send + Sync>() {}

    let first = MotionVector::<f64, R>::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let second = MotionVector::<f64, R>::from_array([-2.0, 1.0, 4.0, 0.5, -3.0, 2.0]);
    let subspace = MotionSubspace::from_columns([first, second]);

    assert_eq!(subspace.columns(), &[first, second]);
    assert!(subspace.is_finite());
    assert!(
        !MotionSubspace::from_columns([
            first,
            MotionVector::<f64, R>::from_array([f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ])
        .is_finite()
    );

    assert_eq!(subspace.apply(&[1.0, 0.0]), first);
    assert_eq!(subspace.apply(&[0.0, 1.0]), second);

    let x = [2.0, -0.5];
    let y = [-1.0, 3.0];
    let a = 1.5;
    let b = -2.0;
    let combined = std::array::from_fn(|i| a * x[i] + b * y[i]);
    assert_motion_close(
        &subspace.apply(&combined),
        &(subspace.apply(&x) * a + subspace.apply(&y) * b),
    );

    let force = ForceVector::<f64, R>::from_array([3.0, -1.0, 2.0, 4.0, 0.5, -2.0]);
    let generalized = subspace.generalized_force(&force);
    let spatial_power = subspace.apply(&x).dot(&force);
    let generalized_power = x[0] * generalized[0] + x[1] * generalized[1];
    assert!((spatial_power - generalized_power).abs() <= 1.0e-12);

    let one = MotionSubspace::<1, f64, R>::from(first);
    assert_eq!(one.apply(&[2.5]), first * 2.5);
    assert_eq!(one.generalized_force(&force), [first.dot(&force)]);

    let fixed = MotionSubspace::<0, f64, R>::from_columns([]);
    assert_eq!(fixed.columns(), &[]);
    assert_eq!(fixed.apply(&[]), MotionVector::<f64, R>::zeros());
    assert_eq!(fixed.generalized_force(&force), [] as [f64; 0]);

    assert_send_sync::<MotionSubspace<2, f64, R>>();
}

#[cfg(any(feature = "builtin", feature = "nalgebra", feature = "glam"))]
fn exercise_transforms<R>()
where
    R: SpatialRepresentation<f64>,
{
    let subspace = MotionSubspace::from_columns([
        MotionVector::<f64, R>::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        MotionVector::<f64, R>::from_array([-2.0, 1.0, 4.0, 0.5, -3.0, 2.0]),
    ]);
    let forces = [
        ForceVector::<f64, R>::from_array([3.0, -1.0, 2.0, 4.0, 0.5, -2.0]),
        ForceVector::<f64, R>::from_array([-2.0, 5.0, 1.0, 0.0, 3.0, 4.0]),
    ];
    let generalized = subspace.generalized_forces(&forces);
    for (row, generalized_row) in generalized.iter().enumerate() {
        for (column, actual) in generalized_row.iter().enumerate() {
            assert!((*actual - subspace.columns()[row].dot(&forces[column])).abs() <= 1.0e-12);
        }
    }
    assert_eq!(subspace.generalized_forces(&[]), [[] as [f64; 0]; 2]);

    let transform = SpatialTransform::<f64, R>::new(
        R::rotation3_from_array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        R::vector3_from_array([3.0, 5.0, 7.0]),
    );
    let transformed = subspace.transformed(&transform);
    let coefficients = [1.5, -0.25];
    assert_motion_close(
        &transformed.apply(&coefficients),
        &transform.transform_motion(&subspace.apply(&coefficients)),
    );
    let transformed_force = transform.transform_force(&forces[0]);
    let original_generalized = subspace.generalized_force(&forces[0]);
    let transformed_generalized = transformed.generalized_force(&transformed_force);
    for (actual, expected) in transformed_generalized
        .into_iter()
        .zip(original_generalized)
    {
        assert!((actual - expected).abs() <= 1.0e-12);
    }

    let fixed = MotionSubspace::<0, f64, R>::from_columns([]);
    assert_eq!(fixed.generalized_forces(&forces), [] as [[f64; 2]; 0]);
    assert_eq!(fixed.transformed(&transform).columns(), &[]);
}

#[cfg(feature = "builtin")]
#[test]
fn builtin_core_operations_preserve_the_subspace_contract() {
    exercise_core::<spatial6::Builtin>();
}

#[cfg(feature = "nalgebra")]
#[test]
fn nalgebra_core_operations_preserve_the_subspace_contract() {
    exercise_core::<spatial6::Nalgebra>();
}

#[cfg(feature = "glam")]
#[test]
fn glam_core_operations_preserve_the_subspace_contract() {
    exercise_core::<spatial6::Glam>();
}

#[cfg(feature = "builtin")]
#[test]
fn builtin_multi_column_and_transform_operations_preserve_the_subspace_contract() {
    exercise_transforms::<spatial6::Builtin>();
}

#[cfg(feature = "nalgebra")]
#[test]
fn nalgebra_multi_column_and_transform_operations_preserve_the_subspace_contract() {
    exercise_transforms::<spatial6::Nalgebra>();
}

#[cfg(feature = "glam")]
#[test]
fn glam_multi_column_and_transform_operations_preserve_the_subspace_contract() {
    exercise_transforms::<spatial6::Glam>();
}

#[cfg(all(
    feature = "serde",
    any(feature = "builtin", feature = "nalgebra", feature = "glam")
))]
fn exercise_serde<R>()
where
    R: SpatialRepresentation<f64>,
    MotionVector<f64, R>: serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    fn round_trip<const N: usize, R>(subspace: MotionSubspace<N, f64, R>)
    where
        R: SpatialRepresentation<f64>,
        MotionVector<f64, R>: serde::Serialize + for<'de> serde::Deserialize<'de>,
    {
        let json = serde_json::to_string(&subspace).unwrap();
        assert_eq!(
            serde_json::from_str::<MotionSubspace<N, f64, R>>(&json).unwrap(),
            subspace
        );
    }

    round_trip(MotionSubspace::<0, f64, R>::from_columns([]));
    round_trip(MotionSubspace::from_columns([
        MotionVector::<f64, R>::zeros(),
    ]));
    round_trip(MotionSubspace::from_columns(
        std::array::from_fn::<_, 33, _>(|i| {
            MotionVector::<f64, R>::from_array([i as f64, 1.0, 2.0, 3.0, 4.0, 5.0])
        }),
    ));
}

#[cfg(all(feature = "builtin", feature = "serde"))]
#[test]
fn builtin_serde_supports_arbitrary_column_counts() {
    exercise_serde::<spatial6::Builtin>();
}

#[cfg(all(feature = "nalgebra", feature = "serde"))]
#[test]
fn nalgebra_serde_supports_arbitrary_column_counts() {
    exercise_serde::<spatial6::Nalgebra>();
}

#[cfg(all(feature = "glam", feature = "serde"))]
#[test]
fn glam_serde_supports_arbitrary_column_counts() {
    exercise_serde::<spatial6::Glam>();
}
