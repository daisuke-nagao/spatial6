use spatial6::{
    ArticulatedBodyInertia, ForceVector, InertiaError, MotionSubspace, MotionVector,
    RigidBodyInertia, SpatialRepresentation, SpatialScalar, SpatialTransform,
};

use crate::custom::ArrayRepresentation;

const ROTATION: [[f64; 3]; 3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
const INERTIA: [[f64; 3]; 3] = [[4.0, 0.2, 0.1], [0.2, 5.0, 0.3], [0.1, 0.3, 6.0]];

fn scalar<T: SpatialScalar>(value: f64) -> T {
    T::from(value).unwrap_or_else(T::zero)
}

fn close<T: SpatialScalar>(left: T, right: T, tolerance: T) -> bool {
    (left - right).abs() <= tolerance * (T::one() + left.abs() + right.abs())
}

fn arrays_close<T: SpatialScalar, const N: usize>(
    left: &[T; N],
    right: &[T; N],
    tolerance: T,
) -> bool {
    left.iter()
        .zip(right)
        .all(|(&left, &right)| close(left, right, tolerance))
}

fn matrix_close<T: SpatialScalar, const N: usize>(
    left: &[[T; N]; N],
    right: &[[T; N]; N],
    tolerance: T,
) -> bool {
    left.iter()
        .zip(right)
        .all(|(left, right)| arrays_close(left, right, tolerance))
}

fn rotation<T: SpatialScalar, R: SpatialRepresentation<T>>() -> R::Rotation3 {
    R::rotation3_from_array(core::array::from_fn(|row| {
        core::array::from_fn(|column| scalar::<T>(ROTATION[row][column]))
    }))
}

fn transform<T: SpatialScalar, R: SpatialRepresentation<T>>() -> SpatialTransform<T, R> {
    SpatialTransform::new(
        rotation::<T, R>(),
        R::vector3_from_array([scalar(3.0), scalar(5.0), scalar(7.0)]),
    )
}

fn motion<T: SpatialScalar, R: SpatialRepresentation<T>>(seed: T) -> MotionVector<T, R> {
    MotionVector::from_array(core::array::from_fn(|index| {
        seed + scalar((index as f64) * 0.25 - 0.5)
    }))
}

fn force<T: SpatialScalar, R: SpatialRepresentation<T>>(seed: T) -> ForceVector<T, R> {
    ForceVector::from_array(core::array::from_fn(|index| {
        seed + scalar((index as f64) * -0.125 + 0.75)
    }))
}

fn rigid<T: SpatialScalar, R: SpatialRepresentation<T>>() -> RigidBodyInertia<T, R> {
    RigidBodyInertia::try_new(
        scalar(2.0),
        R::vector3_from_array([scalar(1.0), scalar(2.0), scalar(3.0)]),
        R::matrix3_from_array(core::array::from_fn(|row| {
            core::array::from_fn(|column| scalar(INERTIA[row][column]))
        })),
    )
    .unwrap_or_else(|_| {
        // This branch is unreachable for the fixed fixture and keeps the wrapper
        // total if a backend changes its validation behavior.
        RigidBodyInertia::try_new(scalar(1.0), R::vector3_zero(), R::matrix3_identity()).unwrap()
    })
}

fn subspace<T: SpatialScalar, R: SpatialRepresentation<T>, const N: usize>(
    seed: T,
) -> MotionSubspace<N, T, R> {
    MotionSubspace::from_columns(core::array::from_fn(|column| {
        MotionVector::from_array(core::array::from_fn(|index| {
            seed + scalar((column * 7 + index + 1) as f64 * 0.0625)
        }))
    }))
}

fn subspace_cases<T: SpatialScalar, R: SpatialRepresentation<T>>(seed: T, tolerance: T) -> bool {
    let force = force::<T, R>(seed);
    let transform = transform::<T, R>();
    let mut ok = true;

    let zero = subspace::<T, R, 0>(seed);
    ok &= zero.columns().is_empty();
    ok &= zero.apply(&[]).to_array() == [T::zero(); 6];
    ok &= zero.generalized_force(&force) == [];
    ok &= zero.transformed(&transform).columns().is_empty();

    let one = MotionSubspace::<1, T, R>::from(motion::<T, R>(seed));
    let one_coefficients = [scalar::<T>(1.5)];
    ok &= arrays_close(
        &one.apply(&one_coefficients).to_array(),
        &(motion::<T, R>(seed) * one_coefficients[0]).to_array(),
        tolerance,
    );
    ok &= close(
        one.generalized_force(&force)[0],
        one.columns()[0].dot(&force),
        tolerance,
    );

    let three = subspace::<T, R, 3>(seed);
    let six = subspace::<T, R, 6>(seed);
    let seven = subspace::<T, R, 7>(seed);
    let coefficients = [scalar::<T>(1.0); 3];
    let transformed = three.transformed(&transform);
    ok &= transformed.is_finite();
    ok &= arrays_close(
        &transformed.apply(&coefficients).to_array(),
        &transform
            .transform_motion(&three.apply(&coefficients))
            .to_array(),
        tolerance,
    );
    ok &= three.generalized_forces(&[force])
        == [
            [three.columns()[0].dot(&force)],
            [three.columns()[1].dot(&force)],
            [three.columns()[2].dot(&force)],
        ];
    ok &= six.is_finite() && seven.is_finite();
    ok &= seven.generalized_force(&force).len() == 7;
    ok
}

fn cases<T: SpatialScalar, R: SpatialRepresentation<T>>(seed: T) -> T {
    let tolerance = if core::mem::size_of::<T>() == core::mem::size_of::<f32>() {
        scalar::<T>(2.0e-4)
    } else {
        scalar::<T>(2.0e-11)
    };
    let mut ok = true;
    let mut checksum = seed;
    let velocity = motion::<T, R>(seed);
    let second_motion = motion::<T, R>(seed + scalar(0.5));
    let applied_force = force::<T, R>(seed + scalar(0.25));

    ok &= velocity.is_finite() && applied_force.is_finite();
    let motion_sum = velocity + second_motion;
    ok &= arrays_close(
        &motion_sum.to_array(),
        &core::array::from_fn(|index| velocity.to_array()[index] + second_motion.to_array()[index]),
        tolerance,
    );
    let mut motion_arithmetic = motion_sum - second_motion;
    motion_arithmetic *= scalar(2.0);
    motion_arithmetic /= scalar(2.0);
    ok &= arrays_close(
        &motion_arithmetic.to_array(),
        &velocity.to_array(),
        tolerance,
    );
    ok &= arrays_close(
        &(-velocity).to_array(),
        &core::array::from_fn(|index| -velocity.to_array()[index]),
        tolerance,
    );
    let other_force = force::<T, R>(seed);
    let force_sum = applied_force + other_force;
    ok &= arrays_close(
        &force_sum.to_array(),
        &core::array::from_fn(|index| {
            applied_force.to_array()[index] + other_force.to_array()[index]
        }),
        tolerance,
    );
    ok &= arrays_close(
        &((force_sum - other_force) * scalar::<T>(0.5) / scalar::<T>(0.5)).to_array(),
        &applied_force.to_array(),
        tolerance,
    );
    ok &= close(
        velocity * applied_force,
        velocity.dot(&applied_force),
        tolerance,
    );
    let negative_zero = -T::zero();
    ok &= negative_zero == T::zero() && negative_zero.is_sign_negative();
    ok &= MotionVector::<T, R>::from_array([negative_zero; 6])
        .to_array()
        .iter()
        .all(|value| value.is_sign_negative());
    ok &= close(
        velocity.dot(&applied_force),
        applied_force.dot(&velocity),
        tolerance,
    );
    let cross_motion = R::matrix6_vector_mul(&velocity.cross_matrix(), &second_motion.to_vector());
    ok &= arrays_close(
        &R::vector6_to_array(&cross_motion),
        &velocity.cross_motion(&second_motion).to_array(),
        tolerance,
    );
    let cross_force =
        R::matrix6_vector_mul(&velocity.cross_dual_matrix(), &applied_force.to_vector());
    ok &= arrays_close(
        &R::vector6_to_array(&cross_force),
        &velocity.cross_force(&applied_force).to_array(),
        tolerance,
    );
    checksum = checksum + velocity.dot(&applied_force);

    let transform = transform::<T, R>();
    let transformed_motion = transform.transform_motion(&velocity);
    let transformed_force = transform.transform_force(&applied_force);
    ok &= close(
        velocity.dot(&applied_force),
        transformed_motion.dot(&transformed_force),
        tolerance,
    );
    ok &= arrays_close(
        &transform
            .inverse()
            .transform_motion(&transformed_motion)
            .to_array(),
        &velocity.to_array(),
        tolerance,
    );
    ok &= arrays_close(
        &transform
            .inverse()
            .transform_force(&transformed_force)
            .to_array(),
        &applied_force.to_array(),
        tolerance,
    );
    let composed = transform.then(&transform);
    ok &= arrays_close(
        &composed.transform_motion(&velocity).to_array(),
        &transform
            .transform_motion(&transform.transform_motion(&velocity))
            .to_array(),
        tolerance,
    );
    let (pose_rotation, pose_position) = transform.to_pose_parts();
    ok &= matrix_close(
        &R::rotation3_to_array(&pose_rotation),
        &R::rotation3_to_array(&R::rotation3_inverse(transform.rotation())),
        tolerance,
    );
    ok &= R::vector3_to_array(&pose_position) == R::vector3_to_array(transform.translation());
    checksum = checksum + transformed_motion.to_array()[0] + transformed_force.to_array()[3];

    let body = rigid::<T, R>();
    let body_matrix = body.matrix();
    let body_force = body.apply(&velocity);
    let matrix_force = ForceVector::<T, R>::from_vector(R::matrix6_vector_mul(
        &body_matrix,
        &velocity.to_vector(),
    ));
    ok &= arrays_close(&body_force.to_array(), &matrix_force.to_array(), tolerance);
    let moved_body = body.transformed(&transform);
    ok &= moved_body.mass().is_finite()
        && R::vector3_is_finite(moved_body.center_of_mass())
        && R::matrix3_is_finite(&moved_body.inertia_at_center_of_mass());
    let derivative = body.time_derivative(&velocity);
    ok &= R::matrix6_to_array(&derivative)
        .iter()
        .flatten()
        .all(|value| value.is_finite());
    ok &= body.bias_force(&velocity).is_finite();
    let acceleration = motion::<T, R>(seed + scalar(0.75));
    let inverse_force = body.inverse_dynamics(&velocity, &acceleration);
    ok &= body
        .forward_dynamics(&velocity, &inverse_force)
        .map(|result| {
            arrays_close(
                &result.to_array(),
                &acceleration.to_array(),
                tolerance * scalar(10.0),
            )
        })
        .unwrap_or(false);
    ok &= body
        .forward_dynamics(&MotionVector::from_array([T::nan(); 6]), &inverse_force)
        .is_none();
    ok &= matches!(
        RigidBodyInertia::<T, R>::try_new(T::zero(), R::vector3_zero(), R::matrix3_identity(),),
        Err(InertiaError::NonPositiveMass)
    );
    ok &= matches!(
        RigidBodyInertia::<T, R>::try_new(-T::one(), R::vector3_zero(), R::matrix3_identity(),),
        Err(InertiaError::NonPositiveMass)
    );
    ok &= matches!(
        RigidBodyInertia::<T, R>::try_new(T::infinity(), R::vector3_zero(), R::matrix3_identity(),),
        Err(InertiaError::NonFinite)
    );
    ok &= matches!(
        RigidBodyInertia::<T, R>::try_new(
            T::one(),
            R::vector3_from_array([T::nan(), T::zero(), T::zero()]),
            R::matrix3_identity(),
        ),
        Err(InertiaError::NonFinite)
    );
    ok &= matches!(
        RigidBodyInertia::<T, R>::try_new(
            T::one(),
            R::vector3_zero(),
            R::matrix3_from_array([
                [-T::one(), T::zero(), T::zero()],
                [T::zero(), T::one(), T::zero()],
                [T::zero(), T::zero(), T::one()],
            ]),
        ),
        Err(InertiaError::NotPositiveDefinite)
    );
    ok &= matches!(
        RigidBodyInertia::<T, R>::try_new(
            T::one(),
            R::vector3_zero(),
            R::matrix3_from_array([
                [T::one(), T::one(), T::zero()],
                [T::zero(), T::one(), T::zero()],
                [T::zero(), T::zero(), T::one()],
            ]),
        ),
        Err(InertiaError::NonSymmetric)
    );
    ok &= matches!(
        RigidBodyInertia::<T, R>::try_new(
            T::one(),
            R::vector3_zero(),
            R::matrix3_from_array([
                [T::nan(), T::zero(), T::zero()],
                [T::zero(), T::one(), T::zero()],
                [T::zero(), T::zero(), T::one()],
            ]),
        ),
        Err(InertiaError::NonFinite)
    );
    checksum = checksum + body.mass() + body_force.to_array()[0] + inverse_force.to_array()[5];

    let articulated =
        ArticulatedBodyInertia::try_from(&body).unwrap_or_else(|_| ArticulatedBodyInertia::zeros());
    let articulated_force = articulated.apply(&velocity);
    ok &= arrays_close(
        &articulated_force.to_array(),
        &body_force.to_array(),
        tolerance * scalar(10.0),
    );
    ok &= articulated.apply_subspace(&subspace::<T, R, 3>(seed)).len() == 3;
    ok &= articulated
        .try_combined(&articulated)
        .map(|combined| {
            let matrix = combined.matrix();
            R::matrix6_to_array(&matrix)
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        })
        .unwrap_or(false);
    ok &= articulated
        .try_rank_one_updated(scalar(0.25), &applied_force)
        .map(|updated| updated.apply(&velocity).is_finite())
        .unwrap_or(false);
    ok &= articulated
        .try_rank_one_updated(T::nan(), &applied_force)
        .is_err();
    ok &= articulated
        .try_transformed(&transform)
        .map(|moved| moved.apply(&velocity).is_finite())
        .unwrap_or(false);
    let bad_articulated = R::matrix6_from_array(core::array::from_fn(|row| {
        core::array::from_fn(|column| {
            if row == column {
                scalar(1.0)
            } else if row == 0 && column == 1 {
                scalar(1.0)
            } else {
                T::zero()
            }
        })
    }));
    ok &= matches!(
        ArticulatedBodyInertia::<T, R>::try_from_matrix(bad_articulated),
        Err(InertiaError::NonSymmetric)
    );
    let nonfinite_articulated = R::matrix6_from_array(core::array::from_fn(|row| {
        core::array::from_fn(|column| {
            if row == column {
                if row == 0 { T::nan() } else { T::one() }
            } else {
                T::zero()
            }
        })
    }));
    ok &= matches!(
        ArticulatedBodyInertia::<T, R>::try_from_matrix(nonfinite_articulated),
        Err(InertiaError::NonFinite)
    );
    let mut boundary = [[T::zero(); 6]; 6];
    for index in 0..6 {
        boundary[index][index] = scalar(1.0);
    }
    boundary[0][1] = scalar::<T>(32.0) * T::epsilon();
    boundary[1][0] = T::zero();
    ok &= ArticulatedBodyInertia::<T, R>::try_from_matrix(R::matrix6_from_array(boundary)).is_ok();
    let mut outside_boundary = boundary;
    outside_boundary[0][1] = scalar::<T>(128.0) * T::epsilon();
    ok &= matches!(
        ArticulatedBodyInertia::<T, R>::try_from_matrix(R::matrix6_from_array(outside_boundary)),
        Err(InertiaError::NonSymmetric)
    );

    let indefinite = R::matrix6_from_array(core::array::from_fn(|row| {
        core::array::from_fn(|column| {
            if row == column {
                if row == 0 { -T::one() } else { T::one() }
            } else {
                T::zero()
            }
        })
    }));
    ok &= ArticulatedBodyInertia::<T, R>::try_from_matrix(indefinite).is_ok();

    ok &= subspace_cases::<T, R>(seed, tolerance);

    // Exercise the primitive square-root path used by the SPD solve.
    let root = (seed.abs() + scalar(4.0)).sqrt();
    ok &= close(
        root * root,
        seed.abs() + scalar(4.0),
        tolerance * scalar(10.0),
    );
    let solved = R::matrix6_solve_positive_definite(&body_matrix, &body_force.to_vector());
    ok &= solved
        .map(|value| {
            let residual = R::matrix6_vector_mul(&body_matrix, &value);
            arrays_close(
                &R::vector6_to_array(&residual),
                &body_force.to_array(),
                tolerance * scalar(10.0),
            )
        })
        .unwrap_or(false);
    let singular = R::matrix6_from_array([[T::zero(); 6]; 6]);
    ok &= R::matrix6_solve_positive_definite(&singular, &body_force.to_vector()).is_none();
    ok &= R::matrix6_solve_positive_definite(&indefinite, &body_force.to_vector()).is_none();
    let nonfinite_matrix = R::matrix6_from_array(core::array::from_fn(|row| {
        core::array::from_fn(|column| {
            if row == column && row == 0 {
                T::infinity()
            } else if row == column {
                T::one()
            } else {
                T::zero()
            }
        })
    }));
    ok &= R::matrix6_solve_positive_definite(&nonfinite_matrix, &body_force.to_vector()).is_none();
    ok &= R::matrix6_solve_positive_definite(&body_matrix, &R::vector6_from_array([T::nan(); 6]))
        .is_none();

    if ok { checksum } else { T::nan() }
}

#[inline(never)]
pub fn run_array_f32(seed: f64) -> f64 {
    cases::<f32, ArrayRepresentation>(seed as f32) as f64
}

#[inline(never)]
pub fn run_array_f64(seed: f64) -> f64 {
    cases::<f64, ArrayRepresentation>(seed)
}

#[cfg(feature = "builtin")]
#[inline(never)]
pub fn run_builtin_f32(seed: f64) -> f64 {
    cases::<f32, spatial6::Builtin>(seed as f32) as f64
}

#[cfg(feature = "builtin")]
#[inline(never)]
pub fn run_builtin_f64(seed: f64) -> f64 {
    cases::<f64, spatial6::Builtin>(seed)
}

#[cfg(feature = "nalgebra")]
#[inline(never)]
pub fn run_nalgebra_f32(seed: f64) -> f64 {
    cases::<f32, spatial6::Nalgebra>(seed as f32) as f64
}

#[cfg(feature = "nalgebra")]
#[inline(never)]
pub fn run_nalgebra_f64(seed: f64) -> f64 {
    cases::<f64, spatial6::Nalgebra>(seed)
}

#[cfg(feature = "glam")]
#[inline(never)]
pub fn run_glam_f32(seed: f64) -> f64 {
    cases::<f32, spatial6::Glam>(seed as f32) as f64
}

#[cfg(feature = "glam")]
#[inline(never)]
pub fn run_glam_f64(seed: f64) -> f64 {
    cases::<f64, spatial6::Glam>(seed)
}

#[cfg(feature = "serde")]
#[inline(never)]
pub fn run_serde(seed: f64) -> f64 {
    crate::serde_cases::run::<ArrayRepresentation>(seed)
}

#[cfg(all(feature = "serde", feature = "builtin"))]
#[inline(never)]
pub fn run_serde_builtin(seed: f64) -> f64 {
    crate::serde_cases::run::<spatial6::Builtin>(seed)
}

#[cfg(all(feature = "serde", feature = "nalgebra"))]
#[inline(never)]
pub fn run_serde_nalgebra(seed: f64) -> f64 {
    crate::serde_cases::run::<spatial6::Nalgebra>(seed)
}

#[cfg(all(feature = "serde", feature = "glam"))]
#[inline(never)]
pub fn run_serde_glam(seed: f64) -> f64 {
    crate::serde_cases::run::<spatial6::Glam>(seed)
}

/// Executes every concrete operation selected by this fixture's feature set.
#[inline(never)]
pub fn run_all(seed: f64) -> f64 {
    #[allow(unused_mut)]
    let mut total = run_array_f32(seed) + run_array_f64(seed);
    #[cfg(feature = "builtin")]
    {
        total += run_builtin_f32(seed) + run_builtin_f64(seed);
    }
    #[cfg(feature = "nalgebra")]
    {
        total += run_nalgebra_f32(seed) + run_nalgebra_f64(seed);
    }
    #[cfg(feature = "glam")]
    {
        total += run_glam_f32(seed) + run_glam_f64(seed);
    }
    #[cfg(feature = "serde")]
    {
        total += run_serde(seed);
    }
    #[cfg(all(feature = "serde", feature = "builtin"))]
    {
        total += run_serde_builtin(seed);
    }
    #[cfg(all(feature = "serde", feature = "nalgebra"))]
    {
        total += run_serde_nalgebra(seed);
    }
    #[cfg(all(feature = "serde", feature = "glam"))]
    {
        total += run_serde_glam(seed);
    }
    total
}
