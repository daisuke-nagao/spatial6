// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(any(feature = "builtin", feature = "nalgebra", feature = "glam"))]
use spatial6::{
    ForceVector, InertiaError, MotionVector, RigidBodyInertia, SpatialRepresentation,
    SpatialTransform,
};

#[cfg(any(feature = "builtin", feature = "nalgebra", feature = "glam"))]
fn assert_close_f64_1e10(actual: f64, expected: f64) {
    let tolerance = 1.0e-10 + 1.0e-10 * actual.abs().max(expected.abs());
    assert!((actual - expected).abs() <= tolerance);
}

#[cfg(all(feature = "builtin", feature = "nalgebra", feature = "glam"))]
fn assert_close_f64_1e9(actual: f64, expected: f64) {
    let tolerance = 1.0e-9 + 1.0e-9 * actual.abs().max(expected.abs());
    assert!((actual - expected).abs() <= tolerance);
}

#[cfg(any(feature = "builtin", feature = "nalgebra", feature = "glam"))]
fn exercise_backend<R>()
where
    R: SpatialRepresentation<f64>,
{
    fn assert_send_sync<T: Send + Sync>() {}

    let vector3 = [1.0, 2.0, 3.0];
    assert_eq!(
        R::vector3_to_array(&R::vector3_from_array(vector3)),
        vector3
    );

    let matrix3 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
    assert_eq!(
        R::matrix3_to_array(&R::matrix3_from_array(matrix3)),
        matrix3
    );
    let matrix3_native = R::matrix3_from_array(matrix3);
    let matrix3_identity = R::matrix3_identity();
    assert_eq!(
        R::matrix3_to_array(&R::matrix3_add(&matrix3_native, &matrix3_identity)),
        [[2.0, 2.0, 3.0], [4.0, 6.0, 6.0], [7.0, 8.0, 11.0]]
    );
    assert_eq!(
        R::rotation3_to_array(&R::rotation3_from_array(matrix3)),
        matrix3
    );

    let matrix6 =
        std::array::from_fn(|row| std::array::from_fn(|column| (row * 6 + column) as f64));
    assert_eq!(
        R::matrix6_to_array(&R::matrix6_from_array(matrix6)),
        matrix6
    );

    let velocity = MotionVector::<f64, R>::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let force = ForceVector::<f64, R>::from_array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    assert_eq!(velocity.dot(&force), 280.0);
    assert_eq!(
        velocity.cross_force(&force).to_array(),
        [-18.0, 36.0, -18.0, -12.0, 24.0, -12.0]
    );

    let transform = SpatialTransform::<f64, R>::new(
        R::rotation3_from_array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        R::vector3_from_array([3.0, 5.0, 7.0]),
    );
    assert_eq!(
        transform.transform_motion(&velocity).to_array(),
        [-2.0, 1.0, 3.0, -7.0, 3.0, 5.0]
    );

    let inertia = RigidBodyInertia::<f64, R>::try_new(
        2.0,
        R::vector3_from_array([1.0, 2.0, 3.0]),
        R::matrix3_from_array([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]),
    )
    .unwrap();
    let other = RigidBodyInertia::<f64, R>::try_new(
        3.0,
        R::vector3_from_array([-2.0, 1.0, 0.5]),
        R::matrix3_from_array([[7.0, 0.2, -0.1], [0.2, 8.0, 0.3], [-0.1, 0.3, 9.0]]),
    )
    .unwrap();
    let combined = inertia.try_combined(&other).unwrap();
    assert_close_f64_1e10(combined.mass(), 5.0);
    for (actual, expected) in R::vector3_to_array(combined.center_of_mass())
        .into_iter()
        .zip([-0.8, 1.4, 1.5])
    {
        assert_close_f64_1e10(actual, expected);
    }
    assert_eq!(
        RigidBodyInertia::<f64, R>::try_new(
            2.0,
            R::vector3_from_array([f64::NAN, 0.0, 0.0]),
            R::matrix3_from_array([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0],]),
        ),
        Err(InertiaError::NonFinite)
    );
    let acceleration = MotionVector::<f64, R>::new(
        R::vector3_from_array([1.0, 0.0, 0.0]),
        R::vector3_from_array([0.0, 1.0, 0.0]),
    );
    assert_eq!(
        inertia.apply(&acceleration).to_array(),
        [24.0, -4.0, -4.0, 0.0, -4.0, 4.0]
    );
    for (actual, expected) in combined
        .apply(&acceleration)
        .to_array()
        .into_iter()
        .zip((inertia.apply(&acceleration) + other.apply(&acceleration)).to_array())
    {
        assert_close_f64_1e10(actual, expected);
    }
    assert_send_sync::<MotionVector<f64, R>>();
    assert_send_sync::<SpatialTransform<f64, R>>();
    assert_send_sync::<RigidBodyInertia<f64, R>>();
}

#[cfg(feature = "builtin")]
#[test]
fn builtin_backend_matches_the_contract() {
    exercise_backend::<spatial6::Builtin>();
}

#[cfg(feature = "nalgebra")]
#[test]
fn nalgebra_backend_uses_native_types_and_matches_the_contract() {
    use spatial6::Nalgebra;

    let _: nalgebra::Vector3<f64> = Nalgebra::vector3_from_array([1.0, 2.0, 3.0]);
    let _: nalgebra::SVector<f64, 6> = Nalgebra::vector6_from_array([0.0; 6]);
    let rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
    let native = nalgebra::Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0);
    assert_eq!(Nalgebra::matrix3_to_array(&native), rows);
    assert_eq!(Nalgebra::matrix3_from_array(rows)[(2, 1)], 8.0);
    let _: nalgebra::SMatrix<f64, 6, 6> = Nalgebra::matrix6_from_array([[0.0; 6]; 6]);
    let _: nalgebra::Rotation3<f64> =
        Nalgebra::rotation3_from_array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    exercise_backend::<Nalgebra>();
}

#[cfg(feature = "glam")]
#[test]
fn glam_backend_uses_native_types_and_matches_the_contract() {
    use spatial6::Glam;

    let _: glam::DVec3 = Glam::vector3_from_array([1.0, 2.0, 3.0]);
    let rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
    let native = glam::DMat3::from_cols(
        glam::DVec3::new(1.0, 4.0, 7.0),
        glam::DVec3::new(2.0, 5.0, 8.0),
        glam::DVec3::new(3.0, 6.0, 10.0),
    );
    assert_eq!(
        <Glam as SpatialRepresentation<f64>>::matrix3_to_array(&native),
        rows
    );
    assert_eq!(
        <Glam as SpatialRepresentation<f64>>::matrix3_from_array(rows).to_cols_array(),
        [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0]
    );
    let _: glam::DMat3 =
        Glam::rotation3_from_array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

    let transform = SpatialTransform::<f64, Glam>::new(
        glam::DMat3::from_cols(
            glam::DVec3::new(0.0, 1.0, 0.0),
            glam::DVec3::new(-1.0, 0.0, 0.0),
            glam::DVec3::new(0.0, 0.0, 1.0),
        ),
        glam::DVec3::ZERO,
    );
    let motion = MotionVector::<f64, Glam>::new(
        glam::DVec3::new(1.0, 2.0, 3.0),
        glam::DVec3::new(4.0, 5.0, 6.0),
    );
    assert_eq!(
        transform.transform_motion(&motion).to_array(),
        [-2.0, 1.0, 3.0, -5.0, 4.0, 6.0]
    );
    exercise_backend::<Glam>();
}

/// The scenario demonstrated by `examples/backend_builtin.rs`,
/// `examples/backend_nalgebra.rs`, and `examples/backend_glam.rs`: those three
/// files hand-duplicate this data as backend-native literals so each stands
/// alone as a self-contained demo. This test is the automated guarantee that
/// they actually compute the same physical result rather than merely printing
/// matching numbers by coincidence — it runs in CI (via `cargo test
/// --all-features`) even though the examples themselves do not.
#[cfg(all(feature = "builtin", feature = "nalgebra", feature = "glam"))]
#[test]
fn all_backends_agree_on_the_backend_examples_scenario() {
    fn single_body_force<R: SpatialRepresentation<f64>>() -> [f64; 6] {
        let inertia = RigidBodyInertia::<f64, R>::try_new(
            2.0,
            R::vector3_from_array([0.5, 0.0, 0.0]),
            R::matrix3_from_array([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]]),
        )
        .unwrap();
        let velocity = MotionVector::<f64, R>::new(
            R::vector3_from_array([0.2, -0.1, 0.4]),
            R::vector3_from_array([0.1, 0.3, -0.2]),
        );
        let acceleration = MotionVector::<f64, R>::new(
            R::vector3_from_array([0.05, 0.1, -0.05]),
            R::vector3_from_array([1.0, 0.0, -0.5]),
        );
        inertia
            .inverse_dynamics(&velocity, &acceleration)
            .to_array()
    }

    let builtin = single_body_force::<spatial6::Builtin>();
    let nalgebra = single_body_force::<spatial6::Nalgebra>();
    let glam = single_body_force::<spatial6::Glam>();

    for index in 0..6 {
        assert_close_f64_1e10(nalgebra[index], builtin[index]);
        assert_close_f64_1e10(glam[index], builtin[index]);
    }
}

/// The scenario demonstrated by `examples/rnea_builtin.rs`, `examples/rnea_nalgebra.rs`,
/// and `examples/rnea_glam.rs`. See
/// `all_backends_agree_on_the_backend_examples_scenario` above for why this
/// exists: it is the CI-run guarantee behind those three hand-duplicated files.
#[cfg(all(feature = "builtin", feature = "nalgebra", feature = "glam"))]
#[test]
fn all_backends_agree_on_the_rnea_examples_scenario() {
    const NUM_LINKS: usize = 3;

    fn joint_torques<R: SpatialRepresentation<f64>>() -> [f64; NUM_LINKS] {
        let link_length = 1.0;
        let mass = 1.0;
        let rotational_inertia =
            R::matrix3_from_array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]);

        let joint_angle: [f64; NUM_LINKS] = [0.3, -0.2, 0.5];
        let joint_rate = [0.4, -0.3, 0.2];
        let joint_accel = [0.1, 0.0, -0.1];

        let joint_axis = MotionVector::<f64, R>::new(
            R::vector3_from_array([0.0, 0.0, 1.0]),
            R::vector3_from_array([0.0, 0.0, 0.0]),
        );

        let transforms: Vec<SpatialTransform<f64, R>> = joint_angle
            .iter()
            .map(|&angle| {
                let (sin, cos) = angle.sin_cos();
                SpatialTransform::<f64, R>::new(
                    R::rotation3_from_array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]),
                    R::vector3_from_array([link_length, 0.0, 0.0]),
                )
            })
            .collect();

        let com = R::vector3_from_array([link_length / 2.0, 0.0, 0.0]);
        let inertia: Vec<RigidBodyInertia<f64, R>> = (0..NUM_LINKS)
            .map(|_| RigidBodyInertia::<f64, R>::try_new(mass, com, rotational_inertia).unwrap())
            .collect();

        let mut velocity = [MotionVector::<f64, R>::zeros(); NUM_LINKS + 1];
        let mut acceleration = [MotionVector::<f64, R>::zeros(); NUM_LINKS + 1];
        acceleration[0] = MotionVector::<f64, R>::new(
            R::vector3_from_array([0.0, 0.0, 0.0]),
            R::vector3_from_array([0.0, 0.0, 9.81]),
        );

        let mut force = [ForceVector::<f64, R>::zeros(); NUM_LINKS + 1];

        for i in 1..=NUM_LINKS {
            let joint_velocity = joint_axis * joint_rate[i - 1];
            let joint_acceleration = joint_axis * joint_accel[i - 1];

            velocity[i] = transforms[i - 1].transform_motion(&velocity[i - 1]) + joint_velocity;
            acceleration[i] = transforms[i - 1].transform_motion(&acceleration[i - 1])
                + joint_acceleration
                + velocity[i].cross_motion(&joint_velocity);
            force[i] = inertia[i - 1].inverse_dynamics(&velocity[i], &acceleration[i]);
        }

        let mut torque = [0.0; NUM_LINKS];
        for i in (1..=NUM_LINKS).rev() {
            torque[i - 1] = joint_axis.dot(&force[i]);
            if i > 1 {
                force[i - 1] += transforms[i - 1].inverse().transform_force(&force[i]);
            }
        }
        torque
    }

    let builtin = joint_torques::<spatial6::Builtin>();
    let nalgebra = joint_torques::<spatial6::Nalgebra>();
    let glam = joint_torques::<spatial6::Glam>();

    for index in 0..NUM_LINKS {
        assert_close_f64_1e9(nalgebra[index], builtin[index]);
        assert_close_f64_1e9(glam[index], builtin[index]);
    }
}

/// The scenario demonstrated by `examples/composite_rigid_body_builtin.rs`,
/// `examples/composite_rigid_body_nalgebra.rs`, and
/// `examples/composite_rigid_body_glam.rs`. See
/// `all_backends_agree_on_the_backend_examples_scenario` above for why this
/// exists: it is the CI-run guarantee behind those three hand-duplicated files.
#[cfg(all(feature = "builtin", feature = "nalgebra", feature = "glam"))]
#[test]
fn all_backends_agree_on_the_composite_rigid_body_examples_scenario() {
    fn composite_force<R: SpatialRepresentation<f64>>() -> [f64; 6] {
        let inertia_1 = RigidBodyInertia::<f64, R>::try_new(
            2.0,
            R::vector3_from_array([0.5, 0.0, 0.0]),
            R::matrix3_from_array([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]]),
        )
        .unwrap();
        let offset = SpatialTransform::<f64, R>::new(
            R::rotation3_from_array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            R::vector3_from_array([1.0, 0.0, 0.0]),
        );
        let inertia_2_at_own_point = RigidBodyInertia::<f64, R>::try_new(
            1.0,
            R::vector3_from_array([0.3, 0.0, 0.0]),
            R::matrix3_from_array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]),
        )
        .unwrap();
        let inertia_2_at_o = inertia_2_at_own_point.transformed(&offset);
        let composite = inertia_1.try_combined(&inertia_2_at_o).unwrap();

        let velocity = MotionVector::<f64, R>::new(
            R::vector3_from_array([0.2, -0.1, 0.4]),
            R::vector3_from_array([0.1, 0.3, -0.2]),
        );
        let acceleration = MotionVector::<f64, R>::new(
            R::vector3_from_array([0.05, 0.1, -0.05]),
            R::vector3_from_array([1.0, 0.0, -0.5]),
        );

        composite
            .inverse_dynamics(&velocity, &acceleration)
            .to_array()
    }

    let builtin = composite_force::<spatial6::Builtin>();
    let nalgebra = composite_force::<spatial6::Nalgebra>();
    let glam = composite_force::<spatial6::Glam>();

    for index in 0..6 {
        assert_close_f64_1e9(nalgebra[index], builtin[index]);
        assert_close_f64_1e9(glam[index], builtin[index]);
    }
}

/// `RigidBodyInertia::forward_dynamics` is the only place
/// `SpatialRepresentation::matrix6_solve_positive_definite` is exercised for a
/// non-diagonal matrix, and `Nalgebra`'s Cholesky-based override of that
/// function (`src/representation.rs`) had no automated coverage anywhere in
/// the repository before this test -- only `examples/articulated_inertia*.rs`
/// exercise it, and nothing in CI ever runs an example's `main` (`cargo
/// test`/`clippy` only type-check them). This test exercises both of that
/// solver's paths -- recovering a known acceleration from a positive-definite
/// system, and correctly refusing a positive-*semi*-definite one (the same
/// Schur-complement joint-locking projection ABA performs, eq. 7.31-7.34) --
/// generically across all three backends.
#[cfg(all(feature = "builtin", feature = "nalgebra", feature = "glam"))]
#[test]
fn all_backends_agree_on_forward_dynamics_and_reject_singular_systems() {
    fn wheel_inertia<R: SpatialRepresentation<f64>>() -> RigidBodyInertia<f64, R> {
        RigidBodyInertia::<f64, R>::try_new(
            3.0,
            R::vector3_from_array([0.2, 0.15, -0.1]),
            R::matrix3_from_array([[0.4, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 0.5]]),
        )
        .unwrap()
    }

    fn recovered_acceleration<R: SpatialRepresentation<f64>>() -> [f64; 6] {
        let inertia = wheel_inertia::<R>();
        let velocity = MotionVector::<f64, R>::new(
            R::vector3_from_array([0.1, -0.2, 0.3]),
            R::vector3_from_array([0.4, 0.0, -0.1]),
        );
        let acceleration = MotionVector::<f64, R>::new(
            R::vector3_from_array([0.2, 0.1, -0.3]),
            R::vector3_from_array([-0.1, 0.4, 0.2]),
        );
        let force = inertia.inverse_dynamics(&velocity, &acceleration);
        inertia
            .forward_dynamics(&velocity, &force)
            .unwrap()
            .to_array()
    }

    fn rejects_locked_direction<R: SpatialRepresentation<f64>>() -> bool {
        let inertia = wheel_inertia::<R>();
        let s = MotionVector::<f64, R>::new(
            R::vector3_from_array([0.0, 0.0, 0.0]),
            R::vector3_from_array([1.0, 0.0, 0.0]),
        );
        let h = inertia.apply(&s);
        let d = s.dot(&h);
        let h_array = h.to_array();
        let mut projected = R::matrix6_to_array(&inertia.matrix());
        for row in 0..6 {
            for column in 0..6 {
                projected[row][column] -= h_array[row] * h_array[column] / d;
            }
        }
        R::matrix6_solve_positive_definite(&R::matrix6_from_array(projected), &h.to_vector())
            .is_none()
    }

    let expected_acceleration = [0.2, 0.1, -0.3, -0.1, 0.4, 0.2];
    for actual in [
        recovered_acceleration::<spatial6::Builtin>(),
        recovered_acceleration::<spatial6::Nalgebra>(),
        recovered_acceleration::<spatial6::Glam>(),
    ] {
        for index in 0..6 {
            assert_close_f64_1e9(actual[index], expected_acceleration[index]);
        }
    }

    assert!(rejects_locked_direction::<spatial6::Builtin>());
    assert!(rejects_locked_direction::<spatial6::Nalgebra>());
    assert!(rejects_locked_direction::<spatial6::Glam>());
}

#[cfg(feature = "glam")]
#[test]
fn glam_backend_supports_f32() {
    use spatial6::Glam;

    fn assert_close(actual: f32, expected: f32) {
        let tolerance = 1.0e-5 + 1.0e-5 * actual.abs().max(expected.abs());
        assert!((actual - expected).abs() <= tolerance);
    }

    let motion = MotionVector::<f32, Glam>::new(
        glam::Vec3::new(1.0, 2.0, 3.0),
        glam::Vec3::new(4.0, 5.0, 6.0),
    );
    let force = ForceVector::<f32, Glam>::new(
        glam::Vec3::new(7.0, 8.0, 9.0),
        glam::Vec3::new(10.0, 11.0, 12.0),
    );

    assert!((motion.dot(&force) - 217.0).abs() <= 1.0e-5);

    let left = RigidBodyInertia::<f32, Glam>::try_new(
        2.0,
        glam::Vec3::new(1.0, 0.0, 0.0),
        glam::Mat3::from_diagonal(glam::Vec3::new(1.0, 2.0, 3.0)),
    )
    .unwrap();
    let right = RigidBodyInertia::<f32, Glam>::try_new(
        3.0,
        glam::Vec3::new(-1.0, 2.0, 0.5),
        glam::Mat3::from_diagonal(glam::Vec3::new(4.0, 5.0, 6.0)),
    )
    .unwrap();
    let combined = left.try_combined(&right).unwrap();
    assert_close(combined.mass(), 5.0);
    for (actual, expected) in combined
        .center_of_mass()
        .to_array()
        .into_iter()
        .zip([-0.2, 1.2, 0.3])
    {
        assert_close(actual, expected);
    }
}
