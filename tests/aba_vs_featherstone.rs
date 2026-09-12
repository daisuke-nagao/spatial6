#![cfg(feature = "builtin")]

// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../benches/support/aba/mod.rs"]
mod aba;

use aba::featherstone_adapter::{
    featherstone_inertia, featherstone_transform, install_featherstone_state, spatial6_inertia,
    spatial6_transform,
};
use aba::fixture::{Axis, ChainFamily, Fixture, LinkSpec};
use aba::reference::{ReferenceResult, preflight_all, preflight_case, rnea_f64, solve_reference};
use aba::spatial6_solver::{Spatial6Model, Spatial6State, spatial6_aba_allocating};

const MAPPING_TOLERANCE: f32 = 64.0 * f32::EPSILON;

fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    let scale = actual.abs().max(expected.abs()).max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance * scale,
        "{actual} != {expected}"
    );
}

fn assert_matrix_close(actual: &nalgebra_featherstone::Matrix6<f32>, expected: [[f32; 6]; 6]) {
    for row in 0..6 {
        for column in 0..6 {
            assert_close(
                actual[(row, column)],
                expected[row][column],
                MAPPING_TOLERANCE,
            );
        }
    }
}

fn assert_vector_close(actual: &[f32], expected: &[f32], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        let difference = f64::from(*actual - *expected).abs();
        assert!(difference <= tolerance, "{actual} != {expected}");
    }
}

fn assert_finite_reference(result: &ReferenceResult) {
    assert!(result.qdd.iter().all(|value| value.is_finite()));
    assert!(result.bias.iter().all(|value| value.is_finite()));
    assert!(
        result
            .mass_matrix
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    );
}

fn rotated_mount() -> LinkSpec {
    LinkSpec::new(
        Axis::Y,
        [0.17, -0.08, 0.11],
        1.0,
        [0.05, 0.01, -0.02],
        [0.003, 0.004, 0.005],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
}

#[test]
fn canonical_fixture_has_both_named_families() {
    assert_eq!(aba::fixture::CHAIN_LENGTHS, [2, 4, 8, 16, 32]);
    assert_eq!(Axis::X.as_array(), [1.0, 0.0, 0.0]);
    for family in [ChainFamily::PlanarZ, ChainFamily::SpatialXyz] {
        let fixture = Fixture::canonical(family, 2).unwrap();
        assert_eq!(
            family.as_str(),
            if family == ChainFamily::PlanarZ {
                "planar_z"
            } else {
                "spatial_xyz"
            }
        );
        assert_eq!(fixture.links.len(), 2);
        assert_eq!(fixture.q.len(), 2);
        assert_eq!(fixture.qd.len(), 2);
        assert_eq!(fixture.tau.len(), 2);
        assert_eq!(fixture.gravity, [0.0, -9.81, 0.0]);
        assert_eq!(
            aba::featherstone_adapter::featherstone_body(&fixture)
                .unwrap()
                .body_count(),
            2
        );
    }
}

#[test]
fn a01_link_inertia_matrices_match_for_nonzero_com() {
    let fixture = Fixture::canonical(ChainFamily::SpatialXyz, 2).unwrap();

    for link in &fixture.links {
        let spatial6 = spatial6_inertia(link).unwrap().matrix();
        let featherstone = featherstone_inertia(link).data;
        assert_matrix_close(&featherstone, spatial6);
    }
}

#[test]
fn a02_transform_matrices_and_basis_applications_match() {
    let link = rotated_mount();
    let q = 0.37;
    let spatial6 = spatial6_transform(&link, q);
    let featherstone = featherstone_transform(&link, q);

    assert_matrix_close(&featherstone.to_matrix_motion(), spatial6.motion_matrix());
    assert_matrix_close(&featherstone.to_matrix_force(), spatial6.force_matrix());

    for basis in 0..6 {
        let mut vector = [0.0; 6];
        vector[basis] = 1.0;
        let motion = spatial6.transform_motion(&spatial6::MotionVector::from_array(vector));
        let force = spatial6.transform_force(&spatial6::ForceVector::from_array(vector));
        for (actual, expected) in featherstone
            .apply_motion(&featherstone::prelude::SpatialVector::from_vector(
                nalgebra_featherstone::Vector6::from_column_slice(&vector),
            ))
            .data
            .iter()
            .zip(motion.to_array())
        {
            assert_close(*actual, expected, MAPPING_TOLERANCE);
        }
        for (actual, expected) in featherstone
            .apply_force(&featherstone::prelude::SpatialVector::from_vector(
                nalgebra_featherstone::Vector6::from_column_slice(&vector),
            ))
            .data
            .iter()
            .zip(force.to_array())
        {
            assert_close(*actual, expected, MAPPING_TOLERANCE);
        }
    }
}

#[test]
fn a03_gravity_adapters_produce_base_acceleration() {
    let gravity = [0.0, -9.81, 0.0];
    let spatial6 = aba::featherstone_adapter::spatial6_base_acceleration(gravity);
    assert_eq!(spatial6.to_array(), [0.0, 0.0, 0.0, 0.0, 9.81, 0.0]);

    let mut body = featherstone::prelude::ArticulatedBody::new();
    aba::featherstone_adapter::set_featherstone_gravity(&mut body, gravity);
    assert_eq!(
        body.gravity.data,
        nalgebra_featherstone::Vector6::new(0.0, 0.0, 0.0, 0.0, 9.81, 0.0)
    );
}

#[test]
fn a04_one_joint_zero_com_zero_gravity_matches_scalar_inertia() {
    let fixture = Fixture::try_new(
        ChainFamily::PlanarZ,
        vec![LinkSpec::new(
            Axis::Z,
            [0.0; 3],
            1.0,
            [0.0; 3],
            [0.7, 0.8, 1.2],
            aba::fixture::identity_rotation(),
        )],
        vec![0.0],
        vec![0.0],
        vec![0.36],
        [0.0; 3],
    )
    .unwrap();
    let model = Spatial6Model::<f32>::from_fixture(&fixture).unwrap();
    let mut state = Spatial6State::<f32>::from_fixture(&fixture).unwrap();
    state.qdd[0] = 123.0;

    let qdd = spatial6_aba_allocating(&model, &mut state).unwrap();
    assert_close(qdd[0], 0.36 / 1.2, 1.0e-6);
    assert_close(state.qdd[0], 0.36 / 1.2, 1.0e-6);
    let mut body = aba::featherstone_adapter::featherstone_body(&fixture).unwrap();
    let featherstone_qdd = featherstone::prelude::aba_forward_dynamics(&mut body)[0];
    assert_close(featherstone_qdd, 0.36 / 1.2, 1.0e-6);
}

#[test]
fn a05_one_z_joint_gravity_matches_com_torque() {
    let mass = 2.0;
    let lever = 0.4;
    let i_zz = 0.3;
    let gravity = [0.0, -9.81, 0.0];
    let fixture = Fixture::try_new(
        ChainFamily::PlanarZ,
        vec![LinkSpec::new(
            Axis::Z,
            [0.0; 3],
            mass,
            [lever, 0.0, 0.0],
            [0.7, 0.8, i_zz],
            aba::fixture::identity_rotation(),
        )],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        gravity,
    )
    .unwrap();
    let model = Spatial6Model::<f32>::from_fixture(&fixture).unwrap();
    let mut state = Spatial6State::<f32>::from_fixture(&fixture).unwrap();

    let qdd = spatial6_aba_allocating(&model, &mut state).unwrap();
    let expected = -mass * -gravity[1] * lever / (i_zz + mass * lever * lever);
    assert_close(qdd[0], expected, 1.0e-5);
    let mut body = aba::featherstone_adapter::featherstone_body(&fixture).unwrap();
    let featherstone_qdd = featherstone::prelude::aba_forward_dynamics(&mut body)[0];
    assert_close(featherstone_qdd, expected, 1.0e-5);
}

#[test]
fn a06_two_link_spatial_fixture_exercises_bias_propagation() {
    let fixture = Fixture::canonical(ChainFamily::SpatialXyz, 2).unwrap();
    assert!(fixture.qd.iter().any(|value| *value != 0.0));
    assert!(fixture.tau.iter().any(|value| *value != 0.0));

    let result = preflight_case(ChainFamily::SpatialXyz, 2).unwrap();
    let reference = solve_reference(&fixture).unwrap();
    assert_finite_reference(&reference);
    assert!(
        result.pairwise_error
            <= 2.0e-3
                + 2.0e-3
                    * result
                        .reference_qdd
                        .iter()
                        .map(|value| value.abs())
                        .fold(0.0, f64::max)
    );
}

#[test]
fn a07_all_timed_fixtures_pass_oracle_pairwise_and_residual_gates() {
    let results = preflight_all().unwrap();
    assert_eq!(results.len(), 10);
    for result in results {
        let output_limit = 2.0e-3
            + 2.0e-3
                * result
                    .reference_qdd
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0, f64::max);
        assert!(result.spatial6_error <= output_limit);
        assert!(result.featherstone_error <= output_limit);
        assert!(result.pairwise_error <= output_limit);
        assert!(result.reference_residual <= 2.0e-4);
        assert!(result.spatial6_residual <= 2.0e-4);
        assert!(result.featherstone_residual <= 2.0e-4);
    }
}

#[test]
fn a08_rnea_generated_torque_is_recovered_by_both_aba_solvers() {
    let base = Fixture::canonical(ChainFamily::SpatialXyz, 4).unwrap();
    let selected_qdd = [0.11_f32, -0.23, 0.37, -0.19];
    let tau_f64 = rnea_f64(&base, &base.q, &base.qd, &selected_qdd, base.gravity).unwrap();
    let tau = tau_f64
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let fixture = Fixture::try_new(
        base.family,
        base.links.clone(),
        base.q.clone(),
        base.qd.clone(),
        tau,
        base.gravity,
    )
    .unwrap();

    let model = Spatial6Model::<f32>::from_fixture(&fixture).unwrap();
    let mut state = Spatial6State::<f32>::from_fixture(&fixture).unwrap();
    let spatial6_qdd = spatial6_aba_allocating(&model, &mut state)
        .unwrap()
        .to_vec();
    let mut body = aba::featherstone_adapter::featherstone_body(&fixture).unwrap();
    let featherstone_qdd = featherstone::prelude::aba_forward_dynamics(&mut body)
        .as_slice()
        .to_vec();

    assert_vector_close(&spatial6_qdd, &selected_qdd, 2.0e-3);
    assert_vector_close(&featherstone_qdd, &selected_qdd, 2.0e-3);
}

#[test]
fn a09_state_a_b_a_recomputes_all_solver_state() {
    let fixture_a = Fixture::canonical(ChainFamily::SpatialXyz, 4).unwrap();
    let fixture_b = Fixture::try_new(
        fixture_a.family,
        fixture_a.links.clone(),
        vec![0.31, -0.27, 0.19, -0.13],
        vec![0.7, -0.4, 0.2, -0.6],
        vec![0.09, -0.06, 0.03, -0.12],
        fixture_a.gravity,
    )
    .unwrap();
    let model = Spatial6Model::<f32>::from_fixture(&fixture_a).unwrap();
    let mut state = Spatial6State::<f32>::from_fixture(&fixture_a).unwrap();
    let output_a = spatial6_aba_allocating(&model, &mut state)
        .unwrap()
        .to_vec();
    state.install_fixture(&fixture_b).unwrap();
    let output_b = spatial6_aba_allocating(&model, &mut state)
        .unwrap()
        .to_vec();
    state.install_fixture(&fixture_a).unwrap();
    let output_a_again = spatial6_aba_allocating(&model, &mut state)
        .unwrap()
        .to_vec();
    assert!(
        output_a
            .iter()
            .zip(&output_b)
            .any(|(left, right)| (left - right).abs() > 1.0e-5)
    );
    assert_vector_close(&output_a_again, &output_a, 1.0e-6);

    let mut body = aba::featherstone_adapter::featherstone_body(&fixture_a).unwrap();
    let featherstone_a = featherstone::prelude::aba_forward_dynamics(&mut body)
        .as_slice()
        .to_vec();
    install_featherstone_state(&mut body, &fixture_b).unwrap();
    let featherstone_b = featherstone::prelude::aba_forward_dynamics(&mut body)
        .as_slice()
        .to_vec();
    install_featherstone_state(&mut body, &fixture_a).unwrap();
    let featherstone_a_again = featherstone::prelude::aba_forward_dynamics(&mut body)
        .as_slice()
        .to_vec();
    assert!(
        featherstone_a
            .iter()
            .zip(&featherstone_b)
            .any(|(left, right)| (left - right).abs() > 1.0e-5)
    );
    assert_vector_close(&featherstone_a_again, &featherstone_a, 1.0e-6);
}

#[test]
fn a10_invalid_inputs_are_rejected_without_committing_partial_qdd() {
    let fixture = Fixture::canonical(ChainFamily::PlanarZ, 2).unwrap();
    let model = Spatial6Model::<f32>::from_fixture(&fixture).unwrap();
    let mut state = Spatial6State::<f32>::from_fixture(&fixture).unwrap();
    state.qdd.fill(17.0);
    state.q[0] = f32::NAN;
    let error = spatial6_aba_allocating(&model, &mut state).unwrap_err();
    assert_eq!(
        error.reason,
        aba::spatial6_solver::AbaBenchReason::NonFinite
    );
    assert_eq!(state.qdd, vec![17.0; 2]);

    assert_eq!(
        Fixture::try_new(
            fixture.family,
            fixture.links.clone(),
            vec![0.0],
            fixture.qd.clone(),
            fixture.tau.clone(),
            fixture.gravity,
        )
        .unwrap_err(),
        aba::fixture::FixtureError::LengthMismatch
    );
    assert_eq!(
        Fixture::try_new(
            fixture.family,
            {
                let mut links = fixture.links.clone();
                links[0].mass = f32::NAN;
                links
            },
            fixture.q.clone(),
            fixture.qd.clone(),
            fixture.tau.clone(),
            fixture.gravity,
        )
        .unwrap_err(),
        aba::fixture::FixtureError::NonFinite
    );
    assert_eq!(
        Fixture::try_new(
            fixture.family,
            {
                let mut links = fixture.links.clone();
                links[0].tree_rotation = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
                links
            },
            fixture.q.clone(),
            fixture.qd.clone(),
            fixture.tau.clone(),
            fixture.gravity,
        )
        .unwrap_err(),
        aba::fixture::FixtureError::InvalidRotation
    );
}
