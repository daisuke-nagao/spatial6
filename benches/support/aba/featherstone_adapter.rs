// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use featherstone::prelude::{
    ArticulatedBody, GenJoint, SpatialInertia as FsInertia, SpatialTransform as FsTransform,
};
use nalgebra_featherstone::{Matrix3 as FsMatrix3, Vector3 as FsVector3};
use spatial6::{
    InertiaError, MotionVector, RigidBodyInertia, SpatialInertia as S6Inertia, SpatialScalar,
    SpatialTransform as S6Transform,
};

use super::fixture::{Axis, Fixture, FixtureError, LinkSpec};

pub fn spatial6_inertia(link: &LinkSpec) -> Result<RigidBodyInertia<f32>, InertiaError> {
    S6Inertia::try_new(
        link.mass,
        link.com,
        diagonal_matrix(link.inertia_com_diagonal),
    )
}

pub fn featherstone_inertia(link: &LinkSpec) -> FsInertia {
    FsInertia::from_mass_inertia(
        link.mass,
        fs_vector(link.com),
        fs_matrix(diagonal_matrix(link.inertia_com_diagonal)),
    )
}

pub fn spatial6_tree_transform(link: &LinkSpec) -> S6Transform<f32> {
    S6Transform::new(link.tree_rotation, link.parent_offset)
}

pub fn featherstone_tree_transform(link: &LinkSpec) -> FsTransform {
    let rotation = fs_matrix(link.tree_rotation);
    let translation = rotation * fs_vector(link.parent_offset);
    FsTransform::from_rotation_translation(rotation, translation)
}

pub fn spatial6_joint_transform<T: SpatialScalar>(axis: Axis, q: T) -> S6Transform<T> {
    let (sin, cos) = q.sin_cos();
    let zero = T::zero();
    let one = T::one();
    let rotation = match axis {
        Axis::X => [[one, zero, zero], [zero, cos, -sin], [zero, sin, cos]],
        Axis::Y => [[cos, zero, sin], [zero, one, zero], [-sin, zero, cos]],
        Axis::Z => [[cos, -sin, zero], [sin, cos, zero], [zero, zero, one]],
    };
    S6Transform::new(rotation, [zero; 3])
}

pub fn featherstone_joint_transform(axis: Axis, q: f32) -> FsTransform {
    GenJoint::Revolute {
        axis: fs_vector(axis.as_array()),
    }
    .transform(&[q])
}

pub fn spatial6_transform(link: &LinkSpec, q: f32) -> S6Transform<f32> {
    spatial6_tree_transform(link).then(&spatial6_joint_transform(link.axis, q))
}

pub fn featherstone_transform(link: &LinkSpec, q: f32) -> FsTransform {
    featherstone_joint_transform(link.axis, q).compose(&featherstone_tree_transform(link))
}

pub fn spatial6_base_acceleration(gravity: [f32; 3]) -> MotionVector<f32> {
    MotionVector::new([0.0; 3], gravity.map(|value| -value))
}

pub fn set_featherstone_gravity(body: &mut ArticulatedBody, gravity: [f32; 3]) {
    body.set_gravity(fs_vector(gravity));
}

pub fn featherstone_body(fixture: &Fixture) -> Result<ArticulatedBody, FixtureError> {
    fixture.validate()?;

    let mut body = ArticulatedBody::new();
    for (index, link) in fixture.links.iter().enumerate() {
        spatial6_inertia(link).map_err(|error| match error {
            InertiaError::NonFinite => FixtureError::NonFinite,
            InertiaError::NonPositiveMass => FixtureError::InvalidMass,
            InertiaError::NonSymmetric | InertiaError::NotPositiveDefinite => {
                FixtureError::InvalidInertia
            }
        })?;
        let parent = if index == 0 { -1 } else { (index - 1) as i32 };
        body.add_body(
            format!("link{index}"),
            parent,
            GenJoint::Revolute {
                axis: fs_vector(link.axis.as_array()),
            },
            featherstone_inertia(link),
            featherstone_tree_transform(link),
        );
    }
    install_featherstone_state(&mut body, fixture)?;
    Ok(body)
}

pub fn install_featherstone_state(
    body: &mut ArticulatedBody,
    fixture: &Fixture,
) -> Result<(), FixtureError> {
    fixture.validate()?;
    set_featherstone_gravity(body, fixture.gravity);
    for index in 0..fixture.links.len() {
        body.set_joint_q(index, &[fixture.q[index]]);
        body.set_joint_qd(index, &[fixture.qd[index]]);
        body.set_joint_tau(index, &[fixture.tau[index]]);
    }
    body.clear_external_forces();
    Ok(())
}

fn diagonal_matrix(diagonal: [f32; 3]) -> [[f32; 3]; 3] {
    [
        [diagonal[0], 0.0, 0.0],
        [0.0, diagonal[1], 0.0],
        [0.0, 0.0, diagonal[2]],
    ]
}

fn fs_vector(value: [f32; 3]) -> FsVector3<f32> {
    FsVector3::new(value[0], value[1], value[2])
}

fn fs_matrix(value: [[f32; 3]; 3]) -> FsMatrix3<f32> {
    FsMatrix3::from_row_slice(&[
        value[0][0],
        value[0][1],
        value[0][2],
        value[1][0],
        value[1][1],
        value[1][2],
        value[2][0],
        value[2][1],
        value[2][2],
    ])
}
