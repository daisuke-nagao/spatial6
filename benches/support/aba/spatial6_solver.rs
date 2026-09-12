// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

use spatial6::{
    ArticulatedBodyInertia, ForceVector, InertiaError, MotionVector, RigidBodyInertia,
    SpatialScalar, SpatialTransform,
};

use super::fixture::{Axis, Fixture, FixtureError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AbaBenchReason {
    InvalidModel,
    InvalidState,
    NonFinite,
    NonPositiveJointInertia,
    Inertia(InertiaError),
    Fixture(FixtureError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AbaBenchError {
    pub phase: &'static str,
    pub joint: Option<usize>,
    pub reason: AbaBenchReason,
}

impl AbaBenchError {
    const fn new(phase: &'static str, joint: Option<usize>, reason: AbaBenchReason) -> Self {
        Self {
            phase,
            joint,
            reason,
        }
    }
}

impl fmt::Display for AbaBenchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.joint {
            Some(joint) => write!(
                formatter,
                "spatial6 ABA {} failed at joint {}: {:?}",
                self.phase, joint, self.reason
            ),
            None => write!(
                formatter,
                "spatial6 ABA {} failed: {:?}",
                self.phase, self.reason
            ),
        }
    }
}

impl std::error::Error for AbaBenchError {}

pub struct Spatial6Model<T: SpatialScalar + From<f32>> {
    rigid_inertias: Vec<RigidBodyInertia<T>>,
    initial_abis: Vec<ArticulatedBodyInertia<T>>,
    subspaces: Vec<MotionVector<T>>,
    tree_transforms: Vec<SpatialTransform<T>>,
    axes: Vec<Axis>,
    gravity: [T; 3],
}

pub struct Spatial6State<T: SpatialScalar + From<f32>> {
    pub q: Vec<T>,
    pub qd: Vec<T>,
    pub tau: Vec<T>,
    pub qdd: Vec<T>,
}

impl<T> Spatial6Model<T>
where
    T: SpatialScalar + From<f32>,
{
    pub fn from_fixture(fixture: &Fixture) -> Result<Self, AbaBenchError> {
        fixture
            .validate()
            .map_err(|error| AbaBenchError::new("model", None, AbaBenchReason::Fixture(error)))?;

        let mut rigid_inertias = Vec::with_capacity(fixture.links.len());
        let mut initial_abis = Vec::with_capacity(fixture.links.len());
        let mut subspaces = Vec::with_capacity(fixture.links.len());
        let mut tree_transforms = Vec::with_capacity(fixture.links.len());
        let mut axes = Vec::with_capacity(fixture.links.len());

        for (index, link) in fixture.links.iter().enumerate() {
            let rigid = RigidBodyInertia::<T>::try_new(
                <T as From<f32>>::from(link.mass),
                cast_vector(link.com),
                cast_matrix(diagonal_matrix(link.inertia_com_diagonal)),
            )
            .map_err(|error| {
                AbaBenchError::new("model", Some(index), AbaBenchReason::Inertia(error))
            })?;
            let abi = ArticulatedBodyInertia::try_from(&rigid).map_err(|error| {
                AbaBenchError::new("model", Some(index), AbaBenchReason::Inertia(error))
            })?;
            rigid_inertias.push(rigid);
            initial_abis.push(abi);
            subspaces.push(MotionVector::new(
                cast_vector(link.axis.as_array()),
                [T::zero(); 3],
            ));
            tree_transforms.push(SpatialTransform::new(
                cast_matrix(link.tree_rotation),
                cast_vector(link.parent_offset),
            ));
            axes.push(link.axis);
        }

        let model = Self {
            rigid_inertias,
            initial_abis,
            subspaces,
            tree_transforms,
            axes,
            gravity: cast_vector(fixture.gravity),
        };
        model.validate()?;
        Ok(model)
    }

    fn validate(&self) -> Result<(), AbaBenchError> {
        let n = self.rigid_inertias.len();
        if n == 0
            || self.initial_abis.len() != n
            || self.subspaces.len() != n
            || self.tree_transforms.len() != n
            || self.axes.len() != n
            || !self.gravity.into_iter().all(|value| value.is_finite())
        {
            return Err(AbaBenchError::new(
                "model",
                None,
                AbaBenchReason::InvalidModel,
            ));
        }

        for index in 0..n {
            if !self.rigid_inertias[index]
                .matrix()
                .iter()
                .flatten()
                .all(|value| value.is_finite())
                || !self.initial_abis[index]
                    .matrix()
                    .iter()
                    .flatten()
                    .all(|value| value.is_finite())
                || !self.subspaces[index].is_finite()
                || !self.tree_transforms[index]
                    .motion_matrix()
                    .iter()
                    .flatten()
                    .all(|value| value.is_finite())
            {
                return Err(AbaBenchError::new(
                    "model",
                    Some(index),
                    AbaBenchReason::NonFinite,
                ));
            }
        }
        Ok(())
    }
}

impl<T> Spatial6State<T>
where
    T: SpatialScalar + From<f32>,
{
    pub fn from_fixture(fixture: &Fixture) -> Result<Self, FixtureError> {
        fixture.validate()?;
        Ok(Self {
            q: fixture
                .q
                .iter()
                .copied()
                .map(<T as From<f32>>::from)
                .collect(),
            qd: fixture
                .qd
                .iter()
                .copied()
                .map(<T as From<f32>>::from)
                .collect(),
            tau: fixture
                .tau
                .iter()
                .copied()
                .map(<T as From<f32>>::from)
                .collect(),
            qdd: vec![T::zero(); fixture.links.len()],
        })
    }

    #[allow(dead_code)]
    pub fn install_fixture(&mut self, fixture: &Fixture) -> Result<(), FixtureError> {
        *self = Self::from_fixture(fixture)?;
        Ok(())
    }
}

pub fn spatial6_aba_allocating<'a, T>(
    model: &Spatial6Model<T>,
    state: &'a mut Spatial6State<T>,
) -> Result<&'a [T], AbaBenchError>
where
    T: SpatialScalar + From<f32>,
{
    let n = model.rigid_inertias.len();
    if state.q.len() != n || state.qd.len() != n || state.tau.len() != n || state.qdd.len() != n {
        return Err(AbaBenchError::new(
            "state",
            None,
            AbaBenchReason::InvalidState,
        ));
    }
    if !state.q.iter().all(|value| value.is_finite())
        || !state.qd.iter().all(|value| value.is_finite())
        || !state.tau.iter().all(|value| value.is_finite())
    {
        return Err(AbaBenchError::new("state", None, AbaBenchReason::NonFinite));
    }

    let mut x_up = vec![SpatialTransform::<T>::identity(); n];
    let mut velocity = vec![MotionVector::<T>::zeros(); n];
    let mut bias_acceleration = vec![MotionVector::<T>::zeros(); n];
    let mut acceleration = vec![MotionVector::<T>::zeros(); n];
    let mut articulated_inertia = vec![ArticulatedBodyInertia::<T>::zeros(); n];
    let mut bias_force = vec![ForceVector::<T>::zeros(); n];
    let mut u_force = vec![ForceVector::<T>::zeros(); n];
    let mut d = vec![T::zero(); n];
    let mut u_scalar = vec![T::zero(); n];

    for index in 0..n {
        let joint = super::featherstone_adapter::spatial6_joint_transform(
            model.axes[index],
            state.q[index],
        );
        x_up[index] = model.tree_transforms[index].then(&joint);
        if !x_up[index]
            .motion_matrix()
            .iter()
            .flatten()
            .all(|value| value.is_finite())
        {
            return Err(AbaBenchError::new(
                "pass1 transform",
                Some(index),
                AbaBenchReason::NonFinite,
            ));
        }

        let joint_velocity = model.subspaces[index] * state.qd[index];
        velocity[index] = if index == 0 {
            joint_velocity
        } else {
            x_up[index].transform_motion(&velocity[index - 1]) + joint_velocity
        };
        bias_acceleration[index] = if index == 0 {
            MotionVector::zeros()
        } else {
            velocity[index].cross_motion(&joint_velocity)
        };
        articulated_inertia[index] = model.initial_abis[index];
        bias_force[index] =
            velocity[index].cross_force(&model.rigid_inertias[index].apply(&velocity[index]));
        if !velocity[index].is_finite()
            || !bias_acceleration[index].is_finite()
            || !bias_force[index].is_finite()
        {
            return Err(AbaBenchError::new(
                "pass1",
                Some(index),
                AbaBenchReason::NonFinite,
            ));
        }
    }

    for index in (0..n).rev() {
        let current_ia = articulated_inertia[index];
        let current_pa = bias_force[index];
        let current_c = bias_acceleration[index];
        let current_x = x_up[index];
        let u = current_ia.apply(&model.subspaces[index]);
        let joint_inertia = model.subspaces[index].dot(&u);
        let torque_residual = state.tau[index] - model.subspaces[index].dot(&current_pa);
        if !u.is_finite() || !joint_inertia.is_finite() || !torque_residual.is_finite() {
            return Err(AbaBenchError::new(
                "pass2 projection",
                Some(index),
                AbaBenchReason::NonFinite,
            ));
        }
        if joint_inertia < <T as From<f32>>::from(1.0e-4_f32) {
            return Err(AbaBenchError::new(
                "pass2 projection",
                Some(index),
                AbaBenchReason::NonPositiveJointInertia,
            ));
        }
        u_force[index] = u;
        d[index] = joint_inertia;
        u_scalar[index] = torque_residual;

        if index > 0 {
            let reduced = current_ia
                .try_rank_one_updated(-T::one() / joint_inertia, &u)
                .map_err(|error| {
                    AbaBenchError::new(
                        "pass2 reduction",
                        Some(index),
                        AbaBenchReason::Inertia(error),
                    )
                })?;
            let reduced_bias =
                current_pa + reduced.apply(&current_c) + u * (torque_residual / joint_inertia);
            if !reduced_bias.is_finite() {
                return Err(AbaBenchError::new(
                    "pass2 reduction",
                    Some(index),
                    AbaBenchReason::NonFinite,
                ));
            }
            let child_to_parent = current_x.inverse();
            let transformed = reduced.try_transformed(&child_to_parent).map_err(|error| {
                AbaBenchError::new(
                    "pass2 transform",
                    Some(index),
                    AbaBenchReason::Inertia(error),
                )
            })?;
            articulated_inertia[index - 1] = articulated_inertia[index - 1]
                .try_combined(&transformed)
                .map_err(|error| {
                    AbaBenchError::new("pass2 combine", Some(index), AbaBenchReason::Inertia(error))
                })?;
            bias_force[index - 1] += child_to_parent.transform_force(&reduced_bias);
            if !bias_force[index - 1].is_finite()
                || !articulated_inertia[index - 1]
                    .matrix()
                    .iter()
                    .flatten()
                    .all(|value| value.is_finite())
            {
                return Err(AbaBenchError::new(
                    "pass2 combine",
                    Some(index),
                    AbaBenchReason::NonFinite,
                ));
            }
        }
    }

    for index in 0..n {
        let base_acceleration = if index == 0 {
            MotionVector::new([T::zero(); 3], model.gravity.map(|value| -value))
        } else {
            acceleration[index - 1]
        };
        let acceleration_before_joint =
            x_up[index].transform_motion(&base_acceleration) + bias_acceleration[index];
        let joint_acceleration =
            (u_scalar[index] - acceleration_before_joint.dot(&u_force[index])) / d[index];
        if !acceleration_before_joint.is_finite() || !joint_acceleration.is_finite() {
            state.qdd.fill(T::zero());
            return Err(AbaBenchError::new(
                "pass3",
                Some(index),
                AbaBenchReason::NonFinite,
            ));
        }
        state.qdd[index] = joint_acceleration;
        acceleration[index] =
            acceleration_before_joint + model.subspaces[index] * joint_acceleration;
        if !acceleration[index].is_finite() {
            state.qdd.fill(T::zero());
            return Err(AbaBenchError::new(
                "pass3",
                Some(index),
                AbaBenchReason::NonFinite,
            ));
        }
    }

    Ok(state.qdd.as_slice())
}

fn diagonal_matrix(diagonal: [f32; 3]) -> [[f32; 3]; 3] {
    [
        [diagonal[0], 0.0, 0.0],
        [0.0, diagonal[1], 0.0],
        [0.0, 0.0, diagonal[2]],
    ]
}

fn cast_vector<T: From<f32>>(value: [f32; 3]) -> [T; 3] {
    value.map(<T as From<f32>>::from)
}

fn cast_matrix<T: From<f32>>(value: [[f32; 3]; 3]) -> [[T; 3]; 3] {
    value.map(|row| row.map(<T as From<f32>>::from))
}
