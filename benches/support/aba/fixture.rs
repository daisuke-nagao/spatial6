// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

pub const CHAIN_LENGTHS: [usize; 5] = [2, 4, 8, 16, 32];
pub const GRAVITY: [f32; 3] = [0.0, -9.81, 0.0];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChainFamily {
    PlanarZ,
    SpatialXyz,
}

impl ChainFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PlanarZ => "planar_z",
            Self::SpatialXyz => "spatial_xyz",
        }
    }

    const fn link_axis(self, index: usize) -> Axis {
        match self {
            Self::PlanarZ => Axis::Z,
            Self::SpatialXyz => match index % 3 {
                0 => Axis::X,
                1 => Axis::Y,
                _ => Axis::Z,
            },
        }
    }

    const fn parent_offset(self) -> [f32; 3] {
        match self {
            Self::PlanarZ => [0.1, 0.0, 0.0],
            Self::SpatialXyz => [0.1, 0.02, -0.01],
        }
    }

    const fn center_of_mass(self) -> [f32; 3] {
        match self {
            Self::PlanarZ => [0.05, 0.0, 0.0],
            Self::SpatialXyz => [0.05, 0.01, -0.02],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub const fn as_array(self) -> [f32; 3] {
        match self {
            Self::X => [1.0, 0.0, 0.0],
            Self::Y => [0.0, 1.0, 0.0],
            Self::Z => [0.0, 0.0, 1.0],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkSpec {
    pub axis: Axis,
    pub parent_offset: [f32; 3],
    pub mass: f32,
    pub com: [f32; 3],
    pub inertia_com_diagonal: [f32; 3],
    pub tree_rotation: [[f32; 3]; 3],
}

impl LinkSpec {
    pub const fn new(
        axis: Axis,
        parent_offset: [f32; 3],
        mass: f32,
        com: [f32; 3],
        inertia_com_diagonal: [f32; 3],
        tree_rotation: [[f32; 3]; 3],
    ) -> Self {
        Self {
            axis,
            parent_offset,
            mass,
            com,
            inertia_com_diagonal,
            tree_rotation,
        }
    }

    fn is_finite(self) -> bool {
        self.mass.is_finite()
            && self.parent_offset.into_iter().all(f32::is_finite)
            && self.com.into_iter().all(f32::is_finite)
            && self.inertia_com_diagonal.into_iter().all(f32::is_finite)
            && self.tree_rotation.into_iter().flatten().all(f32::is_finite)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Fixture {
    pub family: ChainFamily,
    pub links: Vec<LinkSpec>,
    pub q: Vec<f32>,
    pub qd: Vec<f32>,
    pub tau: Vec<f32>,
    pub gravity: [f32; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixtureError {
    Empty,
    LengthMismatch,
    NonFinite,
    InvalidMass,
    InvalidInertia,
    InvalidRotation,
}

impl fmt::Display for FixtureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Empty => "ABA fixture must contain at least one link",
            Self::LengthMismatch => "ABA fixture state lengths must match link count",
            Self::NonFinite => "ABA fixture contains a non-finite value",
            Self::InvalidMass => "ABA fixture mass must be finite and positive",
            Self::InvalidInertia => "ABA fixture COM inertia must be finite and positive",
            Self::InvalidRotation => "ABA fixture tree rotation must be proper orthogonal",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for FixtureError {}

impl Fixture {
    pub fn canonical(family: ChainFamily, length: usize) -> Result<Self, FixtureError> {
        if length == 0 {
            return Err(FixtureError::Empty);
        }

        let links = (0..length)
            .map(|index| {
                LinkSpec::new(
                    family.link_axis(index),
                    family.parent_offset(),
                    1.0,
                    family.center_of_mass(),
                    [0.003, 0.004, 0.005],
                    identity_rotation(),
                )
            })
            .collect();
        let q = (0..length)
            .map(|index| 0.12_f32 * ((index % 7) as i32 - 3) as f32)
            .collect();
        let qd = (0..length)
            .map(|index| 0.20_f32 * ((index % 5) as i32 - 2) as f32)
            .collect();
        let tau = (0..length)
            .map(|index| 0.03_f32 * ((index % 3) as i32 - 1) as f32)
            .collect();

        Self::try_new(family, links, q, qd, tau, GRAVITY)
    }

    pub fn try_new(
        family: ChainFamily,
        links: Vec<LinkSpec>,
        q: Vec<f32>,
        qd: Vec<f32>,
        tau: Vec<f32>,
        gravity: [f32; 3],
    ) -> Result<Self, FixtureError> {
        let fixture = Self {
            family,
            links,
            q,
            qd,
            tau,
            gravity,
        };
        fixture.validate()?;
        Ok(fixture)
    }

    pub fn validate(&self) -> Result<(), FixtureError> {
        if self.links.is_empty() {
            return Err(FixtureError::Empty);
        }
        if self.q.len() != self.links.len()
            || self.qd.len() != self.links.len()
            || self.tau.len() != self.links.len()
        {
            return Err(FixtureError::LengthMismatch);
        }
        if !self.gravity.into_iter().all(f32::is_finite)
            || !self.q.iter().all(|value| value.is_finite())
            || !self.qd.iter().all(|value| value.is_finite())
            || !self.tau.iter().all(|value| value.is_finite())
        {
            return Err(FixtureError::NonFinite);
        }
        for link in &self.links {
            if !link.is_finite() {
                return Err(FixtureError::NonFinite);
            }
            if !link.is_proper_rotation() {
                return Err(FixtureError::InvalidRotation);
            }
            if link.mass <= 0.0 {
                return Err(FixtureError::InvalidMass);
            }
            if link
                .inertia_com_diagonal
                .into_iter()
                .any(|value| value <= 0.0)
            {
                return Err(FixtureError::InvalidInertia);
            }
        }
        Ok(())
    }
}

impl LinkSpec {
    fn is_proper_rotation(self) -> bool {
        let rotation = self.tree_rotation;
        let mut orthogonality_error = 0.0_f32;
        for row in 0..3 {
            for column in 0..3 {
                let value = (0..3)
                    .map(|index| rotation[index][row] * rotation[index][column])
                    .sum::<f32>();
                let expected = if row == column { 1.0 } else { 0.0 };
                orthogonality_error = orthogonality_error.max((value - expected).abs());
            }
        }
        let determinant = rotation[0][0]
            * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]);
        orthogonality_error <= 1.0e-5 && (determinant - 1.0).abs() <= 1.0e-5
    }
}

pub const fn identity_rotation() -> [[f32; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}
