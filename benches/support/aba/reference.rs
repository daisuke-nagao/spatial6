// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use featherstone::prelude::aba_forward_dynamics;
use nalgebra_featherstone::{DMatrix, DVector};

use super::featherstone_adapter::featherstone_body;
use super::fixture::{Axis, CHAIN_LENGTHS, ChainFamily, Fixture, FixtureError};
use super::spatial6_solver::{
    AbaBenchError, Spatial6Model, Spatial6State, spatial6_aba_allocating,
};

type Vector3 = [f64; 3];
type Vector6 = [f64; 6];
type Matrix3 = [[f64; 3]; 3];
type Matrix6 = [[f64; 6]; 6];

#[derive(Clone, Copy)]
struct RawTransform {
    motion: Matrix6,
    force: Matrix6,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceResult {
    pub bias: Vec<f64>,
    pub mass_matrix: Vec<Vec<f64>>,
    pub qdd: Vec<f64>,
    pub symmetry_error: f64,
    pub residual: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReferenceError {
    pub phase: &'static str,
    pub joint: Option<usize>,
    pub message: &'static str,
}

#[derive(Debug)]
pub enum PreflightError {
    Fixture(FixtureError),
    Reference(ReferenceError),
    Spatial6(AbaBenchError),
    Dependency(&'static str),
    Threshold {
        family: ChainFamily,
        length: usize,
        check: &'static str,
        value: f64,
        limit: f64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreflightResult {
    pub family: ChainFamily,
    pub length: usize,
    pub reference_qdd: Vec<f64>,
    pub spatial6_qdd: Vec<f32>,
    pub featherstone_qdd: Vec<f32>,
    pub spatial6_error: f64,
    pub featherstone_error: f64,
    pub pairwise_error: f64,
    pub reference_residual: f64,
    pub spatial6_residual: f64,
    pub featherstone_residual: f64,
}

impl std::fmt::Display for PreflightError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixture(error) => write!(formatter, "invalid preflight fixture: {error}"),
            Self::Reference(error) => error.fmt(formatter),
            Self::Spatial6(error) => error.fmt(formatter),
            Self::Dependency(message) => formatter.write_str(message),
            Self::Threshold {
                family,
                length,
                check,
                value,
                limit,
            } => write!(
                formatter,
                "{family:?}/{length} {check}={value} exceeds {limit}"
            ),
        }
    }
}

impl std::error::Error for PreflightError {}

pub fn preflight_all() -> Result<Vec<PreflightResult>, PreflightError> {
    let mut results = Vec::with_capacity(2 * CHAIN_LENGTHS.len());
    for family in [ChainFamily::PlanarZ, ChainFamily::SpatialXyz] {
        for &length in &CHAIN_LENGTHS {
            results.push(preflight_case(family, length)?);
        }
    }
    Ok(results)
}

pub fn preflight_case(
    family: ChainFamily,
    length: usize,
) -> Result<PreflightResult, PreflightError> {
    let fixture = Fixture::canonical(family, length).map_err(PreflightError::Fixture)?;
    let model = Spatial6Model::<f32>::from_fixture(&fixture).map_err(PreflightError::Spatial6)?;
    let mut state =
        Spatial6State::<f32>::from_fixture(&fixture).map_err(PreflightError::Fixture)?;
    let spatial6_qdd = spatial6_aba_allocating(&model, &mut state)
        .map_err(PreflightError::Spatial6)?
        .to_vec();

    let mut body = featherstone_body(&fixture).map_err(PreflightError::Fixture)?;
    let featherstone_qdd = aba_forward_dynamics(&mut body).as_slice().to_vec();
    if featherstone_qdd.len() != length || !featherstone_qdd.iter().all(|value| value.is_finite()) {
        return Err(PreflightError::Dependency(
            "featherstone ABA returned an invalid output",
        ));
    }

    let reference = solve_reference(&fixture).map_err(PreflightError::Reference)?;
    let spatial6_as_f64 = spatial6_qdd
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let featherstone_as_f64 = featherstone_qdd
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let reference_norm = norm_inf(&reference.qdd);
    let output_limit = 2.0e-3 + 2.0e-3 * reference_norm;
    let spatial6_error = difference_norm(&spatial6_as_f64, &reference.qdd);
    let featherstone_error = difference_norm(&featherstone_as_f64, &reference.qdd);
    let pairwise_error = difference_norm(&spatial6_as_f64, &featherstone_as_f64);
    check_limit(
        family,
        length,
        "spatial6/reference",
        spatial6_error,
        output_limit,
    )?;
    check_limit(
        family,
        length,
        "featherstone/reference",
        featherstone_error,
        output_limit,
    )?;
    check_limit(
        family,
        length,
        "spatial6/featherstone",
        pairwise_error,
        output_limit,
    )?;

    let spatial6_residual = equation_residual(
        &reference.mass_matrix,
        &spatial6_as_f64,
        &reference.bias,
        &fixture.tau,
    );
    let featherstone_residual = equation_residual(
        &reference.mass_matrix,
        &featherstone_as_f64,
        &reference.bias,
        &fixture.tau,
    );
    check_limit(
        family,
        length,
        "reference residual",
        reference.residual,
        2.0e-4,
    )?;
    check_limit(
        family,
        length,
        "spatial6 residual",
        spatial6_residual,
        2.0e-4,
    )?;
    check_limit(
        family,
        length,
        "featherstone residual",
        featherstone_residual,
        2.0e-4,
    )?;

    Ok(PreflightResult {
        family,
        length,
        reference_qdd: reference.qdd,
        spatial6_qdd,
        featherstone_qdd,
        spatial6_error,
        featherstone_error,
        pairwise_error,
        reference_residual: reference.residual,
        spatial6_residual,
        featherstone_residual,
    })
}

fn difference_norm(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn check_limit(
    family: ChainFamily,
    length: usize,
    check: &'static str,
    value: f64,
    limit: f64,
) -> Result<(), PreflightError> {
    if !value.is_finite() || value > limit {
        return Err(PreflightError::Threshold {
            family,
            length,
            check,
            value,
            limit,
        });
    }
    Ok(())
}

impl std::fmt::Display for ReferenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.joint {
            Some(joint) => write!(
                formatter,
                "reference {} failed at joint {}: {}",
                self.phase, joint, self.message
            ),
            None => write!(
                formatter,
                "reference {} failed: {}",
                self.phase, self.message
            ),
        }
    }
}

impl std::error::Error for ReferenceError {}

pub fn solve_reference(fixture: &Fixture) -> Result<ReferenceResult, ReferenceError> {
    fixture.validate().map_err(|_| ReferenceError {
        phase: "fixture",
        joint: None,
        message: "invalid fixture",
    })?;
    let n = fixture.links.len();
    let zero = vec![0.0; n];
    let q = fixture
        .q
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let qd = fixture
        .qd
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let gravity = fixture.gravity.map(f64::from);
    let bias = rnea_raw(fixture, &q, &qd, &zero, gravity)?;

    let mut mass_matrix = vec![vec![0.0; n]; n];
    for column in 0..n {
        let mut qdd = vec![0.0; n];
        qdd[column] = 1.0;
        let values = rnea_raw(fixture, &q, &zero, &qdd, [0.0; 3])?;
        for row in 0..n {
            mass_matrix[row][column] = values[row];
        }
    }

    let symmetry_error = relative_symmetry_error(&mass_matrix);
    if !symmetry_error.is_finite() || symmetry_error > 1.0e-12 {
        return Err(ReferenceError {
            phase: "mass-matrix symmetry",
            joint: None,
            message: "mass matrix is not symmetric",
        });
    }
    for row in 0..n {
        for column in (row + 1)..n {
            let value = (mass_matrix[row][column] + mass_matrix[column][row]) * 0.5;
            mass_matrix[row][column] = value;
            mass_matrix[column][row] = value;
        }
    }

    let matrix_data: Vec<f64> = mass_matrix.iter().flatten().copied().collect();
    let h = DMatrix::from_row_slice(n, n, &matrix_data);
    let rhs = DVector::from_iterator(
        n,
        fixture
            .tau
            .iter()
            .map(|value| f64::from(*value))
            .zip(&bias)
            .map(|(tau, bias)| tau - bias),
    );
    let qdd = h
        .cholesky()
        .ok_or(ReferenceError {
            phase: "factorization",
            joint: None,
            message: "mass matrix is not positive definite",
        })?
        .solve(&rhs);
    let qdd = qdd.as_slice().to_vec();
    if !qdd.iter().all(|value| value.is_finite()) {
        return Err(ReferenceError {
            phase: "factorization",
            joint: None,
            message: "solution is not finite",
        });
    }
    let residual = equation_residual(&mass_matrix, &qdd, &bias, &fixture.tau);
    if !residual.is_finite() {
        return Err(ReferenceError {
            phase: "residual",
            joint: None,
            message: "residual is not finite",
        });
    }

    Ok(ReferenceResult {
        bias,
        mass_matrix,
        qdd,
        symmetry_error,
        residual,
    })
}

pub fn rnea_f64(
    fixture: &Fixture,
    q: &[f32],
    qd: &[f32],
    qdd: &[f32],
    gravity: [f32; 3],
) -> Result<Vec<f64>, ReferenceError> {
    if q.len() != fixture.links.len()
        || qd.len() != fixture.links.len()
        || qdd.len() != fixture.links.len()
    {
        return Err(ReferenceError {
            phase: "input",
            joint: None,
            message: "state length does not match fixture",
        });
    }
    if !q.iter().chain(qd).chain(qdd).all(|value| value.is_finite())
        || !gravity.into_iter().all(f32::is_finite)
    {
        return Err(ReferenceError {
            phase: "input",
            joint: None,
            message: "input contains a non-finite value",
        });
    }
    rnea_raw(
        fixture,
        &q.iter().map(|value| f64::from(*value)).collect::<Vec<_>>(),
        &qd.iter().map(|value| f64::from(*value)).collect::<Vec<_>>(),
        &qdd.iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>(),
        gravity.map(f64::from),
    )
}

pub fn equation_residual(mass_matrix: &[Vec<f64>], qdd: &[f64], bias: &[f64], tau: &[f32]) -> f64 {
    let mut residual = vec![0.0; qdd.len()];
    for row in 0..qdd.len() {
        residual[row] = mass_matrix[row]
            .iter()
            .zip(qdd)
            .map(|(coefficient, value)| coefficient * value)
            .sum::<f64>()
            + bias[row]
            - f64::from(tau[row]);
    }
    norm_inf(&residual)
        / (matrix_norm_inf(mass_matrix) * norm_inf(qdd) + norm_inf(bias) + norm_inf_f32(tau))
            .max(1.0e-12)
}

pub fn norm_inf(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

pub fn matrix_norm_inf(matrix: &[Vec<f64>]) -> f64 {
    matrix
        .iter()
        .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn norm_inf_f32(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| f64::from(*value).abs())
        .fold(0.0, f64::max)
}

fn relative_symmetry_error(matrix: &[Vec<f64>]) -> f64 {
    let difference = matrix
        .iter()
        .enumerate()
        .map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .map(|(column, value)| (value - matrix[column][row]).abs())
                .sum::<f64>()
        })
        .fold(0.0, f64::max);
    difference / matrix_norm_inf(matrix).max(1.0)
}

fn rnea_raw(
    fixture: &Fixture,
    q: &[f64],
    qd: &[f64],
    qdd: &[f64],
    gravity: Vector3,
) -> Result<Vec<f64>, ReferenceError> {
    let n = fixture.links.len();
    let transforms: Vec<RawTransform> = fixture
        .links
        .iter()
        .enumerate()
        .map(|(index, link)| raw_transform(link, q[index]))
        .collect();
    let inertias: Vec<Matrix6> = fixture.links.iter().map(raw_inertia).collect();
    let subspaces: Vec<Vector6> = fixture
        .links
        .iter()
        .map(|link| vector6_from_axis(link.axis))
        .collect();

    let mut velocity = vec![[0.0; 6]; n];
    let mut acceleration = vec![[0.0; 6]; n];
    let mut force = vec![[0.0; 6]; n];
    let base_acceleration = [0.0, 0.0, 0.0, -gravity[0], -gravity[1], -gravity[2]];

    for index in 0..n {
        let joint_velocity = scale_vector(subspaces[index], qd[index]);
        let parent_velocity = if index == 0 {
            [0.0; 6]
        } else {
            velocity[index - 1]
        };
        velocity[index] = add_vector(
            matrix_vector_mul(&transforms[index].motion, &parent_velocity),
            joint_velocity,
        );
        let parent_acceleration = if index == 0 {
            base_acceleration
        } else {
            acceleration[index - 1]
        };
        acceleration[index] = add_vector(
            add_vector(
                matrix_vector_mul(&transforms[index].motion, &parent_acceleration),
                scale_vector(subspaces[index], qdd[index]),
            ),
            matrix_vector_mul(&motion_cross_matrix(velocity[index]), &joint_velocity),
        );
        let inertia_velocity = matrix_vector_mul(&inertias[index], &velocity[index]);
        force[index] = add_vector(
            matrix_vector_mul(&inertias[index], &acceleration[index]),
            matrix_vector_mul(&force_cross_matrix(velocity[index]), &inertia_velocity),
        );
        if !velocity[index]
            .iter()
            .chain(acceleration[index].iter())
            .chain(force[index].iter())
            .all(|value| value.is_finite())
        {
            return Err(ReferenceError {
                phase: "rnea",
                joint: Some(index),
                message: "non-finite intermediate",
            });
        }
    }

    let mut tau = vec![0.0; n];
    for index in (0..n).rev() {
        tau[index] = dot_vector(subspaces[index], force[index]);
        if index > 0 {
            force[index - 1] = add_vector(
                force[index - 1],
                matrix_vector_mul(&transpose(&transforms[index].motion), &force[index]),
            );
        }
        if !tau[index].is_finite() || !force[index].iter().all(|value| value.is_finite()) {
            return Err(ReferenceError {
                phase: "rnea",
                joint: Some(index),
                message: "non-finite backward pass",
            });
        }
    }
    Ok(tau)
}

fn raw_transform(link: &super::fixture::LinkSpec, q: f64) -> RawTransform {
    let tree = transform_matrix(
        cast_matrix3(link.tree_rotation),
        link.parent_offset.map(f64::from),
    );
    let joint = transform_matrix(rotation_matrix(link.axis, q), [0.0; 3]);
    RawTransform {
        motion: matrix_mul(&joint.motion, &tree.motion),
        force: matrix_mul(&joint.force, &tree.force),
    }
}

fn transform_matrix(rotation: Matrix3, translation: Vector3) -> RawTransform {
    let cross = skew(translation);
    let neg_rot_cross = matrix3_scale(&matrix3_mul(&rotation, &cross), -1.0);
    RawTransform {
        motion: blocks(rotation, zero_matrix3(), neg_rot_cross, rotation),
        force: blocks(rotation, neg_rot_cross, zero_matrix3(), rotation),
    }
}

fn raw_inertia(link: &super::fixture::LinkSpec) -> Matrix6 {
    let mass = f64::from(link.mass);
    let center = link.com.map(f64::from);
    let inertia_com = diagonal(link.inertia_com_diagonal.map(f64::from));
    let cross = skew(center);
    let mass_cross = matrix3_scale(&cross, mass);
    let upper_left = matrix3_sub(&inertia_com, &matrix3_mul(&mass_cross, &cross));
    blocks(
        upper_left,
        mass_cross,
        matrix3_scale(&mass_cross, -1.0),
        matrix3_scale(&identity_matrix3(), mass),
    )
}

fn vector6_from_axis(axis: Axis) -> Vector6 {
    let mut value = [0.0; 6];
    let axis = axis.as_array();
    value[..3].copy_from_slice(&axis.map(f64::from));
    value
}

fn rotation_matrix(axis: Axis, angle: f64) -> Matrix3 {
    let (sin, cos) = angle.sin_cos();
    match axis {
        Axis::X => [[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]],
        Axis::Y => [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]],
        Axis::Z => [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
    }
}

fn motion_cross_matrix(value: Vector6) -> Matrix6 {
    let angular = skew([value[0], value[1], value[2]]);
    let linear = skew([value[3], value[4], value[5]]);
    blocks(angular, zero_matrix3(), linear, angular)
}

fn force_cross_matrix(value: Vector6) -> Matrix6 {
    let angular = skew([value[0], value[1], value[2]]);
    let linear = skew([value[3], value[4], value[5]]);
    blocks(angular, linear, zero_matrix3(), angular)
}

fn blocks(
    upper_left: Matrix3,
    upper_right: Matrix3,
    lower_left: Matrix3,
    lower_right: Matrix3,
) -> Matrix6 {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            if row < 3 && column < 3 {
                upper_left[row][column]
            } else if row < 3 {
                upper_right[row][column - 3]
            } else if column < 3 {
                lower_left[row - 3][column]
            } else {
                lower_right[row - 3][column - 3]
            }
        })
    })
}

fn diagonal(value: Vector3) -> Matrix3 {
    [
        [value[0], 0.0, 0.0],
        [0.0, value[1], 0.0],
        [0.0, 0.0, value[2]],
    ]
}

fn identity_matrix3() -> Matrix3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn zero_matrix3() -> Matrix3 {
    [[0.0; 3]; 3]
}

fn skew(value: Vector3) -> Matrix3 {
    [
        [0.0, -value[2], value[1]],
        [value[2], 0.0, -value[0]],
        [-value[1], value[0], 0.0],
    ]
}

fn matrix_vector_mul(matrix: &Matrix6, vector: &Vector6) -> Vector6 {
    std::array::from_fn(|row| {
        matrix[row]
            .iter()
            .zip(vector)
            .map(|(left, right)| left * right)
            .sum()
    })
}

fn matrix_mul(left: &Matrix6, right: &Matrix6) -> Matrix6 {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..6)
                .map(|index| left[row][index] * right[index][column])
                .sum()
        })
    })
}

fn matrix3_scale(matrix: &Matrix3, scalar: f64) -> Matrix3 {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[row][column] * scalar))
}

fn matrix3_sub(left: &Matrix3, right: &Matrix3) -> Matrix3 {
    std::array::from_fn(|row| std::array::from_fn(|column| left[row][column] - right[row][column]))
}

fn matrix3_mul(left: &Matrix3, right: &Matrix3) -> Matrix3 {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|index| left[row][index] * right[index][column])
                .sum()
        })
    })
}

fn transpose(matrix: &Matrix6) -> Matrix6 {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row]))
}

fn add_vector(left: Vector6, right: Vector6) -> Vector6 {
    std::array::from_fn(|index| left[index] + right[index])
}

fn scale_vector(value: Vector6, scalar: f64) -> Vector6 {
    value.map(|component| component * scalar)
}

fn dot_vector(left: Vector6, right: Vector6) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn cast_matrix3(value: [[f32; 3]; 3]) -> Matrix3 {
    value.map(|row| row.map(f64::from))
}
