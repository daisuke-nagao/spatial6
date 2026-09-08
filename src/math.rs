// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::array;

use crate::SpatialScalar;

pub(crate) fn vector_zero<T: SpatialScalar, const N: usize>() -> [T; N] {
    [T::zero(); N]
}

pub(crate) fn vector_add<T: SpatialScalar, const N: usize>(left: [T; N], right: [T; N]) -> [T; N] {
    array::from_fn(|index| left[index] + right[index])
}

pub(crate) fn vector_sub<T: SpatialScalar, const N: usize>(left: [T; N], right: [T; N]) -> [T; N] {
    array::from_fn(|index| left[index] - right[index])
}

pub(crate) fn vector_neg<T: SpatialScalar, const N: usize>(value: [T; N]) -> [T; N] {
    array::from_fn(|index| -value[index])
}

pub(crate) fn vector_scale<T: SpatialScalar, const N: usize>(value: [T; N], scalar: T) -> [T; N] {
    array::from_fn(|index| value[index] * scalar)
}

pub(crate) fn dot<T: SpatialScalar, const N: usize>(left: [T; N], right: [T; N]) -> T {
    (0..N).fold(T::zero(), |sum, index| sum + left[index] * right[index])
}

pub(crate) fn cross<T: SpatialScalar>(left: [T; 3], right: [T; 3]) -> [T; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

pub(crate) fn skew<T: SpatialScalar>(vector: [T; 3]) -> [[T; 3]; 3] {
    [
        [T::zero(), -vector[2], vector[1]],
        [vector[2], T::zero(), -vector[0]],
        [-vector[1], vector[0], T::zero()],
    ]
}

pub(crate) fn matrix_zero<T: SpatialScalar, const N: usize>() -> [[T; N]; N] {
    [[T::zero(); N]; N]
}

pub(crate) fn matrix_identity<T: SpatialScalar, const N: usize>() -> [[T; N]; N] {
    array::from_fn(|row| array::from_fn(|column| if row == column { T::one() } else { T::zero() }))
}

pub(crate) fn matrix_add<T: SpatialScalar, const N: usize>(
    left: [[T; N]; N],
    right: [[T; N]; N],
) -> [[T; N]; N] {
    array::from_fn(|row| array::from_fn(|column| left[row][column] + right[row][column]))
}

pub(crate) fn matrix_sub<T: SpatialScalar, const N: usize>(
    left: [[T; N]; N],
    right: [[T; N]; N],
) -> [[T; N]; N] {
    array::from_fn(|row| array::from_fn(|column| left[row][column] - right[row][column]))
}

pub(crate) fn matrix_neg<T: SpatialScalar, const N: usize>(value: [[T; N]; N]) -> [[T; N]; N] {
    array::from_fn(|row| array::from_fn(|column| -value[row][column]))
}

pub(crate) fn matrix_scale<T: SpatialScalar, const N: usize>(
    value: [[T; N]; N],
    scalar: T,
) -> [[T; N]; N] {
    array::from_fn(|row| array::from_fn(|column| value[row][column] * scalar))
}

pub(crate) fn matrix_transpose<T: SpatialScalar, const N: usize>(
    value: [[T; N]; N],
) -> [[T; N]; N] {
    array::from_fn(|row| array::from_fn(|column| value[column][row]))
}

pub(crate) fn matrix_mul<T: SpatialScalar, const N: usize>(
    left: [[T; N]; N],
    right: [[T; N]; N],
) -> [[T; N]; N] {
    array::from_fn(|row| {
        array::from_fn(|column| {
            (0..N).fold(T::zero(), |sum, index| {
                sum + left[row][index] * right[index][column]
            })
        })
    })
}

pub(crate) fn matrix_vector_mul<T: SpatialScalar, const N: usize>(
    matrix: [[T; N]; N],
    vector: [T; N],
) -> [T; N] {
    array::from_fn(|row| {
        (0..N).fold(T::zero(), |sum, column| {
            sum + matrix[row][column] * vector[column]
        })
    })
}

pub(crate) fn vector_is_finite<T: SpatialScalar, const N: usize>(value: &[T; N]) -> bool {
    value.iter().all(|component| component.is_finite())
}

pub(crate) fn matrix_is_finite<T: SpatialScalar, const N: usize>(value: &[[T; N]; N]) -> bool {
    value
        .iter()
        .flatten()
        .all(|component| component.is_finite())
}

/// Returns the lower Cholesky factor `L` of `matrix` such that `L * Lᵀ == matrix`.
///
/// `matrix` is assumed symmetric; only its lower triangle is read. Returns
/// `None` if `matrix` is not finite or is not positive definite (a
/// non-positive value would appear on the diagonal of `L`).
#[allow(clippy::needless_range_loop)]
fn cholesky<T: SpatialScalar, const N: usize>(matrix: [[T; N]; N]) -> Option<[[T; N]; N]> {
    if !matrix_is_finite(&matrix) {
        return None;
    }

    let mut lower = matrix_zero();
    for row in 0..N {
        for column in 0..=row {
            let mut value = matrix[row][column];
            for index in 0..column {
                value = value - lower[row][index] * lower[column][index];
            }

            if row == column {
                if value <= T::zero() {
                    return None;
                }
                lower[row][column] = value.sqrt();
            } else {
                lower[row][column] = value / lower[column][column];
            }

            if !lower[row][column].is_finite() {
                return None;
            }
        }
    }
    Some(lower)
}

/// Returns whether `matrix` (assumed symmetric) is finite and positive definite.
pub(crate) fn is_positive_definite<T: SpatialScalar, const N: usize>(matrix: [[T; N]; N]) -> bool {
    cholesky(matrix).is_some()
}

/// Solves the symmetric positive-definite system `matrix * x == right` for `x`.
///
/// `matrix` is assumed symmetric. Returns `None` if `right` is not finite, if
/// `matrix` is not finite or not positive definite, or if the computed
/// solution is not finite.
#[allow(clippy::needless_range_loop)]
pub(crate) fn solve_positive_definite<T: SpatialScalar, const N: usize>(
    matrix: [[T; N]; N],
    right: [T; N],
) -> Option<[T; N]> {
    if !vector_is_finite(&right) {
        return None;
    }

    let lower = cholesky(matrix)?;
    let mut intermediate: [T; N] = vector_zero();
    for row in 0..N {
        let mut value = right[row];
        for column in 0..row {
            value = value - lower[row][column] * intermediate[column];
        }
        intermediate[row] = value / lower[row][row];
    }

    let mut solution: [T; N] = vector_zero();
    for row in (0..N).rev() {
        let mut value = intermediate[row];
        for column in (row + 1)..N {
            value = value - lower[column][row] * solution[column];
        }
        solution[row] = value / lower[row][row];
    }

    vector_is_finite(&solution).then_some(solution)
}
