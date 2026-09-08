// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::Debug;

/// A scalar type supported by spatial vector and matrix operations.
///
/// [`num_traits::Float`] supplies the numeric operations required by the crate.
/// Any suitable float type is supported through the blanket implementation.
pub trait SpatialScalar: num_traits::Float + Debug + Send + Sync + 'static {}

impl<T> SpatialScalar for T where T: num_traits::Float + Debug + Send + Sync + 'static {}

/// Maps semantic spatial arrays to backend-specific storage.
///
/// Nested matrix and rotation arrays use row-major semantic order: an element
/// at `[row][column]` is the value at that row and column. Every conversion
/// must preserve this semantic row/column order regardless of the backend's
/// native storage order. `Rotation3` values must represent proper orthogonal
/// rotations; callers and implementers uphold this invariant, and conversions
/// do not validate it.
///
/// # Defining your own representation
///
/// To use your own `Vector6` (or any other backend type) instead of
/// `Builtin`, `Nalgebra`, or `Glam`, implement this trait for a marker
/// type and pass it as the second type parameter, e.g. `MotionVector<f64,
/// MyRepresentation>`. Only the ten `_from_array`/`_to_array` conversions are
/// required; every arithmetic method (`vector3_add`, `matrix6_mul`, and so
/// on) has a default implementation built on top of them, so a minimal
/// representation needs no more than the conversions themselves. Override a
/// default only where your backend has a faster native operation.
///
/// ```
/// use spatial6::{MotionVector, SpatialRepresentation, SpatialScalar};
///
/// #[derive(Copy, Clone, Debug, PartialEq)]
/// struct MyRepresentation;
///
/// impl<T: SpatialScalar> SpatialRepresentation<T> for MyRepresentation {
///     type Vector3 = [T; 3];
///     type Vector6 = [T; 6];
///     type Matrix3 = [[T; 3]; 3];
///     type Matrix6 = [[T; 6]; 6];
///     type Rotation3 = [[T; 3]; 3];
///
///     fn vector3_from_array(value: [T; 3]) -> Self::Vector3 { value }
///     fn vector3_to_array(value: &Self::Vector3) -> [T; 3] { *value }
///     fn vector6_from_array(value: [T; 6]) -> Self::Vector6 { value }
///     fn vector6_to_array(value: &Self::Vector6) -> [T; 6] { *value }
///     fn matrix3_from_array(value: [[T; 3]; 3]) -> Self::Matrix3 { value }
///     fn matrix3_to_array(value: &Self::Matrix3) -> [[T; 3]; 3] { *value }
///     fn matrix6_from_array(value: [[T; 6]; 6]) -> Self::Matrix6 { value }
///     fn matrix6_to_array(value: &Self::Matrix6) -> [[T; 6]; 6] { *value }
///     fn rotation3_from_array(value: [[T; 3]; 3]) -> Self::Rotation3 { value }
///     fn rotation3_to_array(value: &Self::Rotation3) -> [[T; 3]; 3] { *value }
/// }
///
/// let motion = MotionVector::<f64, MyRepresentation>::zeros();
/// assert!(motion.is_finite());
/// ```
///
/// # Overriding defaults
///
/// Every other method on this trait — everything below the ten conversions,
/// listed as "Provided Methods" further down this page — already has a
/// default body expressed in terms of those conversions. Overriding one is
/// ordinary trait syntax: add a method with the same name and signature to
/// your `impl` block, and it replaces the default for that representation.
/// There is no separate opt-in and no marker to flag; the compiler picks
/// whichever definition is present. Reach for this when your backend has a
/// direct operation that skips the round trip through arrays that the
/// default takes, as `Nalgebra` and `Glam` do throughout their own
/// implementations of this trait.
///
/// ```
/// use spatial6::{SpatialRepresentation, SpatialScalar};
///
/// #[derive(Copy, Clone, Debug, PartialEq)]
/// struct MyRepresentation;
///
/// impl<T: SpatialScalar> SpatialRepresentation<T> for MyRepresentation {
///     type Vector3 = [T; 3];
///     type Vector6 = [T; 6];
///     type Matrix3 = [[T; 3]; 3];
///     type Matrix6 = [[T; 6]; 6];
///     type Rotation3 = [[T; 3]; 3];
///
///     fn vector3_from_array(value: [T; 3]) -> Self::Vector3 { value }
///     fn vector3_to_array(value: &Self::Vector3) -> [T; 3] { *value }
///     fn vector6_from_array(value: [T; 6]) -> Self::Vector6 { value }
///     fn vector6_to_array(value: &Self::Vector6) -> [T; 6] { *value }
///     fn matrix3_from_array(value: [[T; 3]; 3]) -> Self::Matrix3 { value }
///     fn matrix3_to_array(value: &Self::Matrix3) -> [[T; 3]; 3] { *value }
///     fn matrix6_from_array(value: [[T; 6]; 6]) -> Self::Matrix6 { value }
///     fn matrix6_to_array(value: &Self::Matrix6) -> [[T; 6]; 6] { *value }
///     fn rotation3_from_array(value: [[T; 3]; 3]) -> Self::Rotation3 { value }
///     fn rotation3_to_array(value: &Self::Rotation3) -> [[T; 3]; 3] { *value }
///
///     // Overrides the default, which round-trips through
///     // `vector3_to_array`/`vector3_from_array` and `crate::math::vector_add`.
///     fn vector3_add(left: &Self::Vector3, right: &Self::Vector3) -> Self::Vector3 {
///         std::array::from_fn(|index| left[index] + right[index])
///     }
/// }
///
/// let sum = MyRepresentation::vector3_add(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
/// assert_eq!(sum, [5.0, 7.0, 9.0]);
/// ```
pub trait SpatialRepresentation<T: SpatialScalar>:
    Copy + Clone + Debug + PartialEq + Send + Sync + 'static
{
    /// Backend storage for a three-component vector.
    type Vector3: Copy + Clone + Debug + PartialEq + Send + Sync + 'static;
    /// Backend storage for a six-component vector.
    type Vector6: Copy + Clone + Debug + PartialEq + Send + Sync + 'static;
    /// Backend storage corresponding to a semantic row-major 3x3 matrix.
    type Matrix3: Copy + Clone + Debug + PartialEq + Send + Sync + 'static;
    /// Backend storage corresponding to a semantic row-major 6x6 matrix.
    type Matrix6: Copy + Clone + Debug + PartialEq + Send + Sync + 'static;
    /// Backend storage for a proper orthogonal 3D rotation.
    type Rotation3: Copy + Clone + Debug + PartialEq + Send + Sync + 'static;

    /// Converts a three-component array into backend storage.
    fn vector3_from_array(value: [T; 3]) -> Self::Vector3;
    /// Converts backend storage into a three-component array.
    fn vector3_to_array(value: &Self::Vector3) -> [T; 3];
    /// Converts a six-component array into backend storage.
    fn vector6_from_array(value: [T; 6]) -> Self::Vector6;
    /// Converts backend storage into a six-component array.
    fn vector6_to_array(value: &Self::Vector6) -> [T; 6];
    /// Converts a semantic row-major 3x3 array into backend storage.
    fn matrix3_from_array(value: [[T; 3]; 3]) -> Self::Matrix3;
    /// Converts backend storage into a semantic row-major 3x3 array.
    fn matrix3_to_array(value: &Self::Matrix3) -> [[T; 3]; 3];
    /// Converts a semantic row-major 6x6 array into backend storage.
    fn matrix6_from_array(value: [[T; 6]; 6]) -> Self::Matrix6;
    /// Converts backend storage into a semantic row-major 6x6 array.
    fn matrix6_to_array(value: &Self::Matrix6) -> [[T; 6]; 6];
    /// Converts a semantic row-major rotation array into backend storage.
    fn rotation3_from_array(value: [[T; 3]; 3]) -> Self::Rotation3;
    /// Converts backend rotation storage into a semantic row-major array.
    fn rotation3_to_array(value: &Self::Rotation3) -> [[T; 3]; 3];

    /// Returns a zero three-component vector.
    fn vector3_zero() -> Self::Vector3 {
        Self::vector3_from_array(crate::math::vector_zero())
    }

    /// Adds two three-component vectors.
    fn vector3_add(left: &Self::Vector3, right: &Self::Vector3) -> Self::Vector3 {
        Self::vector3_from_array(crate::math::vector_add(
            Self::vector3_to_array(left),
            Self::vector3_to_array(right),
        ))
    }

    /// Subtracts two three-component vectors.
    fn vector3_sub(left: &Self::Vector3, right: &Self::Vector3) -> Self::Vector3 {
        Self::vector3_from_array(crate::math::vector_sub(
            Self::vector3_to_array(left),
            Self::vector3_to_array(right),
        ))
    }

    /// Negates a three-component vector.
    fn vector3_neg(value: &Self::Vector3) -> Self::Vector3 {
        Self::vector3_from_array(crate::math::vector_neg(Self::vector3_to_array(value)))
    }

    /// Scales a three-component vector.
    fn vector3_scale(value: &Self::Vector3, scalar: T) -> Self::Vector3 {
        Self::vector3_from_array(crate::math::vector_scale(
            Self::vector3_to_array(value),
            scalar,
        ))
    }

    /// Computes the three-component dot product.
    fn vector3_dot(left: &Self::Vector3, right: &Self::Vector3) -> T {
        crate::math::dot(Self::vector3_to_array(left), Self::vector3_to_array(right))
    }

    /// Computes the three-component cross product.
    fn vector3_cross(left: &Self::Vector3, right: &Self::Vector3) -> Self::Vector3 {
        Self::vector3_from_array(crate::math::cross(
            Self::vector3_to_array(left),
            Self::vector3_to_array(right),
        ))
    }

    /// Returns whether all vector components are finite.
    fn vector3_is_finite(value: &Self::Vector3) -> bool {
        crate::math::vector_is_finite(&Self::vector3_to_array(value))
    }

    /// Returns the skew-symmetric matrix for a three-component vector.
    fn vector3_cross_matrix(value: &Self::Vector3) -> Self::Matrix3 {
        Self::matrix3_from_array(crate::math::skew(Self::vector3_to_array(value)))
    }

    /// Returns a zero 3x3 matrix.
    fn matrix3_zero() -> Self::Matrix3 {
        Self::matrix3_from_array(crate::math::matrix_zero())
    }

    /// Returns a 3x3 identity matrix.
    fn matrix3_identity() -> Self::Matrix3 {
        Self::matrix3_from_array(crate::math::matrix_identity())
    }

    /// Adds two 3x3 matrices.
    fn matrix3_add(left: &Self::Matrix3, right: &Self::Matrix3) -> Self::Matrix3 {
        Self::matrix3_from_array(crate::math::matrix_add(
            Self::matrix3_to_array(left),
            Self::matrix3_to_array(right),
        ))
    }

    /// Subtracts two 3x3 matrices.
    fn matrix3_sub(left: &Self::Matrix3, right: &Self::Matrix3) -> Self::Matrix3 {
        Self::matrix3_from_array(crate::math::matrix_sub(
            Self::matrix3_to_array(left),
            Self::matrix3_to_array(right),
        ))
    }

    /// Scales a 3x3 matrix.
    fn matrix3_scale(value: &Self::Matrix3, scalar: T) -> Self::Matrix3 {
        Self::matrix3_from_array(crate::math::matrix_scale(
            Self::matrix3_to_array(value),
            scalar,
        ))
    }

    /// Multiplies two 3x3 matrices.
    fn matrix3_mul(left: &Self::Matrix3, right: &Self::Matrix3) -> Self::Matrix3 {
        Self::matrix3_from_array(crate::math::matrix_mul(
            Self::matrix3_to_array(left),
            Self::matrix3_to_array(right),
        ))
    }

    /// Multiplies a 3x3 matrix by a three-component vector.
    fn matrix3_vector_mul(matrix: &Self::Matrix3, vector: &Self::Vector3) -> Self::Vector3 {
        Self::vector3_from_array(crate::math::matrix_vector_mul(
            Self::matrix3_to_array(matrix),
            Self::vector3_to_array(vector),
        ))
    }

    /// Returns whether all 3x3 matrix components are finite.
    fn matrix3_is_finite(value: &Self::Matrix3) -> bool {
        crate::math::matrix_is_finite(&Self::matrix3_to_array(value))
    }

    /// Returns whether a 3x3 matrix is finite and positive definite.
    ///
    /// `value` is assumed symmetric. Returns `false` for non-finite input as
    /// well as for input that is finite but not positive definite.
    fn matrix3_is_positive_definite(value: &Self::Matrix3) -> bool {
        crate::math::is_positive_definite(Self::matrix3_to_array(value))
    }

    /// Returns the identity rotation.
    fn rotation3_identity() -> Self::Rotation3 {
        Self::rotation3_from_array(crate::math::matrix_identity())
    }

    /// Returns the inverse rotation.
    ///
    /// Computed as the transpose, which equals the inverse precisely because
    /// `Rotation3` values are proper orthogonal rotations, per the invariant
    /// on [`SpatialRepresentation::Rotation3`].
    fn rotation3_inverse(value: &Self::Rotation3) -> Self::Rotation3 {
        Self::rotation3_from_array(crate::math::matrix_transpose(Self::rotation3_to_array(
            value,
        )))
    }

    /// Composes two rotations in left-to-right matrix order.
    fn rotation3_mul(left: &Self::Rotation3, right: &Self::Rotation3) -> Self::Rotation3 {
        Self::rotation3_from_array(crate::math::matrix_mul(
            Self::rotation3_to_array(left),
            Self::rotation3_to_array(right),
        ))
    }

    /// Applies a rotation to a three-component vector.
    fn rotation3_transform_vector(
        rotation: &Self::Rotation3,
        vector: &Self::Vector3,
    ) -> Self::Vector3 {
        Self::vector3_from_array(crate::math::matrix_vector_mul(
            Self::rotation3_to_array(rotation),
            Self::vector3_to_array(vector),
        ))
    }

    /// Converts a rotation to native 3x3 matrix storage.
    fn rotation3_to_matrix3(value: &Self::Rotation3) -> Self::Matrix3 {
        Self::matrix3_from_array(Self::rotation3_to_array(value))
    }

    /// Assembles a 6x6 matrix from row-major 3x3 blocks.
    fn matrix6_from_blocks(
        upper_left: &Self::Matrix3,
        upper_right: &Self::Matrix3,
        lower_left: &Self::Matrix3,
        lower_right: &Self::Matrix3,
    ) -> Self::Matrix6 {
        let upper_left = Self::matrix3_to_array(upper_left);
        let upper_right = Self::matrix3_to_array(upper_right);
        let lower_left = Self::matrix3_to_array(lower_left);
        let lower_right = Self::matrix3_to_array(lower_right);
        let matrix = std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                let (block, row, column) = if row < 3 && column < 3 {
                    (&upper_left, row, column)
                } else if row < 3 {
                    (&upper_right, row, column - 3)
                } else if column < 3 {
                    (&lower_left, row - 3, column)
                } else {
                    (&lower_right, row - 3, column - 3)
                };
                block[row][column]
            })
        });
        Self::matrix6_from_array(matrix)
    }

    /// Subtracts two 6x6 matrices.
    fn matrix6_sub(left: &Self::Matrix6, right: &Self::Matrix6) -> Self::Matrix6 {
        Self::matrix6_from_array(crate::math::matrix_sub(
            Self::matrix6_to_array(left),
            Self::matrix6_to_array(right),
        ))
    }

    /// Negates a 6x6 matrix.
    fn matrix6_neg(value: &Self::Matrix6) -> Self::Matrix6 {
        Self::matrix6_from_array(crate::math::matrix_neg(Self::matrix6_to_array(value)))
    }

    /// Transposes a 6x6 matrix.
    fn matrix6_transpose(value: &Self::Matrix6) -> Self::Matrix6 {
        Self::matrix6_from_array(crate::math::matrix_transpose(Self::matrix6_to_array(value)))
    }

    /// Multiplies two 6x6 matrices.
    fn matrix6_mul(left: &Self::Matrix6, right: &Self::Matrix6) -> Self::Matrix6 {
        Self::matrix6_from_array(crate::math::matrix_mul(
            Self::matrix6_to_array(left),
            Self::matrix6_to_array(right),
        ))
    }

    /// Multiplies a 6x6 matrix by a six-component vector.
    fn matrix6_vector_mul(matrix: &Self::Matrix6, vector: &Self::Vector6) -> Self::Vector6 {
        Self::vector6_from_array(crate::math::matrix_vector_mul(
            Self::matrix6_to_array(matrix),
            Self::vector6_to_array(vector),
        ))
    }

    /// Solves the symmetric positive-definite system `matrix * x == right`
    /// for `x`.
    ///
    /// `matrix` is assumed symmetric. Returns `None` if `right` is not
    /// finite, if `matrix` is not finite or not positive definite, or if the
    /// computed solution is not finite.
    fn matrix6_solve_positive_definite(
        matrix: &Self::Matrix6,
        right: &Self::Vector6,
    ) -> Option<Self::Vector6> {
        let matrix = Self::matrix6_to_array(matrix);
        let right = Self::vector6_to_array(right);
        crate::math::solve_positive_definite(matrix, right).map(Self::vector6_from_array)
    }
}

#[cfg(feature = "builtin")]
#[derive(Copy, Clone, Debug, PartialEq)]
/// Uses fixed-size Rust arrays for all vector, matrix, and rotation
/// storage: `[T; 3]` for `Vector3`, `[T; 6]` for `Vector6`, `[[T; 3]; 3]`
/// for `Matrix3` and `Rotation3`, and `[[T; 6]; 6]` for `Matrix6`.
pub struct Builtin;

#[cfg(feature = "builtin")]
impl<T: SpatialScalar> SpatialRepresentation<T> for Builtin {
    type Vector3 = [T; 3];
    type Vector6 = [T; 6];
    type Matrix3 = [[T; 3]; 3];
    type Matrix6 = [[T; 6]; 6];
    type Rotation3 = [[T; 3]; 3];

    fn vector3_from_array(value: [T; 3]) -> Self::Vector3 {
        value
    }

    fn vector3_to_array(value: &Self::Vector3) -> [T; 3] {
        *value
    }

    fn vector6_from_array(value: [T; 6]) -> Self::Vector6 {
        value
    }

    fn vector6_to_array(value: &Self::Vector6) -> [T; 6] {
        *value
    }

    fn matrix3_from_array(value: [[T; 3]; 3]) -> Self::Matrix3 {
        value
    }

    fn matrix3_to_array(value: &Self::Matrix3) -> [[T; 3]; 3] {
        *value
    }

    fn matrix6_from_array(value: [[T; 6]; 6]) -> Self::Matrix6 {
        value
    }

    fn matrix6_to_array(value: &Self::Matrix6) -> [[T; 6]; 6] {
        *value
    }

    fn rotation3_from_array(value: [[T; 3]; 3]) -> Self::Rotation3 {
        value
    }

    fn rotation3_to_array(value: &Self::Rotation3) -> [[T; 3]; 3] {
        *value
    }
}

#[cfg(feature = "nalgebra")]
#[derive(Copy, Clone, Debug, PartialEq)]
/// Uses nalgebra's statically sized types as backend storage: `Vector3<T>`
/// for `Vector3`, `SVector<T, 6>` for `Vector6`, `Matrix3<T>` for `Matrix3`,
/// `SMatrix<T, 6, 6>` for `Matrix6`, and `Rotation3<T>` for `Rotation3`.
pub struct Nalgebra;

#[cfg(feature = "nalgebra")]
impl<T> SpatialRepresentation<T> for Nalgebra
where
    T: SpatialScalar + nalgebra::RealField,
{
    type Vector3 = nalgebra::Vector3<T>;
    type Vector6 = nalgebra::SVector<T, 6>;
    type Matrix3 = nalgebra::Matrix3<T>;
    type Matrix6 = nalgebra::SMatrix<T, 6, 6>;
    type Rotation3 = nalgebra::Rotation3<T>;

    fn vector3_from_array(value: [T; 3]) -> Self::Vector3 {
        nalgebra::Vector3::from_row_slice(&value)
    }

    fn vector3_to_array(value: &Self::Vector3) -> [T; 3] {
        let values = value.as_slice();
        [values[0], values[1], values[2]]
    }

    fn vector6_from_array(value: [T; 6]) -> Self::Vector6 {
        nalgebra::SVector::from_row_slice(&value)
    }

    fn vector6_to_array(value: &Self::Vector6) -> [T; 6] {
        let values = value.as_slice();
        [
            values[0], values[1], values[2], values[3], values[4], values[5],
        ]
    }

    fn matrix3_from_array(value: [[T; 3]; 3]) -> Self::Matrix3 {
        let values: [T; 9] = std::array::from_fn(|index| value[index / 3][index % 3]);
        nalgebra::Matrix3::from_row_slice(&values)
    }

    fn matrix3_to_array(value: &Self::Matrix3) -> [[T; 3]; 3] {
        std::array::from_fn(|row| std::array::from_fn(|column| value[(row, column)]))
    }

    fn matrix6_from_array(value: [[T; 6]; 6]) -> Self::Matrix6 {
        let values: [T; 36] = std::array::from_fn(|index| value[index / 6][index % 6]);
        nalgebra::SMatrix::from_row_slice(&values)
    }

    fn matrix6_to_array(value: &Self::Matrix6) -> [[T; 6]; 6] {
        std::array::from_fn(|row| std::array::from_fn(|column| value[(row, column)]))
    }

    fn rotation3_from_array(value: [[T; 3]; 3]) -> Self::Rotation3 {
        let values: [T; 9] = std::array::from_fn(|index| value[index / 3][index % 3]);
        nalgebra::Rotation3::from_matrix_unchecked(nalgebra::Matrix3::from_row_slice(&values))
    }

    fn rotation3_to_array(value: &Self::Rotation3) -> [[T; 3]; 3] {
        let matrix = value.matrix();
        std::array::from_fn(|row| std::array::from_fn(|column| matrix[(row, column)]))
    }

    fn vector3_zero() -> Self::Vector3 {
        nalgebra::Vector3::zeros()
    }

    fn vector3_add(left: &Self::Vector3, right: &Self::Vector3) -> Self::Vector3 {
        *left + *right
    }

    fn vector3_sub(left: &Self::Vector3, right: &Self::Vector3) -> Self::Vector3 {
        *left - *right
    }

    fn vector3_neg(value: &Self::Vector3) -> Self::Vector3 {
        -*value
    }

    fn vector3_scale(value: &Self::Vector3, scalar: T) -> Self::Vector3 {
        *value * scalar
    }

    fn vector3_dot(left: &Self::Vector3, right: &Self::Vector3) -> T {
        left.dot(right)
    }

    fn vector3_cross(left: &Self::Vector3, right: &Self::Vector3) -> Self::Vector3 {
        left.cross(right)
    }

    fn vector3_is_finite(value: &Self::Vector3) -> bool {
        value.iter().all(|component| component.is_finite())
    }

    fn vector3_cross_matrix(value: &Self::Vector3) -> Self::Matrix3 {
        value.cross_matrix()
    }

    fn matrix3_zero() -> Self::Matrix3 {
        nalgebra::Matrix3::zeros()
    }

    fn matrix3_identity() -> Self::Matrix3 {
        nalgebra::Matrix3::identity()
    }

    fn matrix3_add(left: &Self::Matrix3, right: &Self::Matrix3) -> Self::Matrix3 {
        *left + *right
    }

    fn matrix3_sub(left: &Self::Matrix3, right: &Self::Matrix3) -> Self::Matrix3 {
        *left - *right
    }

    fn matrix3_scale(value: &Self::Matrix3, scalar: T) -> Self::Matrix3 {
        *value * scalar
    }

    fn matrix3_mul(left: &Self::Matrix3, right: &Self::Matrix3) -> Self::Matrix3 {
        *left * *right
    }

    fn matrix3_vector_mul(matrix: &Self::Matrix3, vector: &Self::Vector3) -> Self::Vector3 {
        *matrix * *vector
    }

    fn matrix3_is_finite(value: &Self::Matrix3) -> bool {
        value.iter().all(|component| component.is_finite())
    }

    fn matrix3_is_positive_definite(value: &Self::Matrix3) -> bool {
        Self::matrix3_is_finite(value) && nalgebra::Cholesky::new(*value).is_some()
    }

    fn rotation3_identity() -> Self::Rotation3 {
        nalgebra::Rotation3::identity()
    }

    fn rotation3_inverse(value: &Self::Rotation3) -> Self::Rotation3 {
        value.inverse()
    }

    fn rotation3_mul(left: &Self::Rotation3, right: &Self::Rotation3) -> Self::Rotation3 {
        *left * *right
    }

    fn rotation3_transform_vector(
        rotation: &Self::Rotation3,
        vector: &Self::Vector3,
    ) -> Self::Vector3 {
        rotation.transform_vector(vector)
    }

    fn rotation3_to_matrix3(value: &Self::Rotation3) -> Self::Matrix3 {
        *value.matrix()
    }

    fn matrix6_from_blocks(
        upper_left: &Self::Matrix3,
        upper_right: &Self::Matrix3,
        lower_left: &Self::Matrix3,
        lower_right: &Self::Matrix3,
    ) -> Self::Matrix6 {
        nalgebra::SMatrix::from_fn(|row, column| {
            if row < 3 && column < 3 {
                upper_left[(row, column)]
            } else if row < 3 {
                upper_right[(row, column - 3)]
            } else if column < 3 {
                lower_left[(row - 3, column)]
            } else {
                lower_right[(row - 3, column - 3)]
            }
        })
    }

    fn matrix6_sub(left: &Self::Matrix6, right: &Self::Matrix6) -> Self::Matrix6 {
        *left - *right
    }

    fn matrix6_neg(value: &Self::Matrix6) -> Self::Matrix6 {
        -*value
    }

    fn matrix6_transpose(value: &Self::Matrix6) -> Self::Matrix6 {
        value.transpose()
    }

    fn matrix6_mul(left: &Self::Matrix6, right: &Self::Matrix6) -> Self::Matrix6 {
        *left * *right
    }

    fn matrix6_vector_mul(matrix: &Self::Matrix6, vector: &Self::Vector6) -> Self::Vector6 {
        *matrix * *vector
    }

    fn matrix6_solve_positive_definite(
        matrix: &Self::Matrix6,
        right: &Self::Vector6,
    ) -> Option<Self::Vector6> {
        if !matrix.iter().all(|component| component.is_finite())
            || !right.iter().all(|component| component.is_finite())
        {
            return None;
        }
        let decomposition = nalgebra::Cholesky::new(*matrix)?;
        let solution = decomposition.solve(right);
        solution
            .iter()
            .all(|component| component.is_finite())
            .then_some(solution)
    }
}

#[cfg(feature = "glam")]
#[derive(Copy, Clone, Debug, PartialEq)]
/// Uses glam `Vec3`/`DVec3` for `Vector3` and `Mat3`/`DMat3` for both
/// `Matrix3` and `Rotation3` (glam has no dedicated rotation type);
/// `Vector6` and `Matrix6` remain Rust arrays because glam has no 6D types.
pub struct Glam;

#[cfg(feature = "glam")]
impl SpatialRepresentation<f32> for Glam {
    type Vector3 = glam::Vec3;
    type Vector6 = [f32; 6];
    type Matrix3 = glam::Mat3;
    type Matrix6 = [[f32; 6]; 6];
    type Rotation3 = glam::Mat3;

    fn vector3_from_array(value: [f32; 3]) -> Self::Vector3 {
        glam::Vec3::from_array(value)
    }

    fn vector3_to_array(value: &Self::Vector3) -> [f32; 3] {
        value.to_array()
    }

    fn vector6_from_array(value: [f32; 6]) -> Self::Vector6 {
        value
    }

    fn vector6_to_array(value: &Self::Vector6) -> [f32; 6] {
        *value
    }

    fn matrix3_from_array(value: [[f32; 3]; 3]) -> Self::Matrix3 {
        glam::Mat3::from_cols(
            glam::Vec3::new(value[0][0], value[1][0], value[2][0]),
            glam::Vec3::new(value[0][1], value[1][1], value[2][1]),
            glam::Vec3::new(value[0][2], value[1][2], value[2][2]),
        )
    }

    fn matrix3_to_array(value: &Self::Matrix3) -> [[f32; 3]; 3] {
        let columns = value.to_cols_array();
        [
            [columns[0], columns[3], columns[6]],
            [columns[1], columns[4], columns[7]],
            [columns[2], columns[5], columns[8]],
        ]
    }

    fn matrix6_from_array(value: [[f32; 6]; 6]) -> Self::Matrix6 {
        value
    }

    fn matrix6_to_array(value: &Self::Matrix6) -> [[f32; 6]; 6] {
        *value
    }

    fn rotation3_from_array(value: [[f32; 3]; 3]) -> Self::Rotation3 {
        <Self as SpatialRepresentation<f32>>::matrix3_from_array(value)
    }

    fn rotation3_to_array(value: &Self::Rotation3) -> [[f32; 3]; 3] {
        <Self as SpatialRepresentation<f32>>::matrix3_to_array(value)
    }

    fn vector3_zero() -> Self::Vector3 {
        glam::Vec3::ZERO
    }

    fn vector3_add(left: &Self::Vector3, right: &Self::Vector3) -> Self::Vector3 {
        *left + *right
    }

    fn vector3_sub(left: &Self::Vector3, right: &Self::Vector3) -> Self::Vector3 {
        *left - *right
    }

    fn vector3_neg(value: &Self::Vector3) -> Self::Vector3 {
        -*value
    }

    fn vector3_scale(value: &Self::Vector3, scalar: f32) -> Self::Vector3 {
        *value * scalar
    }

    fn vector3_dot(left: &Self::Vector3, right: &Self::Vector3) -> f32 {
        left.dot(*right)
    }

    fn vector3_cross(left: &Self::Vector3, right: &Self::Vector3) -> Self::Vector3 {
        left.cross(*right)
    }

    fn vector3_is_finite(value: &Self::Vector3) -> bool {
        value.is_finite()
    }

    fn vector3_cross_matrix(value: &Self::Vector3) -> Self::Matrix3 {
        glam::Mat3::from_cols(
            glam::Vec3::new(0.0, value.z, -value.y),
            glam::Vec3::new(-value.z, 0.0, value.x),
            glam::Vec3::new(value.y, -value.x, 0.0),
        )
    }

    fn matrix3_zero() -> Self::Matrix3 {
        glam::Mat3::ZERO
    }

    fn matrix3_identity() -> Self::Matrix3 {
        glam::Mat3::IDENTITY
    }

    fn matrix3_add(left: &Self::Matrix3, right: &Self::Matrix3) -> Self::Matrix3 {
        *left + *right
    }

    fn matrix3_sub(left: &Self::Matrix3, right: &Self::Matrix3) -> Self::Matrix3 {
        *left - *right
    }

    fn matrix3_scale(value: &Self::Matrix3, scalar: f32) -> Self::Matrix3 {
        *value * scalar
    }

    fn matrix3_mul(left: &Self::Matrix3, right: &Self::Matrix3) -> Self::Matrix3 {
        *left * *right
    }

    fn matrix3_vector_mul(matrix: &Self::Matrix3, vector: &Self::Vector3) -> Self::Vector3 {
        *matrix * *vector
    }

    fn matrix3_is_finite(value: &Self::Matrix3) -> bool {
        value.is_finite()
    }

    fn rotation3_identity() -> Self::Rotation3 {
        glam::Mat3::IDENTITY
    }

    fn rotation3_inverse(value: &Self::Rotation3) -> Self::Rotation3 {
        value.transpose()
    }

    fn rotation3_mul(left: &Self::Rotation3, right: &Self::Rotation3) -> Self::Rotation3 {
        *left * *right
    }

    fn rotation3_transform_vector(
        rotation: &Self::Rotation3,
        vector: &Self::Vector3,
    ) -> Self::Vector3 {
        *rotation * *vector
    }

    fn rotation3_to_matrix3(value: &Self::Rotation3) -> Self::Matrix3 {
        *value
    }
}

#[cfg(feature = "glam")]
impl SpatialRepresentation<f64> for Glam {
    type Vector3 = glam::DVec3;
    type Vector6 = [f64; 6];
    type Matrix3 = glam::DMat3;
    type Matrix6 = [[f64; 6]; 6];
    type Rotation3 = glam::DMat3;

    fn vector3_from_array(value: [f64; 3]) -> Self::Vector3 {
        glam::DVec3::from_array(value)
    }

    fn vector3_to_array(value: &Self::Vector3) -> [f64; 3] {
        value.to_array()
    }

    fn vector6_from_array(value: [f64; 6]) -> Self::Vector6 {
        value
    }

    fn vector6_to_array(value: &Self::Vector6) -> [f64; 6] {
        *value
    }

    fn matrix3_from_array(value: [[f64; 3]; 3]) -> Self::Matrix3 {
        glam::DMat3::from_cols(
            glam::DVec3::new(value[0][0], value[1][0], value[2][0]),
            glam::DVec3::new(value[0][1], value[1][1], value[2][1]),
            glam::DVec3::new(value[0][2], value[1][2], value[2][2]),
        )
    }

    fn matrix3_to_array(value: &Self::Matrix3) -> [[f64; 3]; 3] {
        let columns = value.to_cols_array();
        [
            [columns[0], columns[3], columns[6]],
            [columns[1], columns[4], columns[7]],
            [columns[2], columns[5], columns[8]],
        ]
    }

    fn matrix6_from_array(value: [[f64; 6]; 6]) -> Self::Matrix6 {
        value
    }

    fn matrix6_to_array(value: &Self::Matrix6) -> [[f64; 6]; 6] {
        *value
    }

    fn rotation3_from_array(value: [[f64; 3]; 3]) -> Self::Rotation3 {
        <Self as SpatialRepresentation<f64>>::matrix3_from_array(value)
    }

    fn rotation3_to_array(value: &Self::Rotation3) -> [[f64; 3]; 3] {
        <Self as SpatialRepresentation<f64>>::matrix3_to_array(value)
    }

    fn vector3_zero() -> Self::Vector3 {
        glam::DVec3::ZERO
    }

    fn vector3_add(left: &Self::Vector3, right: &Self::Vector3) -> Self::Vector3 {
        *left + *right
    }

    fn vector3_sub(left: &Self::Vector3, right: &Self::Vector3) -> Self::Vector3 {
        *left - *right
    }

    fn vector3_neg(value: &Self::Vector3) -> Self::Vector3 {
        -*value
    }

    fn vector3_scale(value: &Self::Vector3, scalar: f64) -> Self::Vector3 {
        *value * scalar
    }

    fn vector3_dot(left: &Self::Vector3, right: &Self::Vector3) -> f64 {
        left.dot(*right)
    }

    fn vector3_cross(left: &Self::Vector3, right: &Self::Vector3) -> Self::Vector3 {
        left.cross(*right)
    }

    fn vector3_is_finite(value: &Self::Vector3) -> bool {
        value.is_finite()
    }

    fn vector3_cross_matrix(value: &Self::Vector3) -> Self::Matrix3 {
        glam::DMat3::from_cols(
            glam::DVec3::new(0.0, value.z, -value.y),
            glam::DVec3::new(-value.z, 0.0, value.x),
            glam::DVec3::new(value.y, -value.x, 0.0),
        )
    }

    fn matrix3_zero() -> Self::Matrix3 {
        glam::DMat3::ZERO
    }

    fn matrix3_identity() -> Self::Matrix3 {
        glam::DMat3::IDENTITY
    }

    fn matrix3_add(left: &Self::Matrix3, right: &Self::Matrix3) -> Self::Matrix3 {
        *left + *right
    }

    fn matrix3_sub(left: &Self::Matrix3, right: &Self::Matrix3) -> Self::Matrix3 {
        *left - *right
    }

    fn matrix3_scale(value: &Self::Matrix3, scalar: f64) -> Self::Matrix3 {
        *value * scalar
    }

    fn matrix3_mul(left: &Self::Matrix3, right: &Self::Matrix3) -> Self::Matrix3 {
        *left * *right
    }

    fn matrix3_vector_mul(matrix: &Self::Matrix3, vector: &Self::Vector3) -> Self::Vector3 {
        *matrix * *vector
    }

    fn matrix3_is_finite(value: &Self::Matrix3) -> bool {
        value.is_finite()
    }

    fn rotation3_identity() -> Self::Rotation3 {
        glam::DMat3::IDENTITY
    }

    fn rotation3_inverse(value: &Self::Rotation3) -> Self::Rotation3 {
        value.transpose()
    }

    fn rotation3_mul(left: &Self::Rotation3, right: &Self::Rotation3) -> Self::Rotation3 {
        *left * *right
    }

    fn rotation3_transform_vector(
        rotation: &Self::Rotation3,
        vector: &Self::Vector3,
    ) -> Self::Vector3 {
        *rotation * *vector
    }

    fn rotation3_to_matrix3(value: &Self::Rotation3) -> Self::Matrix3 {
        *value
    }
}
