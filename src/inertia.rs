// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::ops::Mul;

#[cfg(feature = "builtin")]
use crate::Builtin;
use crate::math::matrix_is_finite;
use crate::transform::SpatialTransform;
use crate::vector::{ForceVector, MotionVector, SpatialMatrix};
use crate::{SpatialRepresentation, SpatialScalar};

/// Errors returned when constructing an inertia from invalid data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InertiaError {
    /// At least one input component is not finite.
    NonFinite,
    /// The mass is zero or negative.
    NonPositiveMass,
    /// A matrix that must be symmetric is outside the comparison tolerance.
    NonSymmetric,
    /// A rigid-body inertia matrix is not positive definite.
    NotPositiveDefinite,
}

impl fmt::Display for InertiaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => formatter.write_str("inertia contains a non-finite value"),
            Self::NonPositiveMass => formatter.write_str("mass must be positive"),
            Self::NonSymmetric => formatter.write_str("inertia matrix must be symmetric"),
            Self::NotPositiveDefinite => {
                formatter.write_str("rigid-body inertia must be positive definite")
            }
        }
    }
}

impl std::error::Error for InertiaError {}

fn symmetry_tolerance<T: SpatialScalar>() -> T {
    let mut factor = T::one();
    for _ in 0..6 {
        factor = factor + factor;
    }
    T::epsilon() * factor
}

fn approximately_equal<T: SpatialScalar>(left: T, right: T) -> bool {
    let mut scale = T::one();
    if left.abs() > scale {
        scale = left.abs();
    }
    if right.abs() > scale {
        scale = right.abs();
    }
    (left - right).abs() <= symmetry_tolerance::<T>() * scale
}

#[allow(clippy::needless_range_loop)]
fn pack_symmetric_matrix<T: SpatialScalar, const N: usize, const P: usize>(
    matrix: &[[T; N]; N],
) -> [T; P] {
    let mut packed = [T::zero(); P];
    let mut index = 0;
    for row in 0..N {
        for column in row..N {
            packed[index] = if row == column {
                matrix[row][column]
            } else {
                (matrix[row][column] + matrix[column][row]) / (T::one() + T::one())
            };
            index += 1;
        }
    }
    packed
}

#[allow(clippy::needless_range_loop)]
fn unpack_symmetric_matrix<T: SpatialScalar, const N: usize, const P: usize>(
    packed: &[T; P],
) -> [[T; N]; N] {
    let mut matrix = [[T::zero(); N]; N];
    let mut index = 0;
    for row in 0..N {
        for column in row..N {
            let value = packed[index];
            matrix[row][column] = value;
            matrix[column][row] = value;
            index += 1;
        }
    }
    matrix
}

#[allow(clippy::needless_range_loop)]
fn symmetric_matrix<T: SpatialScalar, const N: usize>(
    matrix: &[[T; N]; N],
) -> Result<[[T; N]; N], InertiaError> {
    if !matrix_is_finite(matrix) {
        return Err(InertiaError::NonFinite);
    }

    for row in 0..N {
        for column in (row + 1)..N {
            if !approximately_equal(matrix[row][column], matrix[column][row]) {
                return Err(InertiaError::NonSymmetric);
            }
        }
    }

    let mut symmetric = *matrix;
    for row in 0..N {
        for column in (row + 1)..N {
            let value = (matrix[row][column] + matrix[column][row]) / (T::one() + T::one());
            symmetric[row][column] = value;
            symmetric[column][row] = value;
        }
    }
    Ok(symmetric)
}

macro_rules! define_rigid_body_inertia {
    ($($generics:tt)*) => {
        /// A rigid-body spatial inertia stored as ten scalar parameters: mass, center
        /// of mass, and the six independent components of the rotational inertia
        /// about the center of mass. Also available under its usual name,
        /// [`SpatialInertia`].
        ///
        /// `Deserialize` is implemented by hand (see below) so that untrusted input
        /// is re-validated through [`Self::try_new`] rather than bypassing it.
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize))]
        #[cfg_attr(
            feature = "serde",
            serde(bound(serialize = "T: serde::Serialize, R::Vector3: serde::Serialize"))
        )]
        pub struct RigidBodyInertia<$($generics)*> {
            mass: T,
            center_of_mass: R::Vector3,
            inertia_at_center_of_mass: [T; 6],
        }
    };
}

#[cfg(feature = "builtin")]
define_rigid_body_inertia!(T: SpatialScalar = f64, R: SpatialRepresentation<T> = Builtin);
#[cfg(not(feature = "builtin"))]
define_rigid_body_inertia!(T: SpatialScalar, R: SpatialRepresentation<T>);

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "T: serde::Deserialize<'de>, V: serde::Deserialize<'de>"))]
struct RawRigidBodyInertia<T, V> {
    mass: T,
    center_of_mass: V,
    inertia_at_center_of_mass: [T; 6],
}

#[cfg(feature = "serde")]
impl<'de, T, R> serde::Deserialize<'de> for RigidBodyInertia<T, R>
where
    T: SpatialScalar + serde::Deserialize<'de>,
    R: SpatialRepresentation<T>,
    R::Vector3: serde::Deserialize<'de>,
{
    /// Deserializes through [`Self::try_new`], so untrusted input is re-validated
    /// (finite, positive mass, symmetric, positive-definite) rather than
    /// directly populating the private fields.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawRigidBodyInertia::<T, R::Vector3>::deserialize(deserializer)?;
        let inertia_array = unpack_symmetric_matrix::<T, 3, 6>(&raw.inertia_at_center_of_mass);
        let inertia = R::matrix3_from_array(inertia_array);
        Self::try_new(raw.mass, raw.center_of_mass, inertia).map_err(serde::de::Error::custom)
    }
}

impl<T, R> RigidBodyInertia<T, R>
where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
{
    /// Validates and creates a rigid-body inertia from mass, center of mass, and
    /// rotational inertia about the center of mass.
    ///
    /// Validation runs in this order: all components must be finite; the mass must
    /// be positive; `inertia_at_center_of_mass` must be symmetric within a small
    /// relative tolerance; and the symmetrized result must be positive definite.
    ///
    /// # Errors
    ///
    /// Returns [`InertiaError::NonFinite`] if `mass`, `center_of_mass`, or
    /// `inertia_at_center_of_mass` contains a non-finite value.
    /// Returns [`InertiaError::NonPositiveMass`] if `mass` is zero or negative.
    /// Returns [`InertiaError::NonSymmetric`] if `inertia_at_center_of_mass` is not
    /// symmetric within tolerance.
    /// Returns [`InertiaError::NotPositiveDefinite`] if the symmetrized
    /// `inertia_at_center_of_mass` is not positive definite.
    pub fn try_new(
        mass: T,
        center_of_mass: R::Vector3,
        inertia_at_center_of_mass: R::Matrix3,
    ) -> Result<Self, InertiaError> {
        let inertia_array = R::matrix3_to_array(&inertia_at_center_of_mass);
        if !mass.is_finite()
            || !R::vector3_is_finite(&center_of_mass)
            || !R::matrix3_is_finite(&inertia_at_center_of_mass)
        {
            return Err(InertiaError::NonFinite);
        }
        if mass <= T::zero() {
            return Err(InertiaError::NonPositiveMass);
        }
        let inertia_array = symmetric_matrix(&inertia_array)?;
        let symmetric_inertia = R::matrix3_from_array(inertia_array);
        if !R::matrix3_is_positive_definite(&symmetric_inertia) {
            return Err(InertiaError::NotPositiveDefinite);
        }

        Ok(Self::from_parts(mass, center_of_mass, inertia_array))
    }

    /// Combines two rigidly attached bodies expressed in the same frame.
    ///
    /// Frame agreement is a caller-maintained invariant. Use [`Self::transformed`] to
    /// express both inertias in a common frame before combining them.
    ///
    /// With total mass `m = m1 + m2`, center separation `d = c2 - c1`, and
    /// reduced mass `mu = m1 * m2 / m`, the combined center-of-mass inertia is
    /// `I_C = I_C1 + I_C2 - mu * [d]× * [d]×`.
    ///
    /// # Errors
    ///
    /// Returns [`InertiaError::NonFinite`] if the total mass or a derived value
    /// is non-finite. Other validation errors from [`Self::try_new`] are
    /// propagated if the combined inertia is not a valid rigid-body inertia.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "builtin")]
    /// # {
    /// use spatial6::{RigidBodyInertia, SpatialTransform};
    ///
    /// let left = RigidBodyInertia::<f64>::try_new(
    ///     1.0,
    ///     [0.0, 0.0, 0.0],
    ///     [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    /// )?;
    /// let right = RigidBodyInertia::<f64>::try_new(
    ///     2.0,
    ///     [1.0, 0.0, 0.0],
    ///     [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
    /// )?;
    /// let right_to_common = SpatialTransform::<f64>::new(
    ///     [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ///     [1.0, 0.0, 0.0],
    /// );
    ///
    /// let right_in_common = right.transformed(&right_to_common);
    /// let combined = left.try_combined(&right_in_common)?;
    ///
    /// assert_eq!(combined.mass(), 3.0);
    /// # }
    /// # Ok::<(), spatial6::InertiaError>(())
    /// ```
    pub fn try_combined(&self, other: &Self) -> Result<Self, InertiaError> {
        let mass = self.mass + other.mass;
        if !mass.is_finite() {
            return Err(InertiaError::NonFinite);
        }

        let self_weight = self.mass / mass;
        let other_weight = other.mass / mass;
        let center_of_mass = R::vector3_add(
            &R::vector3_scale(&self.center_of_mass, self_weight),
            &R::vector3_scale(&other.center_of_mass, other_weight),
        );

        let center_delta = R::vector3_sub(&other.center_of_mass, &self.center_of_mass);
        let center_cross = R::vector3_cross_matrix(&center_delta);
        let reduced_mass = self.mass * other_weight;
        let parallel_axis =
            R::matrix3_scale(&R::matrix3_mul(&center_cross, &center_cross), reduced_mass);
        let inertia_at_center_of_mass = R::matrix3_sub(
            &R::matrix3_add(
                &self.inertia_at_center_of_mass(),
                &other.inertia_at_center_of_mass(),
            ),
            &parallel_axis,
        );

        Self::try_new(mass, center_of_mass, inertia_at_center_of_mass)
    }

    fn from_parts(
        mass: T,
        center_of_mass: R::Vector3,
        inertia_at_center_of_mass: [[T; 3]; 3],
    ) -> Self {
        Self {
            mass,
            center_of_mass,
            inertia_at_center_of_mass: pack_symmetric_matrix::<T, 3, 6>(&inertia_at_center_of_mass),
        }
    }

    /// Returns the mass.
    pub fn mass(&self) -> T {
        self.mass
    }

    /// Returns the center of mass in this frame.
    pub fn center_of_mass(&self) -> &R::Vector3 {
        &self.center_of_mass
    }

    /// Returns the rotational inertia about the center of mass.
    pub fn inertia_at_center_of_mass(&self) -> R::Matrix3 {
        R::matrix3_from_array(unpack_symmetric_matrix::<T, 3, 6>(
            &self.inertia_at_center_of_mass,
        ))
    }

    /// Returns the full six-by-six spatial inertia matrix.
    ///
    /// For mass `m`, center of mass `c` ([`Self::center_of_mass`]), and rotational
    /// inertia `I_C` about the center of mass ([`Self::inertia_at_center_of_mass`]),
    /// the result is the block matrix `[[I_C - m [c]× [c]×, m [c]×], [-m [c]×, m *
    /// Id]]`, where `Id` is the 3x3 identity matrix.
    pub fn matrix(&self) -> SpatialMatrix<T, R> {
        let center_cross = R::vector3_cross_matrix(&self.center_of_mass);
        let rotational = self.inertia_at_center_of_mass();
        let mass_center_cross = R::matrix3_scale(&center_cross, self.mass);
        let upper_left = R::matrix3_sub(
            &rotational,
            &R::matrix3_mul(&mass_center_cross, &center_cross),
        );
        let upper_right = mass_center_cross;
        let zero = R::matrix3_zero();
        let lower_left = R::matrix3_sub(&zero, &upper_right);
        let identity = R::matrix3_identity();
        let lower_right = R::matrix3_scale(&identity, self.mass);

        R::matrix6_from_blocks(&upper_left, &upper_right, &lower_left, &lower_right)
    }

    /// Applies this inertia to a motion vector without building a six-by-six matrix.
    ///
    /// Equivalent to `self.matrix() * motion`, i.e. `f = I v` as a [`ForceVector`].
    pub fn apply(&self, motion: &MotionVector<T, R>) -> ForceVector<T, R> {
        let center_cross_angular = R::vector3_cross(&self.center_of_mass, &motion.angular);
        let relative_linear = R::vector3_sub(&motion.linear, &center_cross_angular);
        let linear = R::vector3_scale(&relative_linear, self.mass);
        let rotational = self.inertia_at_center_of_mass();
        let rotational_moment = R::matrix3_vector_mul(&rotational, &motion.angular);
        let center_cross_linear = R::vector3_cross(&self.center_of_mass, &linear);
        let moment = R::vector3_add(&rotational_moment, &center_cross_linear);
        ForceVector::new(moment, linear)
    }

    /// Moves this inertia from `transform`'s source frame, which must be this
    /// frame, to the destination frame.
    ///
    /// The center of mass and the rotational inertia at the center of mass are
    /// re-expressed through `transform`'s rotation and translation. Equivalently,
    /// [`Self::matrix`] of the result is the congruence transform `Xf *
    /// self.matrix() * Xm⁻¹`, where `Xm` and `Xf` are respectively `transform`'s
    /// [`motion_matrix`][SpatialTransform::motion_matrix] and
    /// [`force_matrix`][SpatialTransform::force_matrix].
    pub fn transformed(&self, transform: &SpatialTransform<T, R>) -> Self {
        let center_offset = R::vector3_sub(&self.center_of_mass, transform.translation());
        let center_of_mass = R::rotation3_transform_vector(transform.rotation(), &center_offset);
        let rotation = R::rotation3_to_matrix3(transform.rotation());
        let inverse_rotation = R::rotation3_inverse(transform.rotation());
        let inverse_matrix = R::rotation3_to_matrix3(&inverse_rotation);
        let rotational = self.inertia_at_center_of_mass();
        let rotated = R::matrix3_mul(&R::matrix3_mul(&rotation, &rotational), &inverse_matrix);

        Self::from_parts(self.mass, center_of_mass, R::matrix3_to_array(&rotated))
    }

    /// Returns the time derivative of [`Self::matrix`] as the body moves with
    /// `velocity`: `v×* I - I v×`.
    ///
    /// `v×` is `velocity`'s [`cross_matrix`][MotionVector::cross_matrix] and `v×*`
    /// is its [`cross_dual_matrix`][MotionVector::cross_dual_matrix].
    pub fn time_derivative(&self, velocity: &MotionVector<T, R>) -> SpatialMatrix<T, R> {
        let cross_dual = velocity.cross_dual_matrix();
        let cross_motion = velocity.cross_matrix();
        let inertia = self.matrix();
        let dual_product = R::matrix6_mul(&cross_dual, &inertia);
        let product = R::matrix6_mul(&inertia, &cross_motion);
        R::matrix6_sub(&dual_product, &product)
    }

    /// Returns the velocity-dependent bias force `v×*(I v)`, the
    /// Coriolis/centrifugal term in the equations of motion.
    ///
    /// Equivalent to `velocity.cross_force(&self.apply(velocity))`.
    pub fn bias_force(&self, velocity: &MotionVector<T, R>) -> ForceVector<T, R> {
        velocity.cross_force(&self.apply(velocity))
    }

    /// Computes the inverse-dynamics force `I a + v×*(I v)` needed to produce
    /// `acceleration` while moving with `velocity`.
    ///
    /// Equivalent to `self.apply(acceleration) + self.bias_force(velocity)`; see
    /// [`Self::bias_force`] for the second term.
    pub fn inverse_dynamics(
        &self,
        velocity: &MotionVector<T, R>,
        acceleration: &MotionVector<T, R>,
    ) -> ForceVector<T, R> {
        self.apply(acceleration) + self.bias_force(velocity)
    }

    /// Solves forward dynamics for the acceleration produced by `force` while
    /// moving with `velocity`, i.e. the `acceleration` for which
    /// [`Self::inverse_dynamics`] would return `force`.
    ///
    /// Returns `None` if `velocity` or `force` is not finite, or if the resulting
    /// linear system cannot be solved. For any [`Self::try_new`]-validated inertia
    /// (positive mass, positive-definite inertia at the center of mass), the full
    /// six-by-six matrix is always symmetric positive definite, so the solve
    /// failing is only a theoretical/extreme-numerical-conditions case.
    pub fn forward_dynamics(
        &self,
        velocity: &MotionVector<T, R>,
        force: &ForceVector<T, R>,
    ) -> Option<MotionVector<T, R>> {
        if !velocity.is_finite() || !force.is_finite() {
            return None;
        }

        let right = *force - self.bias_force(velocity);
        let matrix = self.matrix();
        let right_vector = right.to_vector();
        R::matrix6_solve_positive_definite(&matrix, &right_vector).map(MotionVector::from_vector)
    }
}

impl<T, R> Mul<MotionVector<T, R>> for RigidBodyInertia<T, R>
where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
{
    type Output = ForceVector<T, R>;

    fn mul(self, rhs: MotionVector<T, R>) -> Self::Output {
        self.apply(&rhs)
    }
}

/// The usual name for [`RigidBodyInertia`].
#[cfg(feature = "builtin")]
pub type SpatialInertia<T = f64, R = Builtin> = RigidBodyInertia<T, R>;

/// The usual name for [`RigidBodyInertia`].
#[cfg(not(feature = "builtin"))]
pub type SpatialInertia<T, R> = RigidBodyInertia<T, R>;
