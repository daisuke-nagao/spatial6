// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::marker::PhantomData;
use std::ops::Mul;

#[cfg(feature = "builtin")]
use crate::Builtin;
use crate::math::matrix_is_finite;
use crate::transform::SpatialTransform;
use crate::vector::{ForceVector, MotionVector, SpatialMatrix};
use crate::{SpatialRepresentation, SpatialScalar};

/// Errors returned when constructing or operating on an inertia.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InertiaError {
    /// At least one input or derived component is not finite.
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

fn safe_average<T: SpatialScalar>(left: T, right: T) -> T {
    let two = T::one() + T::one();
    let sum = left + right;
    if sum.is_finite() {
        sum / two
    } else {
        left / two + right / two
    }
}

#[allow(clippy::needless_range_loop)]
fn articulated_symmetric_matrix<T: SpatialScalar, const N: usize>(
    matrix: &[[T; N]; N],
) -> Result<[[T; N]; N], InertiaError> {
    if !matrix_is_finite(matrix) {
        return Err(InertiaError::NonFinite);
    }

    let mut symmetric = *matrix;
    for row in 0..N {
        for column in (row + 1)..N {
            if !approximately_equal(matrix[row][column], matrix[column][row]) {
                return Err(InertiaError::NonSymmetric);
            }
            let value = safe_average(matrix[row][column], matrix[column][row]);
            symmetric[row][column] = value;
            symmetric[column][row] = value;
        }
    }
    Ok(symmetric)
}

#[allow(clippy::needless_range_loop)]
fn pack_upper_symmetric_matrix<T: SpatialScalar, const N: usize, const P: usize>(
    matrix: &[[T; N]; N],
) -> [T; P] {
    let mut packed = [T::zero(); P];
    let mut index = 0;
    for row in 0..N {
        for column in row..N {
            packed[index] = matrix[row][column];
            index += 1;
        }
    }
    packed
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

macro_rules! define_articulated_body_inertia {
    ($($generics:tt)*) => {
        /// A finite symmetric six-by-six spatial operator stored as its
        /// twenty-one upper-triangular components in row-major order.
        ///
        /// Unlike [`RigidBodyInertia`], an articulated-body inertia may be
        /// indefinite. Public matrix construction validates finiteness and
        /// symmetry, while derived construction validates finiteness and
        /// canonicalizes its mathematically symmetric result. Neither path
        /// certifies positive semidefiniteness or physical realizability.
        /// Frame agreement between an inertia and its operands is maintained
        /// by the caller; methods that change frames document their source and
        /// destination frames explicitly.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct ArticulatedBodyInertia<$($generics)*> {
            packed: [T; 21],
            representation: PhantomData<R>,
        }
    };
}

#[cfg(feature = "builtin")]
define_articulated_body_inertia!(T: SpatialScalar = f64, R: SpatialRepresentation<T> = Builtin);
#[cfg(not(feature = "builtin"))]
define_articulated_body_inertia!(T: SpatialScalar, R: SpatialRepresentation<T>);

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "T: serde::Deserialize<'de>"))]
struct RawArticulatedBodyInertia<T> {
    packed: [T; 21],
}

#[cfg(feature = "serde")]
impl<T, R> serde::Serialize for ArticulatedBodyInertia<T, R>
where
    T: SpatialScalar + serde::Serialize,
    R: SpatialRepresentation<T>,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("ArticulatedBodyInertia", 1)?;
        state.serialize_field("packed", &self.packed)?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, T, R> serde::Deserialize<'de> for ArticulatedBodyInertia<T, R>
where
    T: SpatialScalar + serde::Deserialize<'de>,
    R: SpatialRepresentation<T>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawArticulatedBodyInertia::<T>::deserialize(deserializer)?;
        if raw.packed.iter().any(|value| !value.is_finite()) {
            return Err(serde::de::Error::custom(InertiaError::NonFinite));
        }
        Ok(Self {
            packed: raw.packed,
            representation: PhantomData,
        })
    }
}

impl<T, R> ArticulatedBodyInertia<T, R>
where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
{
    // This path is reserved for arrays derived from formulas symmetric in exact arithmetic.
    #[allow(clippy::needless_range_loop)]
    fn try_from_derived_symmetric_array(matrix: [[T; 6]; 6]) -> Result<Self, InertiaError> {
        if !matrix_is_finite(&matrix) {
            return Err(InertiaError::NonFinite);
        }

        let mut symmetric = matrix;
        for row in 0..6 {
            for column in (row + 1)..6 {
                let value = safe_average(matrix[row][column], matrix[column][row]);
                symmetric[row][column] = value;
                symmetric[column][row] = value;
            }
        }

        Ok(Self {
            packed: pack_upper_symmetric_matrix::<T, 6, 21>(&symmetric),
            representation: PhantomData,
        })
    }

    /// Validates and creates an articulated-body inertia from a six-by-six
    /// spatial matrix. All entries must be finite and every off-diagonal pair
    /// must agree within `64 * T::epsilon() * max(1, |a|, |b|)`. Accepted
    /// off-diagonal pairs are averaged before storage; diagonal entries are
    /// retained unchanged.
    ///
    /// The matrix must be expressed in the same frame as any other operator
    /// with which the result is used. Frame identity is maintained by the
    /// caller.
    ///
    /// # Errors
    ///
    /// Returns [`InertiaError::NonFinite`] if any matrix entry is NaN or
    /// infinite. This check is performed before symmetry checks. Returns
    /// [`InertiaError::NonSymmetric`] if a finite off-diagonal pair is outside
    /// the tolerance above.
    pub fn try_from_matrix(matrix: SpatialMatrix<T, R>) -> Result<Self, InertiaError> {
        let matrix = R::matrix6_to_array(&matrix);
        let matrix = articulated_symmetric_matrix(&matrix)?;
        Ok(Self {
            packed: pack_upper_symmetric_matrix::<T, 6, 21>(&matrix),
            representation: PhantomData,
        })
    }

    /// Returns the zero articulated-body inertia.
    pub fn zeros() -> Self {
        Self {
            packed: [T::zero(); 21],
            representation: PhantomData,
        }
    }

    /// Returns the symmetric six-by-six matrix represented by this value.
    pub fn matrix(&self) -> SpatialMatrix<T, R> {
        R::matrix6_from_array(unpack_symmetric_matrix::<T, 6, 21>(&self.packed))
    }

    /// Applies this inertia to a motion vector, producing a force vector.
    ///
    /// `motion` must use the same frame as this inertia; the returned force is
    /// expressed in that frame. This method performs no finiteness validation
    /// on its input, intermediate values, or result, so non-finite values may
    /// propagate.
    pub fn apply(&self, motion: &MotionVector<T, R>) -> ForceVector<T, R> {
        let matrix = self.matrix();
        let motion = motion.to_vector();
        ForceVector::from_vector(R::matrix6_vector_mul(&matrix, &motion))
    }

    /// Adds the corresponding stored components of another articulated-body
    /// inertia expressed in the same frame.
    ///
    /// Frame agreement is maintained by the caller. The result is a new
    /// operator; neither input is modified.
    ///
    /// # Errors
    ///
    /// Returns [`InertiaError::NonFinite`] if any componentwise sum is
    /// non-finite, including when an addition overflows.
    pub fn try_combined(&self, other: &Self) -> Result<Self, InertiaError> {
        let mut packed = [T::zero(); 21];
        for (index, value) in packed.iter_mut().enumerate() {
            *value = self.packed[index] + other.packed[index];
            if !value.is_finite() {
                return Err(InertiaError::NonFinite);
            }
        }
        Ok(Self {
            packed,
            representation: PhantomData,
        })
    }

    /// Applies the symmetric rank-one update `self + alpha * u * uᵀ`.
    ///
    /// `u` is a force-space vector in the same frame as this inertia. A
    /// negative `alpha` can destroy positive semidefiniteness; this method does
    /// not certify or preserve that property. Finiteness validation does not
    /// certify numerical accuracy.
    ///
    /// # Errors
    ///
    /// Returns [`InertiaError::NonFinite`] if `alpha` or `u` is non-finite, or
    /// if any scaled component, rank-one contribution, or updated stored
    /// component is non-finite. `u` is validated even when `alpha` is zero.
    #[allow(clippy::needless_range_loop)]
    pub fn try_rank_one_updated(
        &self,
        alpha: T,
        u: &ForceVector<T, R>,
    ) -> Result<Self, InertiaError> {
        if !alpha.is_finite() || !u.is_finite() {
            return Err(InertiaError::NonFinite);
        }
        if alpha == T::zero() {
            return Ok(*self);
        }

        let coordinates = u.to_array();
        let mut scaled = [T::zero(); 6];
        for (index, value) in scaled.iter_mut().enumerate() {
            *value = alpha * coordinates[index];
            if !value.is_finite() {
                return Err(InertiaError::NonFinite);
            }
        }

        let mut packed = [T::zero(); 21];
        let mut index = 0;
        for row in 0..6 {
            for column in row..6 {
                let contribution = scaled[row] * coordinates[column];
                if !contribution.is_finite() {
                    return Err(InertiaError::NonFinite);
                }
                packed[index] = self.packed[index] + contribution;
                if !packed[index].is_finite() {
                    return Err(InertiaError::NonFinite);
                }
                index += 1;
            }
        }

        Ok(Self {
            packed,
            representation: PhantomData,
        })
    }

    /// Re-expresses this inertia from the transform's source frame to its
    /// destination frame using the force-space congruence `F * self * Fᵀ`.
    ///
    /// The input inertia must be expressed in the transform's source frame;
    /// the returned inertia is expressed in its destination frame. Here `F`
    /// is `transform.force_matrix()`, which maps source-frame forces to
    /// destination-frame forces. The transform's rotation is assumed to be a
    /// valid proper orthogonal rotation under [`SpatialRepresentation`]'s
    /// invariant; this method does not validate that assumption. Roundoff in
    /// the derived congruence result is canonicalized by averaging opposite
    /// entries before storage. Finiteness checks do not certify numerical
    /// accuracy.
    ///
    /// # Errors
    ///
    /// Returns [`InertiaError::NonFinite`] if `F`, the first product `F *
    /// self`, or the final product is non-finite. An overflowing intermediate
    /// is rejected even if a different evaluation order could produce a finite
    /// final value.
    pub fn try_transformed(
        &self,
        transform: &SpatialTransform<T, R>,
    ) -> Result<Self, InertiaError> {
        let force = transform.force_matrix();
        let force_array = R::matrix6_to_array(&force);
        if !matrix_is_finite(&force_array) {
            return Err(InertiaError::NonFinite);
        }

        let left = R::matrix6_mul(&force, &self.matrix());
        let left_array = R::matrix6_to_array(&left);
        if !matrix_is_finite(&left_array) {
            return Err(InertiaError::NonFinite);
        }

        let transformed = R::matrix6_mul(&left, &R::matrix6_transpose(&force));
        let transformed_array = R::matrix6_to_array(&transformed);
        Self::try_from_derived_symmetric_array(transformed_array)
    }
}

impl<T, R> TryFrom<&RigidBodyInertia<T, R>> for ArticulatedBodyInertia<T, R>
where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
{
    type Error = InertiaError;

    /// Converts a rigid-body inertia to an articulated-body inertia.
    ///
    /// The rigid-body matrix is symmetric in exact arithmetic. Derived
    /// floating-point roundoff is canonicalized by averaging opposite entries
    /// before storage, so `NonSymmetric` is not returned for a conforming
    /// backend. Diagonal entries are retained from the derived matrix.
    ///
    /// # Errors
    ///
    /// Returns [`InertiaError::NonFinite`] if a derived matrix entry is
    /// non-finite, including an intermediate overflow during matrix
    /// construction. A conforming backend does not produce
    /// [`InertiaError::NonSymmetric`] from this conversion.
    fn try_from(rigid: &RigidBodyInertia<T, R>) -> Result<Self, Self::Error> {
        let matrix = R::matrix6_to_array(&rigid.matrix());
        Self::try_from_derived_symmetric_array(matrix)
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
