// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Borrow;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(feature = "builtin")]
use crate::Builtin;
use crate::{SpatialRepresentation, SpatialScalar};

/// The six coordinates used by spatial motion and force vectors.
#[cfg(feature = "builtin")]
pub type SpatialCoordinates<T = f64, R = Builtin> = <R as SpatialRepresentation<T>>::Vector6;

/// The six coordinates used by spatial motion and force vectors.
#[cfg(not(feature = "builtin"))]
pub type SpatialCoordinates<T, R> = <R as SpatialRepresentation<T>>::Vector6;

/// A 6x6 spatial matrix.
#[cfg(feature = "builtin")]
pub type SpatialMatrix<T = f64, R = Builtin> = <R as SpatialRepresentation<T>>::Matrix6;

/// A 6x6 spatial matrix.
#[cfg(not(feature = "builtin"))]
pub type SpatialMatrix<T, R> = <R as SpatialRepresentation<T>>::Matrix6;

macro_rules! define_motion_vector {
    ($($generics:tt)*) => {
        /// A Plücker motion vector, with angular coordinates before linear coordinates.
        ///
        /// ```
        /// # #[cfg(feature = "builtin")] {
        /// use spatial6::MotionVector;
        ///
        /// let motion = MotionVector::<f64>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        /// assert_eq!(motion.to_vector(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        /// # }
        /// ```
        ///
        /// Motion and force vectors remain distinct spaces.
        ///
        /// ```compile_fail
        /// use spatial6::{Builtin, ForceVector, MotionVector};
        ///
        /// let motion = MotionVector::<f64, Builtin>::zeros();
        /// let force = ForceVector::<f64, Builtin>::zeros();
        /// let _ = motion + force;
        /// ```
        ///
        /// A motion vector only has a dual product with a force vector.
        ///
        /// ```compile_fail
        /// use spatial6::{Builtin, MotionVector};
        ///
        /// let motion = MotionVector::<f64, Builtin>::zeros();
        /// let _ = motion.dot(&motion);
        /// ```
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(
            feature = "serde",
            serde(bound(
                serialize = "R::Vector3: serde::Serialize",
                deserialize = "R::Vector3: serde::Deserialize<'de>"
            ))
        )]
        pub struct MotionVector<$($generics)*> {
            /// Angular coordinates.
            pub angular: R::Vector3,
            /// Linear coordinates.
            pub linear: R::Vector3,
        }
    };
}

#[cfg(feature = "builtin")]
define_motion_vector!(T: SpatialScalar = f64, R: SpatialRepresentation<T> = Builtin);
#[cfg(not(feature = "builtin"))]
define_motion_vector!(T: SpatialScalar, R: SpatialRepresentation<T>);

impl<T, R> MotionVector<T, R>
where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
{
    /// Creates a motion vector from its angular and linear components.
    pub fn new(angular: R::Vector3, linear: R::Vector3) -> Self {
        Self { angular, linear }
    }

    /// Returns the zero motion vector.
    pub fn zeros() -> Self {
        Self::new(R::vector3_zero(), R::vector3_zero())
    }

    /// Builds a motion vector from angular-before-linear coordinates.
    pub fn from_vector<V>(vector: V) -> Self
    where
        V: Borrow<R::Vector6>,
    {
        Self::from_array(R::vector6_to_array(vector.borrow()))
    }

    /// Builds a motion vector from backend-independent angular-before-linear coordinates.
    pub fn from_array(vector: [T; 6]) -> Self {
        Self::new(
            R::vector3_from_array([vector[0], vector[1], vector[2]]),
            R::vector3_from_array([vector[3], vector[4], vector[5]]),
        )
    }

    /// Converts this motion vector to angular-before-linear coordinates.
    pub fn to_vector(&self) -> SpatialCoordinates<T, R> {
        R::vector6_from_array(self.to_array())
    }

    /// Returns backend-independent angular-before-linear coordinates.
    pub fn to_array(&self) -> [T; 6] {
        let angular = R::vector3_to_array(&self.angular);
        let linear = R::vector3_to_array(&self.linear);
        [
            angular[0], angular[1], angular[2], linear[0], linear[1], linear[2],
        ]
    }

    /// Returns whether all components are finite.
    pub fn is_finite(&self) -> bool {
        R::vector3_is_finite(&self.angular) && R::vector3_is_finite(&self.linear)
    }

    /// Computes the dual scalar product with a force vector.
    ///
    /// `self.angular·force.moment + self.linear·force.force`.
    pub fn dot(&self, force: &ForceVector<T, R>) -> T {
        R::vector3_dot(&self.angular, &force.moment) + R::vector3_dot(&self.linear, &force.force)
    }

    /// Computes the spatial motion cross product `self × motion`.
    ///
    /// `self × motion = (self.angular × motion.angular, self.angular × motion.linear
    /// + self.linear × motion.angular)`. See [`Self::cross_matrix`] for the matrix form.
    pub fn cross_motion(&self, motion: &Self) -> Self {
        Self::new(
            R::vector3_cross(&self.angular, &motion.angular),
            R::vector3_add(
                &R::vector3_cross(&self.angular, &motion.linear),
                &R::vector3_cross(&self.linear, &motion.angular),
            ),
        )
    }

    /// Computes the dual spatial cross product `self ×* force`.
    ///
    /// `self ×* force = (self.angular × force.moment + self.linear × force.force,
    /// self.angular × force.force)`. See [`Self::cross_dual_matrix`] for the matrix form.
    pub fn cross_force(&self, force: &ForceVector<T, R>) -> ForceVector<T, R> {
        ForceVector::new(
            R::vector3_add(
                &R::vector3_cross(&self.angular, &force.moment),
                &R::vector3_cross(&self.linear, &force.force),
            ),
            R::vector3_cross(&self.angular, &force.force),
        )
    }

    /// Returns the 6x6 matrix form of [`Self::cross_motion`].
    ///
    /// `self.cross_matrix() * motion.to_vector() == self.cross_motion(motion).to_vector()`.
    /// In block form this is `[[A, 0], [L, A]]`, where `A` and `L` are the
    /// skew-symmetric cross-product matrices (`[v]×`, with `[v]× w == v × w`) of
    /// `self.angular` and `self.linear` respectively.
    pub fn cross_matrix(&self) -> SpatialMatrix<T, R> {
        let angular = R::vector3_cross_matrix(&self.angular);
        let linear = R::vector3_cross_matrix(&self.linear);
        let zero = R::matrix3_zero();
        R::matrix6_from_blocks(&angular, &zero, &linear, &angular)
    }

    /// Returns the 6x6 matrix form of the dual cross product `self ×* force`
    /// (see [`Self::cross_force`]).
    ///
    /// Equal to the negated transpose of [`Self::cross_matrix`]:
    /// `self.cross_dual_matrix() * force.to_vector() == self.cross_force(force).to_vector()`.
    /// This is the matrix
    /// [`RigidBodyInertia::time_derivative`](crate::RigidBodyInertia::time_derivative)
    /// uses for its `v×* I` term.
    pub fn cross_dual_matrix(&self) -> SpatialMatrix<T, R> {
        R::matrix6_neg(&R::matrix6_transpose(&self.cross_matrix()))
    }
}

macro_rules! define_force_vector {
    ($($generics:tt)*) => {
        /// A Plücker force vector, with moment coordinates before force coordinates.
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(
            feature = "serde",
            serde(bound(
                serialize = "R::Vector3: serde::Serialize",
                deserialize = "R::Vector3: serde::Deserialize<'de>"
            ))
        )]
        pub struct ForceVector<$($generics)*> {
            /// Moment coordinates.
            pub moment: R::Vector3,
            /// Linear-force coordinates.
            pub force: R::Vector3,
        }
    };
}

#[cfg(feature = "builtin")]
define_force_vector!(T: SpatialScalar = f64, R: SpatialRepresentation<T> = Builtin);
#[cfg(not(feature = "builtin"))]
define_force_vector!(T: SpatialScalar, R: SpatialRepresentation<T>);

impl<T, R> ForceVector<T, R>
where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
{
    /// Creates a force vector from its moment and force components.
    pub fn new(moment: R::Vector3, force: R::Vector3) -> Self {
        Self { moment, force }
    }

    /// Returns the zero force vector.
    pub fn zeros() -> Self {
        Self::new(R::vector3_zero(), R::vector3_zero())
    }

    /// Builds a force vector from moment-before-force coordinates.
    pub fn from_vector<V>(vector: V) -> Self
    where
        V: Borrow<R::Vector6>,
    {
        Self::from_array(R::vector6_to_array(vector.borrow()))
    }

    /// Builds a force vector from backend-independent moment-before-force coordinates.
    pub fn from_array(vector: [T; 6]) -> Self {
        Self::new(
            R::vector3_from_array([vector[0], vector[1], vector[2]]),
            R::vector3_from_array([vector[3], vector[4], vector[5]]),
        )
    }

    /// Converts this force vector to moment-before-force coordinates.
    pub fn to_vector(&self) -> SpatialCoordinates<T, R> {
        R::vector6_from_array(self.to_array())
    }

    /// Returns backend-independent moment-before-force coordinates.
    pub fn to_array(&self) -> [T; 6] {
        let moment = R::vector3_to_array(&self.moment);
        let force = R::vector3_to_array(&self.force);
        [
            moment[0], moment[1], moment[2], force[0], force[1], force[2],
        ]
    }

    /// Returns whether all components are finite.
    pub fn is_finite(&self) -> bool {
        R::vector3_is_finite(&self.moment) && R::vector3_is_finite(&self.force)
    }

    /// Computes the dual scalar product with a motion vector.
    ///
    /// `self.moment·motion.angular + self.force·motion.linear`, the same product as
    /// [`MotionVector::dot`] with the operands swapped.
    pub fn dot(&self, motion: &MotionVector<T, R>) -> T {
        motion.dot(self)
    }
}

macro_rules! impl_vector_arithmetic {
    ($vector:ident, $first:ident, $second:ident) => {
        impl<T, R> Add for $vector<T, R>
        where
            T: SpatialScalar,
            R: SpatialRepresentation<T>,
        {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self::new(
                    R::vector3_add(&self.$first, &rhs.$first),
                    R::vector3_add(&self.$second, &rhs.$second),
                )
            }
        }

        impl<T, R> AddAssign for $vector<T, R>
        where
            T: SpatialScalar,
            R: SpatialRepresentation<T>,
        {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl<T, R> Sub for $vector<T, R>
        where
            T: SpatialScalar,
            R: SpatialRepresentation<T>,
        {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                Self::new(
                    R::vector3_sub(&self.$first, &rhs.$first),
                    R::vector3_sub(&self.$second, &rhs.$second),
                )
            }
        }

        impl<T, R> SubAssign for $vector<T, R>
        where
            T: SpatialScalar,
            R: SpatialRepresentation<T>,
        {
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl<T, R> Neg for $vector<T, R>
        where
            T: SpatialScalar,
            R: SpatialRepresentation<T>,
        {
            type Output = Self;

            fn neg(self) -> Self::Output {
                Self::new(R::vector3_neg(&self.$first), R::vector3_neg(&self.$second))
            }
        }

        impl<T, R> Mul<T> for $vector<T, R>
        where
            T: SpatialScalar,
            R: SpatialRepresentation<T>,
        {
            type Output = Self;

            fn mul(self, rhs: T) -> Self::Output {
                Self::new(
                    R::vector3_scale(&self.$first, rhs),
                    R::vector3_scale(&self.$second, rhs),
                )
            }
        }

        impl<T, R> MulAssign<T> for $vector<T, R>
        where
            T: SpatialScalar,
            R: SpatialRepresentation<T>,
        {
            fn mul_assign(&mut self, rhs: T) {
                *self = *self * rhs;
            }
        }

        impl<T, R> Div<T> for $vector<T, R>
        where
            T: SpatialScalar,
            R: SpatialRepresentation<T>,
        {
            type Output = Self;

            fn div(self, rhs: T) -> Self::Output {
                self * (T::one() / rhs)
            }
        }

        impl<T, R> DivAssign<T> for $vector<T, R>
        where
            T: SpatialScalar,
            R: SpatialRepresentation<T>,
        {
            fn div_assign(&mut self, rhs: T) {
                *self = *self / rhs;
            }
        }
    };
}

impl_vector_arithmetic!(MotionVector, angular, linear);
impl_vector_arithmetic!(ForceVector, moment, force);

impl<T, R> Mul<ForceVector<T, R>> for MotionVector<T, R>
where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
{
    type Output = T;

    fn mul(self, rhs: ForceVector<T, R>) -> Self::Output {
        self.dot(&rhs)
    }
}

impl<T, R> Mul<MotionVector<T, R>> for ForceVector<T, R>
where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
{
    type Output = T;

    fn mul(self, rhs: MotionVector<T, R>) -> Self::Output {
        self.dot(&rhs)
    }
}

impl<R> Mul<MotionVector<f64, R>> for f64
where
    R: SpatialRepresentation<f64>,
{
    type Output = MotionVector<f64, R>;

    fn mul(self, rhs: MotionVector<f64, R>) -> Self::Output {
        rhs * self
    }
}

impl<R> Mul<MotionVector<f32, R>> for f32
where
    R: SpatialRepresentation<f32>,
{
    type Output = MotionVector<f32, R>;

    fn mul(self, rhs: MotionVector<f32, R>) -> Self::Output {
        rhs * self
    }
}

impl<R> Mul<ForceVector<f64, R>> for f64
where
    R: SpatialRepresentation<f64>,
{
    type Output = ForceVector<f64, R>;

    fn mul(self, rhs: ForceVector<f64, R>) -> Self::Output {
        rhs * self
    }
}

impl<R> Mul<ForceVector<f32, R>> for f32
where
    R: SpatialRepresentation<f32>,
{
    type Output = ForceVector<f32, R>;

    fn mul(self, rhs: ForceVector<f32, R>) -> Self::Output {
        rhs * self
    }
}
