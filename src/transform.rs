// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "builtin")]
use crate::Builtin;
use crate::vector::{ForceVector, MotionVector, SpatialMatrix};
use crate::{SpatialRepresentation, SpatialScalar};

macro_rules! define_spatial_transform {
    ($($generics:tt)*) => {
        /// A compact spatial coordinate transform from a source frame to a destination frame.
        ///
        /// Applying the transform (see [`Self::transform_motion`]/[`Self::transform_force`])
        /// re-expresses a vector given in source-frame coordinates as the same physical
        /// quantity in destination-frame coordinates.
        ///
        /// The translation is the vector from the source origin to the destination origin,
        /// expressed in source coordinates. If `E` is the stored rotation and `r` is the
        /// stored translation, the motion and force matrices are respectively
        /// `[[E, 0], [-E[r]×, E]]` and `[[E, -E[r]×], [0, E]]`.
        ///
        /// `rotation` must be a proper orthogonal rotation, per the invariant on
        /// [`SpatialRepresentation::Rotation3`]; this is not validated.
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(
            feature = "serde",
            serde(bound(
                serialize = "R::Rotation3: serde::Serialize, R::Vector3: serde::Serialize",
                deserialize = "R::Rotation3: serde::Deserialize<'de>, R::Vector3: serde::Deserialize<'de>"
            ))
        )]
        pub struct SpatialTransform<$($generics)*> {
            rotation: R::Rotation3,
            translation: R::Vector3,
        }
    };
}

#[cfg(feature = "builtin")]
define_spatial_transform!(T: SpatialScalar = f64, R: SpatialRepresentation<T> = Builtin);
#[cfg(not(feature = "builtin"))]
define_spatial_transform!(T: SpatialScalar, R: SpatialRepresentation<T>);

impl<T, R> SpatialTransform<T, R>
where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
{
    /// Creates a transform from a source frame to a destination frame.
    ///
    /// `rotation` is the source-to-destination rotation and `translation` is the
    /// source-coordinate vector from the source origin to the destination origin.
    /// Neither is validated; see the struct-level invariant on `rotation`.
    pub fn new(rotation: R::Rotation3, translation: R::Vector3) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Returns the identity transform.
    pub fn identity() -> Self {
        Self::new(R::rotation3_identity(), R::vector3_zero())
    }

    /// Returns the source-to-destination rotation.
    pub fn rotation(&self) -> &R::Rotation3 {
        &self.rotation
    }

    /// Returns the source-coordinate vector from the source origin to the destination origin.
    pub fn translation(&self) -> &R::Vector3 {
        &self.translation
    }

    /// Re-expresses a source-frame motion vector in the destination frame.
    ///
    /// Equivalent to `self.motion_matrix() * motion` but computed directly,
    /// without materializing the matrix.
    pub fn transform_motion(&self, motion: &MotionVector<T, R>) -> MotionVector<T, R> {
        MotionVector::new(
            R::rotation3_transform_vector(&self.rotation, &motion.angular),
            R::rotation3_transform_vector(
                &self.rotation,
                &R::vector3_sub(
                    &motion.linear,
                    &R::vector3_cross(&self.translation, &motion.angular),
                ),
            ),
        )
    }

    /// Re-expresses a source-frame force vector in the destination frame.
    ///
    /// Equivalent to `self.force_matrix() * force` but computed directly,
    /// without materializing the matrix. This is the dual of
    /// [`Self::transform_motion`]: it preserves the motion/force dot product,
    /// i.e. `self.transform_motion(m).dot(&self.transform_force(f)) == m.dot(&f)`.
    pub fn transform_force(&self, force: &ForceVector<T, R>) -> ForceVector<T, R> {
        ForceVector::new(
            R::rotation3_transform_vector(
                &self.rotation,
                &R::vector3_sub(
                    &force.moment,
                    &R::vector3_cross(&self.translation, &force.force),
                ),
            ),
            R::rotation3_transform_vector(&self.rotation, &force.force),
        )
    }

    /// Returns the 6x6 matrix form of [`Self::transform_motion`].
    pub fn motion_matrix(&self) -> SpatialMatrix<T, R> {
        let rotation = R::rotation3_to_matrix3(&self.rotation);
        let zero = R::matrix3_zero();
        let rotation_cross = R::matrix3_mul(&rotation, &R::vector3_cross_matrix(&self.translation));
        let lower_left = R::matrix3_sub(&zero, &rotation_cross);
        R::matrix6_from_blocks(&rotation, &zero, &lower_left, &rotation)
    }

    /// Returns the 6x6 matrix form of [`Self::transform_force`].
    pub fn force_matrix(&self) -> SpatialMatrix<T, R> {
        let rotation = R::rotation3_to_matrix3(&self.rotation);
        let zero = R::matrix3_zero();
        let rotation_cross = R::matrix3_mul(&rotation, &R::vector3_cross_matrix(&self.translation));
        let upper_right = R::matrix3_sub(&zero, &rotation_cross);
        R::matrix6_from_blocks(&rotation, &upper_right, &zero, &rotation)
    }

    /// Returns the transform from the destination frame back to the source frame.
    ///
    /// `self.inverse().transform_motion(&self.transform_motion(&m)) == m` for any
    /// motion vector `m` (and likewise for [`Self::transform_force`]), up to
    /// floating-point rounding.
    pub fn inverse(&self) -> Self {
        let rotation = R::rotation3_inverse(&self.rotation);
        let translation = R::vector3_neg(&R::rotation3_transform_vector(
            &self.rotation,
            &self.translation,
        ));
        Self::new(rotation, translation)
    }

    /// Composes this transform with `next`, applying `self` and then `next`.
    ///
    /// `next`'s source frame must be `self`'s destination frame; the result maps
    /// `self`'s source frame directly to `next`'s destination frame:
    /// `self.then(next).transform_motion(&m) == next.transform_motion(&self.transform_motion(&m))`.
    pub fn then(&self, next: &Self) -> Self {
        let rotation = R::rotation3_inverse(&self.rotation);
        Self::new(
            R::rotation3_mul(&next.rotation, &self.rotation),
            R::vector3_add(
                &self.translation,
                &R::rotation3_transform_vector(&rotation, &next.translation),
            ),
        )
    }
}
