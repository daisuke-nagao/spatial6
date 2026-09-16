// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{ForceVector, MotionVector, SpatialRepresentation, SpatialScalar, SpatialTransform};

#[cfg(feature = "builtin")]
use crate::Builtin;

macro_rules! define_motion_subspace {
    ($($generics:tt)*) => {
        /// A fixed-size motion subspace whose columns are spatial motion vectors.
        ///
        /// This represents the linear map `S: R^N -> M^6`. `N` is the generalized
        /// velocity dimension, not necessarily the configuration dimension. All
        /// columns, and any force, inertia, or transform used with them, must be
        /// expressed in consistent frames.
        ///
        /// This type supplies multi-DoF motion and joint-space projection
        /// primitives, including `S x` and `S^T f`; together with an
        /// articulated-body inertia, `I_A S` supplies `U`. It does not solve or
        /// invert the joint-space matrix `D = S^T I_A S`, nor apply the complete
        /// multi-DoF reduction `I_A - U D^{-1} U^T`.
        ///
        /// `N > 6` is allowed for redundant or purely algebraic representations.
        /// Because spatial motion has dimension six, `S` has rank at most six, so the
        /// projected inertia `D` is necessarily singular for `N > 6`.
        /// Algorithms requiring an invertible projected inertia must impose and
        /// validate suitable rank and conditioning constraints for any `N`.
        ///
        /// The dual operation preserves virtual power:
        /// `(S x) · f = x^T (S^T f)`.
        ///
        /// The fixed size does not make the value state-independent; callers may
        /// construct a different `S(q)` for each configuration.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct MotionSubspace<$($generics)*> {
            columns: [MotionVector<T, R>; N],
        }
    };
}

#[cfg(feature = "builtin")]
define_motion_subspace!(const N: usize, T: SpatialScalar = f64, R: SpatialRepresentation<T> = Builtin);
#[cfg(not(feature = "builtin"))]
define_motion_subspace!(const N: usize, T: SpatialScalar, R: SpatialRepresentation<T>);

impl<const N: usize, T, R> MotionSubspace<N, T, R>
where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
{
    /// Creates a motion subspace from its motion columns.
    ///
    /// Algorithms that require `N` independent generalized velocities assume
    /// that the supplied columns are linearly independent.
    pub fn from_columns(columns: [MotionVector<T, R>; N]) -> Self {
        Self { columns }
    }

    /// Returns the motion columns.
    pub fn columns(&self) -> &[MotionVector<T, R>; N] {
        &self.columns
    }

    /// Returns whether every column contains only finite components.
    pub fn is_finite(&self) -> bool {
        self.columns.iter().all(MotionVector::is_finite)
    }

    /// Applies the subspace to generalized-velocity coefficients: `S x`.
    pub fn apply(&self, coefficients: &[T; N]) -> MotionVector<T, R> {
        self.columns
            .iter()
            .zip(coefficients)
            .fold(MotionVector::zeros(), |sum, (column, coefficient)| {
                sum + *column * *coefficient
            })
    }

    /// Maps a spatial force to generalized force: `S^T f`.
    pub fn generalized_force(&self, force: &ForceVector<T, R>) -> [T; N] {
        std::array::from_fn(|index| self.columns[index].dot(force))
    }

    /// Maps spatial-force columns to generalized forces: `S^T F`.
    pub fn generalized_forces<const M: usize>(
        &self,
        forces: &[ForceVector<T, R>; M],
    ) -> [[T; M]; N] {
        std::array::from_fn(|row| {
            std::array::from_fn(|column| self.columns[row].dot(&forces[column]))
        })
    }

    /// Applies a spatial motion transform to every column: `X S`.
    pub fn transformed(&self, transform: &SpatialTransform<T, R>) -> Self {
        Self::from_columns(std::array::from_fn(|index| {
            transform.transform_motion(&self.columns[index])
        }))
    }
}

impl<T, R> From<MotionVector<T, R>> for MotionSubspace<1, T, R>
where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
{
    fn from(column: MotionVector<T, R>) -> Self {
        Self::from_columns([column])
    }
}

#[cfg(feature = "serde")]
impl<const N: usize, T, R> serde::Serialize for MotionSubspace<N, T, R>
where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
    MotionVector<T, R>: serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeTuple;

        let mut tuple = serializer.serialize_tuple(N)?;
        for column in &self.columns {
            tuple.serialize_element(column)?;
        }
        tuple.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, const N: usize, T, R> serde::Deserialize<'de> for MotionSubspace<N, T, R>
where
    T: SpatialScalar,
    R: SpatialRepresentation<T>,
    MotionVector<T, R>: serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ColumnsVisitor<const N: usize, T, R>(std::marker::PhantomData<(T, R)>);

        impl<'de, const N: usize, T, R> serde::de::Visitor<'de> for ColumnsVisitor<N, T, R>
        where
            T: SpatialScalar,
            R: SpatialRepresentation<T>,
            MotionVector<T, R>: serde::Deserialize<'de>,
        {
            type Value = MotionSubspace<N, T, R>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "exactly {N} motion-subspace columns")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut columns: [Option<MotionVector<T, R>>; N] = std::array::from_fn(|_| None);
                for (index, column) in columns.iter_mut().enumerate() {
                    *column = Some(
                        sequence
                            .next_element()?
                            .ok_or_else(|| serde::de::Error::invalid_length(index, &self))?,
                    );
                }
                if sequence.next_element::<serde::de::IgnoredAny>()?.is_some() {
                    return Err(serde::de::Error::invalid_length(N + 1, &self));
                }

                Ok(MotionSubspace::from_columns(std::array::from_fn(|index| {
                    columns[index]
                        .take()
                        .expect("every column was filled above")
                })))
            }
        }

        deserializer.deserialize_tuple(N, ColumnsVisitor(std::marker::PhantomData))
    }
}
