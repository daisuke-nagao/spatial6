// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend-agnostic spatial vectors, transforms, and inertias.
//!
//! The default `builtin` feature uses fixed Rust arrays. Disable default features
//! to provide your own [`SpatialRepresentation`], or enable `nalgebra` and/or
//! `glam` to use `Nalgebra` and `Glam` as the second public type parameter,
//! for example `MotionVector<f64, Nalgebra>`.

mod math;
mod representation;

#[cfg(feature = "builtin")]
pub use representation::Builtin;
#[cfg(feature = "glam")]
pub use representation::Glam;
#[cfg(feature = "nalgebra")]
pub use representation::Nalgebra;
pub use representation::{SpatialRepresentation, SpatialScalar};

mod vector;

pub use vector::{ForceVector, MotionVector, SpatialCoordinates, SpatialMatrix};

mod transform;

pub use transform::SpatialTransform;

mod inertia;

pub use inertia::{ArticulatedBodyInertia, InertiaError, RigidBodyInertia, SpatialInertia};
