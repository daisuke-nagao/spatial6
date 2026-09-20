// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_std]

//! Backend-agnostic spatial vectors, motion subspaces, transforms, and inertias.
//!
//! The default `builtin` feature uses fixed Rust arrays. Disable default features
//! to provide your own [`SpatialRepresentation`], or enable `nalgebra` and/or
//! `glam` to use `Nalgebra` and `Glam` as the second public type parameter,
//! for example `MotionVector<f64, Nalgebra>`.
//!
//! The library source is always `no_std`. The default `std` feature forwards
//! std support to enabled numeric dependencies. Disable defaults for bare-metal
//! use; floating-point fallback support is enabled automatically. With supplied
//! backends and `f32`/`f64`, verified operations require no global allocator.
//! Custom implementations and serialization adapters control their own resource
//! requirements. This crate provides no panic handler or panic-free guarantee.
//!
//! Fixed-storage motion/force pairing with the `builtin` feature:
//!
//! ```
//! # #[cfg(feature = "builtin")]
//! # {
//! use spatial6::{Builtin, ForceVector, MotionVector};
//!
//! let motion = MotionVector::<f64, Builtin>::new([0.0, 0.0, 1.0], [2.0, 0.0, 0.0]);
//! let force = ForceVector::<f64, Builtin>::new([0.0, 0.0, 3.0], [4.0, 0.0, 0.0]);
//! assert_eq!(motion.dot(&force), 11.0);
//! # }
//! ```

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

mod subspace;

pub use subspace::MotionSubspace;

mod transform;

pub use transform::SpatialTransform;

mod inertia;

pub use inertia::{ArticulatedBodyInertia, InertiaError, RigidBodyInertia, SpatialInertia};
