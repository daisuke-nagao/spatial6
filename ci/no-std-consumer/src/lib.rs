#![no_std]

mod cases;
mod custom;

#[cfg(feature = "serde")]
mod serde_cases;

pub use cases::{run_all, run_array_f32, run_array_f64};

#[cfg(feature = "builtin")]
pub use cases::{run_builtin_f32, run_builtin_f64};

#[cfg(feature = "nalgebra")]
pub use cases::{run_nalgebra_f32, run_nalgebra_f64};

#[cfg(feature = "glam")]
pub use cases::{run_glam_f32, run_glam_f64};

#[cfg(feature = "serde")]
pub use cases::run_serde;

#[cfg(all(feature = "serde", feature = "builtin"))]
pub use cases::run_serde_builtin;

#[cfg(all(feature = "serde", feature = "nalgebra"))]
pub use cases::run_serde_nalgebra;

#[cfg(all(feature = "serde", feature = "glam"))]
pub use cases::run_serde_glam;
