//! `spatial6` is backend-agnostic: the same rigid-body inverse-dynamics
//! computation produces identical numbers whether spatial quantities are
//! stored as fixed Rust arrays (`Builtin`, `examples/backend_builtin.rs`),
//! native `nalgebra` types (`examples/backend_nalgebra.rs`), or native `glam`
//! types (`Glam`, this file). Run all three and diff the output:
//!
//! ```sh
//! cargo run --example backend_builtin --no-default-features --features builtin
//! cargo run --example backend_nalgebra --no-default-features --features nalgebra
//! cargo run --example backend_glam --no-default-features --features glam
//! ```

use glam::{DMat3, DVec3};
use spatial6::{Glam, MotionVector, RigidBodyInertia};

fn main() {
    // A single rigid link: 2 kg, center of mass offset along x, isotropic
    // rotational inertia about that center of mass.
    let inertia = RigidBodyInertia::<f64, Glam>::try_new(
        2.0,
        DVec3::new(0.5, 0.0, 0.0),
        DMat3::from_cols(
            DVec3::new(0.2, 0.0, 0.0),
            DVec3::new(0.0, 0.2, 0.0),
            DVec3::new(0.0, 0.0, 0.2),
        ),
    )
    .unwrap();

    let velocity =
        MotionVector::<f64, Glam>::new(DVec3::new(0.2, -0.1, 0.4), DVec3::new(0.1, 0.3, -0.2));
    let acceleration =
        MotionVector::<f64, Glam>::new(DVec3::new(0.05, 0.1, -0.05), DVec3::new(1.0, 0.0, -0.5));

    let force = inertia.inverse_dynamics(&velocity, &acceleration);

    println!("backend: glam (native glam types)");
    for (label, value) in ["mx", "my", "mz", "fx", "fy", "fz"]
        .into_iter()
        .zip(force.to_array())
    {
        println!("  {label} = {value:.6}");
    }
}
