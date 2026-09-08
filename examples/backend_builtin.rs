//! `spatial6` is backend-agnostic: the same rigid-body inverse-dynamics
//! computation produces identical numbers whether spatial quantities are
//! stored as fixed Rust arrays (`Builtin`, this file), native `nalgebra`
//! types (`examples/backend_nalgebra.rs`), or native `glam` types
//! (`examples/backend_glam.rs`). Run all three and diff the output:
//!
//! ```sh
//! cargo run --example backend_builtin --no-default-features --features builtin
//! cargo run --example backend_nalgebra --no-default-features --features nalgebra
//! cargo run --example backend_glam --no-default-features --features glam
//! ```

use spatial6::{MotionVector, SpatialInertia};

fn main() {
    // A single rigid link: 2 kg, center of mass offset along x, isotropic
    // rotational inertia about that center of mass.
    let inertia = SpatialInertia::<f64>::try_new(
        2.0,
        [0.5, 0.0, 0.0],
        [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
    )
    .unwrap();

    let velocity = MotionVector::<f64>::new([0.2, -0.1, 0.4], [0.1, 0.3, -0.2]);
    let acceleration = MotionVector::<f64>::new([0.05, 0.1, -0.05], [1.0, 0.0, -0.5]);

    let force = inertia.inverse_dynamics(&velocity, &acceleration);

    println!("backend: builtin (fixed Rust arrays)");
    for (label, value) in ["mx", "my", "mz", "fx", "fy", "fz"]
        .into_iter()
        .zip(force.to_array())
    {
        println!("  {label} = {value:.6}");
    }
}
