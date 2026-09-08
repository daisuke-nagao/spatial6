//! Chapter 6, Section 6.3 (Featherstone, *Rigid Body Dynamics Algorithms*):
//! the physical interpretation behind the Composite Rigid Body Algorithm
//! (CRBA) — a subtree of links accelerating together as one rigid body has a
//! spatial inertia equal to the simple spatial sum of its members' own
//! inertias, once every member is expressed about the same point.
//!
//! This is the same scenario as `examples/composite_rigid_body_builtin.rs`, rebuilt
//! from native `glam` types (`Glam`) instead of plain arrays (`Builtin`), so
//! it must print numerically identical forces. See also
//! `examples/composite_rigid_body_nalgebra.rs` for the `nalgebra` backend.

use glam::{DMat3, DVec3};
use spatial6::{Glam, MotionVector, SpatialInertia, SpatialTransform};

fn main() {
    // Two links in a subtree, each with its own mass and inertia. `offset`
    // is the transform from link 2's own reference point to link 1's
    // reference point O (translation r = O's position relative to link 2's
    // point, no relative rotation — they move together as one body).
    //
    // glam stores `DMat3` column-major, but this crate treats every matrix
    // as semantically row-major, so each explicit matrix below is built with
    // `from_cols` from the *columns* of its row-major layout, not the rows
    // verbatim — for these particular diagonal matrices the two
    // happen to coincide, but writing it this way keeps the construction
    // correct in general, not just correct by coincidence.
    let inertia_1 = SpatialInertia::<f64, Glam>::try_new(
        2.0,
        DVec3::new(0.5, 0.0, 0.0),
        DMat3::from_cols(
            DVec3::new(0.2, 0.0, 0.0),
            DVec3::new(0.0, 0.2, 0.0),
            DVec3::new(0.0, 0.0, 0.2),
        ),
    )
    .unwrap();
    let offset = SpatialTransform::<f64, Glam>::new(DMat3::IDENTITY, DVec3::new(1.0, 0.0, 0.0));
    let inertia_2_at_own_point = SpatialInertia::<f64, Glam>::try_new(
        1.0,
        DVec3::new(0.3, 0.0, 0.0),
        DMat3::from_cols(
            DVec3::new(0.1, 0.0, 0.0),
            DVec3::new(0.0, 0.1, 0.0),
            DVec3::new(0.0, 0.0, 0.1),
        ),
    )
    .unwrap();
    // Re-expressed about link 1's reference point O, so both inertias can be
    // added directly (Featherstone requires a common point and frame).
    let inertia_2_at_o = inertia_2_at_own_point.transformed(&offset);

    // The composite body's inertia I^c is the plain spatial sum of its
    // members' inertias about the shared point O.
    let composite = inertia_1.try_combined(&inertia_2_at_o).unwrap();

    // Physical check: if the whole subtree accelerates rigidly together
    // (same spatial velocity/acceleration for every member, since they're
    // now expressed about the same point O), the net force produced by
    // treating it as one composite body must equal the sum of the forces
    // each member produces on its own.
    let velocity =
        MotionVector::<f64, Glam>::new(DVec3::new(0.2, -0.1, 0.4), DVec3::new(0.1, 0.3, -0.2));
    let acceleration =
        MotionVector::<f64, Glam>::new(DVec3::new(0.05, 0.1, -0.05), DVec3::new(1.0, 0.0, -0.5));

    let force_composite = composite.inverse_dynamics(&velocity, &acceleration);
    let force_separate = inertia_1.inverse_dynamics(&velocity, &acceleration)
        + inertia_2_at_o.inverse_dynamics(&velocity, &acceleration);

    println!("composite body's own equation of motion:      f = {force_composite:?}");
    println!("sum of each member's individual contribution: f = {force_separate:?}");

    let max_abs_difference = (force_composite - force_separate)
        .to_array()
        .into_iter()
        .fold(0.0_f64, |max, value| max.max(value.abs()));
    assert!(max_abs_difference < 1.0e-9);

    println!("\nA rigidly-moving subtree's composite inertia I^c = sum_j I_j");
    println!("reproduces the combined effect of every member individually,");
    println!("exactly the physical picture CRBA (Chapter 6) relies on.");
}
