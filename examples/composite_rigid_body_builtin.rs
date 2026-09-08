//! Chapter 6, Section 6.3 (Featherstone, *Rigid Body Dynamics Algorithms*):
//! the physical interpretation behind the Composite Rigid Body Algorithm
//! (CRBA) — a subtree of links accelerating together as one rigid body has a
//! spatial inertia equal to the simple spatial sum of its members' own
//! inertias, once every member is expressed about the same point.

use spatial6::{MotionVector, SpatialInertia, SpatialTransform};

fn main() {
    // Two links in a subtree, each with its own mass and inertia. `offset`
    // is the transform from link 2's own reference point to link 1's
    // reference point O (translation r = O's position relative to link 2's
    // point, no relative rotation — they move together as one body).
    let inertia_1 = SpatialInertia::<f64>::try_new(
        2.0,
        [0.5, 0.0, 0.0],
        [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
    )
    .unwrap();
    let offset = SpatialTransform::<f64>::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [1.0, 0.0, 0.0],
    );
    let inertia_2_at_own_point = SpatialInertia::<f64>::try_new(
        1.0,
        [0.3, 0.0, 0.0],
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
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
    let velocity = MotionVector::<f64>::new([0.2, -0.1, 0.4], [0.1, 0.3, -0.2]);
    let acceleration = MotionVector::<f64>::new([0.05, 0.1, -0.05], [1.0, 0.0, -0.5]);

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
