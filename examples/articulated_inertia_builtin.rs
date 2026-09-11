//! Featherstone's articulated-body reduction with the public `spatial6` API.
//!
//! A child link's joint acceleration is eliminated under a specified joint
//! force. The resulting singular spatial operator is transformed into the
//! parent frame and accumulated with the parent's articulated inertia.

use spatial6::{
    ArticulatedBodyInertia, Builtin, ForceVector, MotionVector, SpatialInertia, SpatialTransform,
};

fn max_abs_vector(left: [f64; 6], right: [f64; 6]) -> f64 {
    left.into_iter()
        .zip(right)
        .fold(0.0_f64, |max, (a, b)| max.max((a - b).abs()))
}

fn main() {
    // Keep the same two-link scenario in all three backend examples. Both
    // centers of mass are nonzero; the child is rotated and translated into
    // the parent frame before the subtree inertias are combined.
    let parent_rigid = SpatialInertia::<f64>::try_new(
        4.0,
        [-0.25, 0.2, 0.1],
        [[0.8, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.1]],
    )
    .unwrap();
    let child_rigid = SpatialInertia::<f64>::try_new(
        3.0,
        [0.2, 0.15, -0.1],
        [[0.4, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 0.5]],
    )
    .unwrap();

    let angle: f64 = 0.35;
    let (sin, cos) = angle.sin_cos();
    let child_to_parent = SpatialTransform::<f64>::new(
        [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
        [1.0, -0.4, 0.2],
    );

    let parent: ArticulatedBodyInertia<f64, Builtin> =
        ArticulatedBodyInertia::try_from(&parent_rigid).unwrap();
    let full: ArticulatedBodyInertia<f64, Builtin> =
        ArticulatedBodyInertia::try_from(&child_rigid).unwrap();

    // A prismatic joint has a local-x motion subspace. Eliminating its
    // acceleration while its force is specified gives the ABA rank-one update.
    let s = MotionVector::<f64>::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    let u = full.apply(&s);
    let d = s.dot(&u);
    assert!(d.is_finite() && d > 0.0);
    let reduced = full.try_rank_one_updated(-1.0 / d, &u).unwrap();

    // The eliminated joint direction has no remaining articulated response.
    let eliminated_response = reduced.apply(&s);
    assert!(
        max_abs_vector(
            eliminated_response.to_vector(),
            ForceVector::<f64>::zeros().to_vector(),
        ) < 1.0e-9
    );

    // Propagate the reduced child operator into the parent and accumulate it.
    let child_in_parent = reduced.try_transformed(&child_to_parent).unwrap();
    let accumulated = parent.try_combined(&child_in_parent).unwrap();
    let probe = MotionVector::<f64>::new([0.1, -0.2, 0.3], [0.4, 0.0, -0.1]);
    let propagated = accumulated.apply(&probe);
    let expected = parent.apply(&probe) + child_in_parent.apply(&probe);
    assert!(max_abs_vector(propagated.to_vector(), expected.to_vector()) < 1.0e-9);

    println!("child joint response after acceleration elimination: {eliminated_response:?}");
    println!("parent response after child propagation: {propagated:?}");
    println!("articulated-body reduction and parent propagation succeeded");
}
