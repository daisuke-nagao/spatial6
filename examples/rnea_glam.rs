//! Chapter 5 (Featherstone, *Rigid Body Dynamics Algorithms*): the Recursive
//! Newton-Euler Algorithm (RNEA), computing the joint torques needed to
//! produce a given motion for a 3-link serial chain of revolute joints, with
//! gravity folded into the outward pass via the `a_0 = -a_g` trick.
//!
//! This is the same scenario as `examples/rnea_builtin.rs`, rebuilt from native
//! `glam` types (`Glam`) instead of plain arrays (`Builtin`), so it must
//! print numerically identical torques. See also `examples/rnea_nalgebra.rs`
//! for the `nalgebra` backend.

use glam::{DMat3, DVec3};
use spatial6::{ForceVector, Glam, MotionVector, SpatialInertia, SpatialTransform};

const NUM_LINKS: usize = 3;

fn main() {
    let link_length = 1.0;
    let mass = 1.0;
    let rotational_inertia = DMat3::from_cols(
        DVec3::new(0.1, 0.0, 0.0),
        DVec3::new(0.0, 0.1, 0.0),
        DVec3::new(0.0, 0.0, 0.1),
    );
    let gravity = DVec3::new(0.0, 0.0, -9.81);

    let joint_angle: [f64; NUM_LINKS] = [0.3, -0.2, 0.5];
    let joint_rate = [0.4, -0.3, 0.2];
    let joint_accel = [0.1, 0.0, -0.1];

    // Joint axis is the local z-axis; only angular motion, no offset.
    let joint_axis =
        MotionVector::<f64, Glam>::new(DVec3::new(0.0, 0.0, 1.0), DVec3::new(0.0, 0.0, 0.0));

    // {}^i X_{i-1}: joint rotation about z, composed with a fixed offset of
    // `link_length` along the parent's x-axis. The row-major rotation is
    // [[cos,-sin,0],[sin,cos,0],[0,0,1]]; `DMat3::from_cols` takes columns
    // (glam is column-major internally), which are that matrix's transpose:
    // (cos,sin,0), (-sin,cos,0), (0,0,1).
    let transforms: Vec<SpatialTransform<f64, Glam>> = joint_angle
        .iter()
        .map(|&angle| {
            let (sin, cos) = angle.sin_cos();
            SpatialTransform::<f64, Glam>::new(
                DMat3::from_cols(
                    DVec3::new(cos, sin, 0.0),
                    DVec3::new(-sin, cos, 0.0),
                    DVec3::new(0.0, 0.0, 1.0),
                ),
                DVec3::new(link_length, 0.0, 0.0),
            )
        })
        .collect();

    // Each link's center of mass sits at its own midpoint, not at the joint
    // origin, so gravity actually exerts a moment about each joint.
    let com = DVec3::new(link_length / 2.0, 0.0, 0.0);
    let inertia: Vec<SpatialInertia<f64, Glam>> = (0..NUM_LINKS)
        .map(|_| SpatialInertia::<f64, Glam>::try_new(mass, com, rotational_inertia).unwrap())
        .collect();

    let mut velocity = [MotionVector::<f64, Glam>::zeros(); NUM_LINKS + 1];
    let mut acceleration = [MotionVector::<f64, Glam>::zeros(); NUM_LINKS + 1];
    // The base "accelerates" upward by -g, so every descendant link picks up
    // the equivalent of a downward gravitational force with no per-link
    // external-force term.
    acceleration[0] = MotionVector::<f64, Glam>::new(
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(0.0, 0.0, -gravity[2]),
    );

    let mut force = [ForceVector::<f64, Glam>::zeros(); NUM_LINKS + 1];

    // Outward pass: velocity, acceleration, and each link's own net force.
    for i in 1..=NUM_LINKS {
        let joint_velocity = joint_axis * joint_rate[i - 1];
        let joint_acceleration = joint_axis * joint_accel[i - 1];

        velocity[i] = transforms[i - 1].transform_motion(&velocity[i - 1]) + joint_velocity;
        acceleration[i] = transforms[i - 1].transform_motion(&acceleration[i - 1])
            + joint_acceleration
            + velocity[i].cross_motion(&joint_velocity);
        force[i] = inertia[i - 1].inverse_dynamics(&velocity[i], &acceleration[i]);
    }

    // Inward pass: read off each joint's torque, then push the force this
    // link transmits back into its parent's accumulator.
    let mut torque = [0.0; NUM_LINKS];
    for i in (1..=NUM_LINKS).rev() {
        torque[i - 1] = joint_axis.dot(&force[i]);

        if i > 1 {
            force[i - 1] += transforms[i - 1].inverse().transform_force(&force[i]);
        }
    }

    for (i, value) in torque.iter().enumerate() {
        println!("joint {} torque: {value:.6}", i + 1);
    }
}
