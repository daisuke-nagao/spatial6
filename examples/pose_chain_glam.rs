// SPDX-License-Identifier: MIT OR Apache-2.0

//! Extracts and rebuilds local-to-reference poses through a non-commuting chain.
//! Each transform maps its reference/source frame to its local/destination frame.

use glam::{DMat3, DVec3};
use spatial6::{Glam, SpatialTransform};

fn main() {
    let x_ab = SpatialTransform::<f64, Glam>::new(
        DMat3::from_cols(
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        ),
        DVec3::new(1.0, 2.0, 3.0),
    );
    let x_bc = SpatialTransform::<f64, Glam>::new(
        DMat3::from_cols(
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(0.0, -1.0, 0.0),
        ),
        DVec3::new(4.0, 5.0, 6.0),
    );

    let mut chain = SpatialTransform::<f64, Glam>::identity();
    for (local_frame, relative) in [("B", &x_ab), ("C", &x_bc)] {
        chain = chain.then(relative);
        let pose = chain.to_pose_parts();
        println!(
            "{local_frame} relative to A: Q={:?}, p={:?}",
            pose.0, pose.1
        );
        if local_frame == "B" {
            assert_eq!(
                pose,
                (
                    DMat3::from_cols(
                        DVec3::new(0.0, -1.0, 0.0),
                        DVec3::new(1.0, 0.0, 0.0),
                        DVec3::new(0.0, 0.0, 1.0)
                    ),
                    DVec3::new(1.0, 2.0, 3.0)
                )
            );
        } else {
            assert_eq!(
                pose,
                (
                    DMat3::from_cols(
                        DVec3::new(0.0, -1.0, 0.0),
                        DVec3::new(0.0, 0.0, -1.0),
                        DVec3::new(1.0, 0.0, 0.0)
                    ),
                    DVec3::new(6.0, -2.0, 9.0)
                )
            );
        }
    }

    let (q_ac, p_ac) = chain.to_pose_parts();
    assert_eq!(
        SpatialTransform::<f64, Glam>::from_pose_parts(q_ac, p_ac),
        chain
    );
    assert_eq!(
        chain.inverse().to_pose_parts(),
        (
            DMat3::from_cols(
                DVec3::new(0.0, 0.0, 1.0),
                DVec3::new(-1.0, 0.0, 0.0),
                DVec3::new(0.0, -1.0, 0.0)
            ),
            DVec3::new(-2.0, 9.0, -6.0)
        ),
    );
}
