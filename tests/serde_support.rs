// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "serde")]

use spatial6::{ForceVector, MotionVector, RigidBodyInertia, SpatialTransform};

fn round_trip<T>(value: &T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let json = serde_json::to_string(value).expect("serialize");
    serde_json::from_str(&json).expect("deserialize")
}

#[test]
fn motion_vector_round_trips() {
    let motion = MotionVector::<f64>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    assert_eq!(round_trip(&motion), motion);
}

#[test]
fn force_vector_round_trips() {
    let force = ForceVector::<f64>::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    assert_eq!(round_trip(&force), force);
}

#[test]
fn spatial_transform_round_trips() {
    let transform = SpatialTransform::<f64>::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [1.0, 2.0, 3.0],
    );
    assert_eq!(round_trip(&transform), transform);
}

#[test]
fn rigid_body_inertia_round_trips() {
    let inertia = RigidBodyInertia::<f64>::try_new(
        2.0,
        [0.1, 0.2, 0.3],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    .unwrap();
    assert_eq!(round_trip(&inertia), inertia);
}

#[cfg(feature = "nalgebra")]
#[test]
fn nalgebra_backend_motion_vector_round_trips() {
    use spatial6::Nalgebra;

    let motion = MotionVector::<f64, Nalgebra>::new(
        nalgebra::Vector3::new(1.0, 2.0, 3.0),
        nalgebra::Vector3::new(4.0, 5.0, 6.0),
    );
    assert_eq!(round_trip(&motion), motion);
}

#[cfg(feature = "glam")]
#[test]
fn glam_backend_motion_vector_round_trips() {
    use spatial6::Glam;

    let motion = MotionVector::<f64, Glam>::new(
        glam::DVec3::new(1.0, 2.0, 3.0),
        glam::DVec3::new(4.0, 5.0, 6.0),
    );
    assert_eq!(round_trip(&motion), motion);
}

#[cfg(feature = "nalgebra")]
#[test]
fn nalgebra_backend_spatial_transform_round_trips() {
    use spatial6::Nalgebra;

    let transform = SpatialTransform::<f64, Nalgebra>::new(
        nalgebra::Rotation3::identity(),
        nalgebra::Vector3::new(1.0, 2.0, 3.0),
    );
    assert_eq!(round_trip(&transform), transform);
}

#[cfg(feature = "glam")]
#[test]
fn glam_backend_spatial_transform_round_trips() {
    use spatial6::Glam;

    let transform =
        SpatialTransform::<f64, Glam>::new(glam::DMat3::IDENTITY, glam::DVec3::new(1.0, 2.0, 3.0));
    assert_eq!(round_trip(&transform), transform);
}

#[test]
fn rigid_body_inertia_deserialize_rejects_invalid_data() {
    // Hand-crafted, untrusted JSON with a negative mass: must not silently
    // bypass `RigidBodyInertia::try_new`'s validation.
    let json = r#"{
        "mass": -5.0,
        "center_of_mass": [0.0, 0.0, 0.0],
        "inertia_at_center_of_mass": [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    }"#;
    let result: Result<RigidBodyInertia<f64>, _> = serde_json::from_str(json);
    assert!(result.is_err(), "negative mass must be rejected");
}
