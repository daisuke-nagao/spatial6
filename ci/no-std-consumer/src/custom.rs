use spatial6::{SpatialRepresentation, SpatialScalar};

/// A deliberately boring array-backed representation used in every fixture run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArrayRepresentation;

impl<T: SpatialScalar> SpatialRepresentation<T> for ArrayRepresentation {
    type Vector3 = [T; 3];
    type Vector6 = [T; 6];
    type Matrix3 = [[T; 3]; 3];
    type Matrix6 = [[T; 6]; 6];
    type Rotation3 = [[T; 3]; 3];

    fn vector3_from_array(value: [T; 3]) -> Self::Vector3 {
        value
    }

    fn vector3_to_array(value: &Self::Vector3) -> [T; 3] {
        *value
    }

    fn vector6_from_array(value: [T; 6]) -> Self::Vector6 {
        value
    }

    fn vector6_to_array(value: &Self::Vector6) -> [T; 6] {
        *value
    }

    fn matrix3_from_array(value: [[T; 3]; 3]) -> Self::Matrix3 {
        value
    }

    fn matrix3_to_array(value: &Self::Matrix3) -> [[T; 3]; 3] {
        *value
    }

    fn matrix6_from_array(value: [[T; 6]; 6]) -> Self::Matrix6 {
        value
    }

    fn matrix6_to_array(value: &Self::Matrix6) -> [[T; 6]; 6] {
        *value
    }

    fn rotation3_from_array(value: [[T; 3]; 3]) -> Self::Rotation3 {
        value
    }

    fn rotation3_to_array(value: &Self::Rotation3) -> [[T; 3]; 3] {
        *value
    }
}
