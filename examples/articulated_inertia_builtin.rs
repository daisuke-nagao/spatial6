//! Chapter 7 (Featherstone, *Rigid Body Dynamics Algorithms*): the Articulated
//! Body Algorithm (ABA) represents the effective inertia a subtree presents at
//! a joint as an *articulated-body inertia* -- a general symmetric spatial
//! operator that need not be positive definite, unlike a rigid body's own
//! inertia. Locking a joint's motion direction projects that direction out of
//! the operator via a Schur complement, the step ABA repeats at every joint
//! during its inward pass (eq. 7.31-7.34): `I -= (I*s)(I*s)^T / (s^T*I*s)`.
//!
//! `spatial6` only models rigid-body inertia (`RigidBodyInertia`), so this
//! example reimplements the articulated-body inertia locally to demonstrate
//! the concept end to end.

use spatial6::{
    Builtin, ForceVector, InertiaError, MotionVector, SpatialInertia, SpatialRepresentation,
    SpatialTransform,
};

#[allow(clippy::needless_range_loop)]
fn pack_symmetric_matrix(matrix: &[[f64; 6]; 6]) -> [f64; 21] {
    let mut packed = [0.0; 21];
    let mut index = 0;
    for row in 0..6 {
        for column in row..6 {
            packed[index] = if row == column {
                matrix[row][column]
            } else {
                (matrix[row][column] + matrix[column][row]) / 2.0
            };
            index += 1;
        }
    }
    packed
}

#[allow(clippy::needless_range_loop)]
fn unpack_symmetric_matrix(packed: &[f64; 21]) -> [[f64; 6]; 6] {
    let mut matrix = [[0.0; 6]; 6];
    let mut index = 0;
    for row in 0..6 {
        for column in row..6 {
            let value = packed[index];
            matrix[row][column] = value;
            matrix[column][row] = value;
            index += 1;
        }
    }
    matrix
}

/// A symmetric 6x6 spatial inertia matrix that need not be positive definite,
/// stored as its twenty-one independent scalar parameters.
struct ArticulatedInertia {
    packed: [f64; 21],
}

impl ArticulatedInertia {
    /// Validates and creates an articulated inertia from a finite, symmetric matrix.
    #[allow(clippy::needless_range_loop)]
    fn try_from_matrix(matrix: [[f64; 6]; 6]) -> Result<Self, InertiaError> {
        if matrix.iter().flatten().any(|value| !value.is_finite()) {
            return Err(InertiaError::NonFinite);
        }
        for row in 0..6 {
            for column in (row + 1)..6 {
                if (matrix[row][column] - matrix[column][row]).abs() > 1.0e-9 {
                    return Err(InertiaError::NonSymmetric);
                }
            }
        }
        Ok(Self::from_matrix_trusted(matrix))
    }

    /// Packs an already-symmetric matrix without re-validating it, for results
    /// of operations that are symmetric by construction.
    fn from_matrix_trusted(matrix: [[f64; 6]; 6]) -> Self {
        Self {
            packed: pack_symmetric_matrix(&matrix),
        }
    }

    fn matrix(&self) -> [[f64; 6]; 6] {
        unpack_symmetric_matrix(&self.packed)
    }

    fn apply(&self, motion: &MotionVector<f64>) -> ForceVector<f64> {
        ForceVector::from_vector(Builtin::matrix6_vector_mul(
            &self.matrix(),
            &motion.to_vector(),
        ))
    }

    fn solve(&self, force: &ForceVector<f64>) -> Option<MotionVector<f64>> {
        Builtin::matrix6_solve_positive_definite(&self.matrix(), &force.to_vector())
            .map(MotionVector::from_vector)
    }

    fn transformed(&self, transform: &SpatialTransform<f64>) -> Self {
        let force_matrix = transform.force_matrix();
        let inverse_motion = transform.inverse().motion_matrix();
        let transformed = Builtin::matrix6_mul(
            &Builtin::matrix6_mul(&force_matrix, &self.matrix()),
            &inverse_motion,
        );
        Self::from_matrix_trusted(transformed)
    }

    fn time_derivative(&self, velocity: &MotionVector<f64>) -> Self {
        let cross_dual = velocity.cross_dual_matrix();
        let cross_motion = velocity.cross_matrix();
        let inertia = self.matrix();
        let dual_product = Builtin::matrix6_mul(&cross_dual, &inertia);
        let product = Builtin::matrix6_mul(&inertia, &cross_motion);
        Self::from_matrix_trusted(Builtin::matrix6_sub(&dual_product, &product))
    }
}

fn max_abs_difference(left: [[f64; 6]; 6], right: [[f64; 6]; 6]) -> f64 {
    left.into_iter()
        .flatten()
        .zip(right.into_iter().flatten())
        .fold(0.0_f64, |max, (a, b)| max.max((a - b).abs()))
}

fn max_abs_vector(left: [f64; 6], right: [f64; 6]) -> f64 {
    left.into_iter()
        .zip(right)
        .fold(0.0_f64, |max, (a, b)| max.max((a - b).abs()))
}

fn main() {
    // A wheel-like link: mass 3, center of mass off-axis in all three
    // coordinates (so locking one motion direction below couples into more
    // than a single diagonal entry), diagonal rotational inertia about its
    // own frame origin.
    let inertia = SpatialInertia::<f64>::try_new(
        3.0,
        [0.2, 0.15, -0.1],
        [[0.4, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 0.5]],
    )
    .unwrap();

    // Wrapping a rigid body's own (positive-definite) matrix as an
    // articulated inertia changes nothing observable: apply/solve behave
    // exactly as they do on the `RigidBodyInertia` itself, and round-trip.
    let full = ArticulatedInertia::try_from_matrix(inertia.matrix()).unwrap();
    let motion = MotionVector::<f64>::new([0.1, -0.2, 0.3], [0.4, 0.0, -0.1]);
    let force = full.apply(&motion);
    assert!(max_abs_vector(force.to_vector(), inertia.apply(&motion).to_vector()) < 1.0e-9);
    let recovered_motion = full.solve(&force).unwrap();
    assert!(max_abs_vector(recovered_motion.to_vector(), motion.to_vector()) < 1.0e-9);
    println!("an ordinary rigid body's inertia round-trips through apply/solve unchanged");

    // The reimplemented `time_derivative` must agree with the library's own
    // (independently, already-tested) `RigidBodyInertia::time_derivative`
    // when wrapping the exact same matrix -- not just stay symmetric, which
    // any correctly-*signed* formula would do regardless of a transcription
    // bug in argument order.
    assert!(
        max_abs_difference(
            full.time_derivative(&motion).matrix(),
            inertia.time_derivative(&motion)
        ) < 1.0e-9
    );

    // A prismatic (rail) joint locks the body against accelerating along its
    // own local x axis: `s` is that joint's motion subspace vector. ABA's
    // inward pass absorbs a locked direction into the parent by projecting it
    // out of the child's inertia via a Schur complement:
    // I^A = I - (I*s)(I*s)^T / (s^T*I*s) -- eq. 7.31-7.34.
    let s = MotionVector::<f64>::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    let h = inertia.apply(&s);
    let d = s.dot(&h);
    let h_vector = h.to_vector();
    let mut projected = inertia.matrix();
    for row in 0..6 {
        for column in 0..6 {
            projected[row][column] -= h_vector[row] * h_vector[column] / d;
        }
    }
    let locked = ArticulatedInertia::try_from_matrix(projected).unwrap();

    // Pushing along the locked direction now produces no net response: the
    // joint constraint has completely absorbed it. Unlike a rigid body's own
    // inertia, this operator is only positive *semi*-definite: `apply` (a
    // plain matrix-vector product) still works, but `solve`'s Cholesky
    // factorization requires strict positive-definiteness, so it correctly
    // refuses every force here rather than silently returning a wrong answer.
    // (Real ABA never inverts the constrained operator directly for this
    // reason -- it solves in the reduced, unconstrained subspace instead.)
    let response = locked.apply(&s);
    println!("response to the locked direction:  f = {response:?} (should be ~0)");
    assert!(
        max_abs_vector(
            response.to_vector(),
            ForceVector::<f64>::zeros().to_vector()
        ) < 1.0e-9
    );
    println!(
        "solving the locked (rank-deficient) inertia: {:?}",
        locked.solve(&force)
    );
    assert!(locked.solve(&force).is_none());

    // Congruence transform matches direct matrix composition, exactly the
    // identity ABA relies on to propagate a child's articulated inertia into
    // its parent's frame -- whether or not the operator is still invertible.
    let transform = SpatialTransform::<f64>::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [1.0, 0.0, 0.0],
    );
    let transformed = locked.transformed(&transform);
    let expected = Builtin::matrix6_mul(
        &Builtin::matrix6_mul(&transform.force_matrix(), &locked.matrix()),
        &transform.inverse().motion_matrix(),
    );
    assert!(max_abs_difference(transformed.matrix(), expected) < 1.0e-9);

    // Independent cross-check on the still positive-definite case: matrix
    // congruence must agree with `RigidBodyInertia::transformed`'s completely
    // different, rotation/center-of-mass-based computation -- not just with
    // another copy of the same congruence formula.
    assert!(
        max_abs_difference(
            full.transformed(&transform).matrix(),
            inertia.transformed(&transform).matrix()
        ) < 1.0e-9
    );

    // The spatial time derivative is the same bias operator
    // `RigidBodyInertia::time_derivative` now returns as a raw matrix --
    // constraining a joint doesn't change how the inertia varies with motion,
    // and the result stays symmetric even though it is never re-validated.
    let derivative = locked.time_derivative(&motion).matrix();
    println!("d/dt I^A = {derivative:?}");
    let transposed = Builtin::matrix6_transpose(&derivative);
    assert!(max_abs_difference(derivative, transposed) < 1.0e-9);

    println!("\nAn articulated-body inertia is what ABA (Chapter 7) propagates");
    println!("up a kinematic tree: a rigid body's inertia, with each locked");
    println!("joint direction projected out by a Schur complement.");
}
