# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-09-14

### Added

- A backend-agnostic `ArticulatedBodyInertia` API for finite symmetric spatial
  operators, including rigid-body conversion, application, combination,
  rank-one reduction, and frame transformation.
- `SpatialTransform::to_pose_parts` and `SpatialTransform::from_pose_parts` for
  extracting and rebuilding local-to-reference pose components, with pose-chain
  examples for the builtin, nalgebra, and glam backends.
- A minimal fixed-base serial-chain ABA forward-dynamics example using the
  public `ArticulatedBodyInertia` API.

### Changed

- The articulated-inertia examples now use the public API to demonstrate joint
  acceleration elimination, frame propagation, and parent accumulation.

### Fixed

- `ArticulatedBodyInertia::try_from(&RigidBodyInertia)` now accepts valid
  derived matrices whose off-diagonal asymmetry is caused by floating-point
  roundoff, canonicalizing the symmetric result instead of returning
  `InertiaError::NonSymmetric`.

## [1.0.0] - 2026-09-09

### Added

- A backend-agnostic API for six-dimensional spatial vectors, transforms, and
  rigid-body inertias.
- Builtin fixed-array, nalgebra, and glam backends.
- Rigid-body dynamics algorithms, examples, and benchmarks covering composite
  rigid-body inertia, recursive Newton-Euler inverse dynamics, and articulated-body
  inertia across the supported backends.

[Unreleased]: https://github.com/daisuke-nagao/spatial6/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/daisuke-nagao/spatial6/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/daisuke-nagao/spatial6/releases/tag/v1.0.0
