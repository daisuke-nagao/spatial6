# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A backend-agnostic `ArticulatedBodyInertia` API for symmetric spatial inertia
  operators produced by joint-acceleration elimination.
- `SpatialTransform::to_pose_parts` and `SpatialTransform::from_pose_parts` for
  extracting and rebuilding local-to-reference pose components.

### Changed

- The articulated-inertia examples now demonstrate eliminating joint
  acceleration under a specified joint force through the public API.

## [1.0.0] - 2026-09-09

### Added

- A backend-agnostic API for six-dimensional spatial vectors, transforms, and
  rigid-body inertias.
- Builtin fixed-array, nalgebra, and glam backends.
- Rigid-body dynamics algorithms, examples, and benchmarks covering composite
  rigid-body inertia, recursive Newton-Euler inverse dynamics, and articulated-body
  inertia across the supported backends.

[Unreleased]: https://github.com/daisuke-nagao/spatial6/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/daisuke-nagao/spatial6/releases/tag/v1.0.0
