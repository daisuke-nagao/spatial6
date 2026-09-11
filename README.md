# spatial6

Backend-agnostic 6D spatial vectors, transforms, and inertias for rigid-body
dynamics. Spatial coordinates store angular components before linear components.

```toml
[dependencies]
spatial6 = "1.0"
```

The crate provides `MotionVector`, `ForceVector`, `SpatialTransform`,
`RigidBodyInertia`, and `ArticulatedBodyInertia`.

| Feature | Backend |
| --- | --- |
| `builtin` (default) | Fixed Rust arrays |
| `nalgebra` | Native nalgebra vectors and matrices |
| `glam` | Native glam vectors and matrices |
| `serde` | `Serialize`/`Deserialize` for all public types |
| No default features | User-defined `SpatialRepresentation` |

Enable optional backends with `features = ["nalgebra"]` or `features = ["glam"]`.
Enable `features = ["serde"]` for serialization support; combined with `nalgebra`
or `glam`, it also enables that backend's own `serde` integration so the
underlying vector/matrix types serialize too.

See the [API documentation](https://docs.rs/spatial6) for usage and conventions,
and [`examples/`](examples/) for worked algorithms built on these types:
composite rigid-body inertia, inverse dynamics via the Recursive Newton-Euler
Algorithm, and articulated-body inertia. Each of these three algorithms, plus
a minimal single-rigid-body case, is worked through all three backends in a
`_nalgebra`/`_glam`-suffixed sibling file (`_builtin`/no suffix is the
default), so switching backends can be seen changing only the storage type,
never the answer:

```sh
cargo run --example backend_builtin --no-default-features --features builtin
cargo run --example backend_nalgebra --no-default-features --features nalgebra
cargo run --example backend_glam --no-default-features --features glam

cargo run --example rnea_builtin --no-default-features --features builtin
cargo run --example rnea_nalgebra --no-default-features --features nalgebra
cargo run --example rnea_glam --no-default-features --features glam

cargo run --example composite_rigid_body_builtin --no-default-features --features builtin
cargo run --example composite_rigid_body_nalgebra --no-default-features --features nalgebra
cargo run --example composite_rigid_body_glam --no-default-features --features glam

cargo run --example articulated_inertia_builtin --no-default-features --features builtin
cargo run --example articulated_inertia_nalgebra --no-default-features --features nalgebra
cargo run --example articulated_inertia_glam --no-default-features --features glam
```

`ArticulatedBodyInertia` represents the symmetric inertia operator produced by
eliminating a joint acceleration under a specified joint force. The core
reduction uses only backend-agnostic public operations:

```rust,ignore
let full = ArticulatedBodyInertia::try_from(&rigid)?;
let u = full.apply(&joint_motion);
let d = joint_motion.dot(&u);
assert!(d.is_finite() && d > 0.0);

let reduced = full.try_rank_one_updated(-1.0 / d, &u)?;
let child_in_parent = reduced.try_transformed(&child_to_parent)?;
let accumulated = parent.try_combined(&child_in_parent)?;
```

## Contributing

Install Rust (stable, including rustfmt and Clippy) and
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv tool install pre-commit==4.6.2
pre-commit install
pre-commit run --all-files
```

Before submitting a change, run the same checks CI runs (which treats rustdoc
warnings as errors):

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --no-default-features
cargo test --no-default-features --features nalgebra
cargo test --no-default-features --features glam
cargo test --features serde
cargo test --all-features
cargo doc --no-deps --all-features
```

## License

Copyright 2026 Daisuke Nagao.

Licensed under either the [MIT License](LICENSES/MIT.txt) or the
[Apache License, Version 2.0](LICENSES/Apache-2.0.txt), at your option
(`MIT OR Apache-2.0`). Copyright and SPDX license information is recorded in
Rust file headers.
