# spatial6

Backend-agnostic 6D spatial vectors, transforms, and inertias for rigid-body
dynamics. Spatial coordinates store angular components before linear components.

```toml
[dependencies]
spatial6 = "1"
```

The crate provides `MotionVector`, `ForceVector`, `MotionSubspace`,
`SpatialTransform`, `RigidBodyInertia`, and `ArticulatedBodyInertia`.

| Feature | Backend |
| --- | --- |
| `builtin` (default) | Fixed Rust arrays |
| `std` (default) | Requests std support in enabled numeric dependencies |
| `nalgebra` | Native nalgebra vectors and matrices |
| `glam` | Native glam vectors and matrices |
| `serde` | `Serialize`/`Deserialize` for all public types |
| No default features | User-defined `SpatialRepresentation` |

Enable optional backends with `features = ["nalgebra"]` or `features = ["glam"]`.
Enable `features = ["serde"]` for serialization support; combined with `nalgebra`
or `glam`, it also enables that backend's own `serde` integration so the
underlying vector/matrix types serialize too.

## no_std

The library source is always `no_std`. Its default `std` feature enables std
support in enabled numeric dependencies. Disable default features for bare-metal
use and select a backend explicitly:

```toml
[dependencies.spatial6]
version = "1"
default-features = false
features = ["builtin"]
```

Use `nalgebra` or `glam` instead of `builtin` for those backends; add `serde` for
serialization. With no backend selected, provide a custom
`SpatialRepresentation`. Enabling `std` or `serde` alone does not select a backend.
The floating-point fallback is enabled automatically; no separate `libm` feature
is required.

With the supplied backends and `f32`/`f64`, the verified no_std operations do not
require a global allocator. Their serialization implementations use fixed
storage; custom representations, scalars, serialization adapters, and formatting
sinks determine their own resource requirements. Fixed storage grows with the
dimension of `MotionSubspace<N>` and does not imply a constant stack budget.
The library provides no panic handler and makes no panic-free or hard-real-time
execution claim.

Cargo unifies dependency features: another runtime dependency can enable std or
alloc even when this dependency disables defaults. Inspect the final firmware
graph with an explicit target, for example:

```sh
cargo tree --target riscv32imac-unknown-none-elf --edges normal,no-proc-macro -e features
```

For hosted applications that disable defaults, add `std` explicitly to retain
the std-enabled numeric path. Otherwise the isolated graph uses libm,
which can change performance and rounding near validation thresholds. Algorithms,
validation rules, and public API are unchanged by this feature split; bitwise
reproducibility and identical decisions for every ill-conditioned boundary input
are not promised.
Downstream nalgebra/glam conversion features can also force glam's libm path
through feature unification even when std is enabled.

The independent [consumer](https://github.com/daisuke-nagao/spatial6/tree/HEAD/ci/no-std-consumer)
runs custom and supplied backends for both scalar precisions. The
[link fixture](https://github.com/daisuke-nagao/spatial6/tree/HEAD/ci/no-std-link)
checks allocator-free linking on RISC-V and Cortex-M; it is not a board startup
example. Reproduction commands and evidence are in the repository's
[validation record](https://github.com/daisuke-nagao/spatial6/blob/HEAD/ci/no-std-validation.md).

## Examples

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

cargo run --example motion_subspace_builtin --no-default-features --features builtin
cargo run --example motion_subspace_nalgebra --no-default-features --features nalgebra
cargo run --example motion_subspace_glam --no-default-features --features glam

cargo run --example rnea_builtin --no-default-features --features builtin
cargo run --example rnea_nalgebra --no-default-features --features nalgebra
cargo run --example rnea_glam --no-default-features --features glam

cargo run --example composite_rigid_body_builtin --no-default-features --features builtin
cargo run --example composite_rigid_body_nalgebra --no-default-features --features nalgebra
cargo run --example composite_rigid_body_glam --no-default-features --features glam

cargo run --example articulated_inertia_builtin --no-default-features --features builtin
cargo run --example articulated_inertia_nalgebra --no-default-features --features nalgebra
cargo run --example articulated_inertia_glam --no-default-features --features glam

cargo run --example pose_chain_builtin --no-default-features --features builtin
cargo run --example pose_chain_nalgebra --no-default-features --features nalgebra
cargo run --example pose_chain_glam --no-default-features --features glam
```

`to_pose_parts()` returns the local-to-reference rotation `Q` and the local
origin's position `p` in reference coordinates. Use `from_pose_parts()` to
rebuild a transform from those owned parts; the stored transform rotation and
translation remain the source-to-destination `E`/`r` representation.

```rust,ignore
let (q_local_to_reference, p_in_reference) = transform.to_pose_parts();
let restored = SpatialTransform::from_pose_parts(q_local_to_reference, p_in_reference);
```

`ArticulatedBodyInertia` represents the current articulated inertia, including
the reduced symmetric operator produced by eliminating a joint acceleration
under a specified joint force. The core reduction uses only backend-agnostic
public operations:

```rust,ignore
let full = ArticulatedBodyInertia::try_from(&rigid)?;
let u = full.apply(&joint_motion);
let d = joint_motion.dot(&u);
assert!(d.is_finite() && d > 0.0);

let reduced = full.try_rank_one_updated(-1.0 / d, &u)?;
let child_in_parent = reduced.try_transformed(&child_to_parent)?;
let accumulated = parent.try_combined(&child_in_parent)?;
```

The `aba_builtin` example is a minimal fixed-base serial-chain ABA
forward-dynamics walkthrough using the public `ArticulatedBodyInertia` API.

```sh
cargo run --example aba_builtin --no-default-features --features builtin
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
cargo test --no-default-features --features serde
cargo test --all-features
cargo doc --no-deps --all-features
```

## License

Copyright 2026 Daisuke Nagao.

Licensed under either the [MIT License](LICENSES/MIT.txt) or the
[Apache License, Version 2.0](LICENSES/Apache-2.0.txt), at your option
(`MIT OR Apache-2.0`). Copyright and SPDX license information is recorded in
Rust file headers.
