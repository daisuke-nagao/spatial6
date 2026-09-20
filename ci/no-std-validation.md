# no_std implementation evidence

This is a historical acceptance record for the conversion from the recorded
v1.2.0 baseline to the 1.3.0 package candidate. Version and toolchain values below
describe that run; this file does not certify later revisions, whose workflow and
CI result provide their own evidence.

## Baseline (phase 0)

- Baseline: `690782f45d50f61ef27b4f2b86212e0a9348ee14`, tag `v1.2.0`.
- The tracked checkout was clean. Existing untracked `proposal/` and `.serena/`
  were retained and excluded from implementation commits.
- Package: spatial6 1.2.0, edition 2024, MSRV 1.89. No dependency family upgrade
  or release publication is part of this implementation.
- The acceptance run used host `x86_64-pc-windows-msvc`, stable rustc/Cargo
  1.97.1, and MSRV rustc/Cargo 1.89.0, with
  `riscv32imac-unknown-none-elf` and `thumbv7em-none-eabihf` installed.
- Root is a standalone package; its default resolver follows edition 2024
  (resolver 3). `.cargo/config.toml` only selects wasmtime for wasm32-wasip2.
- Root Cargo.lock is deliberately ignored. Baseline resolved library families:
  num-traits 0.2.19, nalgebra 0.35.0, glam 0.33.6, serde/serde_core 1.0.229,
  simba 0.10.2. Older nalgebra/glam versions belong to development dependencies.
- All seven historical `cargo test --locked` selections passed before changes:
  default; no defaults; nalgebra only; glam only; default + serde; serde only;
  all features. The commands include doctests.
- Production namespace review found 28 `std::` references in five files;
  no direct heap containers, allocation macros, unsafe code, or unit-test
  modules were found. This textual scan is advisory, not a Rust parser.
- Public extension traits and backend identities remain open. The existing
  manifest already disables nalgebra default macros and glam default all-types;
  those capabilities are not removed by this conversion.

## Phase gates

- Phase 1: only `std::` to `core::` namespace replacements, including the
  representation rustdoc example; existing all-feature tests passed.
- Later implementation and acceptance evidence is recorded below as checks run.

- Phase 2: unconditional `#![no_std]`, without a crate-root std test exception;
  default/all-feature tests and warning-free all-feature rustdoc passed. A hosted
  integration test asserts the unchanged `std::error::Error` contract.
- Phase 3: dependency fallback policy implemented without dependency-family
  upgrades; stable and 1.89.0 RISC-V library checks passed with all four non-std
  features. Historical host tests plus std-only, std+nalgebra, and std+glam passed.
- Phase 4: the independent consumer executes every one of the 16 subsets of
  `{builtin, nalgebra, glam, serde}` on the stable host. Stable RISC-V check and
  build matrices and the Rust 1.89.0 RISC-V build matrix passed all 16 subsets;
  the required nine stable ARM rows also passed. Matching host and target-normal
  feature audits and the audit parser's regression suite passed.
- The consumer covers a custom representation and every supplied backend with
  `f32` and `f64`, fixed-capacity serde round trips, malformed sequences,
  numerical residual/error paths, and `MotionSubspace` dimensions 0, 1, 3, 6,
  and 7. RISC-V dev links passed for empty, serde-only, each backend, and all
  features; RISC-V release and ARM dev all-feature links also passed. Link maps
  retained the selected concrete wrappers and contained no allocator symbols.
- Phase 5: stable and Rust 1.89.0 host default/all-feature checks passed, as did
  clippy, rustdoc with all features and selected no-default configurations, the
  changelog check, package verification, and the repository hooks. README,
  rustdoc, and changelog describe migration, feature unification, limitations,
  and the confirmed compatible minor version 1.3.0.
- Disposable fresh resolutions and consumers of the unpacked package passed on
  stable and Rust 1.89.0. Each run regenerated both fixture locks, executed and
  audited all 16 host/RISC-V rows, linked all features in dev and release, and
  byte-compared every packaged production Rust source file. Both toolchains
  resolved the same relevant versions: num-traits 0.2.19, nalgebra 0.35.0,
  glam 0.33.8, serde/serde_core 1.0.229, simba 0.10.2, and libm 0.2.16.

## Dependency and serialization review

The resolved registry sources were inspected independently with Luna Max:

- num-traits 0.2.19 gates `Float` behind std or libm. The explicit lower bound
  and automatic libm feature provide the existing trait on bare metal.
- nalgebra 0.35.0 `libm` forwards to simba; `serde-serialize-no-std` avoids the
  serde/std forwarding in `serde-serialize`. Default macros were already off.
  `DefaultAllocator<Const<R>, Const<C>>` uses `ArrayStorage`; fixed-size Cholesky
  and its cloned RHS stay in fixed storage. These are the actual decomposition
  and solve paths used by the representation overrides.
- glam 0.33.6 `nostd-libm` selects fallback only without std; `libm` forces its
  math path. The existing exposed f32/f64 vectors and matrices remain enabled.
- nalgebra static array serde initializes fixed storage; glam tuple visitors
  and spatial6's own implementations do not introduce a heap buffer.
- `MotionSubspace` fills every `Option` before its final `expect`; early end,
  invalid elements, and excess elements return errors before that point.
  Zero-dimensional subspaces remain supported.
- Downstream nalgebra `convert-glam033` plus `libm` can force glam libm through
  feature unification. The isolated fixture policy does not prohibit users from
  deliberately selecting that graph.

The acceptance run's fixture locks generated by Cargo 1.89 resolved glam 0.33.8,
syn 3.0.6, and unicode-ident 1.0.26. A follow-up registry-source comparison found
glam's f32/f64 math and serde implementation byte-identical from 0.33.6 through
0.33.8. The 0.33.8 changes relevant here are package metadata and documentation;
its features and Rust 1.68.2 minimum are unchanged. The declared 0.33.6 lower
bound already contains the required fallback features.

## Scope

Target compilation and allocator-free links exercise concrete fixture operations;
they do not establish board startup, hardware runtime results, panic freedom,
or resource guarantees for arbitrary custom implementations or applications.
The accepted package candidate was version 1.3.0. Publication and release tagging
were outside the acceptance run.
