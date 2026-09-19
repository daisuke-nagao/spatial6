# no_std implementation evidence

## Baseline (phase 0)

- Baseline: `690782f45d50f61ef27b4f2b86212e0a9348ee14`, tag `v1.2.0`.
- The tracked checkout was clean. Existing untracked `proposal/` and `.serena/`
  were retained and excluded from implementation commits.
- Package: spatial6 1.2.0, edition 2024, MSRV 1.89. No dependency family upgrade
  or release publication is part of this implementation.
- Host: `x86_64-pc-windows-msvc`; stable rustc/Cargo 1.97.1; MSRV rustc/Cargo
  1.89.0. Both toolchains have `riscv32imac-unknown-none-elf` and
  `thumbv7em-none-eabihf` installed for verification.
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

## Scope

Target compilation and allocator-free links exercise concrete fixture operations;
they do not establish board startup, hardware runtime results, panic freedom,
or resource guarantees for arbitrary custom implementations or applications.
The next compatible release is provisionally 1.3.0; the package version remains
1.2.0 until the release owner confirms and performs the release.
