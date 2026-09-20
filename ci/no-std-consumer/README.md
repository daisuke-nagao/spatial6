# spatial6 no-std consumer

This unpublished fixture disables spatial6 default features and exercises the
custom array representation plus each selected backend with f32 and f64. It
covers vector arithmetic, transforms, rigid and articulated inertia, SPD
residuals, square root, subspaces (N = 0, 1, 3, 6, 7), and the fixed-capacity
serde adapter in readable and compact modes. The numerical checks use 2e-4 for
f32 and 2e-11 for f64, with wider factors only for composed operations.

```sh
cargo test --manifest-path ci/no-std-consumer/Cargo.toml --locked --no-default-features
cargo test --manifest-path ci/no-std-consumer/Cargo.toml --locked --no-default-features --features serde
cargo test --manifest-path ci/no-std-consumer/Cargo.toml --locked --no-default-features --features builtin,nalgebra,glam,serde
cargo +1.89.0 check --manifest-path ci/no-std-consumer/Cargo.toml --locked --lib --target riscv32imac-unknown-none-elf --no-default-features --features builtin,nalgebra,glam,serde
```

The hosted test harness may use `std`; the consumer library itself remains
`no_std` and has no allocator or panic handler.
