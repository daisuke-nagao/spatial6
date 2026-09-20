# Allocator-free link check

This independent workspace links the concrete operations in `no-std-consumer`
without a global allocator or allocation error handler. It has no `std` feature.
Both root profiles abort on panic. The panic handler belongs to this application
fixture, never to the spatial6 library.

`build.rs` accepts only `riscv32imac-unknown-none-elf` and
`thumbv7em-none-eabihf`. Both targets use rust-lld's ELF linker and the virtual
layout in `link.x`; ARM unwind index sections are kept separate. The retained
`no_std_entry` calls `run_all` with `black_box` inputs and outputs, which calls
each selected non-generic backend/scalar wrapper. Each build writes `link.map`
under its build-script `OUT_DIR` for retained-code inspection.

The virtual address is an arbitrary link address, **not a board memory map**.
There is no vector table, stack initialization, reset handler, or peripheral
setup. These images are link tests and must not be presented as bootable firmware.
Compiler runtime helpers are real toolchain implementations; no fake math,
memory, or allocation stubs are supplied.

From the repository root:

```sh
python scripts/check_no_std.py --mode link --toolchain stable --target riscv32imac-unknown-none-elf --subset custom-only --subset serde-only --subset backend:builtin --subset backend:nalgebra --subset backend:glam --subset all
python scripts/check_no_std.py --mode link --toolchain stable --target riscv32imac-unknown-none-elf --profile release --subset all
python scripts/check_no_std.py --mode link --toolchain stable --target thumbv7em-none-eabihf --subset all
```

The driver checks the required feature/profile cases and audits the matching
target-normal graph. This is evidence for the retained concrete operations,
not every possible generic instantiation or input, nor execution on hardware.
