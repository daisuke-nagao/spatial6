// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{env, path::PathBuf};

fn main() {
    let target = env::var("TARGET").unwrap();
    assert!(
        matches!(
            target.as_str(),
            "riscv32imac-unknown-none-elf" | "thumbv7em-none-eabihf"
        ),
        "link-only fixture requires a documented bare-metal target"
    );
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed=link.x");
    println!("cargo:rustc-link-arg=-T{}", root.join("link.x").display());
    println!(
        "cargo:rustc-link-arg=-Map={}",
        output.join("link.map").display()
    );
}
