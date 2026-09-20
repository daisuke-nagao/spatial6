// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_std]
#![no_main]

use core::hint::black_box;
use core::panic::PanicInfo;

// Link-only entry: no board initialization or hardware execution is promised.
#[unsafe(no_mangle)]
pub extern "C" fn no_std_entry() -> ! {
    black_box(spatial6_no_std_consumer::run_all(black_box(0.25)));
    loop {
        core::hint::spin_loop();
    }
}

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
