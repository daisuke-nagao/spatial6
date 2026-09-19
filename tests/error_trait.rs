// SPDX-FileCopyrightText: 2026 Daisuke Nagao
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[test]
fn inertia_error_remains_a_host_error() {
    fn assert_error<T: std::error::Error>() {}
    assert_error::<spatial6::InertiaError>();
    let error: &dyn std::error::Error = &spatial6::InertiaError::NonFinite;
    assert_eq!(error.to_string(), "inertia contains a non-finite value");
    assert!(error.source().is_none());
}
