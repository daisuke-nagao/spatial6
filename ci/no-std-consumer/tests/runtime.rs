use spatial6_no_std_consumer::{run_all, run_array_f32, run_array_f64};

#[test]
fn custom_array_operations_run_with_dependency_defaults_disabled() {
    for seed in [0.25, -0.75, 1.5, 1.0e-6, 1.0e3, -0.0] {
        assert!(run_array_f32(seed).is_finite(), "f32 seed {seed}");
        assert!(run_array_f64(seed).is_finite(), "f64 seed {seed}");
    }
}

#[test]
fn retained_entry_runs_all_selected_operations() {
    for seed in [0.375, -0.0, 1.0e-6, 1.0e3] {
        assert!(run_all(seed).is_finite(), "seed {seed}");
    }
}

#[cfg(feature = "builtin")]
#[test]
fn builtin_wrappers_cover_both_precisions() {
    assert!(spatial6_no_std_consumer::run_builtin_f32(0.375).is_finite());
    assert!(spatial6_no_std_consumer::run_builtin_f64(0.375).is_finite());
}

#[cfg(feature = "nalgebra")]
#[test]
fn nalgebra_wrappers_cover_both_precisions() {
    assert!(spatial6_no_std_consumer::run_nalgebra_f32(0.375).is_finite());
    assert!(spatial6_no_std_consumer::run_nalgebra_f64(0.375).is_finite());
}

#[cfg(feature = "glam")]
#[test]
fn glam_wrappers_cover_both_precisions() {
    assert!(spatial6_no_std_consumer::run_glam_f32(0.375).is_finite());
    assert!(spatial6_no_std_consumer::run_glam_f64(0.375).is_finite());
}

#[cfg(feature = "serde")]
#[test]
fn serde_wrapper_exercises_fixed_capacity_round_trips() {
    assert!(spatial6_no_std_consumer::run_serde(0.375).is_finite());
}

#[cfg(all(feature = "serde", feature = "builtin"))]
#[test]
fn builtin_serde_wrapper_exercises_both_precisions() {
    assert!(spatial6_no_std_consumer::run_serde_builtin(0.375).is_finite());
}

#[cfg(all(feature = "serde", feature = "nalgebra"))]
#[test]
fn nalgebra_serde_wrapper_exercises_both_precisions() {
    assert!(spatial6_no_std_consumer::run_serde_nalgebra(0.375).is_finite());
}

#[cfg(all(feature = "serde", feature = "glam"))]
#[test]
fn glam_serde_wrapper_exercises_both_precisions() {
    assert!(spatial6_no_std_consumer::run_serde_glam(0.375).is_finite());
}
