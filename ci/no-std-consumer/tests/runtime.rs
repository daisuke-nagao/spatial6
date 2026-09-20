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
