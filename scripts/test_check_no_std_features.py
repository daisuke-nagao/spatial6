import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts.check_no_std import (
    CONSUMER_MANIFEST,
    FEATURES,
    LinkVerificationFailure,
    cargo_command,
    case_labels,
    expected_link_symbols,
    feature_subsets,
    parse_subset,
    run_matrix,
    verify_link_map,
)
from scripts.check_no_std_features import (
    AuditError,
    audit_records,
    audit_manifest,
    cargo_tree_command,
    parse_tree,
)


def tree(*rows: str) -> str:
    return "\n".join(rows)


def valid_tree(*, selected: str = "builtin", extra: str = "") -> str:
    rows = [
        f"spatial6 v1.2.0 (path+file:///workspace)|{selected}",
        "num-traits v0.2.19 (registry+https://github.com/rust-lang/crates.io-index)|libm",
    ]
    rows.extend(extra.splitlines())
    return tree(*rows)


class FeatureAuditTests(unittest.TestCase):
    def test_feature_subsets_are_the_full_canonical_powerset(self) -> None:
        self.assertEqual(FEATURES, ("builtin", "nalgebra", "glam", "serde"))
        self.assertEqual(len(feature_subsets()), 16)
        self.assertEqual(feature_subsets()[0], ())
        self.assertEqual(feature_subsets()[-1], ("builtin", "nalgebra", "glam", "serde"))

    def test_duplicate_sources_and_versions_remain_distinct(self) -> None:
        records = parse_tree(
            tree(
                "spatial6 v1.2.0 (path+file:///repo)|builtin",
                "num-traits v0.2.19 (registry+one)|libm",
                "num-traits v0.2.20 (registry+two)|libm",
                "num-traits v0.2.19 (registry+three)|libm",
            )
        )
        identities = [record.identity for record in records]
        self.assertEqual(len(set(identities)), 4)
        self.assertEqual(len([record for record in records if record.name == "num-traits"]), 3)

    def test_duplicate_same_identity_and_features_is_not_ambiguous(self) -> None:
        records = parse_tree(
            tree(
                "spatial6 v1.2.0 (path+file:///repo)|builtin",
                "spatial6 v1.2.0 (path+file:///repo)|builtin",
                "num-traits v0.2.19 (registry+one)|libm",
            )
        )
        self.assertEqual(len(records), 3)

    def test_similar_feature_tokens_do_not_satisfy_forbidden_or_required_names(self) -> None:
        records = parse_tree(
            valid_tree(
                selected="nalgebra",
                extra=(
                    "nalgebra v0.35.0 (registry+one)|serde-serialize-no-std,libm-force"
                )
            )
        )
        with self.assertRaisesRegex(AuditError, "libm-force"):
            audit_records(records, ("nalgebra",))

    def test_similar_feature_names_do_not_trigger_std_forbid(self) -> None:
        records = parse_tree(
            valid_tree(
                selected="nalgebra",
                extra="nalgebra v0.35.0 (registry+one)|libm,stdx",
            )
        )
        audit_records(records, ("nalgebra",))

    def test_glam_force_libm_feature_is_rejected_even_with_nostd_libm(self) -> None:
        records = parse_tree(
            valid_tree(
                selected="glam",
                extra="glam v0.33.7 (registry+one)|f64,nostd-libm,libm",
            )
        )
        with self.assertRaisesRegex(AuditError, r"forbidden feature\(s\): libm"):
            audit_records(records, ("glam",))

    def test_conflicting_duplicate_identity_features_are_ambiguous(self) -> None:
        with self.assertRaisesRegex(AuditError, "ambiguous feature records"):
            parse_tree(
                tree(
                    "spatial6 v1.2.0 (path+repo)|builtin",
                    "spatial6 v1.2.0 (path+repo)|nalgebra",
                )
            )

    def test_missing_package_is_a_failure(self) -> None:
        with self.assertRaisesRegex(AuditError, "required package 'num-traits'"):
            audit_records(parse_tree("spatial6 v1.2.0 (path+repo)|builtin"), ("builtin",))

    def test_target_and_host_context_are_retained(self) -> None:
        target = parse_tree(valid_tree(), context="target")
        host = parse_tree(valid_tree(), context="host")
        self.assertTrue(all(record.context == "target" for record in target))
        self.assertTrue(all(record.context == "host" for record in host))
        self.assertEqual(target[0].identity, host[0].identity)

    def test_malformed_output_and_unknown_feature_fail(self) -> None:
        with self.assertRaises(AuditError):
            parse_tree("spatial6 v1.2.0|builtin|extra")
        with self.assertRaises(AuditError):
            audit_records(parse_tree(valid_tree()), ("not-a-feature",))
        with self.assertRaisesRegex(AuditError, "feature set"):
            audit_records(parse_tree(valid_tree(selected="builtin,unknown")), ("builtin",))

    def test_cargo_failure_is_propagated(self) -> None:
        runner = Mock(return_value=subprocess.CompletedProcess(["cargo"], 17, "", "bad cargo"))
        with self.assertRaisesRegex(AuditError, r"Cargo tree failed \(17\)"):
            audit_manifest(
                manifest=Path("ci/no-std-consumer/Cargo.toml"),
                toolchain="1.89.0",
                target="riscv32imac-unknown-none-elf",
                features=("serde",),
                runner=runner,
            )
        command = runner.call_args.args[0]
        self.assertIn("+1.89.0", command)
        self.assertIn("--locked", command)
        self.assertIn("--target", command)

    def test_empty_subset_omits_features_but_keeps_required_flags(self) -> None:
        command = cargo_command(
            mode="check",
            manifest=Path("ci/no-std-consumer/Cargo.toml"),
            toolchain="stable",
            target="riscv32imac-unknown-none-elf",
            features=(),
        )
        self.assertIn("--no-default-features", command)
        self.assertIn("--locked", command)
        self.assertNotIn("--features", command)

    def test_runtime_uses_integration_tests_and_builds_use_lib(self) -> None:
        runtime = cargo_command(
            mode="runtime",
            manifest=CONSUMER_MANIFEST,
            toolchain="stable",
            target="riscv32imac-unknown-none-elf",
            features=(),
        )
        self.assertEqual(runtime[2], "test")
        self.assertNotIn("--lib", runtime)
        self.assertIn("--target", runtime)
        for mode in ("check", "build"):
            command = cargo_command(
                mode=mode,
                manifest=CONSUMER_MANIFEST,
                toolchain="stable",
                target="riscv32imac-unknown-none-elf",
                features=(),
            )
            self.assertIn("--lib", command)

    def test_runtime_matrix_runs_all_subsets_and_matching_audits(self) -> None:
        calls = []

        def runner(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")

        self.assertEqual(
            run_matrix(
                mode="runtime",
                manifest=CONSUMER_MANIFEST,
                toolchain="stable",
                target=None,
                subsets=feature_subsets(),
                profile="dev",
                runner=runner,
            ),
            0,
        )
        cargo_calls = calls[::2]
        audit_calls = calls[1::2]
        self.assertEqual(len(cargo_calls), 16)
        self.assertEqual(len(audit_calls), 16)
        self.assertTrue(all(command[2] == "test" for command in cargo_calls))
        self.assertTrue(all("--lib" not in command for command in cargo_calls))
        self.assertTrue(all("check_no_std_features.py" in command[1] for command in audit_calls))
        self.assertNotIn("--features", cargo_calls[0])
        self.assertNotIn("--features", audit_calls[0])

    def test_audit_command_uses_normal_no_proc_macro_tree(self) -> None:
        command = cargo_tree_command(
            manifest=Path("ci/no-std-consumer/Cargo.toml"),
            toolchain="stable",
            target="riscv32imac-unknown-none-elf",
            features=("serde",),
        )
        self.assertIn("normal,no-proc-macro", command)
        self.assertIn("--no-dedupe", command)
        self.assertIn("{p}|{f}", command)

    def test_link_mode_builds_and_checks_selected_concrete_wrappers(self) -> None:
        command = cargo_command(
            mode="link",
            manifest=Path("ci/no-std-link/Cargo.toml"),
            toolchain="stable",
            target="riscv32imac-unknown-none-elf",
            features=("builtin", "serde"),
            profile="release",
        )
        self.assertEqual(command[2], "build")
        self.assertIn("--release", command)
        self.assertEqual(
            expected_link_symbols(("builtin", "serde")),
            (
                "run_array_f32",
                "run_array_f64",
                "run_builtin_f32",
                "run_builtin_f64",
                "run_serde",
                "run_serde_builtin",
            ),
        )
        self.assertEqual(
            case_labels(("builtin", "serde")),
            (
                "custom:f32",
                "custom:f64",
                "builtin:f32",
                "builtin:f64",
                "serde",
                "serde:builtin",
            ),
        )

    def test_link_map_selection_rejects_stale_candidates_and_false_substrings(self) -> None:
        target = "riscv32imac-unknown-none-elf"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "ci" / "no-std-link" / "Cargo.toml"
            maps = root / "ci" / "no-std-link" / "target" / target / "debug" / "build"
            first = maps / "spatial6-no-std-link-old" / "out" / "link.map"
            second = maps / "spatial6-no-std-link-new" / "out" / "link.map"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            text = "run_array_f32 run_array_f64\n"
            first.write_text(text, encoding="utf-8")
            second.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(LinkVerificationFailure, "expected one current"):
                verify_link_map(manifest, (), target=target)

            second.write_text(
                text + "foo_run_array_f64_extra __rust_allocated core::alloc::Layout\n",
                encoding="utf-8",
            )
            self.assertEqual(
                verify_link_map(manifest, (), target=target, map_path=second), second.resolve()
            )
            second.write_text(text + "__rust_alloc\n", encoding="utf-8")
            with self.assertRaisesRegex(LinkVerificationFailure, "allocator symbol"):
                verify_link_map(manifest, (), target=target, map_path=second)


if __name__ == "__main__":
    unittest.main()
