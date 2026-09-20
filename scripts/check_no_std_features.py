"""Audit the target-normal dependency features used by a no-std consumer.

The audit intentionally checks a small policy surface instead of snapshotting a
complete Cargo tree.  Cargo's flat ``{p}|{f}`` output is parsed strictly so a
format change or an ambiguous package identity fails the check loudly.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:  # Works both as ``python scripts/...`` and as an imported test module.
    from scripts.check_no_std import FEATURES
except ModuleNotFoundError:  # pragma: no cover - direct script execution path.
    from check_no_std import FEATURES

PACKAGE_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.+-]*) v(?P<version>[^\s()|]+)"
    r"(?: \((?P<source>.*)\))?$"
)
FEATURE_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")


class AuditError(ValueError):
    """A malformed tree or a dependency feature policy violation."""

    def __init__(self, message: str, returncode: int = 1):
        super().__init__(message)
        self.returncode = returncode


@dataclass(frozen=True)
class PackageRecord:
    name: str
    version: str
    source: str | None
    features: frozenset[str]
    line_number: int
    context: str

    @property
    def identity(self) -> tuple[str, str, str | None]:
        return self.name, self.version, self.source


def parse_tree(output: str, *, context: str = "target") -> list[PackageRecord]:
    """Parse Cargo's ``{p}|{f}`` output without merging package identities."""

    records: list[PackageRecord] = []
    for line_number, raw_line in enumerate(output.splitlines(), 1):
        if not raw_line.strip():
            continue
        line = raw_line.rstrip()
        if line.count("|") != 1:
            raise AuditError(f"line {line_number}: expected one '|' separator: {raw_line!r}")
        identity_text, feature_text = line.split("|", 1)
        if identity_text != identity_text.strip():
            raise AuditError(f"line {line_number}: unexpected indentation: {raw_line!r}")
        match = PACKAGE_RE.fullmatch(identity_text)
        if match is None:
            raise AuditError(
                f"line {line_number}: malformed package identity: {identity_text!r}"
            )
        tokens = []
        if feature_text.strip():
            for token in feature_text.split(","):
                token = token.strip()
                if not token or FEATURE_RE.fullmatch(token) is None:
                    raise AuditError(
                        f"line {line_number}: malformed feature token: {token!r}"
                    )
                if token in tokens:
                    raise AuditError(
                        f"line {line_number}: duplicate feature token: {token!r}"
                    )
                tokens.append(token)
        source = match.group("source")
        if source is not None and not source.strip():
            raise AuditError(f"line {line_number}: empty package source")
        records.append(
            PackageRecord(
                name=match.group("name"),
                version=match.group("version"),
                source=source,
                features=frozenset(tokens),
                line_number=line_number,
                context=context,
            )
        )
    if not records:
        raise AuditError("Cargo tree output contained no package records")
    seen: dict[tuple[tuple[str, str, str | None], str], frozenset[str]] = {}
    for record in records:
        key = (record.identity, record.context)
        previous = seen.setdefault(key, record.features)
        if previous != record.features:
            raise AuditError(
                f"ambiguous feature records for {record.name} {record.version} "
                f"({record.source or '<unspecified>'}) in {record.context} context"
            )
    return records


def _selected_features(features: Iterable[str]) -> frozenset[str]:
    requested = tuple(features)
    if len(set(requested)) != len(requested):
        raise AuditError("duplicate requested feature")
    selected = frozenset(requested)
    unknown = selected - set(FEATURES)
    if unknown:
        raise AuditError(f"unknown requested feature(s): {', '.join(sorted(unknown))}")
    return selected


def _records(records: Sequence[PackageRecord], name: str) -> list[PackageRecord]:
    return [record for record in records if record.name == name]


def _require(records: Sequence[PackageRecord], name: str) -> list[PackageRecord]:
    found = _records(records, name)
    if not found:
        raise AuditError(f"required package {name!r} is missing")
    return found


def _forbid(record: PackageRecord, forbidden: set[str]) -> None:
    bad = sorted(record.features & forbidden)
    if bad:
        source = record.source or "<unspecified>"
        raise AuditError(
            f"{record.name} {record.version} ({source}) has forbidden feature(s): "
            f"{', '.join(bad)}"
        )


def audit_records(records: Sequence[PackageRecord], features: Iterable[str]) -> None:
    """Assert the no-std dependency feature policy for parsed records."""

    selected = _selected_features(features)
    spatial6 = _require(records, "spatial6")
    num_traits = _require(records, "num-traits")

    for record in spatial6:
        active = set(record.features) - {"std"}
        if active != set(selected):
            raise AuditError(
                f"spatial6 feature set {sorted(active)!r} does not match requested "
                f"{sorted(selected)!r}"
            )

    # Every relevant record is checked.  Duplicate versions/sources stay
    # separate, so a bad instance cannot be hidden by a feature union.
    for record in records:
        _forbid(record, {"std", "alloc"})

    for record in num_traits:
        if "libm" not in record.features:
            raise AuditError(
                f"num-traits {record.version} is missing required feature 'libm'"
            )

    for name in ("nalgebra", "glam", "serde"):
        found = _records(records, name)
        if (name in selected) != bool(found):
            state = "missing" if name in selected else "unexpected"
            raise AuditError(f"{state} selected package {name!r}")

    for record in _records(records, "nalgebra"):
        _forbid(record, {"std", "alloc", "serde-serialize", "libm-force"})
        if "libm" not in record.features:
            raise AuditError(
                f"nalgebra {record.version} is missing required feature 'libm'"
            )
        if "serde" in selected and "serde-serialize-no-std" not in record.features:
            raise AuditError(
                f"nalgebra {record.version} is missing 'serde-serialize-no-std'"
            )

    for record in _records(records, "glam"):
        _forbid(record, {"std", "alloc", "fast-math", "force-libm", "libm"})
        required = {"f64", "nostd-libm"}
        if "serde" in selected:
            required.add("serde")
        missing = sorted(required - record.features)
        if missing:
            raise AuditError(
                f"glam {record.version} is missing required feature(s): {', '.join(missing)}"
            )

    for record in _records(records, "serde"):
        _forbid(record, {"std", "alloc"})

    for record in _records(records, "serde_core"):
        _forbid(record, {"std", "alloc"})

    for record in _records(records, "simba"):
        _forbid(record, {"std", "alloc", "libm_force", "libm-force"})
        if "nalgebra" in selected and "libm" not in record.features:
            raise AuditError(f"simba {record.version} is missing required feature 'libm'")


def cargo_tree_command(
    *,
    manifest: Path,
    toolchain: str,
    target: str | None,
    features: Iterable[str],
    cargo: str = "cargo",
) -> list[str]:
    """Build the reproducible Cargo tree command used by the CLI and driver."""

    selected = _selected_features(features)
    command = [cargo, f"+{toolchain}", "tree", "--manifest-path", str(manifest)]
    command += ["--locked", "--no-default-features"]
    if selected:
        command += ["--features", ",".join(name for name in FEATURES if name in selected)]
    if target:
        command += ["--target", target]
    command += [
        "--edges",
        "normal,no-proc-macro",
        "--prefix",
        "none",
        "--format",
        "{p}|{f}",
        "--no-dedupe",
        "--color",
        "never",
    ]
    return command


def audit_manifest(
    *,
    manifest: Path,
    toolchain: str,
    target: str | None,
    features: Iterable[str],
    cargo: str = "cargo",
    runner=subprocess.run,
) -> list[PackageRecord]:
    command = cargo_tree_command(
        manifest=manifest,
        toolchain=toolchain,
        target=target,
        features=features,
        cargo=cargo,
    )
    result = runner(command, capture_output=True, text=True, check=False)
    if result.returncode:
        details = (result.stderr or result.stdout or "Cargo tree failed").strip()
        raise AuditError(f"Cargo tree failed ({result.returncode}): {details}", result.returncode)
    records = parse_tree(result.stdout, context="target" if target else "host")
    audit_records(records, features)
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--toolchain", default="stable")
    parser.add_argument("--target")
    parser.add_argument("--features", default="", help="comma-separated selected features")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        selected = tuple(filter(None, (part.strip() for part in args.features.split(","))))
        records = audit_manifest(
            manifest=args.manifest,
            toolchain=args.toolchain,
            target=args.target,
            features=selected,
        )
    except (AuditError, OSError) as error:
        print(f"no-std feature audit failed: {error}", file=sys.stderr)
        return getattr(error, "returncode", 1)
    identities = {(record.name, record.version, record.source) for record in records}
    scope = args.target or "host"
    print(
        f"feature audit passed: {len(identities)} package identities, "
        f"toolchain={args.toolchain}, target={scope}, features={args.features or '<empty>'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
