"""Run the reproducible no-std feature matrix and its dependency audits."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


FEATURES = ("builtin", "nalgebra", "glam", "serde")
ROOT = Path(__file__).resolve().parents[1]
CONSUMER_MANIFEST = ROOT / "ci" / "no-std-consumer" / "Cargo.toml"
LINK_MANIFEST = ROOT / "ci" / "no-std-link" / "Cargo.toml"


def feature_subsets() -> tuple[tuple[str, ...], ...]:
    """Return all 16 subsets in stable bit-mask order."""

    return tuple(
        tuple(name for bit, name in enumerate(FEATURES) if mask & (1 << bit))
        for mask in range(1 << len(FEATURES))
    )


class CommandFailure(RuntimeError):
    def __init__(self, command: Sequence[str], returncode: int):
        self.command = tuple(command)
        self.returncode = returncode
        super().__init__(f"command exited with {returncode}")


class LinkVerificationFailure(RuntimeError):
    pass


ALLOCATOR_SYMBOL_RE = re.compile(
    r"(?:"
    r"(?<![A-Za-z0-9_:])alloc::alloc::(?:alloc|dealloc|realloc)(?![A-Za-z0-9_])"
    r"|(?<![A-Za-z0-9_:])_?alloc(?![A-Za-z0-9_:])"
    r"|(?<![A-Za-z0-9_])__rg_alloc(?:_[A-Za-z0-9_]+)?(?![A-Za-z0-9_])"
    r"|(?<![A-Za-z0-9_])__rust_(?:alloc|dealloc|realloc)(?:_[A-Za-z0-9_]+)?"
    r"(?![A-Za-z0-9_])"
    r")"
)


def parse_subset(spec: str) -> tuple[str, ...]:
    value = spec.strip().lower()
    aliases = {
        "": (),
        "empty": (),
        "none": (),
        "custom-only": (),
        "all": FEATURES,
        "serde-only": ("serde",),
    }
    if value in aliases:
        return aliases[value]
    if value.startswith("backend:"):
        value = value.split(":", 1)[1]
    if value.startswith("backend-serde:"):
        backend = value.split(":", 1)[1]
        if backend not in {"builtin", "nalgebra", "glam"}:
            raise ValueError(f"unknown backend in subset {spec!r}")
        return tuple(name for name in FEATURES if name in {backend, "serde"})
    tokens = tuple(part.strip() for part in value.replace("+", ",").split(",") if part.strip())
    if not tokens or any(token not in FEATURES for token in tokens):
        raise ValueError(f"unknown feature subset {spec!r}")
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"duplicate feature in subset {spec!r}")
    return tuple(name for name in FEATURES if name in tokens)


def cargo_command(
    *,
    mode: str,
    manifest: Path,
    toolchain: str,
    target: str | None,
    features: Iterable[str],
    profile: str = "dev",
    cargo: str = "cargo",
) -> list[str]:
    selected = tuple(features)
    unknown = set(selected) - set(FEATURES)
    if unknown:
        raise ValueError(f"unknown features: {', '.join(sorted(unknown))}")
    cargo_mode = {"runtime": "test", "link": "build"}.get(mode, mode)
    command = [cargo, f"+{toolchain}", cargo_mode, "--manifest-path", str(manifest)]
    command += ["--locked", "--no-default-features"]
    if selected:
        command += ["--features", ",".join(name for name in FEATURES if name in selected)]
    if target:
        command += ["--target", target]
    if mode in {"build", "check"}:
        command += ["--lib"]
    if mode == "link" or (mode == "build" and manifest == LINK_MANIFEST):
        # Cargo has no --profile dev shorthand; omitting it is the dev build.
        if profile == "release":
            command += ["--release"]
    if mode == "link":
        command += ["--message-format=json-render-diagnostics"]
    return command


def expected_link_symbols(features: Iterable[str]) -> tuple[str, ...]:
    selected = set(features)
    symbols = ["run_array_f32", "run_array_f64"]
    for backend in ("builtin", "nalgebra", "glam"):
        if backend in selected:
            symbols += [f"run_{backend}_f32", f"run_{backend}_f64"]
    if "serde" in selected:
        symbols.append("run_serde")
        symbols.extend(
            f"run_serde_{backend}"
            for backend in ("builtin", "nalgebra", "glam")
            if backend in selected
        )
    return tuple(symbols)


def case_labels(features: Iterable[str]) -> tuple[str, ...]:
    """Name the concrete scalar/backend/serialization cases in a row."""

    selected = set(features)
    labels = ["custom:f32", "custom:f64"]
    for backend in ("builtin", "nalgebra", "glam"):
        if backend in selected:
            labels += [f"{backend}:f32", f"{backend}:f64"]
    if "serde" in selected:
        labels.append("serde")
        labels.extend(
            f"serde:{backend}"
            for backend in ("builtin", "nalgebra", "glam")
            if backend in selected
        )
    return tuple(labels)


def _profile_root(manifest: Path, target: str | None, profile: str) -> Path:
    target_name = target or "host"
    profile_name = "release" if profile == "release" else "debug"
    return manifest.parent / "target" / target_name / profile_name


def _scoped_link_map(
    manifest: Path, target: str | None, profile: str, map_path: Path | None
) -> Path:
    profile_root = _profile_root(manifest, target, profile).resolve()
    if map_path is None:
        maps = sorted((profile_root / "build").glob("spatial6-no-std-link-*/out/link.map"))
        if len(maps) != 1:
            raise LinkVerificationFailure(
                f"expected one current link.map below {profile_root / 'build'}, found {len(maps)}"
            )
        map_path = maps[0]
    map_path = map_path.resolve()
    try:
        map_path.relative_to(profile_root / "build")
    except ValueError as error:
        raise LinkVerificationFailure(
            f"link map {map_path} is outside target/profile {profile_root}"
        ) from error
    if map_path.name != "link.map" or map_path.parent.name != "out":
        raise LinkVerificationFailure(f"link map is not a build-script output: {map_path}")
    return map_path


def _contains_symbol(text: str, symbol: str) -> bool:
    return re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])", text
    ) is not None


def _allocator_symbol(text: str) -> str | None:
    for line in text.splitlines():
        candidate = line.rsplit("):", 1)[-1]
        if "/" in candidate or "\\" in candidate:
            continue
        match = ALLOCATOR_SYMBOL_RE.search(candidate)
        if match:
            return match.group(0)
    return None


def verify_link_map(
    manifest: Path,
    features: Iterable[str],
    *,
    target: str | None = None,
    profile: str = "dev",
    map_path: Path | None = None,
) -> Path:
    """Require the current target/profile map, wrappers, and no allocator symbols."""

    map_path = _scoped_link_map(manifest, target, profile, map_path)
    try:
        text = map_path.read_text(encoding="utf-8")
    except OSError as error:
        raise LinkVerificationFailure(f"cannot read {map_path}: {error}") from error
    expected = expected_link_symbols(features)
    missing = [symbol for symbol in expected if not _contains_symbol(text, symbol)]
    if missing:
        raise LinkVerificationFailure(
            f"{map_path} is missing selected concrete wrapper symbol(s): {', '.join(missing)}"
        )
    all_wrappers = expected_link_symbols(FEATURES)
    unexpected = [
        symbol
        for symbol in all_wrappers
        if symbol not in expected and _contains_symbol(text, symbol)
    ]
    if unexpected:
        raise LinkVerificationFailure(
            f"{map_path} retains unselected wrapper symbol(s): {', '.join(unexpected)}"
        )
    allocator = _allocator_symbol(text)
    if allocator:
        raise LinkVerificationFailure(
            f"{map_path} contains allocator symbol {allocator!r}"
        )
    print(f"  link map: {map_path} ({len(expected)} wrappers retained)")
    return map_path


def _current_link_map(output: str) -> Path:
    maps: list[Path] = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("reason") != "build-script-executed":
            continue
        target = event.get("target")
        package_id = event.get("package_id", "")
        target_name = target.get("name") if isinstance(target, dict) else None
        if (
            target_name != "spatial6-no-std-link"
            and "#spatial6-no-std-link@" not in package_id
        ):
            continue
        out_dir = event.get("out_dir")
        if isinstance(out_dir, str):
            maps.append(Path(out_dir) / "link.map")
    if len(maps) != 1:
        raise LinkVerificationFailure(
            f"Cargo did not identify exactly one current link-script output (found {len(maps)})"
        )
    return maps[0]


def _audit_command(
    *, manifest: Path, toolchain: str, target: str | None, features: Sequence[str]
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "check_no_std_features.py"),
        "--manifest",
        str(manifest),
        "--toolchain",
        toolchain,
    ]
    if target:
        command += ["--target", target]
    if features:
        command += ["--features", ",".join(features)]
    return command


def run_command(command: Sequence[str], *, runner=subprocess.run):
    result = runner(command, cwd=ROOT, check=False)
    if result.returncode:
        raise CommandFailure(command, result.returncode)
    return result


def run_link_command(
    command: Sequence[str],
    *,
    manifest: Path,
    target: str | None,
    profile: str,
    features: Iterable[str],
    runner=subprocess.run,
) -> Path:
    result = runner(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        raise CommandFailure(command, result.returncode)
    map_path = _current_link_map(result.stdout or "")
    return verify_link_map(
        manifest,
        features,
        target=target,
        profile=profile,
        map_path=map_path,
    )


def run_matrix(
    *,
    mode: str,
    manifest: Path,
    toolchain: str,
    target: str | None,
    subsets: Sequence[tuple[str, ...]],
    profile: str,
    dry_run: bool = False,
    runner=subprocess.run,
) -> int:
    for index, selected in enumerate(subsets, 1):
        label = ",".join(selected) or "<empty>"
        command = cargo_command(
            mode=mode,
            manifest=manifest,
            toolchain=toolchain,
            target=target,
            features=selected,
            profile=profile,
        )
        audit = _audit_command(
            manifest=manifest,
            toolchain=toolchain,
            target=target,
            features=selected,
        )
        cases = ",".join(case_labels(selected))
        print(f"[{index}/{len(subsets)}] {mode} features={label} cases={cases}")
        print(f"  cargo: {shlex.join(command)}")
        if dry_run:
            print(f"  audit: {shlex.join(audit)}")
            continue
        try:
            if mode == "link":
                run_link_command(
                    command,
                    manifest=manifest,
                    target=target,
                    profile=profile,
                    features=selected,
                    runner=runner,
                )
            else:
                run_command(command, runner=runner)
            run_command(audit, runner=runner)
        except (CommandFailure, LinkVerificationFailure) as error:
            print(f"FAILED features={label}: {error}", file=sys.stderr)
            if isinstance(error, CommandFailure):
                print(f"reproduce: {shlex.join(error.command)}", file=sys.stderr)
                return error.returncode or 1
            print(f"reproduce: {shlex.join(command)}", file=sys.stderr)
            return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("runtime", "build", "check", "link"), default="runtime")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--toolchain", default="stable")
    parser.add_argument("--target")
    parser.add_argument("--profile", choices=("dev", "release"), default="dev")
    parser.add_argument(
        "--subset",
        action="append",
        help="feature subset (repeatable; aliases include custom-only, serde-only, all)",
    )
    parser.add_argument("--features", help="one comma-separated feature subset")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.manifest:
        manifest = args.manifest
    elif args.mode == "link":
        manifest = LINK_MANIFEST
    else:
        manifest = CONSUMER_MANIFEST
    if not manifest.is_absolute():
        manifest = ROOT / manifest

    try:
        specs = list(args.subset or ())
        if args.features is not None:
            specs.append(args.features)
        subsets = (
            list(feature_subsets())
            if not specs
            else list(dict.fromkeys(parse_subset(spec) for spec in specs))
        )
    except ValueError as error:
        print(f"invalid feature subset: {error}", file=sys.stderr)
        return 2

    return run_matrix(
        mode=args.mode,
        manifest=manifest,
        toolchain=args.toolchain,
        target=args.target,
        subsets=subsets,
        profile=args.profile,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
