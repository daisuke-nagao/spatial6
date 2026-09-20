"""Verify fresh resolution and unpacked packages without changing fixture locks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib

from check_no_std import FEATURES

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], directory: Path) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=directory, check=True)


def verify(source: Path, directory: Path, toolchain: str, target: str) -> None:
    for name in ("no-std-consumer", "no-std-link"):
        fixture = directory / name
        shutil.copytree(
            ROOT / "ci" / name, fixture,
            ignore=shutil.ignore_patterns("target", "Cargo.lock"),
        )
        if name == "no-std-consumer":
            manifest = fixture / "Cargo.toml"
            text = manifest.read_text(encoding="utf-8")
            old = 'path = "../.."'
            if text.count(old) != 1:
                raise RuntimeError("consumer spatial6 path must be explicit and unique")
            manifest.write_text(
                text.replace(old, f"path = {json.dumps(source.as_posix())}"),
                encoding="utf-8",
            )
        run(["cargo", f"+{toolchain}", "generate-lockfile"], fixture)
        lock = tomllib.loads((fixture / "Cargo.lock").read_text(encoding="utf-8"))
        print(f"RESOLUTION {fixture}: " + ", ".join(
            f"{p['name']} {p['version']}" for p in lock["package"]
        ), flush=True)

    consumer = directory / "no-std-consumer"
    for selected in ((), ("serde",), FEATURES):
        flags = ["--locked", "--no-default-features"]
        if selected:
            flags += ["--features", ",".join(selected)]
        run(["cargo", f"+{toolchain}", "test", *flags], consumer)
        run(["cargo", f"+{toolchain}", "build", "--lib", "--target", target, *flags], consumer)
        audit = [sys.executable, str(ROOT / "scripts/check_no_std_features.py"),
                 "--manifest", str(consumer / "Cargo.toml"), "--toolchain", toolchain]
        if selected:
            audit += ["--features", ",".join(selected)]
        run(audit, directory)
        run([*audit, "--target", target], directory)
    for release in ([], ["--release"]):
        run(["cargo", f"+{toolchain}", "build", "--locked", "--no-default-features",
             "--features", ",".join(FEATURES), "--target", target, *release],
            directory / "no-std-link")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolchain", required=True)
    parser.add_argument("--target", default="riscv32imac-unknown-none-elf")
    args = parser.parse_args()
    output = ROOT / "target" / "no-std-release"
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{args.toolchain}-", dir=output))
    print(f"Evidence retained in {work}", flush=True)
    package = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["package"]

    fresh = work / "fresh-source"
    fresh.mkdir()
    for item in ("src", "tests", "benches", "examples", "LICENSES"):
        shutil.copytree(ROOT / item, fresh / item)
    for item in ("Cargo.toml", "README.md", "CHANGELOG.md"):
        shutil.copy2(ROOT / item, fresh / item)
    verify(fresh, work / "fresh", args.toolchain, args.target)

    package_target = work / "package-target"
    run(["cargo", f"+{args.toolchain}", "package", "--allow-dirty",
         "--target-dir", str(package_target)], ROOT)
    name = f"{package['name']}-{package['version']}"
    with tarfile.open(package_target / "package" / f"{name}.crate") as archive:
        archive.extractall(work / "unpacked", filter="data")
    unpacked = work / "unpacked" / name
    for original in (ROOT / "src").rglob("*.rs"):
        if original.read_bytes() != (unpacked / original.relative_to(ROOT)).read_bytes():
            raise RuntimeError(f"packaged source mismatch: {original.relative_to(ROOT)}")
    verify(unpacked, work / "packaged", args.toolchain, args.target)
    print(f"PASS fresh and packaged no_std consumers: {args.toolchain}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
