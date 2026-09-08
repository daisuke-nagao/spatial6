import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import keepachangelog


ALLOWED_CATEGORIES = {
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
}
RELEASE_HEADING = re.compile(r"## \[([^]]+)](?: - (.+))?")


def validate_changelog(path: Path) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        return [str(error)]

    errors = []
    lines = content.splitlines()
    if not lines or lines[0] != "# Changelog":
        errors.append("the first line must be '# Changelog'")
    if "All notable changes to this project will be documented in this file." not in content:
        errors.append("the standard changelog introduction is missing")
    if "https://keepachangelog.com/en/1.1.0/" not in content:
        errors.append("the Keep a Changelog 1.1.0 notice is missing")
    if "https://semver.org/spec/v2.0.0.html" not in content:
        errors.append("the Semantic Versioning notice is missing")

    releases = []
    for line in (line for line in lines if line.startswith("## ")):
        match = RELEASE_HEADING.fullmatch(line)
        if match:
            releases.append(match.groups())
        else:
            errors.append(f"invalid release heading: {line!r}")

    names = [name for name, _ in releases]
    if not names or names[0] != "Unreleased":
        errors.append("Unreleased must be the first release section")
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate release: {', '.join(duplicates)}")

    categories = [line[4:] for line in lines if line.startswith("### ")]
    unsupported = sorted(set(categories) - ALLOWED_CATEGORIES)
    if unsupported:
        errors.append(f"unsupported category: {', '.join(unsupported)}")

    try:
        changes = keepachangelog.to_dict(path, show_unreleased=True)
    except Exception as error:
        errors.append(f"keepachangelog could not parse the file: {error}")
        return errors

    release_dates = []
    for name, heading_date in releases:
        release = changes.get(name.lower())
        if release is None:
            errors.append(f"release {name!r} could not be parsed")
            continue

        if release.get("uncategorized"):
            errors.append(f"release {name!r} contains uncategorized content")
        for category, entries in release.items():
            if category not in {"metadata", "uncategorized"} and not entries:
                errors.append(f"release {name!r} has an empty {category!r} category")

        metadata = release["metadata"]
        if not metadata.get("url"):
            errors.append(f"release {name!r} is missing a reference link")

        if name == "Unreleased":
            if heading_date:
                errors.append("Unreleased must not have a release date")
            continue

        if "semantic_version" not in metadata:
            errors.append(f"release {name!r} is not a valid semantic version")
        try:
            parsed_date = date.fromisoformat(heading_date or "")
            if parsed_date.isoformat() != heading_date:
                raise ValueError
            release_dates.append(parsed_date)
        except ValueError:
            errors.append(f"release {name!r} must have an ISO 8601 date")

    if release_dates != sorted(release_dates, reverse=True):
        errors.append("release dates must be ordered from newest to oldest")

    return errors


def main(arguments: list[str]) -> int:
    if not arguments:
        print("no changelog file was provided", file=sys.stderr)
        return 1

    failed = False
    for argument in arguments:
        path = Path(argument)
        for error in validate_changelog(path):
            print(f"{path}: {error}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
