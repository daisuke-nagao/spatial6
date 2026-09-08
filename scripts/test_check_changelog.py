import tempfile
import unittest
from pathlib import Path

from scripts.check_changelog import validate_changelog


VALID_UNRELEASED = """# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A new API.

[Unreleased]: https://github.com/daisuke-nagao/spatial6/commits/main
"""


class CheckChangelogTests(unittest.TestCase):
    def validate(self, content: str) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "CHANGELOG.md"
            path.write_text(content, encoding="utf-8")
            return validate_changelog(path)

    def test_valid_unreleased_changelog(self) -> None:
        self.assertEqual(self.validate(VALID_UNRELEASED), [])

    def test_reports_representative_malformed_changelog(self) -> None:
        errors = self.validate(
            """# Wrong title

## [1.0] - 2026-1-2

### Added

## [Unreleased] - 2026-01-03

### Other

- A change.

[Unreleased]: https://example.com/unreleased
"""
        )
        self.assertIn("the first line must be '# Changelog'", errors)
        self.assertIn("the standard changelog introduction is missing", errors)
        self.assertIn("Unreleased must be the first release section", errors)
        self.assertIn("unsupported category: Other", errors)
        self.assertIn("release '1.0' has an empty 'added' category", errors)
        self.assertIn("release '1.0' is not a valid semantic version", errors)
        self.assertIn("Unreleased must not have a release date", errors)
        self.assertIn("release '1.0' is missing a reference link", errors)
        self.assertIn("release '1.0' must have an ISO 8601 date", errors)


if __name__ == "__main__":
    unittest.main()
