#!/usr/bin/env python3
"""Write GitHub release notes from the matching section of CHANGES.md.

The script always exits successfully. It uses the changelog section for the
package version, shortened to GitHub's limit. When that section is missing or
the version cannot be read, it uses the full changelog without shortening it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# GitHub rejects a release body longer than this many characters.
GITHUB_RELEASE_BODY_LIMIT = 125_000
TRUNCATION_NOTICE = "\n\n[... release notes truncated; see CHANGES.md ...]\n"
VERSION_PATTERN = re.compile(r"""__version__\s*=\s*['"]([^'"]+)['"]""")


def read_version(about_path: Path) -> str | None:
    """Return ``__version__`` from the about module, or None if it is unreadable."""
    try:
        text = about_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"warning: could not read {about_path}: {exc}", file=sys.stderr)
        return None
    match = VERSION_PATTERN.search(text)
    if match is None:
        print(f"warning: could not find __version__ in {about_path}", file=sys.stderr)
        return None
    return match.group(1)


def extract_section(changes: str, version: str) -> str | None:
    """Return the changelog section headed by ``version``, or None."""
    lines = changes.splitlines()
    start = None
    for index, line in enumerate(lines[:-1]):
        if line.strip() == version and re.fullmatch(r"=+", lines[index + 1].strip()):
            start = index
            break
    if start is None:
        return None
    end = len(lines)
    for index in range(start + 2, len(lines) - 1):
        if lines[index].strip() and re.fullmatch(r"=+", lines[index + 1].strip()):
            end = index
            break
    section = "\n".join(lines[start:end]).strip()
    if not section:
        return None
    return section + "\n"


def truncate(notes: str, limit: int = GITHUB_RELEASE_BODY_LIMIT) -> str:
    """Shorten ``notes`` so the result is at most ``limit`` characters."""
    if len(notes) <= limit:
        return notes if notes.endswith("\n") else notes + "\n"
    notice = TRUNCATION_NOTICE
    if len(notice) >= limit:
        return notice[:limit]
    budget = limit - len(notice)
    chunk = notes[:budget]
    newline = chunk.rfind("\n")
    if newline > budget // 2:
        chunk = chunk[:newline]
    return chunk.rstrip() + notice


def fallback(version: str | None, reason: str) -> str:
    """Return a short note used only when the changelog file itself is unusable."""
    label = f"TabICL {version}" if version else "TabICL"
    return (
        f"{label}\n\n"
        f"Release notes could not be read from CHANGES.md ({reason}).\n"
    )


def build_notes(changes: str, version: str | None) -> str:
    """Build release notes for ``version`` from changelog text.

    A usable section is shortened to GitHub's body limit. Otherwise the full
    changelog is returned without shortening.
    """
    if version:
        section = extract_section(changes, version)
        if section is not None:
            return truncate(section)
        print(
            f"warning: CHANGES.md has no section for {version}; "
            "using the full changelog",
            file=sys.stderr,
        )
    else:
        print(
            "warning: package version could not be read; using the full changelog",
            file=sys.stderr,
        )
    return changes


def write_notes(changes_path: Path, about_path: Path, output_path: Path) -> str:
    """Write release notes to ``output_path`` and return them.

    Failures while reading the changelog or the version become fallback notes.
    """
    version = read_version(about_path)
    try:
        changes = changes_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"warning: could not read {changes_path}: {exc}", file=sys.stderr)
        notes = fallback(version, f"could not read {changes_path.name}")
    except UnicodeError as exc:
        print(f"warning: could not decode {changes_path}: {exc}", file=sys.stderr)
        notes = fallback(version, f"could not decode {changes_path.name}")
    else:
        notes = build_notes(changes, version)
    try:
        output_path.write_text(notes, encoding="utf-8")
    except OSError as exc:
        print(f"warning: could not write {output_path}: {exc}", file=sys.stderr)
        print(notes, end="" if notes.endswith("\n") else "\n")
    return notes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--changes",
        type=Path,
        default=Path("CHANGES.md"),
        help="path to the changelog (default: CHANGES.md)",
    )
    parser.add_argument(
        "--about",
        type=Path,
        default=Path("src/tabicl/__about__.py"),
        help="path to the module that defines __version__",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("release-notes.md"),
        help="where to write the release notes (default: release-notes.md)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        write_notes(args.changes, args.about, args.output)
    except Exception as exc:  # noqa: BLE001 - release notes must not stop a release
        print(f"warning: release notes failed: {exc}", file=sys.stderr)
        try:
            fallback_notes = args.changes.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            fallback_notes = fallback(None, "release notes could not be prepared")
        try:
            args.output.write_text(fallback_notes, encoding="utf-8")
        except OSError:
            print(fallback_notes, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
