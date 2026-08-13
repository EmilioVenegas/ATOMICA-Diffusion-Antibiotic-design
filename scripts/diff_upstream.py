"""Report which vendored DiffSBDD files diverge from upstream.

`DiffSBDD/` is a modified fork rather than a pinned dependency, so the boundary
between upstream code and this project's contribution is otherwise invisible.
This regenerates the changed-file table in MODIFICATIONS.md.

Usage::

    git clone --depth 1 https://github.com/arneschneuing/DiffSBDD /tmp/upstream-diffsbdd
    python scripts/diff_upstream.py --upstream /tmp/upstream-diffsbdd --fork DiffSBDD
"""

import argparse
import difflib
from pathlib import Path


def changed_lines(a: Path, b: Path) -> int:
    """Count added+removed lines between two files, ignoring the +++/--- header."""
    left = a.read_text(errors="replace").splitlines(keepends=True)
    right = b.read_text(errors="replace").splitlines(keepends=True)
    return sum(
        1
        for line in difflib.unified_diff(left, right, n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--fork", default="DiffSBDD", type=Path)
    parser.add_argument(
        "--suffix", default=".py", help="file extension to compare (default: .py)"
    )
    args = parser.parse_args()

    modified, added, removed = [], [], []

    for path in sorted(args.upstream.rglob(f"*{args.suffix}")):
        if ".git" in path.parts:
            continue
        rel = path.relative_to(args.upstream)
        mine = args.fork / rel
        if not mine.exists():
            removed.append(rel)
            continue
        n = changed_lines(path, mine)
        if n:
            modified.append((n, rel))

    for path in sorted(args.fork.rglob(f"*{args.suffix}")):
        if ".git" in path.parts:
            continue
        rel = path.relative_to(args.fork)
        if not (args.upstream / rel).exists():
            added.append(rel)

    modified.sort(reverse=True)

    print(f"# Divergence from upstream ({len(modified)} modified files)\n")
    print("| File | ± lines |")
    print("|---|---:|")
    for n, rel in modified:
        print(f"| `{rel}` | {n} |")

    if added:
        print("\n**New in fork:** " + ", ".join(f"`{p}`" for p in added))
    if removed:
        print("\n**Removed from fork:** " + ", ".join(f"`{p}`" for p in removed))

    total = sum(n for n, _ in modified)
    print(f"\nTotal changed lines: {total}")


if __name__ == "__main__":
    main()
