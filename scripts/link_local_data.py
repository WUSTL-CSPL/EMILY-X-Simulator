#!/usr/bin/env python3
"""Reuse IQ and terrain from another local workspace without copying large files."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def link_data(source, destination=ROOT):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    plans = []
    for name in ("EMILY-X", "srtm_data"):
        target, link = source/name, destination/name
        if not target.is_dir():
            raise ValueError(f"Missing data directory: {target}")
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise ValueError(f"Already exists; leaving it untouched: {link}")
        if target.resolve() == link:
            raise ValueError("Source must be a different data directory")
        plans.append((link, target.resolve()))
    for link, target in plans:
        link.symlink_to(target, target_is_directory=True)
        print(f"{link.name} -> {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Workspace containing EMILY-X/ and srtm_data/")
    args = parser.parse_args()
    try:
        link_data(args.source)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
