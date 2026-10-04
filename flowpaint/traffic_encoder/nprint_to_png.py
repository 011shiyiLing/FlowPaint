#!/usr/bin/env python3
"""Convert nPrint CSV files into RGBA PNG heatmaps compatible with evaluation pipeline."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image

PNG_PADDED_HEIGHT = int(os.environ.get("PNG_PADDED_HEIGHT", "1024"))
RGB_DROP_SUBSTRINGS = ["ipv4_src", "ipv4_dst", "ipv6_src", "ipv6_dst", "src_ip"]

ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
YELLOW = (255, 255, 0)


def load_nprint(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        try:
            header = next(reader)
        except StopIteration as exc:  # pragma: no cover - empty file guard
            raise ValueError(f"nPrint file {path} is empty") from exc
        rows = [row for row in reader]
    if "Unnamed: 0" in header:
        idx = header.index("Unnamed: 0")
        header.pop(idx)
        for row in rows:
            if len(row) > idx:
                row.pop(idx)
    return header, rows


def kept_column_indices(header: Sequence[str]) -> list[int]:
    kept: list[int] = []
    for idx, column in enumerate(header):
        if any(sub in column for sub in RGB_DROP_SUBSTRINGS):
            continue
        kept.append(idx)
    return kept


def value_to_pixel(value: str | None) -> tuple[int, int, int, int]:
    if value is None or value == "":
        return (0, 0, 255, 255)
    try:
        ivalue = int(float(value))
    except ValueError:
        return (0, 0, 255, 255)
    if ivalue == 1:
        return (*ORANGE, 255)
    if ivalue == 0:
        return (*PURPLE, 255)
    if ivalue == -1:
        return (*YELLOW, 255)
    if ivalue > 1:
        return (*ORANGE, min(255, ivalue))
    if ivalue < -1:
        return (*YELLOW, min(255, abs(ivalue)))
    return (*YELLOW, 255)


def nprint_to_png(nprint_path: Path, png_path: Path, *, pad_rows: bool = True) -> None:
    header, rows = load_nprint(nprint_path)
    kept_idx = kept_column_indices(header)
    width = len(kept_idx)
    height = PNG_PADDED_HEIGHT if pad_rows else max(1, len(rows))
    canvas = np.full((height, width, 4), (0, 0, 255, 255), dtype=np.uint8)

    for row_idx, row in enumerate(rows):
        if row_idx >= height:
            break
        if len(row) < len(header):
            row = row + ["-1"] * (len(header) - len(row))
        for pos, col_idx in enumerate(kept_idx):
            value = row[col_idx] if col_idx < len(row) else None
            canvas[row_idx, pos] = value_to_pixel(value)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGBA").save(png_path)


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    start_index: int = 0,
    max_files: int | None = None,
    skip_existing: bool = False,
    *,
    pad_rows: bool = True,
) -> None:
    files = sorted(path for path in input_dir.iterdir() if path.suffix == ".nprint")
    output_dir.mkdir(parents=True, exist_ok=True)
    if start_index:
        files = files[start_index:]
    if max_files is not None and max_files > 0:
        files = files[:max_files]

    for nprint_file in files:
        output_file = output_dir / nprint_file.with_suffix(".png").name
        if skip_existing and output_file.exists():
            print(f"Skipping existing {output_file}")
            continue
        print(f"Processing {nprint_file.name}")
        nprint_to_png(nprint_file, output_file, pad_rows=pad_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", "-i", required=True, type=Path, help="Directory containing .nprint files")
    parser.add_argument("--output_dir", "-o", required=True, type=Path, help="Directory to save PNG images")
    parser.add_argument("--start", type=int, default=0, help="Index of the first .nprint file to process")
    parser.add_argument("--max-files", type=int, default=0, help="Maximum number of files to process (0 = all)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip conversion if the output PNG already exists")
    parser.add_argument(
        "--pad-rows",
        dest="pad_rows",
        action="store_true",
        help="Pad or crop PNG height to PNG_PADDED_HEIGHT rows (default 1024)",
    )
    parser.add_argument(
        "--no-pad-rows",
        dest="pad_rows",
        action="store_false",
        help="Keep PNG height equal to the number of nPrint rows",
    )
    parser.set_defaults(pad_rows=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_files = args.max_files if args.max_files > 0 else None
    convert_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        start_index=args.start,
        max_files=max_files,
        skip_existing=args.skip_existing,
        pad_rows=args.pad_rows,
    )


if __name__ == "__main__":
    main()
