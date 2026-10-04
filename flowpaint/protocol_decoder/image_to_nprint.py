#!/usr/bin/env python3
"""Convert diffusion-generated PNG heatmaps back into nPrint CSV tables."""

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
PADDING = (0, 0, 255)


def load_reference_nprint(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Reference nPrint {path} is empty") from exc
        rows = [row for row in reader]
    if "Unnamed: 0" in header:
        idx = header.index("Unnamed: 0")
        header.pop(idx)
        for row in rows:
            if len(row) > idx:
                row.pop(idx)
    if "src_ip" in header:
        idx = header.index("src_ip")
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


# Palette for nearest-neighbor decoding: every pixel is assigned the nPrint value
# of the closest encoder color (Euclidean distance in RGB). The blue padding color
# used for rows beyond the end of a short flow decodes to a placeholder.
PALETTE = np.array([ORANGE, PURPLE, YELLOW, PADDING], dtype=np.int32)
PALETTE_VALUES = np.array(["1", "0", "-1", "-1"], dtype=object)


def decode_pixels(pixels: np.ndarray) -> np.ndarray:
    """Map an (H, W, 3|4) pixel array to nPrint values by nearest palette color."""
    rgb = pixels[..., :3].astype(np.int32)
    dist = ((rgb[..., None, :] - PALETTE[None, None, :, :]) ** 2).sum(axis=-1)
    return PALETTE_VALUES[dist.argmin(axis=-1)]


def read_png_rgba(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)


def fit_pixels(pixels: np.ndarray, width: int, padded_height: int, rows: int) -> np.ndarray:
    """Bring a generated image back to the encoder geometry and drop padding rows.

    The encoder pads every flow to ``padded_height`` rows. If the editor returned
    a different resolution, the image is first resized (nearest neighbor) to
    ``padded_height x width``; the first ``rows`` rows are then the flow itself.
    """
    height = max(padded_height, rows)
    if pixels.shape[0] != height or pixels.shape[1] != width:
        image = Image.fromarray(pixels, mode="RGBA")
        image = image.resize((width, height), Image.Resampling.NEAREST)
        pixels = np.array(image, dtype=np.uint8)
    return pixels[:rows]


def write_nprint(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerows(rows)


def reconstruct_nprint(
    reference_path: Path,
    png_path: Path,
    output_path: Path,
) -> None:
    header, reference_rows = load_reference_nprint(reference_path)
    kept_idx = kept_column_indices(header)
    expected_rows = len(reference_rows)
    width = len(kept_idx)

    pixels = fit_pixels(read_png_rgba(png_path), width, PNG_PADDED_HEIGHT, expected_rows)
    values = decode_pixels(pixels)

    rebuilt_rows: List[List[str]] = []
    for row_idx in range(expected_rows):
        if row_idx < len(reference_rows):
            row = list(reference_rows[row_idx])
        else:
            row = ["-1"] * len(header)
        if len(row) < len(header):
            row.extend(["-1"] * (len(header) - len(row)))
        for pos, col_idx in enumerate(kept_idx):
            if col_idx >= len(row):
                row.extend(["-1"] * (col_idx - len(row) + 1))
            row[col_idx] = values[row_idx, pos]
        rebuilt_rows.append(row[: len(header)])

    write_nprint(output_path, header, rebuilt_rows)


def convert_directory(
    reference_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> None:
    png_files = sorted(path for path in input_dir.iterdir() if path.suffix.lower() == ".png")
    if not png_files:
        raise RuntimeError(f"No PNG files found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for png_path in png_files:
        output_path = output_dir / png_path.with_suffix(".nprint").name
        print(f"Processing {png_path.name}")
        reconstruct_nprint(
            reference_path=reference_path,
            png_path=png_path,
            output_path=output_path,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org_nprint", required=True, type=Path, help="Reference .nprint file")
    parser.add_argument("--input_dir", required=True, type=Path, help="Directory containing PNG files")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to store reconstructed .nprint files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_directory(
        reference_path=args.org_nprint,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
