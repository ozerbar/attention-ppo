#!/usr/bin/env python3
"""align_images.py

Stack three images **vertically**, resizing each one to 4×3 inches so the
result can be dropped onto an A0-sized poster without further tweaking.

Usage
-----
$ python visualization/align_images.py img1.png img2.png img3.png -o out.png [--dpi 300]

The script will create *out.png* (defaults to *stacked.png*) where
• each individual image is 4 in × 3 in at the chosen DPI,
• the total canvas is 4 in × 9 in.
"""
from __future__ import annotations

"""USER CONFIGURATION edit only this block"""
# Absolute or relative paths to the input images (top → bottom)
IMG_TOP    = "/home/damlakonur/tum-adlr-01/tum-adlr-01/figures/multi_noise_comparison_ramp.png"
IMG_MIDDLE = "/home/damlakonur/tum-adlr-01/tum-adlr-01/figures/multi_noise_comparison_uniform.png"
IMG_BOTTOM = "/home/damlakonur/tum-adlr-01/tum-adlr-01/figures/multi_noise_comparison_Gaussian.png"

# Where to save the stacked image
OUTPUT_FILE = "stacked.png"

# Resolution of the exported figure
DPI = 300  # typical print quality

# ────────────────────────────────────────────────

from pathlib import Path
from typing import Sequence

from PIL import Image

WIDTH_INCHES = 2.6            # final strip width
HEIGHT_INCHES = 5.55 / 3      # height per image ≈ 3.1833"


def _resize(img: Image.Image, width_px: int, height_px: int) -> Image.Image:
    """Simply stretch/shrink image to *exact* target size (no cropping, may distort)."""
    return img.resize((width_px, height_px), Image.LANCZOS)


def stack_images(paths: Sequence[str], out_path: str, dpi: int = DPI) -> None:
    if len(paths) != 3:
        raise ValueError("Exactly three input images are required.")

    width_px = int(WIDTH_INCHES * dpi)
    height_px = int(HEIGHT_INCHES * dpi)

    resized = [_resize(Image.open(p).convert("RGB"), width_px, height_px) for p in paths]

    canvas = Image.new("RGB", (width_px, height_px * 3), color="white")
    for idx, img in enumerate(resized):
        canvas.paste(img, (0, idx * height_px))

    canvas.save(out_path, dpi=(dpi, dpi))
    print(f"[✓] Saved stacked image to {out_path} ({WIDTH_INCHES}×{HEIGHT_INCHES*3} inches @ {dpi} DPI)")


def main() -> None:
    stack_images([IMG_TOP, IMG_MIDDLE, IMG_BOTTOM], OUTPUT_FILE, DPI)


if __name__ == "__main__":
    main()
