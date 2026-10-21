"""Read the three curves back out of the original MATLAB figures (original/pic/).

The figures plot log10 x (blue), log10 y (red) and log10 z (black, dashed)
against tau. Each line has a dark core with a coloured halo, so every short
dark run in a pixel column is classified by the colour around it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Pixel positions of the axis frame, read off the images, and the axis limits.
FIGURES = {
    "Gambar 2": dict(file="image.png", f_label=0.1, x0=157, x1=1837, t_max=7,
                     y0=102, y1=1430, legend=(1300, 1840, 120, 360)),
    "Gambar 3": dict(file="Screenshot 2024-06-22 at 17.02.48.png", f_label=0.4, x0=149.5, x1=1688.5,
                     t_max=10, y0=93.5, y1=1312.5, legend=(1180, 1660, 130, 350)),
}
Y_TOP, Y_BOTTOM = 3.0, -7.0


def digitize(path: Path, x0, x1, t_max, y0, y1, legend, step: int = 3, **_) -> dict[str, np.ndarray]:
    im = np.asarray(Image.open(path).convert("RGB")).astype(int)
    lx0, lx1, ly0, ly1 = legend
    out = {"x": [], "y": [], "z": []}
    for j in range(int(x0) + 6, int(x1) - 5, step):
        col = im[:, j]
        rows = np.where(col.sum(axis=1) < 420)[0]
        rows = rows[(rows > y0 + 5) & (rows < y1 - 5)]
        if lx0 < j < lx1:
            rows = rows[(rows < ly0) | (rows > ly1)]
        if rows.size == 0:
            continue
        t = (j - x0) / (x1 - x0) * t_max
        for run in np.split(rows, np.where(np.diff(rows) > 2)[0] + 1):
            if len(run) > 14:  # vertical stretch of a jump
                continue
            r, g, b = col[max(run[0] - 6, 0): run[-1] + 7].mean(axis=0)
            core = col[run].mean(axis=0)
            if r - g > 18 and r - b > 12:
                key = "y"
            elif b - r > 12:
                key = "x"
            elif core.max() - core.min() < 25:
                key = "z"
            else:
                continue
            value = Y_TOP - (run.mean() - y0) / (y1 - y0) * (Y_TOP - Y_BOTTOM)
            out[key].append((t, value))
    curves = {}
    for key, pts in out.items():
        pts = np.array(pts)
        # drop tick marks on the frame and the start-up spike
        keep = (pts[:, 0] > 0.12) & (pts[:, 0] < t_max - 0.15) & (pts[:, 1] < 2.6) & (pts[:, 1] > -6.8)
        curves[key] = pts[keep]
    return curves


def digitize_all(pic_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    return {name: digitize(pic_dir / spec["file"], **spec) for name, spec in FIGURES.items()}
