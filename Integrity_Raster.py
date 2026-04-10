#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute a biosphere functional integrity indicator that consists in a convolution of 1000m radius on a binary classified land cover database.  
The program is designed to use a land cover database in RASTER format.

The program takes as input:
- a territory vector layer where to compute biosphere functional integrity (BFI),
- a land-cover raster to be classified,
- class assignments of the land-cover raster values for semi-natural (1), non-semi-natural (0), and null values (NaN)
- settings for the convolution to be made to calculate the functional integrity, and histogram outputs (with default settings provided aligned with the IFB definition in Mohamed et al., 2024).
- settings for multithreading (to improve program's performance) and tile size for processing.

It is organized into three processing blocks.

Block 1    
    - Analyze raster values inside the calculation area, defined as the input
    territory buffered by the effective convolution radius.
    - Display the raster values found in this area and validate that every value
    is assigned to exactly one of the three class groups (0/1/NaN).

Block 2
    Compute functional integrity and export two raster outputs:
    - a raster providing the values used to calculate biosphere functional integrity, consisting in a raster with values 0 / 1 / NaN at the extent of the input vector layer buffered by the convolution radius.
    - a raster providing the value of biosphere functional integrity, in the [0, 1] range, at the extent of the input territory.

Block 3
    Export a histogram of functional integrity values over the territory, with a
    user-defined bin width and threshold. The histogram is exported as:
    - a CSV table,
    - a PNG figure.

Notes
-----
- Exact circular convolution uses SciPy when available.
- If SciPy is unavailable, the program falls back to a box-based local mean.
- Console output uses Python's standard ``logging`` module.
- Docstrings follow a NumPy-style layout compatible with general PEP 257 docstring conventions.

"""

from __future__ import annotations

import csv
import logging
import math
import os
import re
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely.ops import unary_union
from tqdm import tqdm


# =============================================================================
# USER PARAMETERS
# =============================================================================

# Active dataset
RASTER_PATH: str = r"path\to\input_raster.tiff"
VECTOR_PATH: str = r"path\to\territory_vector.gpkg"

# Raster outputs
OUT_TIF: str = r"path\to\integrity_output.tif"
SN_TIF: str = r"path\to\binary_output.tif"

# Histogram outputs
HISTO_CSV: str = r"path\to\histogram.csv"
HISTO_PNG: str = r"path\to\histogram.png"

# Raster class definitions
CLASSES_1: str = ""                        # semi-natural = 1
CLASSES_0: str = ""       # non-semi-natural = 0
CLASSES_NULL: str = ""                             # ignored = NaN

# Functional integrity settings
CONV_SIZE_M: float = 1000.0
PIXEL_MAX_M: float | None = None
N_JOBS: int = 7
TILE_PX: int = 2048
KERNEL_SHAPE: str = "circular_fft"                      # "circular_fft" or "box"

# Histogram settings
BIN_WIDTH: float = 0.01
THRESHOLDS: list[float] = [0.25]
EXCLUDE_ZERO_IN_HISTOGRAM: bool = True

# General settings
STRICT_VALIDATION: bool = True
VERBOSE: bool = True
LOG_LEVEL: str = "INFO"


# =============================================================================
# LOGGING HELPERS
# =============================================================================

LOGGER = logging.getLogger("functional_integrity")


def configure_logging(level: str = "INFO") -> None:
    """Configure console logging for the workflow.

    Parameters
    ----------
    level : str, default="INFO"
        Logging threshold passed to the console handler.
    """
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, level.upper(), logging.INFO))
    LOGGER.propagate = False

    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)


def log_section(title: str) -> None:
    """Log a visible section separator.

    Parameters
    ----------
    title : str
        Section title displayed between separator lines.
    """
    LOGGER.info("%s", "=" * 88)
    LOGGER.info("%s", title)
    LOGGER.info("%s", "=" * 88)


def log_block_start(block_id: int, title: str) -> None:
    """Log the start of a processing block.

    Parameters
    ----------
    block_id : int
        Block index in the workflow.
    title : str
        Human-readable block title.
    """
    log_section(f"START | BLOCK {block_id}/3 | {title}")


def log_block_end(block_id: int, title: str) -> None:
    """Log the end of a processing block.

    Parameters
    ----------
    block_id : int
        Block index in the workflow.
    title : str
        Human-readable block title.
    """
    LOGGER.info("COMPLETE | BLOCK %s/3 | %s", block_id, title)


def ensure_parent_dirs(paths: Iterable[str]) -> None:
    """Create parent directories for output file paths.

    Parameters
    ----------
    paths : Iterable[str]
        Output file paths whose parent directories must exist.
    """
    for path in paths:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)


# =============================================================================
# CLASS SPECIFICATION PARSING
# =============================================================================

_RANGE_RE = re.compile(r"^\s*(-?\d+)\s+thru\s+(-?\d+)\s*$", re.IGNORECASE)


def parse_class_spec(spec: str | None) -> list[tuple[int, int]]:
    """Parse a class specification string into inclusive integer ranges.

    Parameters
    ----------
    spec : str or None
        Raw class specification. Supported syntax includes individual values and
        inclusive ranges written as ``10 thru 20``.

    Returns
    -------
    list of tuple of int
        Inclusive ``(min_value, max_value)`` ranges.
    """
    if spec is None:
        return []

    spec = spec.strip()
    if spec == "" or spec == "*":
        return []

    parts = re.split(r"[;,\n]+", spec)
    ranges: list[tuple[int, int]] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        match = _RANGE_RE.match(part)
        if match:
            a, b = int(match.group(1)), int(match.group(2))
            ranges.append((min(a, b), max(a, b)))
            continue

        for token in part.split():
            token = token.strip()
            if token:
                value = int(token)
                ranges.append((value, value))

    return ranges


def build_mask_from_ranges(arr_int: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    """Build a boolean mask for values covered by inclusive ranges.

    Parameters
    ----------
    arr_int : numpy.ndarray
        Integer raster array.
    ranges : list of tuple of int
        Inclusive value ranges.

    Returns
    -------
    numpy.ndarray
        Boolean mask with the same shape as ``arr_int``.
    """
    if not ranges:
        return np.zeros(arr_int.shape, dtype=bool)

    mask = np.zeros(arr_int.shape, dtype=bool)
    for lower, upper in ranges:
        mask |= (arr_int >= lower) & (arr_int <= upper)
    return mask


def value_in_ranges(value: int, ranges: list[tuple[int, int]]) -> bool:
    """Check whether a value belongs to at least one inclusive range.

    Parameters
    ----------
    value : int
        Value to test.
    ranges : list of tuple of int
        Inclusive value ranges.

    Returns
    -------
    bool
        ``True`` if the value is covered by at least one range.
    """
    return any(lower <= value <= upper for lower, upper in ranges)


# =============================================================================
# GEOMETRY AND WINDOW HELPERS
# =============================================================================


def make_output_profile(
    src_profile: dict,
    window: Window,
    transform: rasterio.Affine,
) -> dict:
    """Create an output raster profile cropped to a given window.

    Parameters
    ----------
    src_profile : dict
        Source raster profile.
    window : rasterio.windows.Window
        Output raster window in source pixel coordinates.
    transform : affine.Affine
        Output raster transform for this cropped window.

    Returns
    -------
    dict
        Updated output profile.
    """
    profile = src_profile.copy()
    profile.update(
        driver="GTiff",
        width=int(window.width),
        height=int(window.height),
        transform=transform,
        dtype="float32",
        nodata=-9999.0,
        count=1,
        compress="LZW",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    return profile


def window_to_slices(window: Window) -> tuple[slice, slice]:
    """Convert a raster window to row and column slices.

    Parameters
    ----------
    window : rasterio.windows.Window
        Input window.

    Returns
    -------
    tuple of slice
        Row and column slices.
    """
    row_slice = slice(int(window.row_off), int(window.row_off + window.height))
    col_slice = slice(int(window.col_off), int(window.col_off + window.width))
    return row_slice, col_slice


# =============================================================================
# BLOCK 1 - RASTER ANALYSIS AND VALIDATION
# =============================================================================


def collect_unique_values(
    src: rasterio.io.DatasetReader,
    mask_raster: np.ndarray,
    window: Window,
    step_px: int = 2048,
) -> np.ndarray:
    """Collect unique raster values inside a processing mask.

    Parameters
    ----------
    src : rasterio.io.DatasetReader
        Open raster dataset.
    mask_raster : numpy.ndarray
        Rasterized binary mask of the buffered calculation area.
    window : rasterio.windows.Window
        Minimal raster window covering the calculation area.
    step_px : int, default=2048
        Tile size used to scan the raster.

    Returns
    -------
    numpy.ndarray
        Sorted array of unique integer raster values.
    """
    unique_vals: np.ndarray | None = None
    tile = int(step_px)
    rows = math.ceil(int(window.height) / tile)
    cols = math.ceil(int(window.width) / tile)

    for row_idx in tqdm(range(rows), desc="Block 1 - scanning raster tiles", unit="row"):
        for col_idx in range(cols):
            win = Window(
                int(window.col_off) + col_idx * tile,
                int(window.row_off) + row_idx * tile,
                min(tile, int(window.width) - col_idx * tile),
                min(tile, int(window.height) - row_idx * tile),
            )
            arr = src.read(1, window=win, resampling=Resampling.nearest)
            row_slice, col_slice = window_to_slices(win)
            msk = mask_raster[row_slice, col_slice]

            if src.nodata is not None:
                arr = arr[(msk == 1) & (arr != src.nodata)]
            else:
                arr = arr[msk == 1]

            if arr.size == 0:
                continue

            values = np.unique(arr.astype(np.int64, copy=False))
            unique_vals = values if unique_vals is None else np.union1d(unique_vals, values)

    if unique_vals is None:
        return np.array([], dtype=np.int64)
    return unique_vals.astype(np.int64)


def analyze_raster_values(
    unique_vals: np.ndarray,
    ranges_1: list[tuple[int, int]],
    ranges_0: list[tuple[int, int]],
    ranges_null: list[tuple[int, int]],
) -> tuple[list[tuple[int, list[str]]], list[int], dict[int, list[str]]]:
    """Analyze raster values against class definitions.

    Parameters
    ----------
    unique_vals : numpy.ndarray
        Unique raster values encountered in the calculation area.
    ranges_1 : list of tuple of int
        Inclusive ranges assigned to ``CLASSES_1``.
    ranges_0 : list of tuple of int
        Inclusive ranges assigned to ``CLASSES_0``.
    ranges_null : list of tuple of int
        Inclusive ranges assigned to ``CLASSES_NULL``.

    Returns
    -------
    overlaps : list of tuple
        Values assigned to more than one class, with their conflicting class
        names.
    unclassified : list of int
        Values present in the raster but assigned to no class.
    details : dict of int to list of str
        Membership details for every encountered value.
    """
    overlaps: list[tuple[int, list[str]]] = []
    unclassified: list[int] = []
    details: dict[int, list[str]] = {}

    for value in unique_vals.tolist():
        memberships: list[str] = []
        if value_in_ranges(value, ranges_1):
            memberships.append("CLASSES_1")
        if value_in_ranges(value, ranges_0):
            memberships.append("CLASSES_0")
        if value_in_ranges(value, ranges_null):
            memberships.append("CLASSES_NULL")

        details[value] = memberships

        if len(memberships) == 0:
            unclassified.append(value)
        elif len(memberships) > 1:
            overlaps.append((value, memberships))

    return overlaps, unclassified, details


def print_raster_analysis(
    unique_vals: np.ndarray,
    overlaps: list[tuple[int, list[str]]],
    unclassified: list[int],
) -> None:
    """Log the Block 1 raster analysis report.

    Parameters
    ----------
    unique_vals : numpy.ndarray
        Unique raster values encountered in the calculation area.
    overlaps : list of tuple
        Values assigned to multiple classes.
    unclassified : list of int
        Values not covered by any class.
    """
    values_str = " ".join(map(str, unique_vals.tolist())) if unique_vals.size else "<no values found>"
    LOGGER.info("REPORT | Raster values found inside the calculation area: %s", values_str)

    if overlaps:
        LOGGER.error("REPORT | Values assigned to multiple classes:")
        for value, memberships in overlaps:
            LOGGER.error("REPORT |   %s -> %s", value, ", ".join(memberships))

    if unclassified:
        LOGGER.error(
            "REPORT | Values present in the raster but not assigned to any class: %s",
            " ".join(map(str, unclassified)),
        )


# =============================================================================
# BLOCK 2 - CONVOLUTION AND FUNCTIONAL INTEGRITY
# =============================================================================


def circular_kernel(size: int) -> np.ndarray:
    """Build a circular kernel as a boolean array.

    Parameters
    ----------
    size : int
        Odd kernel size in pixels.

    Returns
    -------
    numpy.ndarray
        Boolean circular kernel.
    """
    radius = size // 2
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius


def box_integral_mean(valid: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    """Approximate a local mean using integral images and a square window.

    Parameters
    ----------
    valid : numpy.ndarray
        Boolean mask marking valid pixels.
    values : numpy.ndarray
        Float array containing the raster values to average.
    k : int
        Window size in pixels.

    Returns
    -------
    numpy.ndarray
        Local mean raster.

    Notes
    -----
    This function is used as a fallback when FFT-based circular convolution is
    unavailable.
    """
    pad = k // 2

    def int2d(array: np.ndarray) -> np.ndarray:
        return array.cumsum(axis=0).cumsum(axis=1)

    def rect(ii: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> np.number:
        total = ii[y1, x1]
        if y0 > 0:
            total -= ii[y0 - 1, x1]
        if x0 > 0:
            total -= ii[y1, x0 - 1]
        if y0 > 0 and x0 > 0:
            total += ii[y0 - 1, x0 - 1]
        return total

    height, width = values.shape
    val = np.nan_to_num(values, nan=0.0).astype(np.float64)
    cnt = valid.astype(np.int32)
    val_pad = np.pad(val, pad, mode="reflect")
    cnt_pad = np.pad(cnt, pad, mode="reflect")
    ii_val, ii_cnt = int2d(val_pad), int2d(cnt_pad)
    out = np.empty_like(values, dtype=np.float32)

    for y in range(height):
        y0, y1 = y, y + 2 * pad
        for x in range(width):
            x0, x1 = x, x + 2 * pad
            sum_val = rect(ii_val, y0, x0, y1, x1)
            sum_cnt = rect(ii_cnt, y0, x0, y1, x1)
            out[y, x] = sum_val / sum_cnt if sum_cnt > 0 else np.nan

    return out


def circular_mean_fft(valid: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    """Compute an exact circular local mean using FFT convolution.

    Parameters
    ----------
    valid : numpy.ndarray
        Boolean mask marking valid pixels.
    values : numpy.ndarray
        Float array containing the raster values to average.
    k : int
        Circular kernel size in pixels.

    Returns
    -------
    numpy.ndarray
        Local mean raster.

    Raises
    ------
    RuntimeError
        If SciPy is unavailable in the runtime environment.
    """
    try:
        from scipy.signal import fftconvolve
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError("SciPy is not available for fftconvolve.") from exc

    kernel = circular_kernel(k).astype(np.float32)
    vals = np.nan_to_num(values, nan=0.0).astype(np.float32)
    cnt = valid.astype(np.float32)

    sum_vals = fftconvolve(vals, kernel, mode="same")
    sum_cnt = fftconvolve(cnt, kernel, mode="same")

    out = np.divide(
        sum_vals,
        sum_cnt,
        out=np.full(values.shape, np.nan, dtype=np.float32),
        where=(sum_cnt > 0),
    )
    return out.astype(np.float32)


def run_convolution(valid: np.ndarray, values: np.ndarray, k: int, kernel_shape: str) -> np.ndarray:
    """Run the selected local mean implementation.

    Parameters
    ----------
    valid : numpy.ndarray
        Boolean mask marking valid pixels.
    values : numpy.ndarray
        Float array containing the raster values to average.
    k : int
        Kernel size in pixels.
    kernel_shape : {"circular_fft", "box"}
        Convolution mode.

    Returns
    -------
    numpy.ndarray
        Local mean raster.
    """
    if kernel_shape == "circular_fft":
        try:
            return circular_mean_fft(valid, values, k)
        except Exception as exc:
            warnings.warn(
                f"'circular_fft' is unavailable ({exc}). Falling back to 'box'.",
                stacklevel=2,
            )
            LOGGER.warning("FALLBACK | 'circular_fft' unavailable. Using 'box' convolution instead.")
            return box_integral_mean(valid, values, k)

    return box_integral_mean(valid, values, k)


def classify_binary_tile(
    arr: np.ndarray,
    nodata_in: float | int | None,
    ranges_1: list[tuple[int, int]],
    ranges_0: list[tuple[int, int]],
    ranges_null: list[tuple[int, int]],
) -> np.ndarray:
    """Classify a raster tile into 0 / 1 / NaN.

    Parameters
    ----------
    arr : numpy.ndarray
        Input raster tile.
    nodata_in : float, int, or None
        Input raster nodata value.
    ranges_1, ranges_0, ranges_null : list of tuple of int
        Inclusive class ranges.

    Returns
    -------
    numpy.ndarray
        Float tile with values 0, 1, or NaN.
    """
    arr_int = arr.astype(np.int64, copy=False)

    mask_null = build_mask_from_ranges(arr_int, ranges_null)
    mask_1 = build_mask_from_ranges(arr_int, ranges_1)
    mask_0 = build_mask_from_ranges(arr_int, ranges_0)

    sn = np.full(arr_int.shape, np.nan, dtype=np.float32)
    sn[mask_1] = 1.0
    sn[mask_0] = 0.0
    sn[mask_null] = np.nan

    if nodata_in is not None:
        sn[arr == nodata_in] = np.nan

    return sn


def process_integrity_tile(
    arr: np.ndarray,
    nodata_in: float | int | None,
    ranges_1: list[tuple[int, int]],
    ranges_0: list[tuple[int, int]],
    ranges_null: list[tuple[int, int]],
    k_size: int,
    halo_top: int,
    halo_bottom: int,
    halo_left: int,
    halo_right: int,
    kernel_shape: str,
) -> np.ndarray:
    """Process one territory tile and return its integrity values.

    Parameters
    ----------
    arr : numpy.ndarray
        Raster tile including its halo.
    nodata_in : float, int, or None
        Input raster nodata value.
    ranges_1, ranges_0, ranges_null : list of tuple of int
        Inclusive class ranges.
    k_size : int
        Kernel size in pixels.
    halo_top, halo_bottom, halo_left, halo_right : int
        Effective halo sizes around the tile.
    kernel_shape : {"circular_fft", "box"}
        Convolution mode.

    Returns
    -------
    numpy.ndarray
        Functional integrity tile without halo.
    """
    sn = classify_binary_tile(arr, nodata_in, ranges_1, ranges_0, ranges_null)
    valid = ~np.isnan(sn)
    fi = run_convolution(valid, sn, k_size, kernel_shape)

    y0 = halo_top
    x0 = halo_left
    y1 = fi.shape[0] - halo_bottom
    x1 = fi.shape[1] - halo_right
    return fi[y0:y1, x0:x1]


def process_integrity_tile_timed(args: tuple) -> tuple[bool, np.ndarray | str, float, int, int]:
    """Process one integrity tile and attach execution timing metadata.

    Parameters
    ----------
    args : tuple
        Positional arguments forwarded to :func:`process_integrity_tile`, plus a
        final ``(row_idx, col_idx)`` tuple.

    Returns
    -------
    tuple
        Tuple containing success status, payload or error representation,
        elapsed time, row index, and column index.
    """
    t0 = time.perf_counter()
    row_idx, col_idx = args[-1]
    try:
        fi = process_integrity_tile(*args[:-1])
        dt = time.perf_counter() - t0
        return True, fi, dt, row_idx, col_idx
    except Exception as exc:
        dt = time.perf_counter() - t0
        return False, repr(exc), dt, row_idx, col_idx


# =============================================================================
# BLOCK 3 - HISTOGRAM EXPORT
# =============================================================================


def save_histogram_png(
    bins_left: np.ndarray,
    bins_right: np.ndarray,
    counts: np.ndarray,
    out_png_path: str,
    thr_results: list[tuple[float, float]] | None = None,
) -> None:
    """Save the histogram figure as a PNG file.

    Parameters
    ----------
    bins_left, bins_right : numpy.ndarray
        Left and right bin edges.
    counts : numpy.ndarray
        Histogram counts.
    out_png_path : str
        Output PNG path.
    thr_results : list of tuple of float, optional
        Threshold summaries displayed on the chart.
    """
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    colors_hex = [
        "#a50f15", "#de2d26", "#fb6a4a", "#fcae91", "#fee5d9",
        "#F7FCFD", "#E5EFE5", "#D4E2DD", "#C2D5CD", "#B0C7BC",
        "#9FBAAC", "#8DAD9C", "#7CA08C", "#6A937C", "#58866C",
        "#47795C", "#356B4B", "#235E3B", "#12512B", "#00441B",
    ]

    def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    colors = [hex_to_rgb01(color) for color in colors_hex]
    class_width = 1.0 / len(colors)

    centers = (bins_left + bins_right) / 2.0
    widths = bins_right - bins_left

    bar_colors = []
    for left, right in zip(bins_left, bins_right):
        mid = (left + right) / 2.0
        idx = int(mid / class_width)
        idx = min(idx, len(colors) - 1)
        bar_colors.append(colors[idx])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(centers, counts, width=widths, edgecolor="black", color=bar_colors)
    ax.set_xlabel("(Semi)-natural area in the surrounding km²")
    ax.set_ylabel("Pixel count")
    ax.set_title("")
    ax.spines["bottom"].set_position(("data", 0))
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if thr_results:
        threshold, proportion = thr_results[0]
        ax.axvline(x=threshold, linestyle="--", linewidth=0.8)
        text = f"≥ {threshold:.2f} : {proportion * 100:.1f}%"
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            threshold,
            -0.13,
            text,
            transform=trans,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            clip_on=False,
        )

    plt.savefig(out_png_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_histogram(
    path_tif: str,
    bin_width: float = 0.01,
    exclude_zero: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute the histogram of a float raster in the [0, 1] range.

    Parameters
    ----------
    path_tif : str
        Input raster path.
    bin_width : float, default=0.01
        Histogram bin width. Must divide 1.0 exactly.
    exclude_zero : bool, default=True
        If ``True``, pixels equal to zero are excluded from the histogram.

    Returns
    -------
    bins : numpy.ndarray
        Histogram bin edges.
    counts : numpy.ndarray
        Histogram counts.
    total_valid : int
        Total number of valid pixels included in the histogram.

    Raises
    ------
    ValueError
        If ``bin_width`` does not divide 1.0 exactly.
    """
    n_bins = int(round(1.0 / bin_width))
    if not math.isclose(n_bins * bin_width, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("BIN_WIDTH must divide 1.0 exactly (for example 0.01, 0.02, or 0.005).")

    bins = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    total_valid = 0

    with rasterio.open(path_tif) as src:
        nodata = src.nodata
        windows_iter = src.block_windows(1)

        for _, window in tqdm(windows_iter, desc="Block 3 - raster blocks", unit="block"):
            data = src.read(1, window=window, masked=True).astype(np.float32)

            if nodata is not None:
                data = np.ma.masked_equal(data, nodata)
            data = np.ma.masked_invalid(data)

            if exclude_zero:
                data = np.ma.masked_where(data == 0, data)

            if data.count() == 0:
                continue

            arr = np.clip(data.compressed(), 0.0, 1.0)
            block_counts, _ = np.histogram(arr, bins=bins)
            counts += block_counts
            total_valid += arr.size

    return bins, counts, int(total_valid)


def summarize_thresholds(
    counts: np.ndarray,
    total_valid: int,
    thresholds: list[float],
    bin_width: float,
) -> list[tuple[float, float]]:
    """Compute the proportion of pixels above each threshold.

    Parameters
    ----------
    counts : numpy.ndarray
        Histogram counts.
    total_valid : int
        Total number of valid pixels.
    thresholds : list of float
        Threshold values to summarize.
    bin_width : float
        Histogram bin width.

    Returns
    -------
    list of tuple of float
        ``(threshold, proportion_ge_threshold)`` values.
    """
    results: list[tuple[float, float]] = []
    for threshold in thresholds:
        threshold = float(threshold)

        if total_valid == 0:
            results.append((threshold, 0.0))
            continue

        if threshold <= 0.0:
            proportion = 1.0
        elif threshold >= 1.0:
            proportion = counts[-1] / total_valid
        else:
            idx = int(math.floor(threshold / bin_width + 1e-9))
            idx = max(0, min(idx, len(counts) - 1))
            proportion = counts[idx:].sum() / total_valid
        results.append((threshold, proportion))

    return results


def write_histogram_csv(
    output_csv: str,
    bins_left: np.ndarray,
    bins_right: np.ndarray,
    counts: np.ndarray,
    proportions: np.ndarray,
    thr_results: list[tuple[float, float]],
) -> None:
    """Write histogram counts and threshold summaries to CSV.

    Parameters
    ----------
    output_csv : str
        Output CSV path.
    bins_left, bins_right : numpy.ndarray
        Left and right bin edges.
    counts : numpy.ndarray
        Histogram counts.
    proportions : numpy.ndarray
        Histogram proportions.
    thr_results : list of tuple of float
        Threshold summaries.
    """
    parent = os.path.dirname(output_csv)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["bin_label", "bin_left", "bin_right", "count", "proportion"])
        for left, right, count, proportion in zip(bins_left, bins_right, counts, proportions):
            label = f"[{left:.2f},{right:.2f})"
            writer.writerow([label, f"{left:.2f}", f"{right:.2f}", int(count), f"{proportion:.8f}"])

        writer.writerow([])
        writer.writerow(["threshold", "proportion_ge_threshold"])
        for threshold, proportion in thr_results:
            writer.writerow([f">= {threshold:.2f}", f"{proportion:.8f}"])


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def functional_integrity_with_histogram(
    raster_path: str,
    vector_path: str,
    conv_size_m: float,
    classes_1: str,
    classes_0: str,
    classes_null: str,
    out_tif: str,
    sn_tif: str,
    histo_csv: str,
    histo_png: str,
    pixel_max_m: float | None = None,
    n_jobs: int = 7,
    tile_px: int = 2048,
) -> None:
    """Run the complete three-block functional integrity workflow.

    Parameters
    ----------
    raster_path : str
        Input classified raster path.
    vector_path : str
        Input territory vector path.
    conv_size_m : float
        Convolution diameter in meters.
    classes_1 : str
        Class definition for semi-natural pixels.
    classes_0 : str
        Class definition for non-semi-natural pixels.
    classes_null : str
        Class definition for ignored pixels.
    out_tif : str
        Output path for the continuous integrity raster.
    sn_tif : str
        Output path for the binary semi-natural raster.
    histo_csv : str
        Output path for the histogram CSV.
    histo_png : str
        Output path for the histogram PNG.
    pixel_max_m : float or None, default=None
        Pixel size used to derive the kernel size. If ``None``, the native
        raster resolution is used.
    n_jobs : int, default=7
        Number of worker processes.
    tile_px : int, default=2048
        Tile size in pixels for chunked processing.

    Raises
    ------
    ValueError
        If the raster or vector inputs are invalid, or if class validation
        fails under strict validation.
    RuntimeError
        If a worker tile fails during Block 2.
    """
    ensure_parent_dirs([out_tif, sn_tif, histo_csv, histo_png])

    with rasterio.open(raster_path) as src:
        if src.crs is None or not src.crs.is_projected:
            raise ValueError("The input raster must use a projected CRS in meter units.")

        transform = src.transform
        res_x = abs(transform.a)
        res_y = abs(transform.e)
        raster_pixel_size = max(res_x, res_y)
        pixel_max_m = raster_pixel_size if pixel_max_m is None else pixel_max_m

        kernel_size = int(2 * int((conv_size_m / (pixel_max_m * 2))) + 1)
        kernel_size = max(3, kernel_size | 1)
        radius_px = kernel_size // 2
        requested_radius_m = conv_size_m / 2.0
        effective_radius_m = radius_px * pixel_max_m
        buffer_m = effective_radius_m

        LOGGER.info("READ | Opening vector layer: %s", vector_path)
        gdf = gpd.read_file(vector_path)
        if gdf.empty:
            raise ValueError("The input vector layer is empty.")
        if gdf.crs is None:
            raise ValueError("The input vector layer must have a defined CRS.")
        if gdf.crs.to_string() != src.crs.to_string():
            LOGGER.info("TRANSFORM | Reprojecting vector layer to the raster CRS.")
            gdf = gdf.to_crs(src.crs)

        geom_territory = unary_union(gdf.geometry)
        if geom_territory.is_empty:
            raise ValueError("The dissolved territory geometry is empty.")

        geom_buffer = geom_territory.buffer(buffer_m)
        if geom_buffer.is_empty:
            raise ValueError("The buffered territory geometry is empty.")

        mask_territory = features.rasterize(
            [(geom_territory, 1)],
            out_shape=(src.height, src.width),
            transform=src.transform,
            all_touched=False,
            fill=0,
            dtype="uint8",
        )
        mask_processing = features.rasterize(
            [(geom_buffer, 1)],
            out_shape=(src.height, src.width),
            transform=src.transform,
            all_touched=False,
            fill=0,
            dtype="uint8",
        )

        territory_window = rasterio.windows.from_bounds(
            *geom_territory.bounds,
            transform=src.transform,
        ).round_offsets().round_lengths()
        territory_window = territory_window.intersection(Window(0, 0, src.width, src.height))

        processing_window = rasterio.windows.from_bounds(
            *geom_buffer.bounds,
            transform=src.transform,
        ).round_offsets().round_lengths()
        processing_window = processing_window.intersection(Window(0, 0, src.width, src.height))

        integrity_transform = src.window_transform(territory_window)
        binary_transform = src.window_transform(processing_window)

        if VERBOSE:
            LOGGER.info("CONFIG | Raster input                    : %s", raster_path)
            LOGGER.info("CONFIG | Vector input                    : %s", vector_path)
            LOGGER.info("CONFIG | Effective resolution            : %.3f m/px", pixel_max_m)
            LOGGER.info("CONFIG | Requested convolution diameter  : %.1f m", conv_size_m)
            LOGGER.info("CONFIG | Requested convolution radius    : %.1f m", requested_radius_m)
            LOGGER.info("CONFIG | Effective convolution radius    : %.3f m (%s px)", effective_radius_m, radius_px)
            LOGGER.info("CONFIG | Automatic buffer distance       : %.3f m", buffer_m)
            LOGGER.info("CONFIG | Territory window                : %s x %s px", int(territory_window.width), int(territory_window.height))
            LOGGER.info("CONFIG | Buffered calculation window     : %s x %s px", int(processing_window.width), int(processing_window.height))
            LOGGER.info("CONFIG | Parallel jobs                   : %s", n_jobs)
            LOGGER.info("CONFIG | Tile size                       : %s px", tile_px)
            LOGGER.info("CONFIG | Kernel size                     : %s px", kernel_size)
            LOGGER.info("CONFIG | Kernel mode                     : %s", KERNEL_SHAPE)
            LOGGER.info("CONFIG | Integrity output                : %s", out_tif)
            LOGGER.info("CONFIG | Binary output                   : %s", sn_tif)
            LOGGER.info("CONFIG | Histogram CSV                   : %s", histo_csv)
            LOGGER.info("CONFIG | Histogram PNG                   : %s", histo_png)

        ranges_1 = parse_class_spec(classes_1)
        ranges_0 = parse_class_spec(classes_0)
        ranges_null = parse_class_spec(classes_null)

        # ------------------------------------------------------------------
        # BLOCK 1 - Raster analysis and class validation
        # ------------------------------------------------------------------
        log_block_start(1, "Raster analysis and class validation")

        unique_vals = collect_unique_values(
            src,
            mask_processing,
            processing_window,
            step_px=min(tile_px, 2048),
        )
        if unique_vals.size == 0:
            raise ValueError("No raster values were found inside the buffered calculation area.")

        overlaps, unclassified, _ = analyze_raster_values(unique_vals, ranges_1, ranges_0, ranges_null)
        print_raster_analysis(unique_vals, overlaps, unclassified)

        if overlaps and STRICT_VALIDATION:
            lines = [f"{value} -> {', '.join(memberships)}" for value, memberships in overlaps]
            raise ValueError(
                "Some raster values are assigned to multiple classes. "
                "Each value must belong to exactly one class:\n - " + "\n - ".join(lines)
            )

        if unclassified and STRICT_VALIDATION:
            raise ValueError(
                "Some raster values are present in the input raster but missing from "
                "CLASSES_1, CLASSES_0, and CLASSES_NULL:\n - " + "\n - ".join(map(str, unclassified))
            )

        LOGGER.info("VALIDATE | All raster values are classified correctly.")
        log_block_end(1, "Raster analysis and class validation")

        # ------------------------------------------------------------------
        # BLOCK 2A - Binary raster over the buffered calculation area
        # ------------------------------------------------------------------
        log_block_start(2, "Functional integrity computation and raster export")
        LOGGER.info("PREPARE | Writing the binary raster over the buffered calculation area.")

        binary_profile = make_output_profile(src.profile, processing_window, binary_transform)
        nodata_in = src.nodata

        with rasterio.open(sn_tif, "w", **binary_profile) as dst_binary:
            tile_w = tile_h = tile_px
            cols = math.ceil(int(processing_window.width) / tile_w)
            rows = math.ceil(int(processing_window.height) / tile_h)
            total_tiles = rows * cols
            pbar_binary = tqdm(total=total_tiles, desc="Block 2A - binary raster", unit="tile")

            for row_idx in range(rows):
                for col_idx in range(cols):
                    src_win = Window(
                        int(processing_window.col_off) + col_idx * tile_w,
                        int(processing_window.row_off) + row_idx * tile_h,
                        min(tile_w, int(processing_window.width) - col_idx * tile_w),
                        min(tile_h, int(processing_window.height) - row_idx * tile_h),
                    )

                    arr = src.read(1, window=src_win, resampling=Resampling.nearest)
                    sn_tile = classify_binary_tile(arr, nodata_in, ranges_1, ranges_0, ranges_null)

                    row_slice, col_slice = window_to_slices(src_win)
                    tile_mask_processing = mask_processing[row_slice, col_slice]
                    sn_tile[tile_mask_processing == 0] = np.nan

                    out_win = Window(
                        int(src_win.col_off - processing_window.col_off),
                        int(src_win.row_off - processing_window.row_off),
                        int(src_win.width),
                        int(src_win.height),
                    )
                    dst_binary.write(
                        np.nan_to_num(sn_tile, nan=-9999.0).astype(np.float32),
                        1,
                        window=out_win,
                    )

                    if VERBOSE:
                        LOGGER.info(
                            "WRITE | Binary tile r%s c%s written (%sx%s px).",
                            row_idx,
                            col_idx,
                            int(src_win.width),
                            int(src_win.height),
                        )
                    pbar_binary.update(1)

            pbar_binary.close()

        LOGGER.info("WRITE | Output 1/2 written: %s", sn_tif)

        # ------------------------------------------------------------------
        # BLOCK 2B - Integrity raster over the exact territory extent
        # ------------------------------------------------------------------
        LOGGER.info("PREPARE | Writing the integrity raster over the exact territory extent.")

        integrity_profile = make_output_profile(src.profile, territory_window, integrity_transform)

        with rasterio.open(out_tif, "w", **integrity_profile) as dst_integrity:
            tile_w = tile_h = tile_px
            cols = math.ceil(int(territory_window.width) / tile_w)
            rows = math.ceil(int(territory_window.height) / tile_h)
            total_tiles = rows * cols
            max_in_flight = max(1, n_jobs)

            def build_integrity_task(row_idx: int, col_idx: int) -> tuple[tuple, dict[str, Window | int]]:
                """Build one integrity-processing task and its output metadata."""
                win = Window(
                    int(territory_window.col_off) + col_idx * tile_w,
                    int(territory_window.row_off) + row_idx * tile_h,
                    min(tile_w, int(territory_window.width) - col_idx * tile_w),
                    min(tile_h, int(territory_window.height) - row_idx * tile_h),
                )

                halo = radius_px
                win_halo = Window(
                    max(0, int(win.col_off) - halo),
                    max(0, int(win.row_off) - halo),
                    min(src.width - max(0, int(win.col_off) - halo), int(win.width) + 2 * halo),
                    min(src.height - max(0, int(win.row_off) - halo), int(win.height) + 2 * halo),
                )

                halo_top = int(win.row_off - win_halo.row_off)
                halo_left = int(win.col_off - win_halo.col_off)
                halo_bottom = int((win_halo.row_off + win_halo.height) - (win.row_off + win.height))
                halo_right = int((win_halo.col_off + win_halo.width) - (win.col_off + win.width))

                arr = src.read(1, window=win_halo, resampling=Resampling.nearest)

                row_slice_halo, col_slice_halo = window_to_slices(win_halo)
                halo_processing_mask = mask_processing[row_slice_halo, col_slice_halo]

                arr = arr.copy()
                arr[halo_processing_mask == 0] = np.int64(2_147_483_647)
                ranges_null_tile = ranges_null + [(2_147_483_647, 2_147_483_647)]

                if VERBOSE:
                    LOGGER.info(
                        "PREPARE | Integrity tile r%s c%s | write=%sx%s | halo=%sx%s",
                        row_idx,
                        col_idx,
                        int(win.width),
                        int(win.height),
                        int(win_halo.width),
                        int(win_halo.height),
                    )

                task_args = (
                    arr,
                    nodata_in,
                    ranges_1,
                    ranges_0,
                    ranges_null_tile,
                    kernel_size,
                    halo_top,
                    halo_bottom,
                    halo_left,
                    halo_right,
                    KERNEL_SHAPE,
                    (row_idx, col_idx),
                )
                meta = {
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "src_window": win,
                    "out_window": Window(
                        int(win.col_off - territory_window.col_off),
                        int(win.row_off - territory_window.row_off),
                        int(win.width),
                        int(win.height),
                    ),
                }
                return task_args, meta

            def handle_completed_future(future, future_meta: dict[str, Window | int], progress_bar: tqdm) -> None:
                """Write one completed integrity tile to disk."""
                ok, payload, dt, row_idx, col_idx = future.result()

                if not ok:
                    raise RuntimeError(
                        f"Tile r{row_idx} c{col_idx} failed after {dt:.2f}s: {payload}"
                    )

                fi_tile = payload
                src_window = future_meta["src_window"]
                row_slice_core, col_slice_core = window_to_slices(src_window)
                tile_mask_territory = mask_territory[row_slice_core, col_slice_core]

                src_core = src.read(1, window=src_window, resampling=Resampling.nearest)
                src_core_int = src_core.astype(np.int64, copy=False)
                tile_mask_null = build_mask_from_ranges(src_core_int, ranges_null)
                if nodata_in is not None:
                    tile_mask_null |= (src_core == nodata_in)

                fi_tile = fi_tile.astype(np.float32, copy=False)
                fi_tile[tile_mask_territory == 0] = np.nan
                fi_tile[tile_mask_null] = np.nan

                dst_integrity.write(
                    np.nan_to_num(fi_tile, nan=-9999.0).astype(np.float32),
                    1,
                    window=future_meta["out_window"],
                )

                progress_bar.set_postfix_str(f"r{row_idx}c{col_idx} | compute {dt:.2f}s")
                progress_bar.update(1)

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {}
                pbar_integrity = tqdm(total=total_tiles, desc="Block 2B - integrity raster", unit="tile")

                try:
                    for row_idx in range(rows):
                        for col_idx in range(cols):
                            task_args, meta = build_integrity_task(row_idx, col_idx)
                            future = executor.submit(process_integrity_tile_timed, task_args)
                            futures[future] = meta

                            if len(futures) >= max_in_flight:
                                done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                                for completed_future in done:
                                    completed_meta = futures.pop(completed_future)
                                    handle_completed_future(completed_future, completed_meta, pbar_integrity)

                    while futures:
                        done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                        for completed_future in done:
                            completed_meta = futures.pop(completed_future)
                            handle_completed_future(completed_future, completed_meta, pbar_integrity)
                finally:
                    pbar_integrity.close()

        LOGGER.info("WRITE | Output 2/2 written: %s", out_tif)
        log_block_end(2, "Functional integrity computation and raster export")

    # ----------------------------------------------------------------------
    # BLOCK 3 - Histogram export
    # ----------------------------------------------------------------------
    log_block_start(3, "Integrity histogram export")
    LOGGER.info("READ | Reading the integrity raster to build the histogram.")

    bins, counts, total_valid = compute_histogram(
        out_tif,
        bin_width=BIN_WIDTH,
        exclude_zero=EXCLUDE_ZERO_IN_HISTOGRAM,
    )
    bins_left = bins[:-1]
    bins_right = bins[1:]
    proportions = (counts / total_valid) if total_valid > 0 else np.zeros_like(counts, dtype=float)
    thr_results = summarize_thresholds(counts, total_valid, THRESHOLDS, BIN_WIDTH)

    LOGGER.info("WRITE | Writing histogram CSV.")
    write_histogram_csv(histo_csv, bins_left, bins_right, counts, proportions, thr_results)
    LOGGER.info("WRITE | Output 1/2 written: %s", histo_csv)

    LOGGER.info("WRITE | Rendering histogram PNG.")
    png_dir = os.path.dirname(histo_png)
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)
    save_histogram_png(bins_left, bins_right, counts, histo_png, thr_results=thr_results)
    LOGGER.info("WRITE | Output 2/2 written: %s", histo_png)

    LOGGER.info("REPORT | Valid pixels included in the histogram: %s", total_valid)
    for threshold, proportion in thr_results:
        LOGGER.info("REPORT | proportion >= %.2f: %.3f", threshold, proportion)

    log_block_end(3, "Integrity histogram export")
    LOGGER.info("SUCCESS | Workflow completed successfully.")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    configure_logging(LOG_LEVEL)
    functional_integrity_with_histogram(
        raster_path=RASTER_PATH,
        vector_path=VECTOR_PATH,
        conv_size_m=CONV_SIZE_M,
        classes_1=CLASSES_1,
        classes_0=CLASSES_0,
        classes_null=CLASSES_NULL,
        out_tif=OUT_TIF,
        sn_tif=SN_TIF,
        histo_csv=HISTO_CSV,
        histo_png=HISTO_PNG,
        pixel_max_m=PIXEL_MAX_M,
        n_jobs=N_JOBS,
        tile_px=TILE_PX,
    )
