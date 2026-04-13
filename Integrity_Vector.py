#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute a biosphere functional integrity indicator that consists in a convolution of 1000m radius on a binary classified land cover database.  
The program is designed to use a land cover database in VECTOR format.

The program takes as input:
- a territory vector layer where to compute biosphere functional integrity (BFI),
- a land-cover vector layer and associated field to be classified,
- class assignments of the land-cover field values for semi-natural (1), non-semi-natural (0), and null values (NaN).
- settings for the rasterization of the land-cover vector layer (including in particular the resolution)
- settings for the convolution to be made to calculate the functional integrity, and histogram outputs (with default settings provided aligned with the BFI definition in Mohamed et al., 2024).
- settings for multithreading (to improve program's performance) and tile size for processing.

It is organized into four processing blocks.

Block 1
    - Analyze values in the vector layer inside the calculation area, defined as the input
    territory buffered by the effective convolution radius.
    - Display the vector values found in the classification field and validate that every value is assigned to exactly one of the three class (0/1/NaN).
    

Block 2
    Rasterize the input classification field on the bounding box of the
    buffered integrity territory. 

Block 3
    Compute functional integrity and export two raster outputs:
    - a raster providing the values used to calculate biosphere functional integrity, consisting in a raster with values 0 / 1 / NaN at the extent of the input vector layer buffered by the convolution radius.
    - a raster providing the value of biosphere functional integrity, in the [0, 1] range, at the extent of the input territory.
    
Block 4
    Export a histogram of functional integrity values over the territory, with a
    user-defined bin width and threshold. The histogram is exported as:
    - a CSV table,
    - a PNG figure.

Notes
-----
- The vector used for rasterization is independent from the vector used as the
  territory footprint for the integrity computation.
- Rasterization and convolution blocks are designed for multi-core execution.
- Exact circular convolution uses SciPy when available.
- If SciPy is unavailable, the program falls back to a box-based local mean.
- Console output uses Python's standard ``logging`` module.
- Docstrings follow a NumPy-style layout compatible with general PEP 257
  docstring conventions.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import re
import sqlite3
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from multiprocessing import Pool, cpu_count
from numbers import Integral
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import fiona
from fiona.errors import FionaValueError
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import MergeAlg, Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box, mapping, shape
from tqdm import tqdm


# =============================================================================
# USER PARAMETERS
# =============================================================================

# -------------------------
# Inputs
# -------------------------
INPUT_VECTOR: str = r"path/to/file"
INPUT_LAYER: Optional[str] = None         # None = auto-detect the first layer containing FIELD_NAME
FIELD_NAME: str = "layer_name"    # integer field used for classification and rasterization

TERRITORY_VECTOR: str = r"path/to/file"
TERRITORY_LAYER: Optional[str] = None     # optional; use only if the file contains several layers

# -------------------------
# Classes used for integrity
# Same syntax as in the previous integrity script:
# - individual values separated by spaces, commas, or semicolons
# - ranges allowed with: "10 thru 20"
# -------------------------
CLASSES_1: str = ""
CLASSES_0: str = ""
CLASSES_NULL: str = ""

# -------------------------
# Rasterization outputs
# -------------------------
RASTERIZED_TIF: str = r"path/to/file.tif"
RESOLUTION: float = 5.0
ALL_TOUCHED: bool = False
RASTER_NODATA: int = -2147483648          # int32 nodata sentinel for the classified raster

# -------------------------
# Integrity and histogram outputs
# -------------------------
OUTPUT_DIR: str = r"path/to/directory"  # creates or reuses an "Output" subfolder inside
CONV_SIZE_M: float = 1000.0               # kernel diameter in meters
PIXEL_MAX_M: Optional[float] = None       # if None, use the true raster resolution
BUFFER_M: Optional[float] = None          # None = auto-compute from the convolution kernel
OUTPUT_NODATA: float = -9999.0
KERNEL_SHAPE: str = "circular_fft"       # "circular_fft" or "box"

# -------------------------
# Histogram settings
# Same logic and style as the original histogram script.
# -------------------------
HIST_BIN_WIDTH: float = 0.01
HIST_THRESHOLDS: List[float] = [0.25]
HIST_EXCLUDE_ZERO: bool = True

# -------------------------
# Parallelization / tiling
# -------------------------
N_JOBS: int = 7
TILE_PX: int = 2048

# -------------------------
# General settings
# -------------------------
STRICT_VALIDATION: bool = True
VERBOSE: bool = True
LOG_LEVEL: str = "INFO"

# GeoTIFF creation options
RIO_PROFILE_KW: Dict = dict(
    driver="GTiff",
    compress="lzw",
    tiled=True,
    blockxsize=256,
    blockysize=256,
    bigtiff="IF_SAFER",
)


# =============================================================================
# LOGGING HELPERS
# =============================================================================

LOGGER = logging.getLogger("functional_integrity_vector")


def configure_logging(level: str = "INFO", log_file_path: str | None = None) -> None:
    """Configure console and optional file logging for the workflow.

    Parameters
    ----------
    level : str, default="INFO"
        Logging threshold passed to the handlers.
    log_file_path : str or None, default=None
        Optional path for the execution log file.
    """
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, level.upper(), logging.INFO))
    LOGGER.propagate = False

    handler_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(handler_level)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)

    if log_file_path:
        ensure_parent_dirs([log_file_path])
        file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
        file_handler.setLevel(handler_level)
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)


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


def log_block_start(block_id: int, total_blocks: int, title: str) -> None:
    """Log the start of a processing block.

    Parameters
    ----------
    block_id : int
        Block index in the workflow.
    total_blocks : int
        Total number of blocks in the workflow.
    title : str
        Human-readable block title.
    """
    log_section(f"START | BLOCK {block_id}/{total_blocks} | {title}")


def log_block_end(block_id: int, total_blocks: int, title: str) -> None:
    """Log the end of a processing block.

    Parameters
    ----------
    block_id : int
        Block index in the workflow.
    total_blocks : int
        Total number of blocks in the workflow.
    title : str
        Human-readable block title.
    """
    LOGGER.info("COMPLETE | BLOCK %s/%s | %s", block_id, total_blocks, title)


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


def resolve_output_dir(base_dir: str) -> str:
    """Resolve the final output directory used by the workflow.

    Parameters
    ----------
    base_dir : str
        Parent directory configured by the user.

    Returns
    -------
    str
        Absolute path to the output directory.
    """
    if not base_dir or not base_dir.strip():
        raise ValueError("OUTPUT_DIR must point to a parent directory.")

    normalized = os.path.normpath(os.path.abspath(base_dir))
    if os.path.basename(normalized).lower() == "output":
        output_dir = normalized
    else:
        output_dir = os.path.join(normalized, "Output")

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def build_output_paths(base_dir: str) -> dict[str, str]:
    """Build the standardized output paths for the workflow.

    Parameters
    ----------
    base_dir : str
        Parent directory configured by the user.

    Returns
    -------
    dict of str
        Mapping containing the output directory and all generated file paths.
    """
    output_dir = resolve_output_dir(base_dir)
    return {
        "output_dir": output_dir,
        "out_tif": os.path.join(output_dir, "integrity_output.tif"),
        "sn_tif": os.path.join(output_dir, "binary_output.tif"),
        "hist_csv": os.path.join(output_dir, "histogram.csv"),
        "hist_png": os.path.join(output_dir, "histogram.png"),
        "log_file": os.path.join(output_dir, "execution.log"),
    }


def format_values(values: Sequence[int], values_per_line: int = 20) -> str:
    """Format integer values into wrapped console lines.

    Parameters
    ----------
    values : sequence of int
        Values to format.
    values_per_line : int, default=20
        Maximum number of values displayed per output line.

    Returns
    -------
    str
        Wrapped string representation of the values.
    """
    values = list(values)
    if not values:
        return "<empty>"
    lines = []
    for i in range(0, len(values), values_per_line):
        chunk = values[i:i + values_per_line]
        lines.append(" ".join(str(v) for v in chunk))
    return "\n".join(lines)


# =============================================================================
# HELPERS - VECTOR / LAYER HANDLING
# =============================================================================

def pick_layer_with_field(path: str, field: str, layer_hint: Optional[str] = None) -> str:
    """Return the layer name from a vector file.

    - If `layer_hint` is given, only validate that it exists.
    - Otherwise, return the first layer containing `field`.
    """
    layers = fiona.listlayers(path)

    if layer_hint:
        if layer_hint in layers:
            return layer_hint
        raise ValueError(
            f"Layer '{layer_hint}' was not found in {path}. Available layers: {layers}"
        )

    for lyr in layers:
        with fiona.open(path, layer=lyr) as src:
            if src.schema and "properties" in src.schema and field in src.schema["properties"]:
                return lyr

    raise ValueError(
        f"No layer contains field '{field}'. Available layers: {layers}"
    )


def get_bounds_crs(path: str, layer: str) -> Tuple[Tuple[float, float, float, float], CRS]:
    """Return layer bounds and CRS."""
    with fiona.open(path, layer=layer) as src:
        crs = CRS.from_user_input(src.crs) if src.crs else None
        if crs is None:
            raise ValueError("CRS not found in the vector layer.")
        bounds = src.bounds
    return bounds, crs


def pick_first_layer(path: str, layer_hint: Optional[str] = None) -> str:
    """Return the requested layer or the first available layer."""
    layers = fiona.listlayers(path)
    if not layers:
        raise ValueError(f"No layer was found in vector file: {path}")
    if layer_hint:
        if layer_hint in layers:
            return layer_hint
        raise ValueError(
            f"Layer '{layer_hint}' was not found in {path}. Available layers: {layers}"
        )
    return layers[0]


def get_layer_feature_count(path: str, layer: str) -> int:
    """Return the number of features in a vector layer.

    Parameters
    ----------
    path : str
        Input vector path.
    layer : str
        Layer name.

    Returns
    -------
    int
        Number of features in the layer.
    """
    with fiona.open(path, layer=layer) as src:
        return len(src)


def ensure_projected_meters(crs: CRS) -> None:
    """Ensure the CRS is projected in meters."""
    linear_units = (crs.linear_units or "").lower() if hasattr(crs, "linear_units") else ""
    if not crs.is_projected or linear_units not in ("metre", "meter", "metres", "meters"):
        raise ValueError(
            f"CRS {crs} is not a projected CRS in meters. "
            "Please reproject the vector/raster before running this script."
        )


def expand_bounds(bounds: Tuple[float, float, float, float], distance: float) -> Tuple[float, float, float, float]:
    """Expand a bounding box by a constant distance in all directions.

    Parameters
    ----------
    bounds : tuple of float
        Bounding box as ``(minx, miny, maxx, maxy)``.
    distance : float
        Expansion distance in CRS units.

    Returns
    -------
    tuple of float
        Expanded bounding box.
    """
    minx, miny, maxx, maxy = bounds
    return (minx - distance, miny - distance, maxx + distance, maxy + distance)


# =============================================================================
# STEP 1 - VECTOR FIELD ANALYSIS
# =============================================================================

def is_gpkg_path(path: str) -> bool:
    """Return ``True`` when the input path targets a GeoPackage file.

    Parameters
    ----------
    path : str
        Input vector path.

    Returns
    -------
    bool
        ``True`` if the file extension is ``.gpkg``.
    """
    return os.path.splitext(path)[1].lower() == ".gpkg"


def is_integer_schema_type(schema_type: object) -> bool:
    """Return ``True`` if a Fiona schema type describes integer values.

    Parameters
    ----------
    schema_type : object
        Raw schema type read from Fiona.

    Returns
    -------
    bool
        ``True`` for integer-like Fiona schema definitions such as ``int32``.
    """
    if schema_type is None:
        return False
    normalized = str(schema_type).strip().lower()
    return normalized.startswith("int")


def sqlite_quote_identifier(identifier: str) -> str:
    """Quote an SQLite identifier safely.

    Parameters
    ----------
    identifier : str
        SQLite table or field name.

    Returns
    -------
    str
        Safely quoted SQLite identifier.
    """
    return '"' + identifier.replace('"', '""') + '"'


def coerce_class_value_to_int(value: object) -> int:
    """Convert a class value to an integer when possible.

    Parameters
    ----------
    value : object
        Raw field value read from the vector source.

    Returns
    -------
    int
        Integer class value.

    Raises
    ------
    ValueError
        If the value cannot be converted safely to an integer class code.
    """
    if value is None:
        raise ValueError("NULL value")

    if isinstance(value, bool):
        raise ValueError(repr(value))

    if isinstance(value, Integral):
        return int(value)

    if isinstance(value, float):
        if math.isfinite(value) and float(value).is_integer():
            return int(value)
        raise ValueError(repr(value))

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            raise ValueError(repr(value))

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            raise ValueError("EMPTY_STRING")
        if re.fullmatch(r"[+-]?\d+", stripped):
            return int(stripped)
        if re.fullmatch(r"[+-]?\d+\.0+", stripped):
            return int(float(stripped))
        raise ValueError(repr(value))

    raise ValueError(repr(value))


def analyze_vector_field_gpkg_fast(
    path: str,
    layer: str,
    field_name: str,
) -> Tuple[int, str, List[int]]:
    """Analyze a GeoPackage classification field using SQLite queries.

    Parameters
    ----------
    path : str
        Input GeoPackage path.
    layer : str
        GeoPackage layer name.
    field_name : str
        Classification field name.

    Returns
    -------
    tuple
        Feature count, validation mode label, and sorted unique integer values.

    Raises
    ------
    ValueError
        If the field contains NULL values, empty strings, or values that cannot
        be converted safely to integers.
    """
    layer_sql = sqlite_quote_identifier(layer)
    field_sql = sqlite_quote_identifier(field_name)

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()

        cursor.execute(f"SELECT COUNT(*) FROM {layer_sql}")
        total = int(cursor.fetchone()[0])

        cursor.execute(f"SELECT COUNT(*) FROM {layer_sql} WHERE {field_sql} IS NULL")
        null_count = int(cursor.fetchone()[0])

        cursor.execute(
            f"SELECT DISTINCT {field_sql} FROM {layer_sql} "
            f"WHERE {field_sql} IS NOT NULL ORDER BY {field_sql}"
        )
        rows = cursor.fetchall()

    if null_count > 0:
        raise ValueError(
            f"The field '{field_name}' contains {null_count} NULL values. "
            "All features must have a class value before rasterization."
        )

    unique_values: set[int] = set()
    bad_examples: List[str] = []
    empty_count = 0

    for (raw_value,) in rows:
        try:
            coerced = coerce_class_value_to_int(raw_value)
            unique_values.add(coerced)
        except ValueError as exc:
            if str(exc) == "EMPTY_STRING":
                empty_count += 1
            elif len(bad_examples) < 10:
                bad_examples.append(str(exc))

    if empty_count > 0:
        raise ValueError(
            f"The field '{field_name}' contains {empty_count} empty string value(s). "
            "All features must have a class value before rasterization."
        )

    if bad_examples:
        raise ValueError(
            "The classification field contains non-integer values. "
            f"Examples of invalid values: {', '.join(bad_examples)}"
        )

    if not unique_values:
        raise ValueError(f"No valid values were found in field '{field_name}'.")

    return total, "adaptive fast path (GeoPackage SQL)", sorted(unique_values)


def analyze_vector_field_generic(
    path: str,
    layer: str,
    field_name: str,
) -> Tuple[int, str, List[int]]:
    """Analyze a classification field with generic feature-by-feature reading.

    Parameters
    ----------
    path : str
        Input vector path.
    layer : str
        Vector layer name.
    field_name : str
        Classification field name.

    Returns
    -------
    tuple
        Feature count, validation mode label, and sorted unique integer values.

    Raises
    ------
    ValueError
        If the field contains NULL values, empty strings, or values that are
        not integer-compatible.
    """
    unique_values: set[int] = set()
    bad_examples: List[str] = []
    null_count = 0
    empty_count = 0

    with fiona.open(path, layer=layer) as src:
        total = len(src)
        for feat in tqdm(src, total=total, desc="Reading vector features", unit="feature"):
            value = feat["properties"].get(field_name)

            if value is None:
                null_count += 1
                continue

            try:
                coerced = coerce_class_value_to_int(value)
                unique_values.add(coerced)
            except ValueError as exc:
                if str(exc) == "EMPTY_STRING":
                    empty_count += 1
                elif len(bad_examples) < 10:
                    bad_examples.append(str(exc))

    if null_count > 0:
        raise ValueError(
            f"The field '{field_name}' contains {null_count} NULL values. "
            "All features must have a class value before rasterization."
        )

    if empty_count > 0:
        raise ValueError(
            f"The field '{field_name}' contains {empty_count} empty string value(s). "
            "All features must have a class value before rasterization."
        )

    if bad_examples:
        raise ValueError(
            "The classification field is not integer-compatible. "
            f"Examples of invalid values: {', '.join(bad_examples)}"
        )

    if not unique_values:
        raise ValueError(f"No valid values were found in field '{field_name}'.")

    return total, "adaptive fallback (generic feature scan)", sorted(unique_values)


def analyze_vector_field(
    path: str,
    field_name: str,
    layer_name: Optional[str] = None,
) -> Tuple[str, Tuple[float, float, float, float], CRS, int, Optional[str], str, List[int]]:
    """Analyze the classification field in the input vector.

    Parameters
    ----------
    path : str
        Input vector path.
    field_name : str
        Name of the classification field.
    layer_name : str or None, default=None
        Optional layer name. If ``None``, the first layer containing
        ``field_name`` is selected.

    Returns
    -------
    tuple
        Selected layer name, vector bounds, vector CRS, number of features,
        field schema type, validation mode, and sorted unique integer values
        found in the classification field.

    Raises
    ------
    ValueError
        If the field is missing, contains NULL values, or contains values that
        are not integer-compatible.
    """
    layer = pick_layer_with_field(path, field_name, layer_hint=layer_name)
    bounds, crs = get_bounds_crs(path, layer)
    ensure_projected_meters(crs)

    with fiona.open(path, layer=layer) as src:
        if field_name not in src.schema.get("properties", {}):
            available = list(src.schema.get("properties", {}).keys())
            raise ValueError(
                f"Field '{field_name}' was not found in layer '{layer}'. Available fields: {available}"
            )
        schema_type = src.schema["properties"].get(field_name)

    if is_gpkg_path(path):
        try:
            total, validation_mode, sorted_values = analyze_vector_field_gpkg_fast(path, layer, field_name)
        except Exception as exc:
            LOGGER.warning(
                "FALLBACK | Adaptive fast path failed (%s). Switching to generic feature scan.",
                exc,
            )
            total, validation_mode, sorted_values = analyze_vector_field_generic(path, layer, field_name)
    else:
        total, validation_mode, sorted_values = analyze_vector_field_generic(path, layer, field_name)

    return layer, bounds, crs, total, schema_type, validation_mode, sorted_values


def analyze_territory_vector(
    path: str,
    layer_name: Optional[str] = None,
) -> Tuple[str, Tuple[float, float, float, float], CRS, int]:
    """Analyze the territory vector used for the integrity footprint.

    Parameters
    ----------
    path : str
        Territory vector path.
    layer_name : str or None, default=None
        Optional layer name. If ``None``, the first layer is selected.

    Returns
    -------
    tuple
        Selected layer name, vector bounds, vector CRS, and number of features.
    """
    layer = pick_first_layer(path, layer_hint=layer_name)
    bounds, crs = get_bounds_crs(path, layer)
    ensure_projected_meters(crs)
    feature_count = get_layer_feature_count(path, layer)

    return layer, bounds, crs, feature_count


# =============================================================================
# STEP 1 - CLASS PARSING / VALIDATION
# =============================================================================

_RANGE_RE = re.compile(r"^\s*(-?\d+)\s+thru\s+(-?\d+)\s*$", re.IGNORECASE)


def parse_class_spec(spec: str) -> List[Tuple[int, int]]:
    """Parse a class specification string into inclusive integer ranges.

    Parameters
    ----------
    spec : str
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
    ranges: List[Tuple[int, int]] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = _RANGE_RE.match(part)
        if match:
            a, b = int(match.group(1)), int(match.group(2))
            ranges.append((min(a, b), max(a, b)))
        else:
            for token in part.split():
                token = token.strip()
                if token:
                    value = int(token)
                    ranges.append((value, value))

    return ranges


def value_in_ranges(value: int, ranges: Sequence[Tuple[int, int]]) -> bool:
    """Check whether a value belongs to at least one inclusive range.

    Parameters
    ----------
    value : int
        Value to test.
    ranges : sequence of tuple of int
        Inclusive value ranges.

    Returns
    -------
    bool
        ``True`` if the value is covered by at least one range.
    """
    return any(a <= value <= b for a, b in ranges)


def mask_from_ranges(arr_int: np.ndarray, ranges: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Build a boolean mask for values covered by inclusive ranges.

    Parameters
    ----------
    arr_int : numpy.ndarray
        Integer array to classify.
    ranges : sequence of tuple of int
        Inclusive value ranges.

    Returns
    -------
    numpy.ndarray
        Boolean mask with the same shape as ``arr_int``.
    """
    if not ranges:
        return np.zeros(arr_int.shape, dtype=bool)
    mask = np.zeros(arr_int.shape, dtype=bool)
    for a, b in ranges:
        mask |= (arr_int >= a) & (arr_int <= b)
    return mask


def validate_unique_values_against_classes(
    unique_values: Sequence[int],
    classes_1: str,
    classes_0: str,
    classes_null: str,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Validate classification values against user-defined class groups.

    Parameters
    ----------
    unique_values : sequence of int
        Unique values found in the input classification field.
    classes_1, classes_0, classes_null : str
        Class specifications provided by the user.

    Returns
    -------
    tuple of list of tuple of int
        Parsed ranges for ``CLASSES_1``, ``CLASSES_0``, and ``CLASSES_NULL``.

    Raises
    ------
    ValueError
        If a value is missing from all class groups or assigned to more than one
        class group.
    """
    LOGGER.info("VALIDATE | Checking class coverage and class overlaps.")

    ranges_1 = parse_class_spec(classes_1)
    ranges_0 = parse_class_spec(classes_0)
    ranges_null = parse_class_spec(classes_null)

    uncovered: List[int] = []
    duplicated: List[Tuple[int, List[str]]] = []

    for value in unique_values:
        memberships: List[str] = []
        if value_in_ranges(value, ranges_1):
            memberships.append("CLASSES_1")
        if value_in_ranges(value, ranges_0):
            memberships.append("CLASSES_0")
        if value_in_ranges(value, ranges_null):
            memberships.append("CLASSES_NULL")

        if len(memberships) == 0:
            uncovered.append(value)
        elif len(memberships) > 1:
            duplicated.append((value, memberships))

    if uncovered:
        raise ValueError(
            "Some field values are not assigned to any class group.\n"
            f"Unclassified values ({len(uncovered)}): {uncovered}"
        )

    if duplicated:
        details = "\n".join(
            f"  - value {value}: {', '.join(groups)}" for value, groups in duplicated
        )
        raise ValueError(
            "Some field values are assigned to more than one class group.\n"
            f"{details}"
        )

    LOGGER.info("VALIDATE | All classification values are assigned correctly.")
    return ranges_1, ranges_0, ranges_null


# =============================================================================
# STEP 2 - RASTERIZATION
# =============================================================================

def compute_grid(bounds: Tuple[float, float, float, float], res: float) -> Tuple[int, int, rasterio.Affine]:
    """Compute raster dimensions and transform from bounds and resolution.

    Parameters
    ----------
    bounds : tuple of float
        Bounding box as ``(minx, miny, maxx, maxy)``.
    res : float
        Output raster resolution.

    Returns
    -------
    tuple
        Number of rows, number of columns, and affine transform.
    """
    minx, miny, maxx, maxy = bounds
    cols = int(math.ceil((maxx - minx) / res))
    rows = int(math.ceil((maxy - miny) / res))
    transform = from_origin(minx, maxy, res, res)
    return rows, cols, transform


def generate_tile_windows(rows: int, cols: int, tile_px: int) -> List[Window]:
    """Generate raster windows covering a full raster grid.

    Parameters
    ----------
    rows : int
        Raster height in pixels.
    cols : int
        Raster width in pixels.
    tile_px : int
        Tile size in pixels.

    Returns
    -------
    list of rasterio.windows.Window
        Tiled raster windows.
    """
    windows: List[Window] = []
    for row_off in range(0, rows, tile_px):
        height = min(tile_px, rows - row_off)
        for col_off in range(0, cols, tile_px):
            width = min(tile_px, cols - col_off)
            windows.append(Window(col_off=col_off, row_off=row_off, width=width, height=height))
    return windows


def window_bounds(win: Window, transform: rasterio.Affine) -> Tuple[float, float, float, float]:
    """Return the spatial bounds of a raster window.

    Parameters
    ----------
    win : rasterio.windows.Window
        Input raster window.
    transform : affine.Affine
        Raster transform.

    Returns
    -------
    tuple of float
        Window bounds as ``(minx, miny, maxx, maxy)``.
    """
    x0, y0 = transform * (win.col_off, win.row_off)
    x1, y1 = transform * (win.col_off + win.width, win.row_off + win.height)
    minx, maxx = min(x0, x1), max(x0, x1)
    miny, maxy = min(y0, y1), max(y0, y1)
    return (minx, miny, maxx, maxy)


def read_shapes_for_bbox(
    path: str,
    layer: str,
    bbox_xyxy: Tuple[float, float, float, float],
    field: str,
) -> Iterable[Tuple[Dict, int]]:
    """Yield (geometry, value) pairs intersecting the given bbox."""
    bbox_geom = box(*bbox_xyxy)

    try:
        with fiona.open(path, layer=layer) as src:
            for _, feat in src.items(bbox=bbox_xyxy):
                if feat["geometry"] is None:
                    continue
                value = feat["properties"].get(field)
                if value is None:
                    continue
                geom = shape(feat["geometry"])
                if geom.is_empty:
                    continue
                if geom.intersects(bbox_geom):
                    yield mapping(geom), int(value)
    except FionaValueError:
        with fiona.open(path, layer=layer) as src:
            for feat in src:
                if feat["geometry"] is None:
                    continue
                value = feat["properties"].get(field)
                if value is None:
                    continue
                geom = shape(feat["geometry"])
                if geom.is_empty:
                    continue
                if geom.intersects(bbox_geom):
                    yield mapping(geom), int(value)


def rasterize_one_tile(
    tile_win: Window,
    transform: rasterio.Affine,
    path: str,
    layer: str,
    field: str,
    out_dtype: str,
    all_touched: bool,
    nodata_value: int,
) -> Tuple[Window, np.ndarray]:
    """Rasterize one tile of the input vector layer.

    Parameters
    ----------
    tile_win : rasterio.windows.Window
        Tile window in output raster coordinates.
    transform : affine.Affine
        Full output raster transform.
    path : str
        Input vector path.
    layer : str
        Input layer name.
    field : str
        Attribute field used for rasterization.
    out_dtype : str
        Output NumPy dtype.
    all_touched : bool
        Forwarded to :func:`rasterio.features.rasterize`.
    nodata_value : int
        Nodata value used to initialize empty tiles.

    Returns
    -------
    tuple
        Raster window and rasterized NumPy array.
    """
    bbox = window_bounds(tile_win, transform)
    tile_transform = rasterio.windows.transform(tile_win, transform)
    shapes = list(read_shapes_for_bbox(path, layer, bbox, field))

    if not shapes:
        arr = np.full((int(tile_win.height), int(tile_win.width)), nodata_value, dtype=out_dtype)
        return tile_win, arr

    arr = rasterize(
        shapes=shapes,
        out_shape=(int(tile_win.height), int(tile_win.width)),
        transform=tile_transform,
        fill=nodata_value,
        all_touched=all_touched,
        dtype=out_dtype,
        merge_alg=MergeAlg.replace,
    )
    return tile_win, arr


def initialize_raster_with_nodata(dst: rasterio.io.DatasetWriter, nodata_value: float | int) -> None:
    """Safely initialize a new output raster with nodata values."""
    for _, win in tqdm(list(dst.block_windows(1)), desc="Prepare output raster", unit="block"):
        arr = np.full((int(win.height), int(win.width)), nodata_value, dtype=dst.dtypes[0])
        dst.write(arr, 1, window=win)


def rasterize_vector_field(
    input_vector: str,
    input_layer: str,
    field_name: str,
    output_tif: str,
    bounds: Tuple[float, float, float, float],
    crs: CRS,
    resolution: float,
    all_touched: bool,
    nodata_value: int,
    n_jobs: int,
    tile_px: int,
) -> None:
    """Rasterize the classification field on the buffered territory extent.

    Parameters
    ----------
    input_vector : str
        Input vector path.
    input_layer : str
        Input layer name.
    field_name : str
        Attribute field used for rasterization.
    output_tif : str
        Output raster path.
    bounds : tuple of float
        Rasterization bounds.
    crs : rasterio.crs.CRS
        Output raster CRS.
    resolution : float
        Output raster resolution.
    all_touched : bool
        Forwarded to :func:`rasterio.features.rasterize`.
    nodata_value : int
        Output nodata value.
    n_jobs : int
        Number of worker processes.
    tile_px : int
        Tile size in pixels.
    """
    ensure_parent_dirs([output_tif])

    rows, cols, transform = compute_grid(bounds, resolution)
    profile = {
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": "int32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata_value,
        **RIO_PROFILE_KW,
    }

    windows = generate_tile_windows(rows, cols, tile_px)
    worker = partial(
        rasterize_one_tile,
        transform=transform,
        path=input_vector,
        layer=input_layer,
        field=field_name,
        out_dtype="int32",
        all_touched=all_touched,
        nodata_value=nodata_value,
    )

    nprocs = min(max(1, n_jobs), cpu_count())

    LOGGER.info("CONFIG | Rasterization bounds: %s", bounds)
    LOGGER.info("CONFIG | Rasterization resolution: %.3f m", resolution)
    LOGGER.info("CONFIG | Rasterization tile size: %s px", tile_px)
    LOGGER.info("CONFIG | Rasterization workers: %s", nprocs)
    LOGGER.info("PREPARE | Initializing the rasterized output with nodata values.")

    with rasterio.open(output_tif, "w", **profile) as dst:
        initialize_raster_with_nodata(dst, nodata_value)
        with Pool(processes=nprocs) as pool:
            for win, arr in tqdm(
                pool.imap_unordered(worker, windows),
                total=len(windows),
                desc="Block 2 - rasterization tiles",
                unit="tile",
            ):
                dst.write(arr.astype(np.int32, copy=False), 1, window=win)

    LOGGER.info("WRITE | Rasterized output written: %s", output_tif)
    LOGGER.info("REPORT | Raster size: %s x %s px", cols, rows)
    LOGGER.info("REPORT | Resolution: %.3f m", resolution)


# =============================================================================
# STEP 3 - FUNCTIONAL INTEGRITY
# =============================================================================

def iter_shapes_for_mask(
    path: str,
    layer: str,
    buffer_m: float,
) -> Iterable[Tuple[Dict, int]]:
    """Yield vector geometries for rasterized territory masks.

    Parameters
    ----------
    path : str
        Territory vector path.
    layer : str
        Territory layer name.
    buffer_m : float
        Optional buffer distance applied before rasterization.

    Yields
    ------
    tuple
        ``(geometry_mapping, 1)`` pairs suitable for rasterization.
    """
    with fiona.open(path, layer=layer) as src:
        for feat in src:
            geom_dict = feat.get("geometry")
            if geom_dict is None:
                continue
            geom = shape(geom_dict)
            if geom.is_empty:
                continue
            if buffer_m != 0.0:
                geom = geom.buffer(buffer_m)
                if geom.is_empty:
                    continue
            yield mapping(geom), 1


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


def box_integral_mean(valid: np.ndarray, values: np.ndarray, kernel_size: int) -> np.ndarray:
    """Approximate a local mean using integral images and a square window.

    Parameters
    ----------
    valid : numpy.ndarray
        Boolean mask marking valid pixels.
    values : numpy.ndarray
        Float array containing the raster values to average.
    kernel_size : int
        Window size in pixels.

    Returns
    -------
    numpy.ndarray
        Local mean raster.
    """
    pad = kernel_size // 2

    def int2d(a: np.ndarray) -> np.ndarray:
        return a.cumsum(axis=0).cumsum(axis=1)

    def rect(ii: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
        result = ii[y1, x1]
        if y0 > 0:
            result -= ii[y0 - 1, x1]
        if x0 > 0:
            result -= ii[y1, x0 - 1]
        if y0 > 0 and x0 > 0:
            result += ii[y0 - 1, x0 - 1]
        return result

    height, width = values.shape
    values_num = np.nan_to_num(values, nan=0.0).astype(np.float64)
    counts = valid.astype(np.int32)

    values_pad = np.pad(values_num, pad, mode="reflect")
    counts_pad = np.pad(counts, pad, mode="reflect")
    ii_val = int2d(values_pad)
    ii_cnt = int2d(counts_pad)

    out = np.empty_like(values, dtype=np.float32)
    for y in range(height):
        y0, y1 = y, y + 2 * pad
        for x in range(width):
            x0, x1 = x, x + 2 * pad
            s_val = rect(ii_val, y0, x0, y1, x1)
            s_cnt = rect(ii_cnt, y0, x0, y1, x1)
            out[y, x] = s_val / s_cnt if s_cnt > 0 else np.nan
    return out


def circular_mean_fft(valid: np.ndarray, values: np.ndarray, kernel_size: int) -> np.ndarray:
    """Compute an exact circular local mean using FFT convolution.

    Parameters
    ----------
    valid : numpy.ndarray
        Boolean mask marking valid pixels.
    values : numpy.ndarray
        Float array containing the raster values to average.
    kernel_size : int
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
    except Exception as exc:
        raise RuntimeError("SciPy is not available for fftconvolve.") from exc

    kernel = circular_kernel(kernel_size).astype(np.float32)
    values_num = np.nan_to_num(values, nan=0.0).astype(np.float32)
    counts = valid.astype(np.float32)

    sum_values = fftconvolve(values_num, kernel, mode="same")
    sum_counts = fftconvolve(counts, kernel, mode="same")

    out = np.divide(
        sum_values,
        sum_counts,
        out=np.full(values.shape, np.nan, dtype=np.float32),
        where=(sum_counts > 0),
    )
    return out.astype(np.float32)


def run_convolution(valid: np.ndarray, values: np.ndarray, kernel_size: int, kernel_shape: str) -> np.ndarray:
    """Run the selected local mean implementation.

    Parameters
    ----------
    valid : numpy.ndarray
        Boolean mask marking valid pixels.
    values : numpy.ndarray
        Float array containing the raster values to average.
    kernel_size : int
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
            return circular_mean_fft(valid, values, kernel_size)
        except Exception as exc:
            warnings.warn(f"circular_fft is unavailable ({exc}); falling back to 'box'.")
            return box_integral_mean(valid, values, kernel_size)
    return box_integral_mean(valid, values, kernel_size)


def process_tile(
    arr: np.ndarray,
    nodata_in: Optional[float],
    ranges_1: Sequence[Tuple[int, int]],
    ranges_0: Sequence[Tuple[int, int]],
    ranges_null: Sequence[Tuple[int, int]],
    kernel_size: int,
    halo_top: int,
    halo_bottom: int,
    halo_left: int,
    halo_right: int,
    kernel_shape: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return:
    - fi_core: local functional integrity
    - sn_core: local semi-natural raster (0 / 1 / NaN)
    """
    a = arr.astype(np.int64, copy=False)

    mask_null = mask_from_ranges(a, ranges_null)
    mask_1 = mask_from_ranges(a, ranges_1)
    mask_0 = mask_from_ranges(a, ranges_0)

    sn = np.full(a.shape, np.nan, dtype=np.float32)
    sn[mask_1] = 1.0
    sn[mask_0] = 0.0
    sn[mask_null] = np.nan

    if nodata_in is not None:
        sn[a == nodata_in] = np.nan

    valid = ~np.isnan(sn)
    fi = run_convolution(valid, sn, kernel_size, kernel_shape)

    y0 = halo_top
    x0 = halo_left
    y1 = fi.shape[0] - halo_bottom
    x1 = fi.shape[1] - halo_right

    return fi[y0:y1, x0:x1], sn[y0:y1, x0:x1]


def process_tile_timed(args):
    """Process one integrity tile and attach execution timing metadata.

    Parameters
    ----------
    args : tuple
        Positional arguments forwarded to :func:`process_tile`, plus a final
        ``(row_idx, col_idx)`` tuple.

    Returns
    -------
    tuple
        Tuple containing success status, payload or error representation,
        elapsed time, row index, and column index.
    """
    t0 = time.perf_counter()
    rr, cc = args[-1]
    try:
        fi, sn = process_tile(*args[:-1])
        dt = time.perf_counter() - t0
        return True, (fi, sn), dt, rr, cc
    except Exception as exc:
        dt = time.perf_counter() - t0
        return False, repr(exc), dt, rr, cc


def compute_kernel_size(conv_size_m: float, pixel_size_m: float) -> int:
    diameter_px = max(1.0, conv_size_m / pixel_size_m)
    k_size = max(3, int(round(diameter_px)))
    if k_size % 2 == 0:
        k_size += 1
    return k_size


def resolve_buffer_m(
    conv_size_m: float,
    pixel_size_m: float,
    buffer_m: Optional[float],
) -> Tuple[float, int, float, str]:
    """Resolve the analysis buffer.

    Returns
    -------
    resolved_buffer_m : float
        Buffer used for rasterization / analysis.
    kernel_size : int
        Odd kernel size in pixels.
    effective_radius_m : float
        Effective convolution radius in meters after raster discretization.
    source_label : str
        'auto' or 'user'.
    """
    kernel_size = compute_kernel_size(conv_size_m, pixel_size_m)
    radius_px = kernel_size // 2
    effective_radius_m = radius_px * pixel_size_m

    if buffer_m is None:
        resolved_buffer_m = effective_radius_m + pixel_size_m
        source_label = "auto"
    else:
        resolved_buffer_m = float(buffer_m)
        source_label = "user"

    return resolved_buffer_m, kernel_size, effective_radius_m, source_label


def compute_integrity(
    raster_path: str,
    territory_vector_path: str,
    territory_layer: str,
    out_tif: str,
    sn_tif: str,
    territory_bounds: Tuple[float, float, float, float],
    ranges_1: Sequence[Tuple[int, int]],
    ranges_0: Sequence[Tuple[int, int]],
    ranges_null: Sequence[Tuple[int, int]],
    conv_size_m: float,
    pixel_max_m: Optional[float],
    buffer_m: float,
    tile_px: int,
    n_jobs: int,
    output_nodata: float,
    kernel_shape: str,
) -> None:
    """Compute functional integrity and export the two raster outputs.

    Parameters
    ----------
    raster_path : str
        Input classified raster path.
    territory_vector_path : str
        Territory vector path.
    territory_layer : str
        Territory layer name.
    out_tif : str
        Output path for the continuous integrity raster.
    sn_tif : str
        Output path for the binary semi-natural raster.
    territory_bounds : tuple of float
        Territory bounds in the raster CRS.
    ranges_1, ranges_0, ranges_null : sequence of tuple of int
        Inclusive value ranges for the three class groups.
    conv_size_m : float
        Convolution diameter in meters.
    pixel_max_m : float or None
        Pixel size used to derive the kernel size. If ``None``, the raster
        resolution is used.
    buffer_m : float
        Buffered calculation distance in meters.
    tile_px : int
        Tile size in pixels.
    n_jobs : int
        Number of worker processes.
    output_nodata : float
        Output nodata value for float rasters.
    kernel_shape : {"circular_fft", "box"}
        Convolution mode.
    """
    ensure_parent_dirs([out_tif, sn_tif])

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("The classified raster has no CRS.")
        ensure_projected_meters(src.crs)

        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        pixel_size = pixel_max_m if pixel_max_m is not None else max(res_x, res_y)
        kernel_size = compute_kernel_size(conv_size_m, pixel_size)
        radius = kernel_size // 2

        analysis_window = Window(0, 0, src.width, src.height)
        output_window = rasterio.windows.from_bounds(*territory_bounds, transform=src.transform)
        output_window = output_window.round_offsets().round_lengths()
        output_window = output_window.intersection(analysis_window)

        output_transform = rasterio.windows.transform(output_window, src.transform)
        output_height = int(output_window.height)
        output_width = int(output_window.width)

        if output_width <= 0 or output_height <= 0:
            raise ValueError("The territory extent does not intersect the rasterized input.")

        LOGGER.info("CONFIG | Classified raster input: %s", raster_path)
        LOGGER.info("CONFIG | Pixel size used: %.3f m", pixel_size)
        LOGGER.info("CONFIG | Kernel diameter: %.3f m", conv_size_m)
        LOGGER.info("CONFIG | Kernel size: %s px (radius %s px)", kernel_size, radius)
        LOGGER.info("CONFIG | Buffered raster size: %s x %s px", src.width, src.height)
        LOGGER.info("CONFIG | Integrity window size: %s x %s px", output_width, output_height)
        LOGGER.info("CONFIG | Binary output size: %s x %s px", src.width, src.height)
        LOGGER.info("CONFIG | Buffer distance used: %.3f m", buffer_m)
        LOGGER.info("CONFIG | Parallel jobs: %s", n_jobs)
        LOGGER.info("CONFIG | Tile size: %s px", tile_px)
        LOGGER.info("CONFIG | Kernel mode: %s", kernel_shape)

        LOGGER.info("PREPARE | Building the buffered and territory masks.")
        buffer_mask = rasterize(
            shapes=iter_shapes_for_mask(territory_vector_path, territory_layer, buffer_m=buffer_m),
            out_shape=(src.height, src.width),
            transform=src.transform,
            all_touched=False,
            fill=0,
            dtype="uint8",
        ).astype(bool)

        territory_mask = rasterize(
            shapes=iter_shapes_for_mask(territory_vector_path, territory_layer, buffer_m=0.0),
            out_shape=(output_height, output_width),
            transform=output_transform,
            all_touched=False,
            fill=0,
            dtype="uint8",
        ).astype(bool)

        profile_out = src.profile.copy()
        profile_out.update(
            dtype="float32",
            nodata=output_nodata,
            count=1,
            height=output_height,
            width=output_width,
            transform=output_transform,
            compress="LZW",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )

        profile_sn = src.profile.copy()
        profile_sn.update(
            dtype="float32",
            nodata=output_nodata,
            count=1,
            compress="LZW",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )

        with rasterio.open(out_tif, "w", **profile_out) as dst_out, rasterio.open(sn_tif, "w", **profile_sn) as dst_sn:
            LOGGER.info("PREPARE | Initializing the integrity output raster.")
            initialize_raster_with_nodata(dst_out, output_nodata)
            LOGGER.info("PREPARE | Initializing the binary output raster.")
            initialize_raster_with_nodata(dst_sn, output_nodata)

            tile_w = tile_h = tile_px
            cols = math.ceil(output_window.width / tile_w)
            rows = math.ceil(output_window.height / tile_h)
            total_tiles = rows * cols
            nodata_in = src.nodata

            futures = {}
            with ProcessPoolExecutor(max_workers=max(1, n_jobs)) as executor:
                for rr in range(rows):
                    for cc in range(cols):
                        win_out_local = Window(
                            col_off=cc * tile_w,
                            row_off=rr * tile_h,
                            width=min(tile_w, output_window.width - cc * tile_w),
                            height=min(tile_h, output_window.height - rr * tile_h),
                        )

                        win_out_global = Window(
                            col_off=output_window.col_off + win_out_local.col_off,
                            row_off=output_window.row_off + win_out_local.row_off,
                            width=win_out_local.width,
                            height=win_out_local.height,
                        )

                        halo = radius
                        win_halo_global = Window(
                            col_off=max(0, win_out_global.col_off - halo),
                            row_off=max(0, win_out_global.row_off - halo),
                            width=min(src.width - max(0, win_out_global.col_off - halo), win_out_global.width + 2 * halo),
                            height=min(src.height - max(0, win_out_global.row_off - halo), win_out_global.height + 2 * halo),
                        )

                        halo_top = int(win_out_global.row_off - win_halo_global.row_off)
                        halo_left = int(win_out_global.col_off - win_halo_global.col_off)
                        halo_bottom = int((win_halo_global.row_off + win_halo_global.height) - (win_out_global.row_off + win_out_global.height))
                        halo_right = int((win_halo_global.col_off + win_halo_global.width) - (win_out_global.col_off + win_out_global.width))

                        t_read0 = time.perf_counter()
                        arr = src.read(1, window=win_halo_global, resampling=Resampling.nearest)
                        read_dt = time.perf_counter() - t_read0

                        mask_tile = buffer_mask[
                            int(win_halo_global.row_off): int(win_halo_global.row_off + win_halo_global.height),
                            int(win_halo_global.col_off): int(win_halo_global.col_off + win_halo_global.width),
                        ]

                        sentinel = np.int64(nodata_in if nodata_in is not None else 2_147_483_647)
                        arr = arr.copy()
                        arr[~mask_tile] = sentinel
                        ranges_null_tile = list(ranges_null) + [(int(sentinel), int(sentinel))]

                        if VERBOSE:
                            LOGGER.info(
                                "PREPARE | Integrity tile r%s c%s | write=%sx%s px | halo=%sx%s px | read=%.2fs",
                                rr,
                                cc,
                                int(win_out_local.width),
                                int(win_out_local.height),
                                int(win_halo_global.width),
                                int(win_halo_global.height),
                                read_dt,
                            )

                        future = executor.submit(
                            process_tile_timed,
                            (
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
                                kernel_shape,
                                (rr, cc),
                            ),
                        )
                        futures[future] = {
                            "rr": rr,
                            "cc": cc,
                            "write_window": win_out_local,
                            "write_window_global": win_out_global,
                        }

                with tqdm(total=total_tiles, desc="Block 3A - integrity raster", unit="tile") as pbar:
                    for future in as_completed(futures):
                        meta = futures[future]
                        ok, payload, calc_dt, rr, cc = future.result()

                        if not ok:
                            raise RuntimeError(
                                f"Tile r{rr} c{cc} failed after {calc_dt:.2f}s: {payload}"
                            )

                        fi_tile, _ = payload

                        terr_tile = territory_mask[
                            int(meta["write_window"].row_off): int(meta["write_window"].row_off + meta["write_window"].height),
                            int(meta["write_window"].col_off): int(meta["write_window"].col_off + meta["write_window"].width),
                        ]

                        raw_core = src.read(1, window=meta["write_window_global"], resampling=Resampling.nearest)
                        null_core = mask_from_ranges(raw_core.astype(np.int64, copy=False), ranges_null)

                        fi_tile = fi_tile.astype(np.float32, copy=False)
                        fi_tile[~terr_tile] = np.nan
                        fi_tile[null_core] = np.nan

                        t_write0 = time.perf_counter()
                        dst_out.write(
                            np.nan_to_num(fi_tile, nan=output_nodata).astype(np.float32, copy=False),
                            1,
                            window=meta["write_window"],
                        )
                        write_dt = time.perf_counter() - t_write0

                        pbar.set_postfix_str(f"r{rr}c{cc} | compute {calc_dt:.2f}s | write {write_dt:.2f}s")
                        pbar.update(1)

            LOGGER.info("WRITE | Integrity output written: %s", out_tif)
            LOGGER.info("PREPARE | Exporting the binary raster over the buffered calculation area.")

            sn_windows = generate_tile_windows(src.height, src.width, tile_px)
            for win in tqdm(sn_windows, desc="Block 3B - binary raster", unit="tile"):
                arr = src.read(1, window=win, resampling=Resampling.nearest)
                arr_int = arr.astype(np.int64, copy=False)

                sn_tile = np.full(arr.shape, np.nan, dtype=np.float32)
                sn_tile[mask_from_ranges(arr_int, ranges_1)] = 1.0
                sn_tile[mask_from_ranges(arr_int, ranges_0)] = 0.0
                sn_tile[mask_from_ranges(arr_int, ranges_null)] = np.nan
                if nodata_in is not None:
                    sn_tile[arr == nodata_in] = np.nan

                mask_tile = buffer_mask[
                    int(win.row_off): int(win.row_off + win.height),
                    int(win.col_off): int(win.col_off + win.width),
                ]
                sn_tile[~mask_tile] = np.nan

                dst_sn.write(
                    np.nan_to_num(sn_tile, nan=output_nodata).astype(np.float32, copy=False),
                    1,
                    window=win,
                )

        LOGGER.info("WRITE | Binary output written: %s", sn_tif)


# =============================================================================
# STEP 4 - HISTOGRAM
# =============================================================================

def _save_histogram_png(
    bins_left: np.ndarray,
    bins_right: np.ndarray,
    counts: np.ndarray,
    out_png_path: str,
    thr_results: Optional[List[Tuple[float, float]]] = None,
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
    from matplotlib.transforms import blended_transform_factory

    # 20-colour palette by 0.05 classes, as in the original histogram script
    colors_hex = [
        "#a50f15",
        "#de2d26",
        "#fb6a4a",
        "#fcae91",
        "#fee5d9",
        "#F7FCFD",
        "#E5EFE5",
        "#D4E2DD",
        "#C2D5CD",
        "#B0C7BC",
        "#9FBAAC",
        "#8DAD9C",
        "#7CA08C",
        "#6A937C",
        "#58866C",
        "#47795C",
        "#356B4B",
        "#235E3B",
        "#12512B",
        "#00441B",
    ]

    def _hex_to_rgb01(h: str) -> Tuple[float, float, float]:
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    colors = [_hex_to_rgb01(h) for h in colors_hex]
    class_width = 1.0 / len(colors)

    centers = (bins_left + bins_right) / 2.0
    widths = bins_right - bins_left

    bar_colors = []
    for left, right in zip(bins_left, bins_right):
        mid = (left + right) / 2.0
        idx = int(mid / class_width)
        if idx >= len(colors):
            idx = len(colors) - 1
        bar_colors.append(colors[idx])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(centers, counts, width=widths, edgecolor="black", color=bar_colors)
    ax.set_xlabel("(Semi)-natural area in the surrounding km²")
    ax.set_ylabel("Pixel count")
    ax.set_title("")
    ax.spines["bottom"].set_position(("data", 0))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if thr_results:
        threshold, prop = thr_results[0]
        text = f"≥ {threshold:.2f} : {prop * 100:.1f}%"
        ax.axvline(x=threshold, linestyle="--", linewidth=0.8)

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
    plt.close(fig)


def _compute_histogram_counts(
    path_tif: str,
    bin_width: float = 0.01,
    exclude_zero: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
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
    tuple
        Histogram bin edges, histogram counts, and total number of valid pixels.
    """
    n_bins = int(round(1.0 / bin_width))
    if not math.isclose(n_bins * bin_width, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("HIST_BIN_WIDTH must divide 1.0 exactly (e.g. 0.01, 0.02, 0.005).")

    bins = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    total_valid = 0

    with rasterio.open(path_tif) as src:
        nodata = src.nodata
        windows = list(src.block_windows(1))

        for _, window in tqdm(windows, desc="Block 4 - raster blocks", unit="block"):
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
            total_valid += int(arr.size)

    return bins, counts, total_valid


def _summarize_thresholds(
    counts: np.ndarray,
    bins: np.ndarray,
    total_valid: int,
    thresholds: Sequence[float],
    bin_width: float,
) -> List[Tuple[float, float]]:
    """Compute the proportion of pixels above each threshold.

    Parameters
    ----------
    counts : numpy.ndarray
        Histogram counts.
    bins : numpy.ndarray
        Histogram bin edges.
    total_valid : int
        Total number of valid pixels.
    thresholds : sequence of float
        Threshold values to summarize.
    bin_width : float
        Histogram bin width.

    Returns
    -------
    list of tuple of float
        ``(threshold, proportion_ge_threshold)`` values.
    """
    results: List[Tuple[float, float]] = []
    for t in thresholds:
        threshold = float(t)
        if total_valid == 0:
            results.append((threshold, 0.0))
            continue
        if threshold <= 0.0:
            prop = 1.0
        elif threshold >= 1.0:
            prop = counts[-1] / total_valid
        else:
            idx = int(math.floor(threshold / bin_width + 1e-9))
            idx = max(0, min(idx, len(counts) - 1))
            prop = counts[idx:].sum() / total_valid
        results.append((threshold, float(prop)))
    return results


def export_histogram(
    raster_path: str,
    csv_path: str,
    png_path: str,
    bin_width: float,
    thresholds: Sequence[float],
    exclude_zero: bool,
) -> None:
    """Export histogram products from the integrity raster.

    Parameters
    ----------
    raster_path : str
        Input integrity raster path.
    csv_path : str
        Output CSV path.
    png_path : str
        Output PNG path.
    bin_width : float
        Histogram bin width.
    thresholds : sequence of float
        Thresholds used for summary proportions.
    exclude_zero : bool
        If ``True``, pixels equal to zero are excluded from the histogram.
    """
    ensure_parent_dirs([csv_path, png_path])

    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"Input raster not found for histogram export: {raster_path}")

    bins, counts, total_valid = _compute_histogram_counts(
        raster_path,
        bin_width=bin_width,
        exclude_zero=exclude_zero,
    )

    if total_valid == 0:
        raise ValueError("No valid pixels found in the integrity raster (all were 0, NaN, or NoData).")

    bins_left = bins[:-1]
    bins_right = bins[1:]
    labels = [f"[{left:.2f},{right:.2f})" for left, right in zip(bins_left, bins_right)]
    labels[-1] = f"[{bins_left[-1]:.2f},{bins_right[-1]:.2f}]"

    proportions = counts / total_valid
    thr_results = _summarize_thresholds(
        counts=counts,
        bins=bins,
        total_valid=total_valid,
        thresholds=thresholds,
        bin_width=bin_width,
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_label", "bin_left", "bin_right", "count", "proportion"])
        for label, left, right, count, prop in zip(labels, bins_left, bins_right, counts, proportions):
            writer.writerow([label, f"{left:.2f}", f"{right:.2f}", int(count), f"{prop:.8f}"])
        writer.writerow([])
        writer.writerow(["threshold", "proportion_ge_threshold"])
        for threshold, prop in thr_results:
            writer.writerow([f">= {threshold:.2f}", f"{prop:.8f}"])

    LOGGER.info("WRITE | Histogram CSV written: %s", csv_path)
    _save_histogram_png(
        bins_left=bins_left,
        bins_right=bins_right,
        counts=counts,
        out_png_path=png_path,
        thr_results=thr_results,
    )
    LOGGER.info("WRITE | Histogram PNG written: %s", png_path)
    LOGGER.info(
        "REPORT | Valid pixels included in the histogram (%s): %s",
        "non-zero only" if exclude_zero else "all values",
        total_valid,
    )
    for threshold, prop in thr_results:
        LOGGER.info("REPORT | proportion >= %.2f: %.3f", threshold, prop)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    """Run the complete four-block vector-based integrity workflow."""
    output_paths = build_output_paths(OUTPUT_DIR)
    configure_logging(LOG_LEVEL, output_paths["log_file"])
    start = time.perf_counter()

    (
        layer,
        bounds,
        crs,
        input_feature_count,
        field_schema_type,
        validation_mode,
        unique_values,
    ) = analyze_vector_field(
        path=INPUT_VECTOR,
        field_name=FIELD_NAME,
        layer_name=INPUT_LAYER,
    )

    territory_layer, territory_bounds, territory_crs, territory_feature_count = analyze_territory_vector(
        path=TERRITORY_VECTOR,
        layer_name=TERRITORY_LAYER,
    )

    if territory_crs != crs:
        raise ValueError(
            "The rasterization vector and the integrity territory vector do not use the same CRS. "
            f"Rasterization CRS: {crs}; territory CRS: {territory_crs}. "
            "Please reproject them to the same projected CRS before running the script."
        )

    pixel_size_for_kernel = PIXEL_MAX_M if PIXEL_MAX_M is not None else RESOLUTION
    resolved_buffer_m, kernel_size, effective_radius_m, buffer_source = resolve_buffer_m(
        conv_size_m=CONV_SIZE_M,
        pixel_size_m=pixel_size_for_kernel,
        buffer_m=BUFFER_M,
    )

    if resolved_buffer_m < effective_radius_m:
        raise ValueError(
            "BUFFER_M is smaller than the effective convolution radius. "
            f"BUFFER_M={resolved_buffer_m:.3f} m, required minimum={effective_radius_m:.3f} m."
        )

    rasterize_bounds = expand_bounds(territory_bounds, resolved_buffer_m)

    LOGGER.info("READ | Territory vector path: %s", TERRITORY_VECTOR)
    LOGGER.info("READ | Territory layer: %s", territory_layer)
    LOGGER.info("READ | Territory features: %s", territory_feature_count)
    LOGGER.info("READ | Territory bounds: %s", territory_bounds)

    LOGGER.info("CONFIG | Input vector bounds: %s", bounds)
    LOGGER.info("CONFIG | Integrity territory bounds: %s", territory_bounds)
    LOGGER.info("CONFIG | Rasterization CRS: %s", crs)
    LOGGER.info("CONFIG | Rasterization resolution: %.3f m", RESOLUTION)
    LOGGER.info("CONFIG | Requested convolution diameter: %.3f m", CONV_SIZE_M)
    LOGGER.info("CONFIG | Requested convolution radius: %.3f m", CONV_SIZE_M / 2.0)
    LOGGER.info("CONFIG | Pixel size for kernel: %.3f m", pixel_size_for_kernel)
    LOGGER.info("CONFIG | Kernel size: %s px", kernel_size)
    LOGGER.info("CONFIG | Effective convolution radius: %.3f m", effective_radius_m)
    LOGGER.info("CONFIG | Buffer source: %s", buffer_source)
    LOGGER.info("CONFIG | Buffer distance used: %.3f m", resolved_buffer_m)
    LOGGER.info("CONFIG | Buffered calculation extent: %s", rasterize_bounds)
    LOGGER.info("CONFIG | Output directory: %s", output_paths["output_dir"])
    LOGGER.info("CONFIG | Execution log: %s", output_paths["log_file"])
    LOGGER.info("CONFIG | Rasterization output: %s", RASTERIZED_TIF)
    LOGGER.info("CONFIG | Integrity output: %s", output_paths["out_tif"])
    LOGGER.info("CONFIG | Binary output: %s", output_paths["sn_tif"])
    LOGGER.info("CONFIG | Parallel jobs: %s", N_JOBS)
    LOGGER.info("CONFIG | Tile size: %s px", TILE_PX)
    LOGGER.info("CONFIG | Kernel mode: %s", KERNEL_SHAPE)

    log_block_start(1, 4, "Input vector analysis and class validation")

    LOGGER.info("READ | Input vector path: %s", INPUT_VECTOR)
    LOGGER.info("READ | Selected input layer: %s", layer)
    LOGGER.info("READ | Classification field: %s", FIELD_NAME)
    LOGGER.info("READ | Field schema type: %s", field_schema_type)
    LOGGER.info("READ | Number of features: %s", input_feature_count)
    LOGGER.info("VALIDATE | Field validation mode: %s", validation_mode)
    LOGGER.info("REPORT | Unique field values to classify:\n%s", format_values(unique_values))

    ranges_1, ranges_0, ranges_null = validate_unique_values_against_classes(
        unique_values=unique_values,
        classes_1=CLASSES_1,
        classes_0=CLASSES_0,
        classes_null=CLASSES_NULL,
    )

    log_block_end(1, 4, "Input vector analysis and class validation")

    log_block_start(2, 4, "Rasterization of the classification field")
    rasterize_vector_field(
        input_vector=INPUT_VECTOR,
        input_layer=layer,
        field_name=FIELD_NAME,
        output_tif=RASTERIZED_TIF,
        bounds=rasterize_bounds,
        crs=crs,
        resolution=RESOLUTION,
        all_touched=ALL_TOUCHED,
        nodata_value=RASTER_NODATA,
        n_jobs=N_JOBS,
        tile_px=TILE_PX,
    )
    log_block_end(2, 4, "Rasterization of the classification field")

    log_block_start(3, 4, "Functional integrity computation and raster export")
    compute_integrity(
        raster_path=RASTERIZED_TIF,
        territory_vector_path=TERRITORY_VECTOR,
        territory_layer=territory_layer,
        out_tif=output_paths["out_tif"],
        sn_tif=output_paths["sn_tif"],
        territory_bounds=territory_bounds,
        ranges_1=ranges_1,
        ranges_0=ranges_0,
        ranges_null=ranges_null,
        conv_size_m=CONV_SIZE_M,
        pixel_max_m=PIXEL_MAX_M,
        buffer_m=resolved_buffer_m,
        tile_px=TILE_PX,
        n_jobs=N_JOBS,
        output_nodata=OUTPUT_NODATA,
        kernel_shape=KERNEL_SHAPE,
    )
    log_block_end(3, 4, "Functional integrity computation and raster export")

    log_block_start(4, 4, "Integrity histogram export")
    LOGGER.info("CONFIG | Histogram CSV: %s", output_paths["hist_csv"])
    LOGGER.info("CONFIG | Histogram PNG: %s", output_paths["hist_png"])
    LOGGER.info("CONFIG | Histogram bin width: %s", HIST_BIN_WIDTH)
    LOGGER.info("CONFIG | Histogram thresholds: %s", HIST_THRESHOLDS)
    LOGGER.info("CONFIG | Exclude zero values from histogram: %s", HIST_EXCLUDE_ZERO)

    export_histogram(
        raster_path=output_paths["out_tif"],
        csv_path=output_paths["hist_csv"],
        png_path=output_paths["hist_png"],
        bin_width=HIST_BIN_WIDTH,
        thresholds=HIST_THRESHOLDS,
        exclude_zero=HIST_EXCLUDE_ZERO,
    )
    log_block_end(4, 4, "Integrity histogram export")

    elapsed = time.perf_counter() - start
    LOGGER.info("SUCCESS | Workflow completed successfully.")
    LOGGER.info("REPORT | Total runtime: %.1f s", elapsed)


if __name__ == "__main__":
    main()
