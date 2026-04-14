#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute an overlay between two polygon vector layers.

The program is designed to use two VECTOR polygon layers:
- a primary layer whose features define the base output geometry,
- a secondary layer consulted only when required by the recoding rules.

The program takes as input:
- a primary vector dataset,
- a secondary vector dataset,
- a primary field,
- a secondary field,
- rule-based recoding instructions,
- an output GeoPackage path.

It is organized into three processing blocks.

Block 1
    Read the input datasets, validate that the requested fields exist, verify
    that a spatial overlay can be attempted safely, inspect the values found in
    the primary and secondary fields, and report whether these values are fully
    referenced by the recoding rules.

Block 2
    Run the spatial overlay only for the subset of primary features whose
    primary-field values require consultation of the secondary layer according
    to the rules. Features that do not require overlay bypass this step.

Block 3
    Compute the output code field, merge overlaid and bypassed features, and
    write the final result to a GeoPackage.

Notes
-----
- The script is designed for polygon and multipolygon layers.
- Field values are normalized to strings internally during rule matching so
  that rule comparisons remain robust when storage dtypes differ between the
  primary and secondary fields.
- Rules are evaluated in order. If several rules match the same feature, the
  last matching rule overrides the previous matching outputs.
- If the output field is reused later as the classification field in
  ``Integrity_Vector.py``, its values must remain integers or values that can
  be converted safely to integers.
- If geometry preprocessing is set to ``"off"`` or ``"light"`` and the
  overlay fails, the script retries once with full geometry cleaning.
- Console output uses Python's standard ``logging`` module.
- Docstrings follow a NumPy-style layout compatible with general PEP 257
  docstring conventions.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import sys
from typing import Any, Iterable

import fiona
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import MultiPolygon
from tqdm import tqdm

# =============================================================================
# 1. USER PARAMETERS
# =============================================================================

# Input datasets.
INPUT_PRIMARY_DATASET: str = (
    r"path/to/primary_layer"
)
INPUT_SECONDARY_DATASET: str = (
    r"path/to/secondary_layer"
)

# Output dataset.
OUTPUT_GPKG: str = (
    r"path/to/output.gpkg"
)
OUTPUT_LAYER_NAME: str = "primary_secondary_intersection"

# Input fields.
PRIMARY_FIELD: str = "field_name"
SECONDARY_FIELD: str = "field_name"

# Output field.
OUTPUT_CODE_FIELD: str = "Code_Primary_Secondary"

# Geometry preprocessing before overlay.
# - "off"   : no preprocessing before overlay.
# - "light" : drop null/empty and non-polygon geometries only.
# - "full"  : light preprocessing + fix invalid geometries with buffer(0).
GEOMETRY_PREPROCESSING_MODE: str = "light"

# Recoding rules.
#
# Each rule is evaluated in order.
# If several rules match the same polygon, the last matching rule wins.
# - primary: expected value in PRIMARY_FIELD
# - secondary_in: accepted values in SECONDARY_FIELD
# - secondary_not_in: excluded values in SECONDARY_FIELD (optional)
# - output: value written to OUTPUT_CODE_FIELD if the rule matches
#           (keep it integer-compatible if the result will feed Integrity_Vector.py)
RULES: list[dict[str, Any]] = [
    {"primary": 12, "secondary_in": {3301, 3302, 8001}, "secondary_not_in": None, "output": 120}, # example rule
]

# Parallelization parameters.
N_CORES: int = 7
CHUNK_SIZE: int = 2000

# General settings.
LOG_LEVEL: str = "INFO"

VALID_GEOMETRY_PREPROCESSING_MODES = {"off", "light", "full"}


# =============================================================================
# LOGGING HELPERS
# =============================================================================

LOGGER = logging.getLogger("vector_overlay")


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


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory of an output path when needed.

    Parameters
    ----------
    path : str
        Output file path.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# =============================================================================
# 2. EXCEPTIONS AND NORMALIZATION HELPERS
# =============================================================================

class UserInputError(Exception):
    """Raised when an input dataset or parameter prevents the workflow."""


class OverlayExecutionError(Exception):
    """Raised when the spatial overlay fails at execution time."""


def normalize_value(value: Any) -> str | None:
    """Return a normalized string representation of a value."""
    if pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return str(value).strip()

    text = str(value).strip()

    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
        return str(numeric)
    except Exception:
        return text


def normalize_values(values: Iterable[Any] | None) -> set[str] | None:
    """Normalize an iterable of values into a set of strings."""
    if values is None:
        return None
    return {v for v in (normalize_value(value) for value in values) if v is not None}


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a pandas Series into comparable string values."""
    return series.map(normalize_value)


# =============================================================================
# 3. RULE HANDLING
# =============================================================================

REQUIRED_RULE_KEYS = {"primary", "secondary_in", "secondary_not_in", "output"}


def compile_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and compile user rules."""
    if not rules:
        raise UserInputError("RULES is empty. At least one rule is required.")

    compiled: list[dict[str, Any]] = []

    for index, rule in enumerate(rules, start=1):
        missing_keys = REQUIRED_RULE_KEYS - set(rule)
        if missing_keys:
            raise UserInputError(
                f"Rule #{index} is missing required key(s): {sorted(missing_keys)}."
            )

        primary_key = normalize_value(rule["primary"])
        secondary_in = normalize_values(rule.get("secondary_in"))
        secondary_not_in = normalize_values(rule.get("secondary_not_in"))

        if primary_key is None:
            raise UserInputError(f"Rule #{index} has an empty 'primary' value.")

        compiled.append(
            {
                **rule,
                "primary_key": primary_key,
                "secondary_in_keys": secondary_in,
                "secondary_not_in_keys": secondary_not_in,
            }
        )

    return compiled


def get_primary_values_requiring_overlay(compiled_rules: list[dict[str, Any]]) -> list[str]:
    """Return the normalized primary values that require the secondary layer."""
    return sorted({rule["primary_key"] for rule in compiled_rules})


def get_rule_value_references(
    compiled_rules: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    """Return all normalized values referenced by the rules."""
    primary_values = {rule["primary_key"] for rule in compiled_rules}
    secondary_values: set[str] = set()

    for rule in compiled_rules:
        if rule["secondary_in_keys"]:
            secondary_values.update(rule["secondary_in_keys"])
        if rule["secondary_not_in_keys"]:
            secondary_values.update(rule["secondary_not_in_keys"])

    return primary_values, secondary_values


# =============================================================================
# 4. INPUT READING AND GEOMETRY PREPARATION
# =============================================================================


def get_first_layer_name(path: str) -> str | None:
    """Return the first layer name when the file contains multiple layers."""
    try:
        layers = fiona.listlayers(path)
    except Exception:
        return None

    if not layers:
        return None
    return layers[0]


def read_vector_dataset(path: str, role: str) -> tuple[gpd.GeoDataFrame, str | None]:
    """Read a vector dataset."""
    if not os.path.exists(path):
        raise UserInputError(f"{role} dataset not found: {path}")

    layer_name = get_first_layer_name(path)

    try:
        if layer_name is not None:
            gdf = gpd.read_file(path, layer=layer_name)
        else:
            gdf = gpd.read_file(path)
    except Exception as exc:
        raise UserInputError(f"Unable to read {role} dataset '{path}': {exc}") from exc

    return gdf, layer_name


def ensure_field_exists(gdf: gpd.GeoDataFrame, field_name: str, role: str) -> None:
    """Raise an error when a requested field is missing."""
    if field_name not in gdf.columns:
        raise UserInputError(
            f"Field '{field_name}' was not found in the {role} dataset. "
            f"Available fields: {list(gdf.columns)}"
        )


def _extract_polygonal_geometry(geom):
    """Return the polygonal part of a geometry when possible."""
    if geom is None or geom.is_empty:
        return None

    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom

    if geom.geom_type == "GeometryCollection":
        polygons = []
        for part in geom.geoms:
            if part.is_empty:
                continue
            if part.geom_type == "Polygon":
                polygons.append(part)
            elif part.geom_type == "MultiPolygon":
                polygons.extend(list(part.geoms))
        if not polygons:
            return None
        if len(polygons) == 1:
            return polygons[0]
        return MultiPolygon(polygons)

    return None


def filter_non_empty_polygon_geometries(
    gdf: gpd.GeoDataFrame,
    label: str,
    *,
    convert_geometry_collections: bool = True,
    print_details: bool = True,
) -> gpd.GeoDataFrame:
    """Keep only non-empty polygonal geometries.

    This is the light preprocessing mode: it removes unusable geometries without
    attempting to repair invalid ones.
    """
    filtered = gdf.copy()
    initial_count = len(filtered)

    mask_not_null = filtered.geometry.notnull() & (~filtered.geometry.is_empty)
    removed_empty = initial_count - int(mask_not_null.sum())
    filtered = filtered.loc[mask_not_null].copy()

    if convert_geometry_collections:
        filtered["geometry"] = filtered["geometry"].apply(_extract_polygonal_geometry)
    else:
        polygon_mask = filtered.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        filtered = filtered.loc[polygon_mask].copy()

    filtered = filtered[filtered.geometry.notnull()].copy()
    filtered = filtered[~filtered.geometry.is_empty].copy()

    polygon_mask = filtered.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    removed_non_polygon = len(filtered) - int(polygon_mask.sum())
    filtered = filtered.loc[polygon_mask].copy()

    if print_details:
        LOGGER.info("PREPARE | Geometry preprocessing for %s (light mode).", label)
        if removed_empty:
            LOGGER.info("PREPARE | Removed %s null or empty geometries.", removed_empty)
        if removed_non_polygon:
            LOGGER.info("PREPARE | Removed %s non-polygon geometries.", removed_non_polygon)
        LOGGER.info("PREPARE | %s polygon features kept out of %s.", len(filtered), initial_count)

    return filtered


def clean_geometries(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    """Keep only valid polygonal geometries.

    This is the full preprocessing mode: it applies the light filter and then
    repairs invalid geometries with ``buffer(0)``.
    """
    cleaned = filter_non_empty_polygon_geometries(
        gdf,
        label=label,
        convert_geometry_collections=True,
        print_details=False,
    )
    initial_count = len(gdf)

    LOGGER.info("PREPARE | Geometry preprocessing for %s (full mode).", label)

    removed_before = initial_count - len(cleaned)
    if removed_before:
        LOGGER.info(
            "PREPARE | Removed null, empty, non-polygon, or non-polygonal collection geometries before repair: %s.",
            removed_before,
        )

    invalid_mask = ~cleaned.geometry.is_valid
    if invalid_mask.any():
        LOGGER.info("PREPARE | Fixing %s invalid geometries with buffer(0).", int(invalid_mask.sum()))
        cleaned.loc[invalid_mask, "geometry"] = cleaned.loc[invalid_mask, "geometry"].buffer(0)

    cleaned["geometry"] = cleaned["geometry"].apply(_extract_polygonal_geometry)
    cleaned = cleaned[cleaned.geometry.notnull()].copy()
    cleaned = cleaned[~cleaned.geometry.is_empty].copy()
    cleaned = cleaned[cleaned.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    LOGGER.info("PREPARE | %s valid polygon features kept out of %s.", len(cleaned), initial_count)
    return cleaned


def ensure_usable_geometry(
    gdf: gpd.GeoDataFrame,
    label: str,
    *,
    context: str,
) -> None:
    """Raise an error if no usable polygon geometry remains."""
    if gdf.empty:
        raise UserInputError(
            f"Spatial intersection is not possible: no usable polygon geometry remains "
            f"in the {label} dataset after {context}."
        )


def prepare_datasets_for_overlay(
    primary_gdf: gpd.GeoDataFrame,
    secondary_gdf: gpd.GeoDataFrame,
    geometry_preprocessing_mode: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Validate and optionally prepare datasets before overlay."""
    mode = geometry_preprocessing_mode.lower()
    if mode not in VALID_GEOMETRY_PREPROCESSING_MODES:
        raise UserInputError(
            "GEOMETRY_PREPROCESSING_MODE must be one of: "
            f"{sorted(VALID_GEOMETRY_PREPROCESSING_MODES)}."
        )

    if primary_gdf.crs is None:
        raise UserInputError(
            "Spatial intersection is not possible safely because the primary "
            "dataset has no CRS. Define a CRS before running the script."
        )
    if secondary_gdf.crs is None:
        raise UserInputError(
            "Spatial intersection is not possible safely because the secondary "
            "dataset has no CRS. Define a CRS before running the script."
        )

    if primary_gdf.crs != secondary_gdf.crs:
        LOGGER.info(
            "TRANSFORM | CRS mismatch detected. Reprojecting the primary dataset to the secondary CRS."
        )
        primary_gdf = primary_gdf.to_crs(secondary_gdf.crs)

    if mode == "off":
        LOGGER.info(
            "CONFIG | Geometry preprocessing before overlay: off. The script will try the overlay directly."
        )
        ensure_usable_geometry(primary_gdf, "primary", context="reading")
        ensure_usable_geometry(secondary_gdf, "secondary", context="reading")
        return primary_gdf, secondary_gdf

    if mode == "light":
        primary_gdf = filter_non_empty_polygon_geometries(primary_gdf, label="primary layer")
        secondary_gdf = filter_non_empty_polygon_geometries(secondary_gdf, label="secondary layer")
        ensure_usable_geometry(primary_gdf, "primary", context="light preprocessing")
        ensure_usable_geometry(secondary_gdf, "secondary", context="light preprocessing")
        return primary_gdf, secondary_gdf

    primary_gdf = clean_geometries(primary_gdf, label="primary layer")
    secondary_gdf = clean_geometries(secondary_gdf, label="secondary layer")
    ensure_usable_geometry(primary_gdf, "primary", context="full preprocessing")
    ensure_usable_geometry(secondary_gdf, "secondary", context="full preprocessing")
    return primary_gdf, secondary_gdf


# =============================================================================
# 5. DATASET INSPECTION
# =============================================================================


def summarize_dataset(gdf: gpd.GeoDataFrame, field_name: str, label: str) -> None:
    """Print a short summary of a dataset."""
    geometry_types = sorted(gdf.geometry.geom_type.dropna().unique().tolist())
    LOGGER.info("REPORT | %s dataset summary", label)
    LOGGER.info("REPORT | Features: %s", len(gdf))
    LOGGER.info("REPORT | CRS: %s", gdf.crs)
    LOGGER.info("REPORT | Field: %s", field_name)
    LOGGER.info("REPORT | Field dtype: %s", gdf[field_name].dtype)
    LOGGER.info("REPORT | Geometry types: %s", geometry_types)


def get_unique_non_null_values(series: pd.Series) -> list[Any]:
    """Return sorted unique non-null values, with a string fallback sort."""
    values = [value for value in series.dropna().unique().tolist()]
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: str(value))


def print_field_values(series: pd.Series, field_name: str, label: str) -> list[Any]:
    """Print all unique non-null values found in a field."""
    values = get_unique_non_null_values(series)
    LOGGER.info(
        "REPORT | Unique values found in %s field %s (%s values): %s",
        label,
        field_name,
        len(values),
        values,
    )
    return values


def warn_unreferenced_values(
    values: list[Any],
    referenced_values: set[str],
    label: str,
    field_name: str,
) -> None:
    """Warn when some field values are not referenced by the rules."""
    missing = [value for value in values if normalize_value(value) not in referenced_values]
    if missing:
        LOGGER.warning(
            "VALIDATE | %s value(s) in %s field %s are not referenced by the recoding rules: %s",
            len(missing),
            label,
            field_name,
            missing,
        )
    else:
        LOGGER.info(
            "VALIDATE | All values found in %s field %s are referenced by the recoding rules.",
            label,
            field_name,
        )


def print_field_dtype_warning(primary_series: pd.Series, secondary_series: pd.Series) -> None:
    """Warn when primary and secondary field dtypes differ."""
    primary_dtype = primary_series.dtype
    secondary_dtype = secondary_series.dtype

    if primary_dtype != secondary_dtype:
        LOGGER.warning("VALIDATE | Primary and secondary fields do not share the same pandas dtype.")
        LOGGER.warning("VALIDATE | Primary field dtype: %s", primary_dtype)
        LOGGER.warning("VALIDATE | Secondary field dtype: %s", secondary_dtype)
        LOGGER.warning(
            "VALIDATE | This does not prevent spatial overlay. Rule matching is performed after normalizing values to strings."
        )
        LOGGER.warning(
            "VALIDATE | The final output field dtype will depend on the values kept from the primary field and the output values written by the rules."
        )
    else:
        LOGGER.info("VALIDATE | Primary and secondary fields share the same pandas dtype: %s.", primary_dtype)


# =============================================================================
# 6. RECODING
# =============================================================================


def compute_code_vectorized(
    df: gpd.GeoDataFrame,
    primary_field: str,
    secondary_field: str,
    compiled_rules: list[dict[str, Any]],
) -> pd.Series:
    """Compute the output code field.

    Matching is done on normalized string representations of the field values.
    The original primary value is kept as the default output. Rules are applied
    sequentially, so if several rules match the same row, the last matching
    rule overrides the previous ones.
    """
    primary_raw = df[primary_field]
    secondary_raw = df[secondary_field]

    primary_keys = normalize_series(primary_raw)
    secondary_keys = normalize_series(secondary_raw)

    code = primary_raw.copy()
    secondary_present = secondary_keys.notna()

    # Rules are applied in order; later matching rules intentionally override
    # previous matches on the same row.
    for rule in compiled_rules:
        mask = secondary_present & (primary_keys == rule["primary_key"])

        if rule["secondary_in_keys"] is not None:
            mask = mask & secondary_keys.isin(rule["secondary_in_keys"])

        if rule["secondary_not_in_keys"] is not None:
            mask = mask & (~secondary_keys.isin(rule["secondary_not_in_keys"]))

        code.loc[mask] = rule["output"]

    return code


# =============================================================================
# 7. OVERLAY EXECUTION
# =============================================================================

_SECONDARY_GDF: gpd.GeoDataFrame | None = None


def _init_worker(secondary_gdf: gpd.GeoDataFrame) -> None:
    """Store the secondary GeoDataFrame in each worker process."""
    global _SECONDARY_GDF
    _SECONDARY_GDF = secondary_gdf


def keep_non_empty_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only non-empty polygonal geometries after overlay."""
    if gdf is None or gdf.empty:
        return gdf

    filtered = gdf[gdf.geometry.notnull()].copy()
    filtered = filtered[~filtered.geometry.is_empty].copy()
    filtered["geometry"] = filtered["geometry"].apply(_extract_polygonal_geometry)
    filtered = filtered[filtered.geometry.notnull()].copy()
    filtered = filtered[~filtered.geometry.is_empty].copy()
    filtered = filtered[filtered.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    return filtered


def _overlay_worker(args: tuple[int, gpd.GeoDataFrame, str]) -> gpd.GeoDataFrame | None:
    """Run overlay on one chunk of the primary layer."""
    chunk_index, primary_chunk, how = args
    del chunk_index

    if primary_chunk.empty:
        return None

    try:
        result = gpd.overlay(primary_chunk, _SECONDARY_GDF, how=how, keep_geom_type=False)
    except Exception as exc:
        raise OverlayExecutionError(str(exc)) from exc

    return keep_non_empty_polygons(result)


def parallel_overlay(
    primary_gdf: gpd.GeoDataFrame,
    secondary_gdf: gpd.GeoDataFrame,
    how: str,
    n_cores: int,
    chunk_size: int,
    progress_label: str,
) -> gpd.GeoDataFrame:
    """Run a spatial overlay in parallel on chunks of the primary dataset."""
    total = len(primary_gdf)
    if total == 0:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=primary_gdf.crs)

    tasks: list[tuple[int, gpd.GeoDataFrame, str]] = []
    n_chunks = math.ceil(total / chunk_size)

    for index in range(n_chunks):
        start = index * chunk_size
        stop = min((index + 1) * chunk_size, total)
        tasks.append((index, primary_gdf.iloc[start:stop].copy(), how))

    results: list[gpd.GeoDataFrame] = []

    try:
        with mp.Pool(
            processes=n_cores,
            initializer=_init_worker,
            initargs=(secondary_gdf,),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(_overlay_worker, tasks),
                total=n_chunks,
                desc=progress_label,
            ):
                if result is not None and not result.empty:
                    results.append(result)
    except OverlayExecutionError:
        raise
    except Exception as exc:
        raise OverlayExecutionError(str(exc)) from exc

    if not results:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=primary_gdf.crs)

    return gpd.GeoDataFrame(
        pd.concat(results, ignore_index=True),
        geometry="geometry",
        crs=primary_gdf.crs,
    )


def run_overlay_with_optional_full_retry(
    primary_target: gpd.GeoDataFrame,
    secondary_gdf: gpd.GeoDataFrame,
    base_primary_gdf: gpd.GeoDataFrame,
    base_secondary_gdf: gpd.GeoDataFrame,
    geometry_preprocessing_mode: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, str]:
    """Run overlay and retry once with full geometry cleaning if needed.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame, geopandas.GeoDataFrame, str]
        Overlaid result, prepared primary dataset, prepared secondary dataset,
        and the effective preprocessing mode used for the successful run.
    """
    mode_used = geometry_preprocessing_mode.lower()

    try:
        overlaid = parallel_overlay(
            primary_target,
            secondary_gdf,
            how="identity",
            n_cores=N_CORES,
            chunk_size=CHUNK_SIZE,
            progress_label="Identity overlay",
        )
        return overlaid, base_primary_gdf, base_secondary_gdf, mode_used
    except OverlayExecutionError as first_error:
        if mode_used == "full":
            raise UserInputError(
                "Spatial intersection failed during execution even after full geometry cleaning. "
                "Possible causes include corrupted geometries, topology errors raised by the "
                f"geometry engine, or insufficient memory for the dataset size. Original error: {first_error}"
            ) from first_error

        LOGGER.warning("FALLBACK | Spatial overlay failed with the current geometry preprocessing mode.")
        LOGGER.warning("FALLBACK | Current mode: %s", mode_used)
        LOGGER.warning("FALLBACK | Original error: %s", first_error)
        LOGGER.warning("FALLBACK | Retrying once with full geometry cleaning.")

        retry_primary_gdf, retry_secondary_gdf = prepare_datasets_for_overlay(
            base_primary_gdf,
            base_secondary_gdf,
            geometry_preprocessing_mode="full",
        )

        retry_primary_keys = normalize_series(retry_primary_gdf[PRIMARY_FIELD])
        retry_overlay_mask = retry_primary_keys.isin(get_primary_values_requiring_overlay(compile_rules(RULES)))
        retry_primary_target = retry_primary_gdf.loc[retry_overlay_mask].copy()
        retry_primary_target["_source_id"] = retry_primary_target.index

        try:
            overlaid = parallel_overlay(
                retry_primary_target,
                retry_secondary_gdf,
                how="identity",
                n_cores=N_CORES,
                chunk_size=CHUNK_SIZE,
                progress_label="Identity overlay (retry with full cleaning)",
            )
        except OverlayExecutionError as second_error:
            raise UserInputError(
                "Spatial intersection failed during execution. The script retried once with full "
                "geometry cleaning, but the overlay still failed. Possible causes include corrupted "
                "geometries, topology errors raised by the geometry engine, or insufficient memory "
                f"for the dataset size. First error: {first_error}. Retry error: {second_error}"
            ) from second_error

        return overlaid, retry_primary_gdf, retry_secondary_gdf, "full"


# =============================================================================
# 8. POST-OVERLAY FIELD RECOVERY
# =============================================================================


def find_field(columns: Iterable[str], base_name: str) -> str | None:
    """Find a column by exact name or overlay-generated suffix."""
    columns = list(columns)
    if base_name in columns:
        return base_name

    candidates = [column for column in columns if column.startswith(base_name)]
    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        LOGGER.warning(
            "VALIDATE | Several columns match %s after overlay: %s. Using %s.",
            base_name,
            candidates,
            candidates[0],
        )
        return candidates[0]

    return None


# =============================================================================
# 9. MAIN PROGRAM
# =============================================================================


def main() -> None:
    """Run the full overlay and recoding workflow."""
    LOGGER.info("READ | Opening input datasets.")

    compiled_rules = compile_rules(RULES)
    primary_rule_values, secondary_rule_values = get_rule_value_references(compiled_rules)
    primary_values_requiring_overlay = get_primary_values_requiring_overlay(compiled_rules)

    raw_primary_gdf, primary_layer_name = read_vector_dataset(
        INPUT_PRIMARY_DATASET,
        role="primary",
    )
    raw_secondary_gdf, secondary_layer_name = read_vector_dataset(
        INPUT_SECONDARY_DATASET,
        role="secondary",
    )

    LOGGER.info("CONFIG | Primary dataset: %s", INPUT_PRIMARY_DATASET)
    LOGGER.info("CONFIG | Secondary dataset: %s", INPUT_SECONDARY_DATASET)
    LOGGER.info("CONFIG | Output GeoPackage: %s", OUTPUT_GPKG)
    LOGGER.info("CONFIG | Output layer name: %s", OUTPUT_LAYER_NAME)
    LOGGER.info("CONFIG | Primary field: %s", PRIMARY_FIELD)
    LOGGER.info("CONFIG | Secondary field: %s", SECONDARY_FIELD)
    LOGGER.info("CONFIG | Output code field: %s", OUTPUT_CODE_FIELD)
    LOGGER.info("CONFIG | Geometry preprocessing mode: %s", GEOMETRY_PREPROCESSING_MODE)
    LOGGER.info("CONFIG | Parallel jobs: %s", N_CORES)
    LOGGER.info("CONFIG | Chunk size: %s", CHUNK_SIZE)

    if primary_layer_name is not None:
        LOGGER.info("READ | Primary layer read: %s", primary_layer_name)
    if secondary_layer_name is not None:
        LOGGER.info("READ | Secondary layer read: %s", secondary_layer_name)

    ensure_field_exists(raw_primary_gdf, PRIMARY_FIELD, role="primary")
    ensure_field_exists(raw_secondary_gdf, SECONDARY_FIELD, role="secondary")

    prepared_primary_gdf, prepared_secondary_gdf = prepare_datasets_for_overlay(
        raw_primary_gdf,
        raw_secondary_gdf,
        geometry_preprocessing_mode=GEOMETRY_PREPROCESSING_MODE,
    )

    log_block_start(1, "Field inspection and overlay validation")
    summarize_dataset(prepared_primary_gdf, PRIMARY_FIELD, label="Primary")
    summarize_dataset(prepared_secondary_gdf, SECONDARY_FIELD, label="Secondary")
    print_field_dtype_warning(
        prepared_primary_gdf[PRIMARY_FIELD],
        prepared_secondary_gdf[SECONDARY_FIELD],
    )

    primary_values = print_field_values(prepared_primary_gdf[PRIMARY_FIELD], PRIMARY_FIELD, "primary")
    secondary_values = print_field_values(prepared_secondary_gdf[SECONDARY_FIELD], SECONDARY_FIELD, "secondary")

    warn_unreferenced_values(primary_values, primary_rule_values, "primary", PRIMARY_FIELD)
    warn_unreferenced_values(secondary_values, secondary_rule_values, "secondary", SECONDARY_FIELD)

    LOGGER.info(
        "REPORT | Primary values that require overlay according to the rules: %s",
        primary_values_requiring_overlay,
    )
    log_block_end(1, "Field inspection and overlay validation")

    log_block_start(2, "Spatial overlay")

    prepared_primary_keys = normalize_series(prepared_primary_gdf[PRIMARY_FIELD])
    overlay_mask = prepared_primary_keys.isin(primary_values_requiring_overlay)

    primary_target = prepared_primary_gdf.loc[overlay_mask].copy()
    primary_rest = prepared_primary_gdf.loc[~overlay_mask].copy()
    primary_target["_source_id"] = primary_target.index

    LOGGER.info("PREPARE | Primary features requiring overlay: %s", len(primary_target))
    LOGGER.info("PREPARE | Primary features bypassing overlay: %s", len(primary_rest))

    if not primary_target.empty:
        LOGGER.info("PREPARE | Running identity overlay in parallel.")
        overlaid, prepared_primary_gdf, prepared_secondary_gdf, mode_used = run_overlay_with_optional_full_retry(
            primary_target,
            prepared_secondary_gdf,
            raw_primary_gdf,
            raw_secondary_gdf,
            GEOMETRY_PREPROCESSING_MODE,
        )

        if mode_used != GEOMETRY_PREPROCESSING_MODE.lower():
            LOGGER.info(
                "FALLBACK | Overlay finally succeeded with geometry preprocessing mode: %s",
                mode_used,
            )

        if mode_used != GEOMETRY_PREPROCESSING_MODE.lower():
            prepared_primary_keys = normalize_series(prepared_primary_gdf[PRIMARY_FIELD])
            overlay_mask = prepared_primary_keys.isin(primary_values_requiring_overlay)
            primary_target = prepared_primary_gdf.loc[overlay_mask].copy()
            primary_rest = prepared_primary_gdf.loc[~overlay_mask].copy()
            primary_target["_source_id"] = primary_target.index

        overlaid = keep_non_empty_polygons(overlaid)

        if "_source_id" not in overlaid.columns:
            overlaid["_source_id"] = pd.NA

        overlaid_primary_field = find_field(overlaid.columns, PRIMARY_FIELD)
        overlaid_secondary_field = find_field(overlaid.columns, SECONDARY_FIELD)

        if overlaid_primary_field is None or overlaid_secondary_field is None:
            raise UserInputError(
                "The overlay result does not contain the expected primary and/or secondary "
                f"field. Available columns: {list(overlaid.columns)}"
            )

        expected_ids = set(primary_target["_source_id"].tolist())
        obtained_ids = set(overlaid["_source_id"].dropna().tolist())
        missing_ids = expected_ids - obtained_ids

        if missing_ids:
            LOGGER.warning(
                "REPORT | %s overlaid primary features were lost during overlay. Their original geometries will be reinserted with an empty secondary value.",
                len(missing_ids),
            )
            missing = primary_target.loc[primary_target["_source_id"].isin(missing_ids)].copy()

            if overlaid_primary_field not in missing.columns:
                missing[overlaid_primary_field] = missing[PRIMARY_FIELD]
            if overlaid_secondary_field not in missing.columns:
                missing[overlaid_secondary_field] = pd.NA

            common_columns = sorted(set(overlaid.columns) | set(missing.columns))
            overlaid = overlaid.reindex(columns=common_columns)
            missing = missing.reindex(columns=common_columns)
            overlaid = pd.concat([overlaid, missing], ignore_index=True)
    else:
        overlaid = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=prepared_primary_gdf.crs)
        overlaid_primary_field = PRIMARY_FIELD
        overlaid_secondary_field = SECONDARY_FIELD

    log_block_end(2, "Spatial overlay")

    log_block_start(3, "Output code computation and GeoPackage export")

    if not overlaid.empty:
        overlaid[OUTPUT_CODE_FIELD] = compute_code_vectorized(
            overlaid,
            primary_field=overlaid_primary_field,
            secondary_field=overlaid_secondary_field,
            compiled_rules=compiled_rules,
        )

    if SECONDARY_FIELD not in primary_rest.columns:
        primary_rest[SECONDARY_FIELD] = pd.NA
    primary_rest[OUTPUT_CODE_FIELD] = primary_rest[PRIMARY_FIELD]

    common_columns = sorted(set(overlaid.columns) | set(primary_rest.columns))
    overlaid = overlaid.reindex(columns=common_columns)
    primary_rest = primary_rest.reindex(columns=common_columns)

    result = pd.concat([overlaid, primary_rest], ignore_index=True)
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=prepared_secondary_gdf.crs)

    if "_source_id" in result.columns:
        result = result.drop(columns=["_source_id"])

    result = keep_non_empty_polygons(result)
    LOGGER.info("REPORT | Final output feature count: %s", len(result))
    LOGGER.info("REPORT | Final output field dtype (%s): %s", OUTPUT_CODE_FIELD, result[OUTPUT_CODE_FIELD].dtype)

    ensure_parent_dir(OUTPUT_GPKG)

    if os.path.exists(OUTPUT_GPKG):
        LOGGER.info("WRITE | The output GeoPackage already exists and will be overwritten.")
        os.remove(OUTPUT_GPKG)

    result.to_file(OUTPUT_GPKG, layer=OUTPUT_LAYER_NAME, driver="GPKG")
    LOGGER.info("WRITE | Output written: %s (layer=%s)", OUTPUT_GPKG, OUTPUT_LAYER_NAME)
    log_block_end(3, "Output code computation and GeoPackage export")
    LOGGER.info("SUCCESS | Workflow completed successfully.")


if __name__ == "__main__":
    mp.freeze_support()
    configure_logging(LOG_LEVEL)
    try:
        main()
    except UserInputError as exc:
        LOGGER.error("FAIL | %s", exc)
        sys.exit(1)
