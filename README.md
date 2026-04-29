# Biosphere Functional Integrity for Human-Modified Landscapes

This repository provides three Python scripts to compute a **biosphere functional integrity** indicator from land-cover data and to prepare/refine the input classification when needed.

The repository contains two alternative workflows for the same final calculation:

- **`Integrity_Vector.py`**: use this when your land-cover database is in **vector** format.
- **`Integrity_Raster.py`**: use this when your land-cover database is already in **raster** format.

It also contains one preparation/refinement workflow:

- **`Intersection_Vectors.py`**: use this to **overlay two polygon databases** and create an improved classification before running the integrity calculation. This is useful when a broad land-cover category may contain both **semi-natural** and **dominated by human** situations, and a second database can help separate them more precisely.

---

## Why this repository exists

These scripts were developed to support territorial analyses of **functional integrity** in human-modified landscapes.

The core idea is to estimate, for each location in a territory, the **share of (semi-)natural habitat in the surrounding landscape**. The workflow is inspired by Mohamed et al. (2024), who reviewed the evidence for several nature’s contributions to people depanding on surrounding landscape. They concluded that **at least 20%–25% of complex, diverse (semi-)natural habitat per km²** is needed to sustain multiple ecological functions in human-modified landscapes and that this provision becomes **very low or almost absent below 10% habitat**.

The conclusion from this paper relies on three families of nature's contributions:

- **pollination**,
- **pest and disease regulation**,
- **positive effects on human health and well-being**.

In practice, this repository proposes a GIS workflow to compute this indicator :
1. classifies land-cover into **semi-natural = 1**, **dominated by human = 0**, and **land cover ignored in calculation = NaN**,
2. computes a moving-window share of semi-natural habitat around each pixel,
3. exports maps and histograms that can be used for territorial diagnosis.

---

## Repository contents

| File | Purpose | Typical use case |
|---|---|---|
| `Integrity_Vector.py` | Computes the integrity indicator from a **vector land-cover layer** | Your source database is a polygon layer (GeoPackage / shapefile) |
| `Integrity_Raster.py` | Computes the same integrity indicator from a **raster land-cover layer** | Your source database is a raser layer |
| `Intersection_Vectors.py` | Overlays two polygon datasets and builds a refined output classification | A main category in the primary database may mix both class `0` and class `1`, and a secondary database helps disambiguate it |

---

## Conceptual workflow

### 1. Optional refinement of the land-cover classification

Some land-cover classes are too broad for direct functional integrity analysis. A category may contain both:
- areas that should be treated as **semi-natural**,
- and areas that should be treated as **dominated by human**.

`Intersection_Vectors.py` addresses this issue by enabling to intersect a given land cover database that contains such ambiguity with another land cover database solving this typical ambiguity. The program work as following:

- taking a **primary polygon database**,
- overlaying it with a **secondary polygon database**,
- applying **rule-based recoding**,
- and producing a refined output field that can then be used as the classification input for the integrity scripts.

This step improves the ecological meaning of the final binary classification.

### 2. Reclassify land cover into 1 / 0 / NaN

Both integrity scripts start from a categorical land-cover dataset and assign each value to one of three groups:
- **`1`**: semi-natural land cover,
- **`0`**: land cover dominated by human,
- **`NaN`**: land cover ignored in calculation.

This classification is controlled by user-editable parameter strings such as `CLASSES_1`, `CLASSES_0`, and `CLASSES_NULL`.

### 3. Compute the local integrity value

The local integrity value is computed as the **mean share of semi-natural habitat** within a moving neighborhood around each pixel.

Operationally, the scripts use a **1000 m convolution diameter** (that is, a **500 m radius** neighborhood by default), which is aligned with the implementation described in the methods of Mohamed et al. (2024).

A pixel value therefore represents a local proportion ranging from **0 to 1**:
- **0** = no semi-natural habitat in the surrounding neighborhood,
- **1** = entirely semi-natural surrounding neighborhood,

### 4. Compare results with a reference threshold

The scripts export histogram outputs so that users can compare the territory against reference thresholds such as:
- **0.20**,
- **0.25** (default value)
By default, histogram counts and threshold summaries are computed on **all valid integrity pixels in the territory, including zeros**.
These two values represent the uncertainty zone defined by Mohamed et al. (2024). Below 0.20–0.25, biosphere functional integrity is no longer sustained.
---

## What each script does
### `Intersection_Vectors.py`

Use this script **before** the integrity calculation when a single land-cover dataset is not precise enough.

Main steps:
1. read a **primary** polygon database and a **secondary** polygon database,
2. validate geometry and required fields,
3. overlay only the subset of primary polygons that need refinement. The polygons that does not need refinement keep the value of the primary field,
4. apply ordered **rule-based recoding** in this overlay to define a value for each polygon from the primary and secondary fields,
5. write an output GeoPackage with a refined code field.

Important rule behavior:
- rules are evaluated in order,
- if several rules match the same polygon, the **last matching rule wins**,
- if the output field is later used as the classification field in `Integrity_Vector.py`, the resulting codes must remain **integers** or **integer-compatible** values.

Typical use:
- the **primary database** provides the main territorial land-cover geometry,
- the **secondary database** adds ecological detail for ambiguous categories (e.g. pasture VS grassland),
- the output field becomes the classification field used later by `Integrity_Vector`.

### `Integrity_Vector.py`

Use this script when the input land-cover database is a **vector dataset**.

Main steps:

1. read the territory layer and the land-cover vector layer,
2. validate the input field and class assignments,
3. rasterize the classified vector layer,
4. compute the binary raster and the integrity raster,
5. export:
   - a **binary raster** (`0 / 1 / NaN`),
   - an **integrity raster** (`0–1`),
   - a **CSV histogram**,
   - a **PNG histogram**.


### `Integrity_Raster.py`

Use this script when the input land-cover database is already a **raster**.

Main steps:

1. read the territory vector,
2. inspect raster values inside the calculation area and validate class assignments,
3. compute the binary raster and the integrity raster,
4. export the same histogram products as in the vector version.

By default, histogram counts and threshold summaries include **all valid pixels**, including pixels equal to `0`.

---

## Inputs expected by the scripts

### Intersection script

The overlay script requires:

- a **primary vector dataset**,
- a **secondary vector dataset**,
- one field in each dataset used for recoding,
- an ordered list of **rules** specifying how primary and secondary values should be combined.

### Integrity scripts

Both integrity scripts require:

- a **territory vector** defining where the indicator is computed,
- a classified land-cover database:
  - **vector** for `Integrity_Vector`,
  - **raster** for `Integrity_Raster`,
- the definition of which classes correspond to:
  - semi-natural habitat,
  - land cover dominated by human,
  - land cover ignored in calculation.

---

## Outputs produced

Depending on the script, the repository produces some or all of the following:

- **refined vector dataset** (`.gpkg`) from the overlay step,
- **binary habitat raster** (`0 / 1 / NaN`),
- **functional integrity raster** (`0–1`),
- **histogram CSV**,
- **histogram PNG**.

These outputs can be used for:

- territorial diagnosis,
- comparison between territories,
- monitoring against thresholds such as **20%** or **25%**,
- producing maps and summary graphics for reports.

---

## How to use

### 1. Edit the user parameters

Each script is designed to be configured directly in its **USER PARAMETERS** section.

Replace the placeholder paths such as:

```python
path\to\input_vector.gpkg
path\to\input_raster.tif
path\to\territory.gpkg
path\to\output.tif
```

with your own file locations.

Other parameter will have to be edited, they are explain directly in each program.

### 2. Choose the appropriate workflow

- Use **`Intersection_Vectors.py`** if you first need to refine a land-cover classification.
- Use **`Integrity_Vector.py`** if your final classified source is a vector layer.
- Use **`Integrity_Raster.py`** if your final classified source is already a raster.

### 3. Run the script

---

## Important assumptions and limitations
- The scripts assume a **projected CRS in meters** for distance-based processing.
- The integrity value depends on the **quality of the input classification**. If broad categories mix ecologically different situations, use the overlay workflow first.
- The scripts produce one indicator on biosphere functional integrity, not a full ecological assessment of habitat quality, species composition, or ecosystem condition.
- Thresholds such as **20%** and **25%** should be interpreted as **generic reference levels**, and local ecological context still matters.

---

## Scientific reference

This repository is inspired by:

> Mohamed, A., DeClerck, F., Verburg, P. H., Obura, D., Abrams, J. F., Zafra-Calvo, N., Rocha, J., Estrada-Carmona, N., Fremier, A., Jones, S. K., Meier, I. C., & Stewart-Koster, B. (2024). *Securing Nature’s Contributions to People requires at least 20%–25% (semi-)natural habitat in human-modified landscapes*. **One Earth, 7**, 59–71.

Article link:

- https://doi.org/10.1016/j.oneear.2023.12.008

---

## Support

Please contact emile.balembois@emse.fr for any questions.
