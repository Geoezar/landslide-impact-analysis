# Automated Spatial Pipeline for Landslide Impact Analysis Integrating Satellite DEM and OpenStreetMap

## Overview

This repository contains a fully automated, command-line Python pipeline for DEM processing-lineage sensitivity analysis in landslide impact assessment. The pipeline targets the Devrek region of Zonguldak, Turkey, and measures how the choice of cloud-accessible Digital Elevation Model changes terrain derivatives and OpenStreetMap infrastructure-exposure counts — without any manual desktop GIS step.

**Research question:**
> How do SRTM, NASADEM, and AW3D30 alter topographic derivatives and at-risk building counts for the same Devrek study area, and how do those differences relate to the 12.5 m reference context of Yilmaz, Topal, and Suzen (2012)?

The project does not produce a complete landslide susceptibility model. It is an automated exposure and DEM-sensitivity workflow that extends the reference study from a data-engineering perspective.

---

## Scientific Context

The pipeline is anchored to a published reference study (Yilmaz et al. 2012) that used a 12.5 m local DEM and a field-mapped landslide inventory for Devrek. This pipeline does not reproduce that full susceptibility model. It tests how three cloud-accessible 30 m DEM products — representing different Remote Sensing processing lineages — change slope, aspect, profile curvature, plan curvature, and building-exposure counts when everything else (bounding box, thresholds, OSM extract, and code) stays fixed.

**Active DEM catalogue:**

| Key | GEE Asset | Sensor Type | Role |
|---|---|---|---|
| `SRTM_30m` | `USGS/SRTMGL1_003` | C-band InSAR DSM | Baseline 30 m global product |
| `NASADEM_30m` | `NASA/NASADEM_HGT/001` | Reprocessed SRTM-family DEM | Processing-improvement control |
| `ALOS_AW3D30_30m` | `JAXA/ALOS/AW3D30/V4_1` | Optical-stereo DSM (ALOS PRISM) | Different product lineage |

The ASTER GED asset was rejected because it is an emissivity and land-surface-temperature product, not a DEM suitable for slope or curvature derivation.

**Study area bounding box (WGS84):**
```json
{ "west": 31.913731, "south": 41.169295, "east": 32.034410, "north": 41.258297 }
```

Raster calculations run in UTM Zone 36N (`EPSG:32636`) so slope, curvature, and road-length values are metric.

---

## Pipeline Architecture

The pipeline has four sequential phases orchestrated by `main.py`:

```
Google Earth Engine
  → core/data_fetcher.py     Phase 1: Download DEMs as GeoTIFF
  → core/raster_math.py      Phase 2: Slope, aspect, profile curvature, plan curvature
  → core/vector_analyzer.py  Phase 3: OSM buildings and infrastructure exposure
  → core/reporter.py         Phase 4: CSV tables, sensitivity figure, LaTeX package
  → core/visualizer.py       Phase 4: PNG maps (report-facing and verification-facing)
  → core/talk_script.py      Phase 4: Figure-guided talk script
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `main.py` | Configuration, output cleanup, phase orchestration, runtime logging, manifest, inventory |
| `core/data_fetcher.py` | GEE authentication, DEM export, UTM reprojection, geemap/URL fallback |
| `core/raster_math.py` | Vectorized slope (Horn 1981 via richdem), aspect, Zevenbergen–Thorne curvatures |
| `core/vector_analyzer.py` | OSMnx building fetch, centroid sampling, 30 m road interval sampling |
| `core/reporter.py` | Summary CSV, sensitivity figure, LaTeX report source package |
| `core/visualizer.py` | No-GIS PNGs: slope comparison, at-risk overlays, cluster maps, agreement matrix |
| `core/talk_script.py` | Figure-guided narration for an 8–10 minute academic presentation |

### Output structure

Every run writes exclusively to `outputs/` and overwrites only project-owned generated files:

```
outputs/
  dem/                 Downloaded GeoTIFFs (SRTM, NASADEM, AW3D30)
  topo/                Slope, aspect, profile curvature, plan curvature rasters
  vector/              OSM building footprints and at-risk GeoPackages
  maps/
    report/            Report-facing PNG figures
    verification/      Layer-check PNG figures
  report/
    summary_table.csv
    sensitivity_report.png
    infrastructure_exposure_summary.csv
    cluster_exposure_summary.csv
    dem_processing_lineage_report.zip   ← LaTeX report source package
    run_manifest.json
    output_inventory.csv
    runtime_summary.csv
  presentation/
    presentation_script.md
  pipeline.log
```

---

## Risk Metrics

- **Primary metric:** building centroid on `slope >= 15 deg`
- **Secondary subset:** building centroid on `slope >= 15 deg` AND `180 <= aspect <= 360 deg` (SW/W/NW facing, matching the Devrek aspect pattern from Yilmaz et al. 2012)
- **Supporting metrics:** exposed road length (m) sampled every 30 m, civil utility and public facility counts

---

## Dependencies

The pipeline uses `uv` for environment and package management. All packages are resolved at run time — there is no separate `requirements.txt`.

**Required packages:**

| Package | Purpose |
|---|---|
| `earthengine-api` | Google Earth Engine API |
| `geemap` | GEE image export to local GeoTIFF |
| `rasterio` | GeoTIFF I/O and spatial metadata |
| `numpy` | Raster array operations |
| `richdem` | Hydrologically correct slope and aspect (Horn 1981); falls back to numpy gradient if unavailable |
| `osmnx` | OpenStreetMap building and infrastructure fetch |
| `geopandas` | Vector data handling |
| `shapely` | Geometry operations |
| `matplotlib` | All PNG figures |
| `requests` | URL-based GEE export fallback |
| `pytest` | Test suite |

**System requirements:**
- Python 3.11 or later
- `uv` package manager ([installation](https://github.com/astral-sh/uv))
- A Google account with access to [Google Earth Engine](https://earthengine.google.com/) (free for research)
- Active internet connection (GEE export and OSM Overpass API)

---

## Setup and Execution

### 1. Install uv

```powershell
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

Confirm installation:
```powershell
uv --version
```

### 2. Configure Your Google Earth Engine Project

Supply your own Google Cloud project ID at runtime. The repository does not
contain or default to the maintainer's project identifier.

```powershell
$env:EE_PROJECT = "your-google-cloud-project-id"
```

`EE_PROJECT` is a project identifier, not an API key. Never commit API keys,
OAuth client secrets, service-account JSON files, or Earth Engine credential
files to this repository.

### 3. Authenticate with Google Earth Engine

Run once to store credentials locally:
```powershell
uv run --with earthengine-api python -c "import ee; ee.Authenticate()"
```

A browser window opens for Google login. After approval, credentials are cached and reused automatically on subsequent runs.

Earth Engine stores its authentication state outside this repository through
the official client workflow. The pipeline uses the `EE_PROJECT` value from
the current process and never writes it into generated public manifests.

### 4. Run the pipeline

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv run python main.py
```

The first run takes longer because uv resolves and caches all packages. Subsequent runs use the local `.uv-cache`.

### 5. Run the test suite

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv run --with pytest pytest -q
```

Tests do not require GEE credentials or an internet connection. They use synthetic in-memory data and `tmp_path` fixtures.

---

## Adapting This Pipeline to a Different Study Area

To apply this pipeline to a different region, modify the following components in `main.py`:

| Variable | What to change |
|---|---|
| `DEVREK_BBOX_WGS84` | Replace with the `{"west", "south", "east", "north"}` bounding box of your study area (WGS84 decimal degrees) |
| `SLOPE_THRESHOLD_DEG` | Adjust the slope threshold based on local landslide literature for your region |
| `DEM_CATALOGUE` | Add, remove, or replace DEM entries; each entry needs `asset`, `band`, `gsd_m`, `sensor_type`, `note`, and `is_collection` |

In `core/data_fetcher.py`, change the `project` argument in `DataFetcher._init_gee` to your GEE project ID.

All outputs, figures, and the LaTeX report are generated automatically from these configuration values without further code changes.

---

## Limitations

- No field validation or ground truthing was performed.
- No landslide inventory overlay, lithology, drainage density, rainfall, fault distance, soil, or geotechnical variables are included.
- DEM inputs are DSM/DEM products, not a bare-earth DTM; trees, buildings, and canopy may affect the elevation surface.
- OSM completeness and sparsity bias in rural Turkish regions may cause undercounting of buildings, minor roads, and civil infrastructure.
- OSM is a live database and may change between runs; extraction timestamps are recorded in `run_manifest.json`.
- Building exposure uses centroid sampling; road exposure uses 30 m interval midpoint sampling — both are transparent approximations, not engineering survey measurements.

---

## License and External Data

The original source code and project documentation in this repository are
licensed under the [MIT License](LICENSE). This code license does **not** grant
rights to third-party datasets, downloaded imagery, OpenStreetMap data, or
derived data products. Those materials remain subject to their respective
provider terms.

| Source | Role | Attribution and terms |
|---|---|---|
| OpenStreetMap | Buildings, roads, utilities, and public facilities | © OpenStreetMap contributors; data available under the [Open Data Commons Open Database License](https://www.openstreetmap.org/copyright) |
| SRTM | Baseline DEM | NASA / USGS / JPL-Caltech; [Earth Engine dataset page](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003) |
| NASADEM | Reprocessed SRTM-family DEM | NASA / USGS / JPL-Caltech; [Earth Engine dataset page](https://developers.google.com/earth-engine/datasets/catalog/NASA_NASADEM_HGT_001) |
| AW3D30 | Optical-stereo DSM | AW3D30 (JAXA); [Earth Engine dataset page](https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_AW3D30_V4_1) and [JAXA data terms](https://earth.jaxa.jp/en/data/policy/) |
| Sentinel-2 | Optional true-color context | Copernicus Sentinel-2 / ESA; [Earth Engine dataset page](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) |
| Google Earth Engine | Cloud processing platform | Google Earth Engine platform and dataset catalogue |

Generated OSM-derived public figures carry a compact
`© OpenStreetMap contributors` notice. Figures that actually use Sentinel-2
context carry a compact `Copernicus Sentinel-2 / ESA` source note. DEM-only
figures keep source information in this README and in report captions or
metadata instead of adding long legal footers to internal verification maps.

---

## References

- Yilmaz, C., Topal, T., & Suzen, M. L. (2012). GIS-based landslide susceptibility mapping using bivariate statistical analysis in Devrek (Zonguldak-Turkey). *Environmental Earth Sciences*, 65, 2161–2178.
- Gorelick, N., et al. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment*, 202, 18–27.
- Zevenbergen, L. W., & Thorne, C. R. (1987). Quantitative analysis of land surface topography. *Earth Surface Processes and Landforms*, 12(1), 47–56.
- Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. *Computers, Environment and Urban Systems*, 65, 126–139.
- Horn, B. K. P. (1981). Hill shading and the reflectance map. *Proceedings of the IEEE*, 69(1), 14–47.

---

## Academic Context

This pipeline was developed for **GEOE 431 – Introduction to Remote Sensing** at Middle East Technical University, under the supervision of Prof. Dr. M. Lutfi Suzen.
