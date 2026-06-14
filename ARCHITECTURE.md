# System Architecture

This project is a command-line spatial data pipeline. It downloads cloud DEM products, computes terrain derivatives, samples OpenStreetMap infrastructure, and produces no-GIS report figures and a LaTeX source package.

## Data Flow

```text
Google Earth Engine DEM products
  -> core/data_fetcher.py
  -> outputs/dem/*.tif
  -> core/raster_math.py
  -> outputs/topo/*_slope.tif, *_aspect.tif, *_profile_curvature.tif, *_plan_curvature.tif
  -> core/vector_analyzer.py + OpenStreetMap
  -> outputs/vector/*.gpkg
  -> core/visualizer.py and core/reporter.py
  -> outputs/report/*, outputs/maps/*
```

## Modules

### `main.py`

Owns configuration, output cleanup, phase orchestration, runtime logging, manifest writing, output inventory, and final package creation. It keeps all outputs under the project-local `outputs/` directory.

### `core/data_fetcher.py`

Exports `SRTM_30m`, `NASADEM_30m`, and `ALOS_AW3D30_30m` from Google Earth Engine. The outputs are clipped to the Devrek bounding box and reprojected to UTM Zone 36N for metric terrain calculations.

### `core/raster_math.py`

Computes vectorized terrain derivatives:

- slope,
- aspect,
- profile curvature,
- plan curvature.

The implementation keeps in-memory arrays available for the reporter so the report does not reopen every raster unnecessarily.

### `core/vector_analyzer.py`

Fetches OpenStreetMap buildings and civil infrastructure with OSMnx. It counts building exposure by centroid sampling and computes supporting road exposure by 30 m interval midpoint sampling after projection to `EPSG:32636`. It records OSM extraction timestamps because OSM is a live database.

### `core/visualizer.py`

Creates no-GIS PNG figures directly from GeoTIFF and GeoPackage outputs:

- report-facing slope comparisons, exposure overlays, cluster maps, zoom panels, agreement matrices, and sensitivity charts,
- verification-facing hillshade, slope, aspect, profile curvature, and plan curvature maps for every DEM.

Sentinel-2 context imagery uses `COPERNICUS/S2_SR_HARMONIZED` with QA60 opaque-cloud/cirrus masking and SCL cloud-shadow/cloud-class filtering. Hillshade is used as fallback if imagery export fails.

### `core/reporter.py`

Creates:

- `summary_table.csv`,
- `sensitivity_report.png`,
- `dem_processing_lineage_report.zip`,
- LaTeX source, references, figures, tables, compile notes, and literature notes.

The report explains every included figure and table. The reference study is discussed as context, not inserted as a fake pipeline row.

## Scientific Guardrails

- Do not use the rejected ASTER GED asset as a DEM.
- Do not label AW3D30 as the initially assumed high-resolution radar DEM.
- Do not claim a complete landslide susceptibility model.
- Keep limitations explicit: no field validation, no inventory overlay, no lithology/drainage/rainfall/fault-distance/soil/geotechnical variables, DSM surface effects, live OSM, rural OSM sparsity, centroid sampling, and 30 m road midpoint sampling.

## Output Semantics

Report figures are meant for interpretation. Verification figures are meant for layer checks. Runtime, manifest, and inventory files support reproducibility, but the scientific core remains the DEM-derived terrain and infrastructure-exposure sensitivity comparison.
