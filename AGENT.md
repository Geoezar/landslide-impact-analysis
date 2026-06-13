# Agent Handoff Notes

Use this file as the operational brief for future agents working in this repository.

## Active Scientific Position

The project is a Remote Sensing data-engineering pipeline for **DEM processing-lineage sensitivity in automated landslide impact analysis**. Do not revert the framing to a broad radar-versus-optical claim.

The active DEM set is fixed unless the user explicitly changes it:

- `SRTM_30m`
- `NASADEM_30m`
- `ALOS_AW3D30_30m`

Do not revive ASTER unless the user supplies a valid ASTER GDEM file and explicitly asks for a local-source workflow. The rejected Google Earth Engine ASTER GED asset is not a DEM for this analysis.

Do not describe AW3D30 as the initially assumed high-resolution radar DEM. The active GEE product is an ALOS PRISM optical-stereo DSM at about 30 m.

## Current Study Area

```python
DEVREK_BBOX_WGS84 = {
    "west": 31.913731,
    "south": 41.169295,
    "east": 32.034410,
    "north": 41.258297,
}
```

OSMnx 2.x expects `(left, bottom, right, top)`, so the current `(west, south, east, north)` order is correct.

## Architecture

```text
main.py                  Orchestrates all phases
core/data_fetcher.py     GEE export and UTM reprojection
core/raster_math.py      Slope, aspect, profile curvature, plan curvature
core/vector_analyzer.py  OSM buildings, roads, utilities, exposure sampling
core/visualizer.py       No-GIS report and verification PNG figures
core/reporter.py         CSV summaries and LaTeX report source package
core/talk_script.py      Long figure-guided speaking script only
```

There is no dashboard and no web UI. The pipeline is CLI-first.

## Reporting Rules

- The LaTeX report must not mention delivery decks, presentation files, or speaker notes.
- Every figure and table included in the report needs a caption and a direct explanatory paragraph.
- `Reference_12.5m` must not appear as a table row. The reference study is context, not a pipeline output.
- OSM timestamps must remain in `run_manifest.json` and in the report text.
- Keep the rural OSM completeness/sparsity limitation explicit.

## Output Policy

Outputs are project-local under `outputs/`. The run setup removes only project-owned generated files and stale legacy artifacts, then overwrites the current outputs.

Expected high-value outputs:

```text
outputs/report/dem_processing_lineage_report.zip
outputs/report/summary_table.csv
outputs/report/sensitivity_report.png
outputs/maps/report/*.png
outputs/maps/verification/*.png
outputs/presentation/presentation_script.md
```

The `outputs/presentation/` directory contains only the long figure-guided script. Do not regenerate a deck package from code.

## Verification Commands

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv run --with pytest pytest -q
uv run python -m py_compile main.py core\data_fetcher.py core\raster_math.py core\vector_analyzer.py core\reporter.py core\visualizer.py core\talk_script.py tests\conftest.py tests\test_reporter.py tests\test_vector_analyzer.py tests\test_raster_math.py tests\test_visualizer.py tests\test_config.py tests\test_talk_script.py
uv run python main.py
```

## Architecture/Algorithmic Push

The next scientifically meaningful upgrades are not cosmetic. They are a landslide inventory overlay, lithology/drainage/fault-distance layers, field validation, and uncertainty reporting for OSM completeness. Keep buildings as the primary metric; roads and utilities are supporting indicators.
