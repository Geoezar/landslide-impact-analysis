"""
DEM Processing-Lineage Sensitivity in Automated Landslide Impact Analysis
Integrating Cloud-Based DEMs and OpenStreetMap

Academic Context : GEOE 431 - Introduction to Remote Sensing
Supervisor       : Prof. Dr. M. Lutfi Suzen (METU)
Study Area       : Devrek - Zonguldak, Turkey

Reference:
  Yilmaz, Topal & Suzen (2012) used a 12.5 m DEM and field-mapped
  landslide inventory for Devrek. This pipeline does not reproduce that
  full susceptibility model; it tests how free, cloud-accessible DEM
  processing lineages change terrain derivatives and OSM building exposure.

Research Question:
  How do cloud-based DEMs from different processing lineages for the same
  study area (SRTM, NASADEM, AW3D30) change topographic derivatives and
  infrastructure-at-risk counts, and how well do these differences align
  with the 12.5 m Devrek reference study of Yilmaz et al. (2012)?
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import sys
import time
import zipfile
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any


# Project-wide configuration -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Devrek study area - WGS84 (EPSG:4326)
# Source supplied by the user as DMS coordinates, converted to decimal degrees.
DEVREK_BBOX_WGS84 = {
    "west": 31.913731,
    "south": 41.169295,
    "east": 32.034410,
    "north": 41.258297,
}

# Slope threshold validated against Yilmaz et al. (2012):
# Landslides concentrated at 5-17 deg in Caycuma Formation.
SLOPE_THRESHOLD_DEG = 15.0

# DEM catalogue - GEE asset paths plus expected native product scale.
DEM_CATALOGUE = {
    "SRTM_30m": {
        "asset": "USGS/SRTMGL1_003",
        "band": "elevation",
        "gsd_m": 30,
        "sensor_type": "C-band InSAR DSM",
        "note": "Baseline free 30 m global SRTM product in the GEE public catalog.",
        "is_collection": False,
    },
    "NASADEM_30m": {
        "asset": "NASA/NASADEM_HGT/001",
        "band": "elevation",
        "gsd_m": 30,
        "sensor_type": "Reprocessed SRTM-family DEM",
        "note": (
            "NASADEM reprocesses SRTM with auxiliary ASTER GDEM, ICESat GLAS, "
            "and PRISM inputs; used as a 30 m processing-lineage control."
        ),
        "is_collection": False,
    },
    "ALOS_AW3D30_30m": {
        "asset": "JAXA/ALOS/AW3D30/V4_1",
        "band": "DSM",
        "gsd_m": 30,
        "sensor_type": "Optical stereo DSM (ALOS PRISM AW3D30)",
        "note": (
            "Cloud-accessible AW3D30 DSM. It is not a native local 12.5 m "
            "high-resolution radar DEM; that product would require an external "
            "download workflow outside this no-download pipeline."
        ),
        "is_collection": True,
    },
}

log = logging.getLogger("main")


def _is_under(path: Path, root: Path) -> bool:
    path_resolved = path.resolve()
    root_resolved = root.resolve()
    return path_resolved == root_resolved or root_resolved in path_resolved.parents


def _safe_remove_path(path: Path, root: Path = OUTPUT_DIR) -> None:
    """Remove a generated file or directory only if it resolves under root."""
    if not path.exists():
        return
    if not _is_under(path, root):
        raise ValueError(f"Refusing to remove path outside output root: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _prepare_output_tree() -> None:
    """
    Clean only project-owned generated files before a run.

    This keeps reruns deterministic without wiping the whole outputs tree.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    static_files = [
        "pipeline.log",
        "dem/ALOS_12m.tif",
        "vector/at_risk_buildings_ALOS_12m.gpkg",
        "report/sensitivity_report.png",
        "report/summary_table.csv",
        "report/runtime_summary.csv",
        "report/run_manifest.json",
        "report/output_inventory.csv",
        "report/map_index.md",
        "report/dem_processing_lineage_report.zip",
        "report/latex.zip",
        "report/literature_matrix.md",
        "report/references_to_collect.md",
        "report/infrastructure_exposure_summary.csv",
        "report/cluster_exposure_summary.csv",
        "presentation/presentation_script.md",
        "presentation/devrek_remote_sensing_presentation.zip",
    ]
    for rel_path in static_files:
        _safe_remove_path(OUTPUT_DIR / rel_path)

    for pattern in (
        "topo/ALOS_12m_*.tif",
        "maps/report/*.png",
        "maps/verification/*.png",
        "report/figures/*.png",
        "report/tables/*.csv",
        "report/latex/*",
        "presentation/beamer/*",
        "presentation/figures/*",
        "dem/ASTER*.tif",
        "topo/ASTER*.tif",
        "vector/*ASTER*.gpkg",
        "report/*ASTER*",
    ):
        for target in OUTPUT_DIR.glob(pattern):
            _safe_remove_path(target)

    for rel_path in ("presentation/beamer", "presentation/figures"):
        _safe_remove_path(OUTPUT_DIR / rel_path)


def _configure_logging() -> None:
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
        handler.close()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OUTPUT_DIR / "pipeline.log", mode="w", encoding="utf-8"),
        ],
    )


def _phase_row(name: str, start: float) -> dict[str, Any]:
    return {"phase": name, "seconds": round(time.perf_counter() - start, 3)}


def _package_versions() -> dict[str, str]:
    packages = [
        "earthengine-api",
        "geemap",
        "rasterio",
        "numpy",
        "osmnx",
        "geopandas",
        "shapely",
        "matplotlib",
        "pytest",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _write_runtime_summary(runtime_rows: list[dict[str, Any]]) -> Path:
    report_dir = OUTPUT_DIR / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "runtime_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "seconds"])
        writer.writeheader()
        writer.writerows(runtime_rows)
    return path


def _write_run_manifest(
    runtime_rows: list[dict[str, Any]],
    report_outputs: dict[str, Path],
    visual_outputs: dict[str, Any],
    impact_results: dict[str, dict[str, Any]] | None = None,
) -> Path:
    report_dir = OUTPUT_DIR / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": ".",
        "output_root": _repository_relative_path(OUTPUT_DIR),
        "research_question": (
            "How do cloud-based DEMs from different processing lineages for the same "
            "study area (SRTM, NASADEM, AW3D30) change topographic derivatives and "
            "infrastructure-at-risk counts, and how well do these differences align "
            "with the 12.5 m Devrek reference study of Yilmaz et al. (2012)?"
        ),
        "bbox_wgs84": DEVREK_BBOX_WGS84,
        "processing_crs": "EPSG:32636",
        "slope_threshold_deg": SLOPE_THRESHOLD_DEG,
        "aspect_filter_deg": [180.0, 360.0],
        "dem_catalogue": DEM_CATALOGUE,
        "osm_buildings_extracted_at_utc": _first_osm_timestamp(
            impact_results or {}, "osm_buildings_extracted_at_utc"
        ),
        "osm_infrastructure_extracted_at_utc": _first_osm_timestamp(
            impact_results or {}, "osm_infrastructure_extracted_at_utc"
        ),
        "runtime_seconds": runtime_rows,
        "package_versions": _package_versions(),
        "report_outputs": _serialize_artifact_paths(report_outputs),
        "visual_outputs": _serialize_artifact_paths(visual_outputs),
    }
    path = report_dir / "run_manifest.json"
    text = json.dumps(manifest, indent=2)
    _assert_public_manifest_safe(text)
    path.write_text(text, encoding="utf-8")
    return path


def _repository_relative_path(path: Path | str) -> str:
    """Return a public, POSIX-style path relative to the repository root."""
    candidate = Path(path).resolve()
    root = PROJECT_ROOT.resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Artifact path is outside the repository root: {path}") from exc
    return relative.as_posix() or "."


def _serialize_artifact_paths(value: Any) -> Any:
    """Recursively preserve artifact structure while removing absolute paths."""
    if isinstance(value, Path):
        return _repository_relative_path(value)
    if isinstance(value, dict):
        return {key: _serialize_artifact_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_artifact_paths(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return _repository_relative_path(value)
    return value


_PUBLIC_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]+Users[\\/]", re.IGNORECASE),
    re.compile(r"/home/", re.IGNORECASE),
    re.compile(r"/Users/", re.IGNORECASE),
    re.compile(r"\bOneDrive\b", re.IGNORECASE),
    re.compile(r"\bDesktop\b", re.IGNORECASE),
)


def _assert_public_manifest_safe(text: str) -> None:
    """Reject manifest text that exposes a local user or desktop path."""
    for pattern in _PUBLIC_PATH_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"Public manifest contains a private path marker: {pattern.pattern}")


def _verify_manifest_in_zip(zip_path: Path) -> None:
    """Verify the manifest actually packaged into the final LaTeX archive."""
    with zipfile.ZipFile(zip_path) as archive:
        try:
            manifest_text = archive.read("run_manifest.json").decode("utf-8")
        except KeyError as exc:
            raise ValueError("LaTeX ZIP is missing run_manifest.json") from exc
    _assert_public_manifest_safe(manifest_text)


def _first_osm_timestamp(results: dict[str, dict[str, Any]], key: str) -> str:
    for result in results.values():
        if key in result and result[key]:
            return str(result[key])
        infrastructure = result.get("infrastructure", {})
        if key in infrastructure and infrastructure[key]:
            return str(infrastructure[key])
    return ""


def _write_output_inventory() -> Path:
    report_dir = OUTPUT_DIR / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "output_inventory.csv"
    files = []
    for item in sorted(OUTPUT_DIR.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(OUTPUT_DIR)
        if relative.parts and relative.parts[0] == "presentation":
            continue
        files.append(item)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes"])
        writer.writeheader()
        for item in files:
            writer.writerow(
                {
                    "relative_path": item.relative_to(OUTPUT_DIR).as_posix(),
                    "size_bytes": item.stat().st_size,
                }
            )
    return path


def _zip_directory(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(source_dir.rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(source_dir))


def run_pipeline() -> None:
    """
    Orchestrates the full pipeline in four sequential phases.

    Phase 1 - Data fetching: DEMs from GEE to local GeoTIFF.
    Phase 2 - Raster math: slope, aspect, and curvature per DEM.
    Phase 3 - Vector analysis: OSM buildings plus centroid slope/aspect sampling.
    Phase 4 - Reporting: CSV, figures, maps, audit files, and LaTeX package.
    """
    _prepare_output_tree()
    _configure_logging()
    runtime_rows: list[dict[str, Any]] = []

    log.info("=" * 70)
    log.info("GEOE 431 Landslide Pipeline | Study area: Devrek, Zonguldak")
    log.info("=" * 70)
    log.info(
        "Research question: DEM processing-lineage sensitivity against Yilmaz et al. (2012)."
    )
    log.info("Output root: %s", OUTPUT_DIR)
    log.info(
        "BBox WGS84: W=%.6f S=%.6f E=%.6f N=%.6f",
        DEVREK_BBOX_WGS84["west"],
        DEVREK_BBOX_WGS84["south"],
        DEVREK_BBOX_WGS84["east"],
        DEVREK_BBOX_WGS84["north"],
    )

    phase_start = time.perf_counter()
    log.info("[PHASE 1] Fetching DEMs from Google Earth Engine ...")
    from core.data_fetcher import DataFetcher

    fetcher = DataFetcher(
        bbox=DEVREK_BBOX_WGS84,
        dem_catalogue=DEM_CATALOGUE,
        output_dir=OUTPUT_DIR / "dem",
    )
    dem_paths = fetcher.fetch_all()
    fetcher.validate(dem_paths)
    runtime_rows.append(_phase_row("Phase 1 - DEM download", phase_start))
    log.info("[PHASE 1] Complete. %d DEM files ready.", len(dem_paths))

    phase_start = time.perf_counter()
    log.info("[PHASE 2] Computing topographic derivatives ...")
    from core.raster_math import RasterMath

    topo_layers = {}
    for dem_name, dem_path in dem_paths.items():
        rm = RasterMath(dem_path=dem_path, output_dir=OUTPUT_DIR / "topo")
        topo_layers[dem_name] = rm.compute_all()
    runtime_rows.append(_phase_row("Phase 2 - Topographic derivatives", phase_start))
    log.info("[PHASE 2] Complete.")

    phase_start = time.perf_counter()
    log.info("[PHASE 3] Fetching OSM buildings and sampling slope/aspect rasters ...")
    from core.vector_analyzer import VectorAnalyzer

    va = VectorAnalyzer(
        bbox=DEVREK_BBOX_WGS84,
        slope_threshold_deg=SLOPE_THRESHOLD_DEG,
        output_dir=OUTPUT_DIR / "vector",
    )
    impact_results = {}
    for dem_name, layers in topo_layers.items():
        log.info("  Processing: %s", dem_name)
        result = va.count_at_risk_buildings(
            slope_raster=layers["slope"],
            aspect_raster=layers["aspect"],
        )
        result["infrastructure"] = va.analyze_infrastructure_exposure(
            slope_raster=layers["slope"],
            aspect_raster=layers["aspect"],
        )
        impact_results[dem_name] = result
        log.info(
            "  [%s] Total=%d | At-risk=%d (%.1f%%) | Aspect-filtered=%d (%.1f%%)",
            dem_name,
            result["total"],
            result["at_risk"],
            result["pct_at_risk"],
            result["aspect_filtered_at_risk"],
            result["pct_aspect_filtered"],
        )
    runtime_rows.append(_phase_row("Phase 3 - OSM impact analysis", phase_start))
    log.info("[PHASE 3] Complete.")

    phase_start = time.perf_counter()
    log.info("[PHASE 4] Generating reports, no-GIS maps, and package outputs ...")
    from core.reporter import Reporter
    from core.visualizer import Visualizer

    rpt = Reporter(
        dem_catalogue=DEM_CATALOGUE,
        impact_results=impact_results,
        topo_layers=topo_layers,
        output_dir=OUTPUT_DIR / "report",
        reference={
            "slope_mean_deg": 17.0,
            "slope_std_deg": 10.0,
            "n_landslides": 26,
            "prof_curv_std": 0.003,
        },
    )
    report_outputs = rpt.generate()

    visualizer = Visualizer(
        dem_catalogue=DEM_CATALOGUE,
        dem_paths=dem_paths,
        topo_layers=topo_layers,
        impact_results=impact_results,
        output_dir=OUTPUT_DIR / "maps",
        report_dir=OUTPUT_DIR / "report",
        bbox=DEVREK_BBOX_WGS84,
    )
    visual_outputs = visualizer.generate()
    runtime_rows.append(_phase_row("Phase 4 - Reporting and visualization", phase_start))

    runtime_path = _write_runtime_summary(runtime_rows)
    manifest_path = _write_run_manifest(
        runtime_rows,
        report_outputs,
        visual_outputs,
        impact_results,
    )
    inventory_path = _write_output_inventory()
    package_outputs = rpt.generate_latex_package(
        artifacts={
            "summary_csv": report_outputs["csv"],
            "sensitivity_figure": report_outputs["figure"],
            "standalone_figures": report_outputs.get("standalone_figures", []),
            "runtime_csv": runtime_path,
            "manifest_json": manifest_path,
            "inventory_csv": inventory_path,
            "map_index": visual_outputs["index"],
            "report_figures": visual_outputs["report_figures"],
            "verification_figures": visual_outputs["verification_figures"],
            "additional_tables": visual_outputs.get("tables", []),
        },
        runtime_rows=runtime_rows,
    )
    inventory_path = _write_output_inventory()
    latex_dir = package_outputs.get("latex_dir")
    latex_zip = package_outputs.get("latex_zip")
    if latex_dir and latex_zip:
        shutil.copy2(inventory_path, Path(latex_dir) / "tables" / "output_inventory.csv")
        _zip_directory(Path(latex_dir), Path(latex_zip))
        _verify_manifest_in_zip(Path(latex_zip))
    report_outputs.update(package_outputs)

    log.info("[PHASE 4] Complete. See %s", OUTPUT_DIR / "report")

    log.info("=" * 70)
    log.info(
        "PROCESSING-LINEAGE SENSITIVITY SUMMARY (slope threshold=%.0f deg)",
        SLOPE_THRESHOLD_DEG,
    )
    log.info("=" * 70)
    log.info(
        "  %-18s  %6s  %8s  %10s  %12s  %10s",
        "DEM",
        "GSD(m)",
        "At-risk",
        "Pct(%)",
        "AspectRisk",
        "AspectPct",
    )
    log.info("  " + "-" * 74)
    for dem_name, res in impact_results.items():
        gsd = DEM_CATALOGUE[dem_name]["gsd_m"]
        log.info(
            "  %-18s  %6d  %8d  %10.1f  %12d  %10.1f",
            dem_name,
            gsd,
            res["at_risk"],
            res["pct_at_risk"],
            res["aspect_filtered_at_risk"],
            res["pct_aspect_filtered"],
        )
    log.info("=" * 70)
    log.info("LaTeX source package: %s", report_outputs["latex_zip"])
    log.info("Pipeline finished successfully.")


if __name__ == "__main__":
    run_pipeline()
