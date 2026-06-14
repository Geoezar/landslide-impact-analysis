"""
core/vector_analyzer.py
=======================
Fetches building footprints from OpenStreetMap via OSMnx, then samples
terrain rasters at building centroids to estimate infrastructure exposure.

The primary metric is slope >= 15 deg. A secondary academic subset also
requires 180 <= aspect <= 360 deg, matching the SW/W/NW aspect emphasis in
the Devrek landslide literature.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("vector_analyzer")


class VectorAnalyzer:
    """
    Downloads OSM buildings and counts how many fall on slopes exceeding
    the operational threshold.
    """

    def __init__(
        self,
        bbox: dict,
        slope_threshold_deg: float,
        output_dir: Path,
    ):
        self.bbox = bbox
        self.threshold = slope_threshold_deg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._buildings = None
        self._infrastructure = None
        self.osm_buildings_extracted_at_utc: str | None = None
        self.osm_infrastructure_extracted_at_utc: str | None = None

    def fetch_buildings(self) -> object:
        """
        Download building footprints from OpenStreetMap for the bbox.
        Buildings are fetched once per process and cached in memory.
        """
        if self._buildings is not None:
            log.info("  Using session-cached buildings (%d polygons).", len(self._buildings))
            return self._buildings

        try:
            # pyrefly: ignore [missing-import]
            import osmnx as ox
        except ImportError as exc:
            raise ImportError("osmnx not installed. Run: uv pip install osmnx") from exc

        ox.settings.max_query_area_size = 200_000_000
        ox.settings.timeout = 180

        gpkg_path = self.output_dir / "devrek_buildings.gpkg"
        if gpkg_path.exists():
            log.info("  Removing previous buildings file: %s", gpkg_path.name)
            gpkg_path.unlink()

        log.info("  Querying OSM Overpass API for buildings (single request) ...")
        log.info(
            "  bbox: N=%.4f S=%.4f E=%.4f W=%.4f",
            self.bbox["north"],
            self.bbox["south"],
            self.bbox["east"],
            self.bbox["west"],
        )
        log.info("  Expected wait: 5-15 seconds ...")

        try:
            gdf = ox.features_from_bbox(
                bbox=self._osmnx_bbox(),
                tags={"building": True},
            )
        except Exception as exc:
            log.error("OSM fetch failed: %s", exc)
            raise

        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        gdf = gdf.reset_index(drop=True)
        self.osm_buildings_extracted_at_utc = self._utc_now()
        log.info("  Downloaded %d building polygons.", len(gdf))

        gdf.to_file(gpkg_path, driver="GPKG")
        log.info("  Saved to: %s", gpkg_path.name)
        self._buildings = gdf
        return gdf

    def fetch_infrastructure(self) -> object:
        """
        Download non-building civil infrastructure from OSM.

        Buildings remain the primary academic exposure metric. Roads and
        utility-like features are supporting indicators for report-scale
        interpretation, so failures here should not stop the core pipeline.
        """
        if self._infrastructure is not None:
            log.info(
                "  Using session-cached infrastructure (%d features).",
                len(self._infrastructure),
            )
            return self._infrastructure

        try:
            import geopandas as gpd
            # pyrefly: ignore [missing-import]
            import osmnx as ox
        except ImportError as exc:
            raise ImportError("osmnx/geopandas not installed. Run: uv pip install osmnx geopandas") from exc

        ox.settings.max_query_area_size = 200_000_000
        ox.settings.timeout = 180

        gpkg_path = self.output_dir / "devrek_civil_infrastructure.gpkg"
        if gpkg_path.exists():
            log.info("  Removing previous infrastructure file: %s", gpkg_path.name)
            gpkg_path.unlink()

        tags = {
            "highway": True,
            "power": True,
            "man_made": [
                "water_tower",
                "storage_tank",
                "reservoir_covered",
                "water_works",
                "wastewater_plant",
            ],
            "amenity": ["hospital", "school", "fire_station"],
        }
        log.info("  Querying OSM Overpass API for roads and civil utilities ...")
        try:
            gdf = ox.features_from_bbox(bbox=self._osmnx_bbox(), tags=tags)
        except Exception as exc:
            log.warning("  OSM infrastructure fetch failed; support metrics skipped: %s", exc)
            gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        self.osm_infrastructure_extracted_at_utc = self._utc_now()

        if not gdf.empty:
            gdf = gdf.reset_index(drop=True)
            gdf["exposure_category"] = gdf.apply(self._categorize_infrastructure, axis=1)
            gdf = gdf[
                gdf.geometry.geom_type.isin(
                    [
                        "Point",
                        "MultiPoint",
                        "LineString",
                        "MultiLineString",
                        "Polygon",
                        "MultiPolygon",
                    ]
                )
            ].copy()
            gdf.to_file(gpkg_path, driver="GPKG")
            log.info("  Saved infrastructure features: %s", gpkg_path.name)

        self._infrastructure = gdf
        return gdf

    def _osmnx_bbox(self) -> tuple[float, float, float, float]:
        """
        Return bbox order for OSMnx 2.x features_from_bbox.

        OSMnx 2.1.0 documents bbox as (left, bottom, right, top), i.e.
        (west, south, east, north).
        """
        return (
            self.bbox["west"],
            self.bbox["south"],
            self.bbox["east"],
            self.bbox["north"],
        )

    def count_at_risk_buildings(
        self,
        slope_raster: Path,
        aspect_raster: Path | None = None,
    ) -> dict:
        """
        Count buildings whose centroid sits on slope >= threshold.
        """
        # pyrefly: ignore [missing-import]
        import rasterio

        buildings = self.fetch_buildings()
        if self.osm_buildings_extracted_at_utc is None:
            self.osm_buildings_extracted_at_utc = self._utc_now()

        with rasterio.open(slope_raster) as src:
            raster_crs = src.crs
            transform = src.transform
            slope_data = src.read(1).astype(np.float32)
            nodata = src.nodata

        aspect_data = None
        aspect_nodata = None
        if aspect_raster is not None:
            with rasterio.open(aspect_raster) as src:
                if src.crs != raster_crs:
                    raise ValueError(
                        f"Aspect raster CRS {src.crs} does not match slope raster CRS {raster_crs}."
                    )
                if src.transform != transform:
                    raise ValueError("Aspect raster transform does not match slope raster transform.")
                aspect_data = src.read(1).astype(np.float32)
                aspect_nodata = src.nodata

        log.info("  Reprojecting %d buildings to %s ...", len(buildings), raster_crs)
        buildings_utm = buildings.to_crs(raster_crs)

        slope_samples = []
        aspect_samples = []
        for geom in buildings_utm.geometry:
            try:
                cx, cy = geom.centroid.x, geom.centroid.y
                col, row = ~transform * (cx, cy)
                col, row = int(col), int(row)
                rows, cols = slope_data.shape
                if 0 <= row < rows and 0 <= col < cols:
                    val = slope_data[row, col]
                    if nodata is not None and val == nodata:
                        slope_samples.append(np.nan)
                        aspect_samples.append(np.nan)
                    else:
                        slope_samples.append(float(val))
                        if aspect_data is None:
                            aspect_samples.append(np.nan)
                        else:
                            aspect_val = aspect_data[row, col]
                            if aspect_nodata is not None and aspect_val == aspect_nodata:
                                aspect_samples.append(np.nan)
                            else:
                                aspect_samples.append(float(aspect_val))
                else:
                    slope_samples.append(np.nan)
                    aspect_samples.append(np.nan)
            except Exception:
                slope_samples.append(np.nan)
                aspect_samples.append(np.nan)

        slope_arr = np.array(slope_samples, dtype=np.float32)
        aspect_arr = np.array(aspect_samples, dtype=np.float32)
        valid_mask = ~np.isnan(slope_arr)
        at_risk_mask = valid_mask & (slope_arr >= self.threshold)
        aspect_valid_mask = (
            ~np.isnan(aspect_arr) & (aspect_arr >= 180.0) & (aspect_arr <= 360.0)
        )
        aspect_filtered_mask = at_risk_mask & aspect_valid_mask

        total = int(valid_mask.sum())
        at_risk = int(at_risk_mask.sum())
        pct = (at_risk / total * 100) if total > 0 else 0.0
        aspect_filtered = int(aspect_filtered_mask.sum()) if aspect_data is not None else 0
        pct_aspect = (aspect_filtered / total * 100) if total > 0 else 0.0

        log.info(
            "  Slope threshold: %.0f deg | Valid buildings: %d | At-risk: %d (%.1f%%)",
            self.threshold,
            total,
            at_risk,
            pct,
        )
        if aspect_data is not None:
            log.info(
                "  Aspect filter: 180-360 deg | At-risk subset: %d (%.1f%%)",
                aspect_filtered,
                pct_aspect,
            )

        dem_tag = Path(slope_raster).stem.replace("_slope", "")
        out_path = self.output_dir / f"at_risk_buildings_{dem_tag}.gpkg"
        if out_path.exists():
            out_path.unlink()
        at_risk_gdf = buildings_utm[at_risk_mask].copy()
        at_risk_gdf["slope_deg"] = slope_arr[at_risk_mask]
        if aspect_data is not None:
            at_risk_gdf["aspect_deg"] = aspect_arr[at_risk_mask]
            at_risk_gdf["aspect_filtered"] = aspect_filtered_mask[at_risk_mask]
        at_risk_gdf.to_file(out_path, driver="GPKG")
        log.info("  At-risk GeoPackage: %s", out_path.name)

        return {
            "total": total,
            "at_risk": at_risk,
            "pct_at_risk": round(pct, 2),
            "aspect_filtered_at_risk": aspect_filtered,
            "pct_aspect_filtered": round(pct_aspect, 2),
            "slope_samples": slope_arr,
            "aspect_samples": aspect_arr,
            "at_risk_path": str(out_path),
            "osm_buildings_extracted_at_utc": self.osm_buildings_extracted_at_utc,
        }

    def analyze_infrastructure_exposure(
        self,
        slope_raster: Path,
        aspect_raster: Path | None = None,
        sample_spacing_m: float = 30.0,
    ) -> dict[str, Any]:
        """
        Estimate exposed road length and civil utility counts.

        Line features are divided into approximately 30 m intervals and sampled
        at interval midpoints. Points and polygons are sampled at their point
        location or centroid. This is a transparent exposure approximation, not
        a replacement for a detailed engineering survey.
        """
        # pyrefly: ignore [missing-import]
        import rasterio

        infrastructure = self.fetch_infrastructure()
        if self.osm_infrastructure_extracted_at_utc is None:
            self.osm_infrastructure_extracted_at_utc = self._utc_now()
        summary = self._empty_infrastructure_summary()
        if infrastructure.empty:
            summary["at_risk_path"] = ""
            summary["osm_infrastructure_extracted_at_utc"] = (
                self.osm_infrastructure_extracted_at_utc
            )
            return summary

        with rasterio.open(slope_raster) as src:
            raster_crs = src.crs
            if raster_crs is None or raster_crs.is_geographic:
                raise ValueError(
                    "Road and infrastructure lengths must be computed in a projected CRS; "
                    f"got {raster_crs}."
                )
            transform = src.transform
            slope_data = src.read(1).astype(np.float32)
            nodata = src.nodata

        aspect_data = None
        aspect_nodata = None
        if aspect_raster is not None:
            with rasterio.open(aspect_raster) as src:
                if src.crs != raster_crs:
                    raise ValueError(
                        f"Aspect raster CRS {src.crs} does not match slope raster CRS {raster_crs}."
                    )
                if src.transform != transform:
                    raise ValueError("Aspect raster transform does not match slope raster transform.")
                aspect_data = src.read(1).astype(np.float32)
                aspect_nodata = src.nodata

        infra_utm = infrastructure.to_crs(raster_crs).copy()
        risk_flags: list[bool] = []
        aspect_flags: list[bool] = []
        exposed_lengths: list[float] = []
        aspect_lengths: list[float] = []

        for _, row in infra_utm.iterrows():
            category = row.get("exposure_category", "other")
            geom = row.geometry
            if category == "road":
                total_length, exposed_length, aspect_length = self._line_exposure_lengths(
                    geom,
                    slope_data,
                    aspect_data,
                    transform,
                    nodata,
                    aspect_nodata,
                    sample_spacing_m,
                )
                summary["roads_total_m"] += total_length
                summary["roads_at_risk_m"] += exposed_length
                summary["roads_aspect_filtered_m"] += aspect_length
                is_risk = exposed_length > 0.0
                is_aspect = aspect_length > 0.0
            else:
                summary[f"{category}_total"] = summary.get(f"{category}_total", 0) + 1
                slope_value, aspect_value = self._sample_geometry(
                    geom, slope_data, aspect_data, transform, nodata, aspect_nodata
                )
                is_risk = bool(np.isfinite(slope_value) and slope_value >= self.threshold)
                is_aspect = bool(
                    is_risk
                    and np.isfinite(aspect_value)
                    and 180.0 <= aspect_value <= 360.0
                )
                if is_risk:
                    summary[f"{category}_at_risk"] = summary.get(f"{category}_at_risk", 0) + 1
                if is_aspect:
                    summary[f"{category}_aspect_filtered"] = summary.get(
                        f"{category}_aspect_filtered", 0
                    ) + 1
                total_length = exposed_length = aspect_length = 0.0

            risk_flags.append(is_risk)
            aspect_flags.append(is_aspect)
            exposed_lengths.append(exposed_length)
            aspect_lengths.append(aspect_length)

        summary["roads_total_m"] = round(summary["roads_total_m"], 1)
        summary["roads_at_risk_m"] = round(summary["roads_at_risk_m"], 1)
        summary["roads_aspect_filtered_m"] = round(summary["roads_aspect_filtered_m"], 1)
        summary["osm_infrastructure_extracted_at_utc"] = (
            self.osm_infrastructure_extracted_at_utc
        )

        dem_tag = Path(slope_raster).stem.replace("_slope", "")
        out_path = self.output_dir / f"at_risk_infrastructure_{dem_tag}.gpkg"
        if out_path.exists():
            out_path.unlink()

        infra_utm["at_risk"] = risk_flags
        infra_utm["aspect_filtered"] = aspect_flags
        infra_utm["exposed_length_m"] = exposed_lengths
        infra_utm["aspect_exposed_length_m"] = aspect_lengths
        at_risk = infra_utm[infra_utm["at_risk"]].copy()
        if not at_risk.empty:
            at_risk.to_file(out_path, driver="GPKG")
            summary["at_risk_path"] = str(out_path)
        else:
            summary["at_risk_path"] = ""

        log.info(
            "  Road exposure: %.1f m at-risk | %.1f m slope+aspect",
            summary["roads_at_risk_m"],
            summary["roads_aspect_filtered_m"],
        )
        return summary

    @staticmethod
    def _categorize_infrastructure(row: Any) -> str:
        if VectorAnalyzer._has_osm_value(row.get("highway")):
            return "road"
        if VectorAnalyzer._has_osm_value(row.get("power")):
            return "power"
        if VectorAnalyzer._has_osm_value(row.get("man_made")):
            return "water_utility"
        if VectorAnalyzer._has_osm_value(row.get("amenity")):
            return "public_facility"
        return "other"

    @staticmethod
    def _has_osm_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        if isinstance(value, str) and not value.strip():
            return False
        return True

    @staticmethod
    def _empty_infrastructure_summary() -> dict[str, Any]:
        return {
            "roads_total_m": 0.0,
            "roads_at_risk_m": 0.0,
            "roads_aspect_filtered_m": 0.0,
            "power_total": 0,
            "power_at_risk": 0,
            "power_aspect_filtered": 0,
            "water_utility_total": 0,
            "water_utility_at_risk": 0,
            "water_utility_aspect_filtered": 0,
            "public_facility_total": 0,
            "public_facility_at_risk": 0,
            "public_facility_aspect_filtered": 0,
            "other_total": 0,
            "other_at_risk": 0,
            "other_aspect_filtered": 0,
            "at_risk_path": "",
            "osm_infrastructure_extracted_at_utc": "",
        }

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _line_exposure_lengths(
        self,
        geom: object,
        slope_data: np.ndarray,
        aspect_data: np.ndarray | None,
        transform: object,
        nodata: float | int | None,
        aspect_nodata: float | int | None,
        sample_spacing_m: float,
    ) -> tuple[float, float, float]:
        total_length = float(getattr(geom, "length", 0.0) or 0.0)
        exposed_length = 0.0
        aspect_length = 0.0
        for line in self._iter_lines(geom):
            length = float(line.length)
            if length <= 0:
                continue
            distance = 0.0
            while distance < length:
                next_distance = min(distance + sample_spacing_m, length)
                interval_length = next_distance - distance
                midpoint = line.interpolate(distance + interval_length / 2.0)
                slope_value, aspect_value = self._sample_point(
                    midpoint.x,
                    midpoint.y,
                    slope_data,
                    aspect_data,
                    transform,
                    nodata,
                    aspect_nodata,
                )
                if np.isfinite(slope_value) and slope_value >= self.threshold:
                    exposed_length += interval_length
                    if (
                        np.isfinite(aspect_value)
                        and 180.0 <= aspect_value <= 360.0
                    ):
                        aspect_length += interval_length
                distance = next_distance
        return total_length, exposed_length, aspect_length

    @staticmethod
    def _iter_lines(geom: object) -> list[object]:
        geom_type = getattr(geom, "geom_type", "")
        if geom_type == "LineString":
            return [geom]
        if geom_type == "MultiLineString":
            return list(geom.geoms)
        return []

    def _sample_geometry(
        self,
        geom: object,
        slope_data: np.ndarray,
        aspect_data: np.ndarray | None,
        transform: object,
        nodata: float | int | None,
        aspect_nodata: float | int | None,
    ) -> tuple[float, float]:
        geom_type = getattr(geom, "geom_type", "")
        if geom_type in {"Polygon", "MultiPolygon"}:
            point = geom.centroid
        elif geom_type == "MultiPoint":
            point = list(geom.geoms)[0] if len(geom.geoms) else geom.centroid
        else:
            point = geom
        return self._sample_point(
            point.x,
            point.y,
            slope_data,
            aspect_data,
            transform,
            nodata,
            aspect_nodata,
        )

    @staticmethod
    def _sample_point(
        x: float,
        y: float,
        slope_data: np.ndarray,
        aspect_data: np.ndarray | None,
        transform: object,
        nodata: float | int | None,
        aspect_nodata: float | int | None,
    ) -> tuple[float, float]:
        col, row = ~transform * (x, y)
        col, row = int(col), int(row)
        rows, cols = slope_data.shape
        if not (0 <= row < rows and 0 <= col < cols):
            return float("nan"), float("nan")
        slope_value = slope_data[row, col]
        if nodata is not None and slope_value == nodata:
            return float("nan"), float("nan")
        aspect_value = float("nan")
        if aspect_data is not None:
            raw_aspect = aspect_data[row, col]
            if aspect_nodata is None or raw_aspect != aspect_nodata:
                aspect_value = float(raw_aspect)
        return float(slope_value), aspect_value
