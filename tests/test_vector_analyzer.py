from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

from core.vector_analyzer import VectorAnalyzer


def test_osmnx_bbox_uses_left_bottom_right_top_order(tmp_path):
    analyzer = VectorAnalyzer(
        bbox={"west": 31.913731, "south": 41.169295, "east": 32.034410, "north": 41.258297},
        slope_threshold_deg=15.0,
        output_dir=Path(tmp_path),
    )

    assert analyzer._osmnx_bbox() == (31.913731, 41.169295, 32.034410, 41.258297)


def test_count_at_risk_buildings_with_aspect_filter(tmp_path):
    transform = from_origin(0, 20, 10, 10)
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32636",
        "transform": transform,
        "nodata": np.nan,
    }
    slope_path = tmp_path / "SRTM_30m_slope.tif"
    aspect_path = tmp_path / "SRTM_30m_aspect.tif"

    with rasterio.open(slope_path, "w", **profile) as dst:
        dst.write(np.array([[20.0, 20.0], [10.0, 20.0]], dtype=np.float32), 1)
    with rasterio.open(aspect_path, "w", **profile) as dst:
        dst.write(np.array([[200.0, 90.0], [250.0, 300.0]], dtype=np.float32), 1)

    buildings = gpd.GeoDataFrame(
        geometry=[
            box(1, 11, 2, 12),    # slope risk, aspect risk
            box(11, 11, 12, 12),  # slope risk, aspect not risk
            box(1, 1, 2, 2),      # no slope risk
            box(11, 1, 12, 2),    # slope risk, aspect risk
        ],
        crs="EPSG:32636",
    )
    analyzer = VectorAnalyzer(
        bbox={"west": 31.88, "south": 41.19, "east": 32.00, "north": 41.27},
        slope_threshold_deg=15.0,
        output_dir=tmp_path,
    )
    analyzer.fetch_buildings = lambda: buildings

    result = analyzer.count_at_risk_buildings(slope_path, aspect_path)

    assert result["total"] == 4
    assert result["at_risk"] == 3
    assert result["aspect_filtered_at_risk"] == 2
    assert result["pct_aspect_filtered"] == 50.0
    assert result["osm_buildings_extracted_at_utc"]
    assert Path(result["at_risk_path"]).exists()


def test_infrastructure_exposure_samples_lines_and_points(tmp_path):
    transform = from_origin(0, 30, 10, 10)
    profile = {
        "driver": "GTiff",
        "height": 3,
        "width": 3,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32636",
        "transform": transform,
        "nodata": np.nan,
    }
    slope_path = tmp_path / "SRTM_30m_slope.tif"
    aspect_path = tmp_path / "SRTM_30m_aspect.tif"

    with rasterio.open(slope_path, "w", **profile) as dst:
        dst.write(
            np.array(
                [
                    [20.0, 20.0, 10.0],
                    [20.0, 10.0, 10.0],
                    [10.0, 10.0, 20.0],
                ],
                dtype=np.float32,
            ),
            1,
        )
    with rasterio.open(aspect_path, "w", **profile) as dst:
        dst.write(
            np.array(
                [
                    [220.0, 90.0, 90.0],
                    [260.0, 90.0, 90.0],
                    [90.0, 90.0, 300.0],
                ],
                dtype=np.float32,
            ),
            1,
        )

    infrastructure = gpd.GeoDataFrame(
        {
            "highway": ["residential", np.nan, np.nan],
            "power": [np.nan, "tower", np.nan],
            "man_made": [np.nan, np.nan, "water_tower"],
            "amenity": [np.nan, np.nan, np.nan],
            "exposure_category": ["road", "power", "water_utility"],
        },
        geometry=[
            LineString([(1, 25), (25, 25)]),
            Point(5, 25),
            Point(25, 5),
        ],
        crs="EPSG:32636",
    )

    analyzer = VectorAnalyzer(
        bbox={"west": 31.88, "south": 41.19, "east": 32.00, "north": 41.27},
        slope_threshold_deg=15.0,
        output_dir=tmp_path,
    )
    analyzer.fetch_infrastructure = lambda: infrastructure

    result = analyzer.analyze_infrastructure_exposure(
        slope_path,
        aspect_path,
        sample_spacing_m=10.0,
    )

    assert result["roads_total_m"] > 0
    assert result["roads_at_risk_m"] > 0
    assert result["roads_aspect_filtered_m"] > 0
    assert result["power_at_risk"] == 1
    assert result["water_utility_at_risk"] == 1
    assert result["osm_infrastructure_extracted_at_utc"]
    assert Path(result["at_risk_path"]).exists()


def test_infrastructure_exposure_refuses_geographic_length_reporting(tmp_path):
    transform = from_origin(31.9, 41.3, 0.001, 0.001)
    profile = {
        "driver": "GTiff",
        "height": 3,
        "width": 3,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": np.nan,
    }
    slope_path = tmp_path / "SRTM_30m_slope.tif"
    with rasterio.open(slope_path, "w", **profile) as dst:
        dst.write(np.full((3, 3), 20.0, dtype=np.float32), 1)

    infrastructure = gpd.GeoDataFrame(
        {"highway": ["residential"], "exposure_category": ["road"]},
        geometry=[LineString([(31.901, 41.299), (31.902, 41.299)])],
        crs="EPSG:4326",
    )
    analyzer = VectorAnalyzer(
        bbox={"west": 31.88, "south": 41.19, "east": 32.00, "north": 41.27},
        slope_threshold_deg=15.0,
        output_dir=tmp_path,
    )
    analyzer.fetch_infrastructure = lambda: infrastructure

    with pytest.raises(ValueError, match="projected CRS"):
        analyzer.analyze_infrastructure_exposure(slope_path)
