from pathlib import Path
import inspect

import numpy as np
import rasterio
from rasterio.transform import from_origin

from core.visualizer import Visualizer


def _write_raster(path: Path, data: np.ndarray) -> None:
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32636",
        "transform": from_origin(400000, 4560000, 30, 30),
        "nodata": np.nan,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def test_visualizer_generates_report_and_verification_pngs(tmp_path):
    dem_name = "SRTM_30m"
    dem_path = tmp_path / f"{dem_name}.tif"
    slope_path = tmp_path / f"{dem_name}_slope.tif"
    aspect_path = tmp_path / f"{dem_name}_aspect.tif"
    prof_path = tmp_path / f"{dem_name}_profile_curvature.tif"
    plan_path = tmp_path / f"{dem_name}_plan_curvature.tif"

    base = np.array([[100.0, 110.0], [120.0, 130.0]], dtype=np.float32)
    _write_raster(dem_path, base)
    _write_raster(slope_path, np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
    _write_raster(aspect_path, np.array([[180.0, 220.0], [270.0, 330.0]], dtype=np.float32))
    _write_raster(prof_path, np.array([[0.001, -0.002], [0.003, -0.004]], dtype=np.float32))
    _write_raster(plan_path, np.array([[0.001, -0.002], [0.003, -0.004]], dtype=np.float32))

    visualizer = Visualizer(
        dem_catalogue={dem_name: {"gsd_m": 30}},
        dem_paths={dem_name: dem_path},
        topo_layers={
            dem_name: {
                "slope": slope_path,
                "aspect": aspect_path,
                "profile_curvature": prof_path,
                "plan_curvature": plan_path,
            }
        },
        impact_results={dem_name: {"at_risk": 1, "aspect_filtered_at_risk": 1}},
        output_dir=tmp_path / "maps",
        report_dir=tmp_path / "report",
    )

    outputs = visualizer.generate()

    assert outputs["index"].exists()
    assert outputs["report_figures"]
    assert outputs["verification_figures"]
    assert outputs["cluster_csv"].exists()
    assert outputs["infrastructure_csv"].exists()
    report_names = {path.name for path in outputs["report_figures"]}
    assert "exposure_cluster_overview.png" in report_names
    assert "exposure_cluster_zoom_panel.png" in report_names
    assert "dem_exposure_agreement_matrix.png" in report_names
    assert "infrastructure_exposure_summary.png" in report_names
    assert "SRTM_30m_slope_histogram.png" in report_names
    assert "profile_curvature_lineage_sensitivity.png" in report_names
    assert "infrastructure_risk_by_dem_product.png" in report_names
    for path in outputs["report_figures"] + outputs["verification_figures"]:
        assert path.exists()
        assert path.stat().st_size > 0


def test_sentinel2_cloud_mask_uses_qa60_and_scl_classes():
    source = inspect.getsource(Visualizer._mask_sentinel2_clouds)

    assert 'select("QA60")' in source
    assert "1 << 10" in source
    assert "1 << 11" in source
    assert 'select("SCL")' in source
    for scl_class in ("scl.neq(3)", "scl.neq(8)", "scl.neq(9)", "scl.neq(10)", "scl.neq(11)"):
        assert scl_class in source


def test_infrastructure_summary_omits_zero_only_utility_series():
    source = inspect.getsource(Visualizer._plot_infrastructure_summary)

    assert "include_utilities = any(value > 0 for value in utilities)" in source
    assert "if include_utilities and utility_x is not None" in source


def test_public_attribution_is_scoped_to_data_used_by_each_figure():
    helper_source = inspect.getsource(Visualizer._add_public_attribution)
    overlay_source = inspect.getsource(Visualizer._plot_at_risk_overlay)
    zoom_source = inspect.getsource(Visualizer._plot_cluster_zoom_panel)
    verification_source = inspect.getsource(Visualizer._plot_elevation_hillshade)
    dem_only_source = inspect.getsource(Visualizer._plot_profile_curvature_chart)

    assert "© OpenStreetMap contributors" in helper_source
    assert "Copernicus Sentinel-2 / ESA" in helper_source
    assert "_add_public_attribution(fig, osm=True)" in overlay_source
    assert "sentinel=rgb_context is not None" in zoom_source
    for method in (
        Visualizer._plot_exposure_cluster_overview,
        Visualizer._plot_dem_exposure_agreement_matrix,
        Visualizer._plot_infrastructure_summary,
        Visualizer._plot_slope_histogram,
        Visualizer._plot_infrastructure_risk_chart,
    ):
        assert "_add_public_attribution(fig, osm=True)" in inspect.getsource(method)
    assert "_add_public_attribution" not in verification_source
    assert "_add_public_attribution" not in dem_only_source
