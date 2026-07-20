import csv
import inspect
import re
import zipfile
from pathlib import Path

import numpy as np

import main
from core.reporter import Reporter


def test_combined_public_figure_has_osm_attribution():
    source = inspect.getsource(Reporter._write_sensitivity_figure)

    assert "© OpenStreetMap contributors" in source


def test_reporter_generates_csv_and_png(tmp_path):
    dem_catalogue = {
        "SRTM_30m": {"gsd_m": 30, "sensor_type": "Radar InSAR"},
        "NASADEM_30m": {"gsd_m": 30, "sensor_type": "SRTM Reprocessed DEM"},
        "ALOS_AW3D30_30m": {"gsd_m": 30, "sensor_type": "Optical Stereo DSM"},
    }
    impact_results = {
        "SRTM_30m": {"total": 10, "at_risk": 2, "pct_at_risk": 20.0},
        "NASADEM_30m": {
            "total": 10,
            "at_risk": 4,
            "pct_at_risk": 40.0,
            "aspect_filtered_at_risk": 3,
            "pct_aspect_filtered": 30.0,
        },
        "ALOS_AW3D30_30m": {
            "total": 10,
            "at_risk": 5,
            "pct_at_risk": 50.0,
            "aspect_filtered_at_risk": 4,
            "pct_aspect_filtered": 40.0,
        },
    }
    topo_layers = {}
    for idx, dem_name in enumerate(dem_catalogue):
        slope = np.array(
            [[10.0 + idx, 15.0 + idx, np.nan], [20.0 + idx, 25.0 + idx, 30.0 + idx]],
            dtype=np.float32,
        )
        profile_curvature = np.array(
            [[0.001 * (idx + 1), 0.002 * (idx + 1)], [np.nan, 0.003 * (idx + 1)]],
            dtype=np.float32,
        )
        plan_curvature = profile_curvature / 2.0
        topo_layers[dem_name] = {
            "_arrays": {
                "slope": slope,
                "profile_curvature": profile_curvature,
                "plan_curvature": plan_curvature,
            }
        }

    reporter = Reporter(
        dem_catalogue=dem_catalogue,
        impact_results=impact_results,
        topo_layers=topo_layers,
        output_dir=tmp_path / "report",
        reference={
            "slope_mean_deg": 17.0,
            "slope_std_deg": 10.0,
            "prof_curv_std": 0.003,
        },
    )

    outputs = reporter.generate()

    assert outputs["csv"].exists()
    assert outputs["csv"].stat().st_size > 0
    assert outputs["figure"].exists()
    assert outputs["figure"].stat().st_size > 0

    with outputs["csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["DEM"] for row in rows] == [
        "SRTM_30m",
        "NASADEM_30m",
        "ALOS_AW3D30_30m",
    ]
    assert rows[2]["Buildings_AtRisk"] == "5"
    assert rows[2]["Aspect_Filtered_AtRisk"] == "4"
    assert rows[2]["Pct_Aspect_Filtered"] == "40.000"
    assert "Reference_12.5m" not in {row["DEM"] for row in rows}


def test_reporter_uses_dem_keyed_rows_not_position(tmp_path):
    dem_catalogue = {
        "SRTM_30m": {"gsd_m": 30, "sensor_type": "Radar InSAR"},
        "NASADEM_30m": {"gsd_m": 30, "sensor_type": "SRTM Reprocessed DEM"},
        "ALOS_AW3D30_30m": {"gsd_m": 30, "sensor_type": "Optical Stereo DSM"},
    }
    topo_layers = {
        dem_name: {
            "_arrays": {
                "slope": np.array([[10.0, 20.0]], dtype=np.float32),
                "profile_curvature": np.array([[0.001, 0.002]], dtype=np.float32),
                "plan_curvature": np.array([[0.001, 0.002]], dtype=np.float32),
            }
        }
        for dem_name in dem_catalogue
    }
    reporter = Reporter(
        dem_catalogue=dem_catalogue,
        impact_results={},
        topo_layers=topo_layers,
        output_dir=tmp_path,
        reference={"slope_mean_deg": 17.0, "slope_std_deg": 10.0, "prof_curv_std": 0.003},
    )
    rows = [
        {"DEM": "ALOS_AW3D30_30m", "ProfCurv_Std": 3.0, "Buildings_AtRisk": 30, "Pct_AtRisk": 30.0},
        {"DEM": "SRTM_30m", "ProfCurv_Std": 1.0, "Buildings_AtRisk": 10, "Pct_AtRisk": 10.0},
        {"DEM": "NASADEM_30m", "ProfCurv_Std": 2.0, "Buildings_AtRisk": 20, "Pct_AtRisk": 20.0},
    ]

    row_by_dem = reporter._row_by_dem(rows)

    assert row_by_dem["NASADEM_30m"]["Buildings_AtRisk"] == 20


def test_reporter_generates_latex_source_package(tmp_path, monkeypatch):
    dem_catalogue = {
        "SRTM_30m": {"gsd_m": 30, "sensor_type": "Radar InSAR"},
    }
    topo_layers = {
        "SRTM_30m": {
            "_arrays": {
                "slope": np.array([[10.0, 20.0]], dtype=np.float32),
                "profile_curvature": np.array([[0.001, 0.002]], dtype=np.float32),
                "plan_curvature": np.array([[0.001, 0.002]], dtype=np.float32),
            }
        }
    }
    reporter = Reporter(
        dem_catalogue=dem_catalogue,
        impact_results={"SRTM_30m": {"total": 1, "at_risk": 1}},
        topo_layers=topo_layers,
        output_dir=tmp_path / "report",
        reference={"slope_mean_deg": 17.0, "slope_std_deg": 10.0, "prof_curv_std": 0.003},
    )
    outputs = reporter.generate()
    runtime_csv = tmp_path / "runtime_summary.csv"
    runtime_csv.write_text("phase,seconds\nPhase 1,1.0\n", encoding="utf-8")
    inventory_csv = tmp_path / "output_inventory.csv"
    inventory_csv.write_text("relative_path,size_bytes\n", encoding="utf-8")
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "outputs")
    manifest_json = main._write_run_manifest(
        runtime_rows=[],
        report_outputs={"csv": outputs["csv"]},
        visual_outputs={},
    )
    map_index = tmp_path / "map_index.md"
    map_index.write_text("# maps\n", encoding="utf-8")
    infrastructure_csv = tmp_path / "infrastructure_exposure_summary.csv"
    infrastructure_csv.write_text("DEM,Roads_AtRisk_m\nSRTM_30m,0\n", encoding="utf-8")
    figure_names = [
        "slope_map_comparison.png",
        "SRTM_30m_at_risk_overlay.png",
        "exposure_cluster_overview.png",
        "exposure_cluster_zoom_panel.png",
        "dem_exposure_agreement_matrix.png",
        "infrastructure_exposure_summary.png",
        "SRTM_30m_elevation_hillshade.png",
        "SRTM_30m_slope.png",
        "SRTM_30m_aspect.png",
        "SRTM_30m_profile_curvature.png",
        "SRTM_30m_plan_curvature.png",
        "SRTM_30m_slope_histogram.png",
        "profile_curvature_lineage_sensitivity.png",
        "infrastructure_risk_by_dem_product.png",
    ]
    fake_figures = []
    for name in figure_names:
        path = tmp_path / name
        path.write_bytes(b"fake-png")
        fake_figures.append(path)

    package = reporter.generate_latex_package(
        artifacts={
            "summary_csv": outputs["csv"],
            "sensitivity_figure": outputs["figure"],
            "runtime_csv": runtime_csv,
            "inventory_csv": inventory_csv,
            "manifest_json": manifest_json,
            "map_index": map_index,
            "report_figures": fake_figures[:8],
            "verification_figures": fake_figures[8:],
            "additional_tables": [infrastructure_csv],
        },
        runtime_rows=[{"phase": "Phase 1", "seconds": 1.0}],
    )

    assert package["latex_zip"].exists()
    assert package["latex_zip"].stat().st_size > 0
    main._verify_manifest_in_zip(package["latex_zip"])
    with zipfile.ZipFile(package["latex_zip"]) as archive:
        names = set(archive.namelist())
        packaged_manifest = archive.read("run_manifest.json").decode("utf-8")
    assert "main.tex" in names
    assert "references.bib" in names
    assert "tables/summary_table.csv" in names
    assert "tables/infrastructure_exposure_summary.csv" in names
    for private_marker in ("C:\\Users\\", "/home/", "/Users/", "OneDrive", "Desktop"):
        assert private_marker.lower() not in packaged_manifest.lower()

    forbidden = "".join(["12665", "-011-", "1196-4"])
    with zipfile.ZipFile(package["latex_zip"]) as archive:
        text_names = [
            name
            for name in archive.namelist()
            if Path(name).suffix.lower() in {".tex", ".bib", ".md", ".csv", ".json"}
        ]
        for name in text_names:
            content = archive.read(name).decode("utf-8")
            assert forbidden not in content

    main_tex = (package["latex_dir"] / "main.tex").read_text(encoding="utf-8")
    references = (package["latex_dir"] / "references.bib").read_text(encoding="utf-8")
    assert "\\bibliography{references}" in main_tex
    assert "\\nocite{*}" in main_tex
    assert "Reference_12.5m" not in main_tex
    assert re.search(r"\bpresentation\b", main_tex.lower()) is None
    assert "beamer" not in main_tex.lower()
    assert re.search(r"\bslide\b", main_tex.lower()) is None
    assert "OSM completeness and sparsity bias in rural Turkish regions" in main_tex
    assert "MIT" in main_tex
    assert "Open Database License" in main_tex
    assert "OpenStreetMap contributors" in main_tex
    assert "gorelick2017gee" in references
    assert "sentinel2gee" in references
    assert "zevenbergen1987" in references
    for name in figure_names:
        assert f"figures/{name}" in main_tex
    assert "This hillshade-elevation figure shows" in main_tex
    assert "This overlay shows where" in main_tex
    assert "The agreement matrix converts" in main_tex
