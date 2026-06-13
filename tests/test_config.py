from pathlib import Path

import pytest

import main


def test_project_uses_local_outputs_and_correct_bbox():
    assert main.OUTPUT_DIR == main.PROJECT_ROOT / "outputs"
    assert main.DEVREK_BBOX_WGS84 == {
        "west": 31.913731,
        "south": 41.169295,
        "east": 32.034410,
        "north": 41.258297,
    }


def test_active_dem_catalogue_is_processing_lineage_set():
    assert list(main.DEM_CATALOGUE) == [
        "SRTM_30m",
        "NASADEM_30m",
        "ALOS_AW3D30_30m",
    ]
    assert "ALOS_12m" not in main.DEM_CATALOGUE
    assert "ASTER_30m" not in main.DEM_CATALOGUE
    assert main.DEM_CATALOGUE["ALOS_AW3D30_30m"]["gsd_m"] == 30


def test_safe_remove_refuses_paths_outside_output_root(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("do not remove", encoding="utf-8")

    with pytest.raises(ValueError):
        main._safe_remove_path(Path(outside), root=root)

    assert outside.exists()


def test_run_manifest_records_osm_timestamps(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "outputs")
    report_dir = main.OUTPUT_DIR / "report"
    report_dir.mkdir(parents=True)

    path = main._write_run_manifest(
        runtime_rows=[],
        report_outputs={},
        visual_outputs={},
        impact_results={
            "SRTM_30m": {
                "osm_buildings_extracted_at_utc": "2026-05-12T10:00:00+00:00",
                "infrastructure": {
                    "osm_infrastructure_extracted_at_utc": "2026-05-12T10:01:00+00:00"
                },
            }
        },
    )

    text = path.read_text(encoding="utf-8")
    assert "osm_buildings_extracted_at_utc" in text
    assert "2026-05-12T10:00:00+00:00" in text
    assert "osm_infrastructure_extracted_at_utc" in text
    assert "2026-05-12T10:01:00+00:00" in text
