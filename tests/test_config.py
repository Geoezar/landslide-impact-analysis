import json
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
    project_root = tmp_path / "repo"
    monkeypatch.setattr(main, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(main, "OUTPUT_DIR", project_root / "outputs")
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


def test_run_manifest_serializes_nested_artifacts_as_repository_relative_paths(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "repo"
    output_root = project_root / "outputs"
    report_dir = output_root / "report"
    map_dir = output_root / "maps" / "report"
    report_dir.mkdir(parents=True)
    map_dir.mkdir(parents=True)
    summary = report_dir / "summary.csv"
    figure = map_dir / "figure.png"

    monkeypatch.setattr(main, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(main, "OUTPUT_DIR", output_root)

    path = main._write_run_manifest(
        runtime_rows=[],
        report_outputs={"csv": summary, "nested": {"figures": [figure]}},
        visual_outputs={"report_figures": [figure]},
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["project_root"] == "."
    assert manifest["output_root"] == "outputs"
    assert manifest["report_outputs"] == {
        "csv": "outputs/report/summary.csv",
        "nested": {"figures": ["outputs/maps/report/figure.png"]},
    }
    assert manifest["visual_outputs"] == {
        "report_figures": ["outputs/maps/report/figure.png"]
    }


def test_manifest_rejects_artifact_outside_repository(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    output_root = project_root / "outputs"
    (output_root / "report").mkdir(parents=True)
    outside = tmp_path / "outside.txt"

    monkeypatch.setattr(main, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(main, "OUTPUT_DIR", output_root)

    with pytest.raises(ValueError, match="outside the repository root"):
        main._write_run_manifest([], {"outside": outside}, {})


@pytest.mark.parametrize(
    "private_path",
    [
        r"C:\Users\person\project",
        "/home/person/project",
        "/Users/person/project",
        r"D:\OneDrive\project",
        r"D:\Desktop\project",
    ],
)
def test_public_manifest_rejects_private_path_markers(private_path):
    with pytest.raises(ValueError, match="private path marker"):
        main._assert_public_manifest_safe(json.dumps({"path": private_path}))
