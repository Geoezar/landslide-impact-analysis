from pathlib import Path

from core.talk_script import TalkScriptBuilder


def test_talk_script_builder_writes_expanded_script_only(tmp_path):
    builder = TalkScriptBuilder(
        output_dir=tmp_path / "presentation",
        summary_rows=[
            {
                "DEM": "SRTM_30m",
                "Buildings_AtRisk": 52,
                "Aspect_Filtered_AtRisk": 51,
            },
            {
                "DEM": "NASADEM_30m",
                "Buildings_AtRisk": 46,
                "Aspect_Filtered_AtRisk": 44,
            },
            {
                "DEM": "ALOS_AW3D30_30m",
                "Buildings_AtRisk": 86,
                "Aspect_Filtered_AtRisk": 78,
            },
        ],
        runtime_rows=[{"phase": "Phase 1", "seconds": 1.0}],
        report_outputs={"figure": tmp_path / "sensitivity_report.png"},
        visual_outputs={"report_figures": [tmp_path / "slope_map_comparison.png"]},
    )

    outputs = builder.generate()
    script_path = outputs["talk_script"]
    text = script_path.read_text(encoding="utf-8")

    assert script_path.exists()
    assert "Target duration: **8-10 minutes**" in text
    assert "Figure cue" in text
    assert "NASADEM classifies 46 buildings" in text
    assert "ALOS_AW3D30_30m" not in text or "AW3D30" in text
    assert not (Path(tmp_path) / "presentation" / "beamer").exists()
    assert not (Path(tmp_path) / "presentation" / "devrek_remote_sensing_presentation.zip").exists()
    assert "Slide" not in text
