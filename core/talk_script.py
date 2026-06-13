"""
core/talk_script.py
===================
Writes a figure-guided speaking script from the latest pipeline artifacts.

This module intentionally does not create a TeX deck or any additional visual
package. The user will prepare the visual delivery manually; the pipeline only
provides a long, structured narration tied to generated figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TalkScriptBuilder:
    """Generate an 8-10 minute figure-guided narration file."""

    def __init__(
        self,
        output_dir: Path,
        summary_rows: list[dict[str, Any]],
        runtime_rows: list[dict[str, Any]],
        report_outputs: dict[str, Path],
        visual_outputs: dict[str, Any],
    ) -> None:
        self.output_dir = Path(output_dir)
        self.summary_rows = summary_rows
        self.runtime_rows = runtime_rows
        self.report_outputs = report_outputs
        self.visual_outputs = visual_outputs
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self) -> dict[str, Path]:
        script_path = self.output_dir / "presentation_script.md"
        script_path.write_text(self._script_text(), encoding="utf-8")
        return {"talk_script": script_path}

    def _row(self, dem_name: str) -> dict[str, Any]:
        for row in self.summary_rows:
            if row.get("DEM") == dem_name:
                return row
        return {}

    def _figure(self, filename: str) -> str:
        for path in [self.report_outputs.get("figure")] + self.visual_outputs.get(
            "report_figures", []
        ):
            if path and Path(path).name == filename:
                return Path(path).as_posix()
        return filename

    def _counts_sentence(self) -> str:
        nasa = self._row("NASADEM_30m").get("Buildings_AtRisk", 0)
        srtm = self._row("SRTM_30m").get("Buildings_AtRisk", 0)
        aw3d = self._row("ALOS_AW3D30_30m").get("Buildings_AtRisk", 0)
        nasa_aspect = self._row("NASADEM_30m").get("Aspect_Filtered_AtRisk", 0)
        srtm_aspect = self._row("SRTM_30m").get("Aspect_Filtered_AtRisk", 0)
        aw3d_aspect = self._row("ALOS_AW3D30_30m").get("Aspect_Filtered_AtRisk", 0)
        return (
            f"NASADEM classifies {nasa} buildings as exposed, SRTM classifies "
            f"{srtm}, and AW3D30 classifies {aw3d}. After adding the "
            f"aspect-filtered subset, those values become {nasa_aspect}, "
            f"{srtm_aspect}, and {aw3d_aspect}."
        )

    def _runtime_sentence(self) -> str:
        if not self.runtime_rows:
            return "The runtime table is available after a completed pipeline run."
        total = sum(float(row.get("seconds", 0.0)) for row in self.runtime_rows)
        return f"The measured end-to-end runtime in the latest run is approximately {total:.1f} seconds."

    def _script_text(self) -> str:
        counts = self._counts_sentence()
        runtime = self._runtime_sentence()
        return f"""# Figure-Guided Talk Script

Target duration: **8-10 minutes**. The talk should remain centered on Remote Sensing product choice, reproducible Python automation, and DEM-derived infrastructure exposure. Do not present the work as a full landslide susceptibility model.

## 1. Opening: Research Question and Scope

**Figure cue:** Use `slope_map_comparison.png` if a visual is needed early: `{self._figure("slope_map_comparison.png")}`.

**Say:** This project is a Remote Sensing and spatial data-engineering study built for the Devrek region of Zonguldak. The central question is not whether I can draw a visually attractive map. The central question is whether the chosen elevation product changes the terrain derivatives and, through those derivatives, the number of OpenStreetMap buildings and infrastructure elements classified as exposed. The active scientific framing is DEM processing-lineage sensitivity. In other words, for the same geographic area, the same slope threshold, the same aspect subset, and the same Python code, I compare SRTM, NASADEM, and AW3D30 to see how much the elevation source itself changes the result.

**Key sentence to emphasize:** The project does not compete with the field-based Devrek susceptibility study; it automates and tests one important exposure component inspired by that literature.

**Transition:** Before discussing the pipeline, explain why Devrek is a defensible study area and why the reference study matters.

**Expected question:** Are you claiming to produce a complete landslide-risk map?

**Answer:** No. This is an automated exposure and DEM-sensitivity workflow. It does not include lithology, drainage density, rainfall, fault distance, soil mechanics, historical landslide inventory overlay, or field validation.

## 2. Why Devrek and Why the Reference Study Matters

**Figure cue:** Use the study-area maps or the cluster overview: `{self._figure("exposure_cluster_overview.png")}`.

**Say:** Devrek is useful because it is not an arbitrary test rectangle. The reference study by Yilmaz, Topal, and Suzen documented active landslides in the region and used a broader susceptibility methodology that included field mapping, geology, drainage, roads, faults, terrain parameters, and a local 12.5 m DEM. That makes the region academically grounded. My project takes two lessons from that work. First, terrain parameters such as slope, aspect, and curvature matter. Second, settlement and infrastructure exposure matter because unstable slopes become a planning problem when people build on or near them.

**Clarify the boundary:** The reference study is stronger geologically because it includes field evidence and multiple causative factors. My work is stronger as a reproducible automation pipeline. These are different contributions.

**Transition:** Now move from literature context to Remote Sensing product correction.

**Expected question:** Why not reuse the same 12.5 m reference DEM?

**Answer:** The reference DEM is not a public cloud DEM that can be automatically pulled from Google Earth Engine in this no-external-download workflow. Reusing it would change the project from a cloud pipeline into a local-data reproduction task.

## 3. Remote Sensing Product Corrections

**Figure cue:** Use the summary table in the report package and the sensitivity dashboard: `{self._figure("sensitivity_report.png")}`.

**Say:** A major scientific correction in the project was refusing to use a wrong data set. The initially considered ASTER GED asset in Google Earth Engine is not a DEM suitable for slope and curvature derivation. It is an emissivity and land-surface-temperature related product, so treating it as an optical DEM would produce invalid terrain statistics. A second correction concerns AW3D30. The active GEE asset is an ALOS PRISM optical-stereo DSM at about 30 m, not the initially assumed high-resolution radar DEM. Because of that, the final comparison is not advertised as a native 12.5 m radar experiment. It is stated honestly as SRTM baseline, NASADEM reprocessed SRTM-family control, and AW3D30 optical-stereo DSM lineage.

**Why this matters:** In Remote Sensing, metadata is not a footnote. Sensor type, processing lineage, vertical surface representation, and nominal scale determine what the terrain derivatives mean.

**Transition:** After the product correction, explain what each product contributes.

**Expected question:** Did removing the invalid product weaken the project?

**Answer:** No. It strengthened the project. A scientifically wrong DEM would make the comparison meaningless. The final product set is more defensible because every active input is an official, cloud-accessible elevation product.

## 4. DEM Lineages and Resolution Meaning

**Figure cue:** Use the standalone slope histograms:
- `{self._figure("SRTM_30m_slope_histogram.png")}`
- `{self._figure("NASADEM_30m_slope_histogram.png")}`
- `{self._figure("ALOS_AW3D30_30m_slope_histogram.png")}`

**Say:** SRTM is the baseline global 30 m C-band InSAR DSM. NASADEM is a reprocessed SRTM-family product and acts as a control for processing improvement rather than an independent sensor family. AW3D30 is an ALOS PRISM optical-stereo DSM and represents a different processing lineage. Although all active products are treated at about 30 m in this workflow, they do not produce identical terrain fields. Ground sampling distance tells us the nominal grid scale, but the actual derivative behavior also depends on smoothing, gap filling, stereo reconstruction, DSM surface effects, and how local terrain breaks are preserved or generalized.

**Interpretation:** The slope histograms show how much of each DEM surface falls above the 15 degree operational threshold. The right tail of the distribution is crucial because buildings sampled on that tail become exposed in the automated count.

**Transition:** The pipeline computes these derivatives and connects them to OSM infrastructure.

**Expected question:** If the products are all about 30 m, why compare them?

**Answer:** Because equal nominal grid spacing does not mean equal topographic behavior. The project tests whether processing lineage alone changes slope, curvature, and exposure counts.

## 5. Automated No-GIS Python Pipeline

**Figure cue:** Use `sensitivity_report.png` for the full analytical chain and `map_index.md` for the generated file inventory.

**Say:** The workflow has four phases. In Phase 1, Google Earth Engine and geemap export SRTM, NASADEM, and AW3D30 over the corrected Devrek bounding box. The rasters are reprojected to UTM Zone 36N so slope, curvature, and road-length calculations happen in a metric coordinate system. In Phase 2, rasterio and NumPy compute slope, aspect, profile curvature, and plan curvature. In Phase 3, OSMnx and GeoPandas fetch OpenStreetMap buildings, roads, and civil infrastructure. Building exposure is counted by centroid sampling. Road exposure is measured by 30 m interval midpoint sampling only after projection to EPSG:32636. In Phase 4, Matplotlib and the reporter produce CSV tables, PNG figures, GeoPackage outputs, a LaTeX source package, audit files, and this narration file.

**Engineering point:** {runtime} The important claim is reproducibility: the pipeline overwrites owned outputs and can be rerun without manual GIS screenshots.

**Transition:** After the pipeline explanation, show what the topographic derivative maps mean.

**Expected question:** Why use Python instead of a desktop GIS?

**Answer:** Desktop GIS is valuable for exploration, but Python makes the exact workflow repeatable, testable, and easier to audit. That matters when the same analysis must be rerun after OSM edits or DEM-product changes.

## 6. Terrain Derivative Figures

**Figure cue:** Use the verification maps for one DEM first, then compare across DEMs:
- elevation/hillshade maps,
- slope maps,
- aspect maps,
- profile curvature maps,
- plan curvature maps.

**Say:** The verification maps are not decorative. Elevation and hillshade show whether the terrain surface has been downloaded and projected correctly. Slope is the primary operational risk driver because the project classifies buildings on cells with slope greater than or equal to 15 degrees. Aspect is used as a literature-aligned subset from 180 to 360 degrees, corresponding broadly to south-west, west, and north-west facing slopes. Profile curvature and plan curvature describe terrain shape. Profile curvature relates to acceleration or deceleration along the slope direction, while plan curvature describes lateral convergence and divergence. These derivatives allow me to explain why one DEM flags more buildings than another instead of only reporting counts.

**Figure cue:** Use `profile_curvature_lineage_sensitivity.png`: `{self._figure("profile_curvature_lineage_sensitivity.png")}`.

**Interpretation:** Higher profile-curvature variance indicates that a product preserves more local surface roughness, although DSM noise from trees and buildings can also contribute. This is why curvature must be interpreted together with product lineage and limitations.

**Transition:** Now connect the terrain fields to exposed buildings.

**Expected question:** Are bright or rough areas automatically landslides?

**Answer:** No. These are terrain factors and exposure-screening inputs. Confirmed landslides would require inventory overlay and field validation.

## 7. Infrastructure Exposure Maps

**Figure cue:** Use the three at-risk overlay maps:
- `{self._figure("SRTM_30m_at_risk_overlay.png")}`
- `{self._figure("NASADEM_30m_at_risk_overlay.png")}`
- `{self._figure("ALOS_AW3D30_30m_at_risk_overlay.png")}`

**Say:** The at-risk overlay maps answer where the automated rule intersects OSM buildings. The background uses the slope palette, and the exposed buildings are highlighted directly. I removed the cyan threshold borders because they were visually dominant and could distract from the actual exposed structures. The maps should be read as screening evidence. They show which buildings are counted by each DEM product under the same threshold, not confirmed damage or field-verified hazard.

**Figure cue:** Use `dem_exposure_agreement_matrix.png`: `{self._figure("dem_exposure_agreement_matrix.png")}`.

**Interpretation:** The agreement matrix is useful because total counts alone can hide spatial disagreement. A DEM may agree with the others in one cluster but diverge strongly in another. This tells the audience that DEM lineage changes both the number and the spatial distribution of exposed infrastructure.

**Transition:** Full-area maps are useful, but small buildings can disappear at report scale. Move to cluster zooms.

**Expected question:** Why use building centroids instead of full footprint intersection?

**Answer:** Centroid sampling is transparent and robust for a first automated exposure metric. It avoids counting the same building multiple times across noisy raster boundaries, but it is still an approximation.

## 8. Cluster Overview and Sentinel-2 Context

**Figure cue:** Use `exposure_cluster_overview.png` and `exposure_cluster_zoom_panel.png`:
- `{self._figure("exposure_cluster_overview.png")}`
- `{self._figure("exposure_cluster_zoom_panel.png")}`

**Say:** The cluster overview solves a scale problem. Across the full Devrek bounding box, individual building markers can be too small to interpret. The cluster map groups exposed buildings into numbered zones so the viewer can see where the exposure concentrates. The zoom panel then enlarges the highest clusters. When Google Earth Engine access succeeds, the zoom context uses Sentinel-2 SR Harmonized true-color imagery. The code masks QA60 opaque cloud and cirrus bits and also uses the SCL band to remove cloud shadows and cloud classes before building a summer median composite. When that image retrieval fails, the system falls back to hillshade.

**Visual design point:** The zoom markers use high-contrast colors with haloed edges because vegetation and tile roofs can hide low-contrast symbols.

**Transition:** After showing where the exposed structures are, give the quantitative result.

**Expected question:** Are these Google Maps tiles?

**Answer:** No. The true-color context comes from Sentinel-2 through Google Earth Engine, and the fallback is generated from DEM hillshade.

## 9. Quantitative Result and Main Finding

**Figure cue:** Use `infrastructure_risk_by_dem_product.png` and `infrastructure_exposure_summary.png`:
- `{self._figure("infrastructure_risk_by_dem_product.png")}`
- `{self._figure("infrastructure_exposure_summary.png")}`

**Say:** {counts} This is the key result. The study area, the OSM extract, the slope threshold, the aspect filter, and the code remain fixed. The input DEM changes. That alone shifts the building exposure count from the most conservative result to the most sensitive result. The supporting infrastructure chart adds road exposure lengths and civil utility counts where OSM contains those features, but buildings remain the primary academic metric because they are the most interpretable human-exposure layer.

**Interpretation:** AW3D30 produces the highest exposure count in this run. This does not prove AW3D30 is more correct. It proves that the processing lineage changes the automated exposure output enough that relying on one DEM product would be risky.

**Transition:** Close with limitations before claiming contribution.

**Expected question:** Can this result be treated as ground truth?

**Answer:** No. It is a sensitivity result. Ground truth would require field validation, landslide inventory overlay, and comparison with official or cadastral infrastructure records.

## 10. Limitations, Contribution, and Closing

**Figure cue:** Return to `sensitivity_report.png` or the report summary table.

**Say:** The limitations are explicit. There is no field validation, no landslide inventory overlay, and no lithology, drainage density, rainfall, fault distance, soil, or geotechnical modeling. The DEM inputs are not the same local 12.5 m DTM used by the reference study. They are public DEM/DSM products, so trees, buildings, and canopy can affect the surface. OSM is live and may change between runs. More importantly for Devrek and rural Turkey, OSM completeness and sparsity bias may cause undercounting of buildings, minor roads, and civil infrastructure. The report records OSM extraction timestamps so future reruns can be interpreted correctly.

**Contribution:** The project contributes a reproducible no-GIS Remote Sensing data pipeline. It turns cloud DEMs and OSM infrastructure into derivative maps, exposure counts, cluster evidence, report figures, and audit files. Its practical value is that future hazard or planning studies can quickly test whether DEM product choice changes their exposure result before they move to a full susceptibility model.

**Closing line:** The main message is simple: DEM provenance is not just metadata. In this Devrek workflow, it changes the number and location of buildings classified as exposed.
"""
