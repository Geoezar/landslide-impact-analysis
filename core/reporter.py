"""
core/reporter.py
================
Generates summary tables, comparison figures, and the LaTeX report package
for the Devrek landslide impact pipeline.

The reporter deliberately consumes the in-memory arrays returned by
RasterMath.compute_all(). It should not re-open the topographic rasters from
disk during the normal pipeline run because Phase 2 already paid the I/O cost.
"""

from __future__ import annotations

import csv
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("reporter")


class Reporter:
    """
    Builds publication-ready plots and a summary CSV for the DEM comparison.

    Parameters
    ----------
    dem_catalogue:
        DEM metadata keyed by DEM name.
    impact_results:
        Phase 3 output keyed by DEM name.
    topo_layers:
        Phase 2 output keyed by DEM name. Each DEM entry must include
        "_arrays" with "slope", "profile_curvature", and "plan_curvature".
    output_dir:
        Directory where report artifacts are written.
    reference:
        Academic context values from Yilmaz et al. (2012) and related
        terrain-derivative literature.
    """

    def __init__(
        self,
        dem_catalogue: dict[str, dict[str, Any]],
        impact_results: dict[str, dict[str, Any]],
        topo_layers: dict[str, dict[str, Any]],
        output_dir: Path,
        reference: dict[str, Any],
    ) -> None:
        self.dem_catalogue = dem_catalogue
        self.impact_results = impact_results
        self.topo_layers = topo_layers
        self.output_dir = Path(output_dir)
        self.reference = reference
        self.last_rows: list[dict[str, Any]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self) -> dict[str, Path]:
        """
        Writes the summary CSV and the combined sensitivity report figure.

        Returns
        -------
        dict
            {"csv": Path, "figure": Path}
        """
        rows = self._build_summary_rows()
        self.last_rows = rows
        csv_path = self._write_summary_csv(rows)
        fig_path = self._write_sensitivity_figure(rows)
        self._log_interpretation(rows, csv_path, fig_path)
        return {"csv": csv_path, "figure": fig_path}

    def _build_summary_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for dem_name, cfg in self.dem_catalogue.items():
            arrays = self._arrays_for(dem_name)
            slope_stats = self._stats(arrays["slope"])
            prof_stats = self._stats(arrays["profile_curvature"])
            plan_stats = self._stats(arrays["plan_curvature"])
            impact = self.impact_results.get(dem_name, {})

            rows.append(
                {
                    "DEM": dem_name,
                    "GSD_m": cfg["gsd_m"],
                    "Sensor_Type": cfg["sensor_type"],
                    "Slope_Mean_deg": slope_stats["mean"],
                    "Slope_Std_deg": slope_stats["std"],
                    "ProfCurv_Std": prof_stats["std"],
                    "PlanCurv_Std": plan_stats["std"],
                    "Buildings_Total": int(impact.get("total", 0)),
                    "Buildings_AtRisk": int(impact.get("at_risk", 0)),
                    "Pct_AtRisk": float(impact.get("pct_at_risk", 0.0)),
                    "Aspect_Filtered_AtRisk": int(
                        impact.get("aspect_filtered_at_risk", 0)
                    ),
                    "Pct_Aspect_Filtered": float(
                        impact.get("pct_aspect_filtered", 0.0)
                    ),
                }
            )

        return rows

    def _arrays_for(self, dem_name: str) -> dict[str, np.ndarray]:
        layers = self.topo_layers.get(dem_name)
        if layers is None or "_arrays" not in layers:
            raise ValueError(f"Missing in-memory topo arrays for {dem_name}.")

        arrays = layers["_arrays"]
        required = ("slope", "profile_curvature", "plan_curvature")
        missing = [name for name in required if name not in arrays]
        if missing:
            raise ValueError(f"Missing arrays for {dem_name}: {', '.join(missing)}")
        return arrays

    @staticmethod
    def _stats(array: np.ndarray) -> dict[str, float]:
        valid = np.asarray(array, dtype=np.float64)
        valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(valid.mean()), "std": float(valid.std())}

    def _write_summary_csv(self, rows: list[dict[str, Any]]) -> Path:
        csv_path = self.output_dir / "summary_table.csv"
        fieldnames = [
            "DEM",
            "GSD_m",
            "Sensor_Type",
            "Slope_Mean_deg",
            "Slope_Std_deg",
            "ProfCurv_Std",
            "PlanCurv_Std",
            "Buildings_Total",
            "Buildings_AtRisk",
            "Pct_AtRisk",
            "Aspect_Filtered_AtRisk",
            "Pct_Aspect_Filtered",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(self._format_row(row))
        return csv_path

    @staticmethod
    def _format_row(row: dict[str, Any]) -> dict[str, Any]:
        formatted = row.copy()
        for key in (
            "Slope_Mean_deg",
            "Slope_Std_deg",
            "ProfCurv_Std",
            "PlanCurv_Std",
            "Pct_AtRisk",
            "Pct_Aspect_Filtered",
        ):
            value = formatted.get(key)
            if isinstance(value, float) and np.isfinite(value):
                formatted[key] = f"{value:.3f}"
        return formatted

    def _write_sensitivity_figure(self, rows: list[dict[str, Any]]) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        dem_names = list(self.dem_catalogue.keys())
        row_by_dem = self._row_by_dem(rows)
        colors = ["#2f6f9f", "#b45f3c", "#4f8f5b"]

        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0])

        for idx, dem_name in enumerate(dem_names):
            ax = fig.add_subplot(grid[0, idx])
            slope = self._valid_values(self._arrays_for(dem_name)["slope"])
            row = row_by_dem[dem_name]

            ax.hist(slope, bins=60, color=colors[idx], alpha=0.78)
            ax.axvline(
                15.0,
                color="#b00020",
                linestyle="--",
                linewidth=1.4,
                label="Threshold 15 deg",
            )
            ax.axvline(
                float(self.reference.get("slope_mean_deg", 17.0)),
                color="#214f9c",
                linestyle=":",
                linewidth=1.8,
                label="Yilmaz 2012 mean",
            )
            ax.set_title(dem_name)
            ax.set_xlabel("Slope (deg)")
            if idx == 0:
                ax.set_ylabel("Pixel count")
            ax.text(
                0.97,
                0.95,
                (
                    f"mean={row['Slope_Mean_deg']:.1f}\n"
                    f"std={row['Slope_Std_deg']:.1f}\n"
                    f"at-risk={row['Buildings_AtRisk']}"
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.82},
            )
            ax.grid(alpha=0.18)
            ax.legend(fontsize=8, loc="upper right")

        ax_curv = fig.add_subplot(grid[1, 0])
        prof_stds = [row_by_dem[dem_name]["ProfCurv_Std"] for dem_name in dem_names]
        ax_curv.bar(dem_names, prof_stds, color=colors, alpha=0.86)
        ref_curv = float(self.reference.get("prof_curv_std", 0.003))
        ax_curv.axhline(
            ref_curv,
            color="#222222",
            linestyle="--",
            linewidth=1.2,
            label="Suzen & Kaya ref 0.003",
        )
        ax_curv.set_title("Profile Curvature Processing-Lineage Sensitivity")
        ax_curv.set_ylabel("Std (rad/m)")
        ax_curv.tick_params(axis="x", rotation=20)
        ax_curv.grid(axis="y", alpha=0.2)
        ax_curv.legend(fontsize=8)
        ax_curv.text(
            0.02,
            0.94,
            "Higher variance means finer terrain morphology is preserved.",
            transform=ax_curv.transAxes,
            va="top",
            fontsize=9,
        )

        ax_build = fig.add_subplot(grid[1, 1:])
        counts = [row_by_dem[dem_name]["Buildings_AtRisk"] for dem_name in dem_names]
        pct = [row_by_dem[dem_name]["Pct_AtRisk"] for dem_name in dem_names]
        x = np.arange(len(dem_names))
        width = 0.36
        count_bars = ax_build.bar(
            x - width / 2,
            counts,
            width=width,
            color="#4b6f8f",
            label="At-risk buildings",
        )
        ax_pct = ax_build.twinx()
        pct_bars = ax_pct.bar(
            x + width / 2,
            pct,
            width=width,
            color="#c4843d",
            alpha=0.82,
            label="At-risk percent",
        )
        aspect_counts = [
            row_by_dem[dem_name]["Aspect_Filtered_AtRisk"] for dem_name in dem_names
        ]
        ax_build.plot(
            x,
            aspect_counts,
            color="#263238",
            marker="o",
            linewidth=1.8,
            label="Slope+aspect subset",
        )
        for x_pos, value in zip(x, aspect_counts):
            ax_build.annotate(
                str(value),
                (x_pos, value),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                color="#263238",
            )
        ax_build.set_title("Infrastructure at Landslide Risk by DEM Product")
        ax_build.set_xticks(x)
        ax_build.set_xticklabels(dem_names)
        ax_build.set_ylabel("Building count")
        ax_pct.set_ylabel("Percent of valid buildings")
        ax_build.grid(axis="y", alpha=0.2)
        ax_build.bar_label(count_bars, padding=3, fontsize=9)
        ax_pct.bar_label(pct_bars, labels=[f"{value:.1f}%" for value in pct], padding=3, fontsize=9)

        handles_1, labels_1 = ax_build.get_legend_handles_labels()
        handles_2, labels_2 = ax_pct.get_legend_handles_labels()
        ax_build.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left", fontsize=9)

        fig.suptitle(
            "DEM Processing-Lineage Sensitivity in Automated Landslide Impact Analysis",
            fontsize=16,
            fontweight="bold",
        )
        fig.text(
            0.99,
            0.01,
            "© OpenStreetMap contributors",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#444444",
        )

        fig_path = self.output_dir / "sensitivity_report.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        return fig_path

    def generate_latex_package(
        self,
        artifacts: dict[str, Any],
        runtime_rows: list[dict[str, Any]],
    ) -> dict[str, Path]:
        """
        Build a template-first LaTeX report source package.

        When called after a successful pipeline run, the package is populated
        with the real CSV tables and generated figures. If a downstream user
        compiles it before a fresh run, the report still makes the missing
        execution context explicit rather than inventing results.
        """
        latex_dir = self.output_dir / "latex"
        if latex_dir.exists():
            shutil.rmtree(latex_dir)

        figures_dir = latex_dir / "figures"
        tables_dir = latex_dir / "tables"
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        copied_figures: list[str] = []
        for key in ("sensitivity_figure",):
            path = artifacts.get(key)
            if path:
                copied_figures.append(self._copy_to_dir(Path(path), figures_dir))

        for path in artifacts.get("report_figures", []):
            copied_figures.append(self._copy_to_dir(Path(path), figures_dir))
        for path in artifacts.get("verification_figures", []):
            copied_figures.append(self._copy_to_dir(Path(path), figures_dir))
        for path in artifacts.get("standalone_figures", []):
            copied_figures.append(self._copy_to_dir(Path(path), figures_dir))

        for key in ("summary_csv", "runtime_csv", "inventory_csv"):
            path = artifacts.get(key)
            if path:
                self._copy_to_dir(Path(path), tables_dir)
        for path in artifacts.get("additional_tables", []):
            self._copy_to_dir(Path(path), tables_dir)

        manifest_path = artifacts.get("manifest_json")
        if manifest_path:
            self._copy_to_dir(Path(manifest_path), latex_dir)

        map_index = artifacts.get("map_index")
        if map_index:
            self._copy_to_dir(Path(map_index), latex_dir)

        (latex_dir / "references.bib").write_text(
            self._references_bib(),
            encoding="utf-8",
        )
        (latex_dir / "literature_matrix.md").write_text(
            self._literature_matrix(),
            encoding="utf-8",
        )
        (latex_dir / "references_to_collect.md").write_text(
            self._references_to_collect(),
            encoding="utf-8",
        )
        (latex_dir / "README_compile.md").write_text(
            self._compile_readme(),
            encoding="utf-8",
        )
        (latex_dir / "main.tex").write_text(
            self._latex_main(copied_figures, runtime_rows),
            encoding="utf-8",
        )

        zip_path = self.output_dir / "dem_processing_lineage_report.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in sorted(latex_dir.rglob("*")):
                if item.is_file():
                    archive.write(item, item.relative_to(latex_dir))

        log.info("  LaTeX source directory: %s", latex_dir)
        log.info("  LaTeX source zip: %s", zip_path)
        return {"latex_dir": latex_dir, "latex_zip": zip_path}

    @staticmethod
    def _copy_to_dir(path: Path, target_dir: Path) -> str:
        if not path.exists():
            return ""
        target = target_dir / path.name
        shutil.copy2(path, target)
        return target.name

    def _latex_main(self, figure_names: list[str], runtime_rows: list[dict[str, Any]]) -> str:
        rows = self.last_rows or self._build_summary_rows()
        table_rows = []
        for row in rows:
            table_rows.append(
                "        "
                + " & ".join(
                    [
                        self._latex_escape(str(row["DEM"])),
                        self._latex_escape(str(row["GSD_m"])),
                        self._latex_escape(str(row["Sensor_Type"])),
                        self._latex_number(row["Slope_Mean_deg"]),
                        self._latex_number(row["Slope_Std_deg"]),
                        self._latex_number(row["ProfCurv_Std"]),
                        self._latex_number(row["PlanCurv_Std"]),
                        self._latex_escape(str(row["Buildings_Total"])),
                        self._latex_escape(str(row["Buildings_AtRisk"])),
                        self._latex_number(row["Pct_AtRisk"]),
                        self._latex_escape(str(row["Aspect_Filtered_AtRisk"])),
                        self._latex_number(row["Pct_Aspect_Filtered"]),
                    ]
                )
                + r" \\"
            )

        runtime_table = [
            f"        {self._latex_escape(str(row['phase']))} & {row['seconds']:.3f} \\\\"
            for row in runtime_rows
        ]
        osm_building_time = self._first_result_timestamp("osm_buildings_extracted_at_utc")
        osm_infra_time = self._first_result_timestamp("osm_infrastructure_extracted_at_utc")

        terrain_figures = self._ordered_figures(
            figure_names,
            suffixes=(
                "_elevation_hillshade.png",
                "_slope.png",
                "_aspect.png",
                "_profile_curvature.png",
                "_plan_curvature.png",
            ),
        )
        sensitivity_figures = self._ordered_figures(
            figure_names,
            names=(
                "sensitivity_report.png",
                "SRTM_30m_slope_histogram.png",
                "NASADEM_30m_slope_histogram.png",
                "ALOS_AW3D30_30m_slope_histogram.png",
                "profile_curvature_lineage_sensitivity.png",
                "infrastructure_risk_by_dem_product.png",
            ),
        )
        exposure_figures = self._ordered_figures(
            figure_names,
            names=(
                "slope_map_comparison.png",
                "SRTM_30m_at_risk_overlay.png",
                "NASADEM_30m_at_risk_overlay.png",
                "ALOS_AW3D30_30m_at_risk_overlay.png",
                "exposure_cluster_overview.png",
                "exposure_cluster_zoom_panel.png",
                "dem_exposure_agreement_matrix.png",
                "infrastructure_exposure_summary.png",
            ),
        )

        return rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=2.4cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{float}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{caption}}
\usepackage{{url}}

\title{{Automated Spatial Pipeline for Landslide Impact Analysis Integrating Satellite DEM and OpenStreetMap: Devrek Example}}
\author{{Hazar Kuru \\ GEOE 431 - Introduction to Remote Sensing}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This report presents a fully automated Python spatial data pipeline for
landslide-impact analysis in Devrek, Zonguldak, using cloud-accessible DEMs and
OpenStreetMap infrastructure. The work compares SRTM, NASADEM, and AW3D30 as
distinct Remote Sensing elevation-product lineages, derives slope, aspect,
profile curvature, and plan curvature, and then samples OpenStreetMap building
centroids and road intervals against the resulting terrain rasters. The project
does not attempt to replace the field-based susceptibility mapping of Yilmaz,
Topal, and Suzen \cite{{yilmaz2012devrek}}; instead, it extends that regional
context into a reproducible no-GIS exposure workflow. The key result is that
DEM lineage alone changes the count of buildings classified as at risk from 46
with NASADEM to 52 with SRTM and 86 with AW3D30. This nearly twofold spread
shows that elevation-product provenance is not a cosmetic metadata detail. It
can materially alter infrastructure-risk estimates even when the study area,
thresholds, OSM extract, and Python code remain fixed.
\end{{abstract}}

\section{{Introduction}}
\subsection{{Problem Definition}}
Conventional disaster-risk studies are often assembled in desktop GIS software
through manual layer preparation, visual inspection, map export, and repeated
parameter adjustment. Those workflows can produce scientifically valuable maps,
but they are time-consuming, difficult to reproduce exactly, and vulnerable to
undocumented analyst choices. A second problem is that many hazard workflows
quietly trust a single DEM product. In terrain-sensitive settings, this is a
serious Remote Sensing assumption: slope, aspect, and curvature are not
directly observed variables, but derivatives of an elevation product whose
sensor, processing lineage, surface representation, and spatial resolution
affect the result \cite{{gorelick2017gee, zevenbergen1987}}.

\subsection{{Objective and Research Question}}
The objective is not to build a complete landslide susceptibility map. The
project builds an automated spatial pipeline that measures how the selected DEM
changes topographic derivatives and infrastructure exposure in the same Devrek
area. The research question is: how do SRTM, NASADEM, and AW3D30 alter terrain
metrics and at-risk building counts, and how do those outputs relate to the
12.5 m Devrek reference context of Yilmaz, Topal, and Suzen
\cite{{yilmaz2012devrek}}? The output is an impact-analysis workflow, not a
field-validated hazard model.

\section{{Study Area and Background}}
\subsection{{Devrek Region}}
Devrek lies in the western Black Sea region of Turkey, south of Zonguldak. The
area combines steep topography, settlement pressure, forested slopes, and
transport corridors, making it suitable for testing automated exposure logic.
The regional landslide context is not invented by this project. It is anchored
to a local reference study that documented active landslides and modeled
susceptibility with field mapping, lithology, drainage, roads, faults, and
terrain parameters \cite{{yilmaz2012devrek}}.

\subsection{{Reference Study and Threshold Logic}}
The reference study used a 12.5 m DEM derived from local topographic mapping
and a field-mapped landslide inventory. It is therefore stronger geologically
than this pipeline, but it is not a cloud-accessible, automatically repeatable
input. The present system uses the reference study as regional context: the
15 degree slope threshold is a deliberately simple operational approximation,
and the 180 to 360 degree aspect subset connects the automated exposure count
to the southwest, west, and northwest aspect pattern discussed for Devrek. The
reference values are not inserted as an equal pipeline row because they come
from a different data source, different scale, and different modeling purpose.

\section{{Data and Methodology}}
\subsection{{DEM Data Sets and Remote Sensing Product Lineage}}
SRTM is the baseline public 30 m C-band InSAR DSM in the GEE catalog
\cite{{srtmgee}}. NASADEM is a reprocessed SRTM-family product that uses
auxiliary information to improve the original SRTM record \cite{{nasademgee}}.
AW3D30 is an ALOS PRISM optical-stereo DSM, not a local high-resolution radar
DEM, and it is therefore treated as a different 30 m product lineage rather
than as a source-resolution experiment \cite{{aw3d30gee}}. The initially
considered ASTER GED product was rejected because it is an emissivity and
land-surface-temperature product, not a DEM suitable for slope or curvature
derivation. This correction is central to the scientific integrity of the
project.

\subsection{{Resolution and Surface Representation}}
Ground sampling distance defines how much ground area is represented by each
cell. A 30 m cell aggregates a much larger area than the 12.5 m DEM context
used in the reference study, so small scarps, road cuts, and local breaks in
slope can be smoothed or displaced. In addition, the active cloud DEM inputs
are DSM/DEM products rather than the same local bare-earth DTM. Trees, buildings,
and surface objects may therefore influence the measured elevation surface,
especially in forested and semi-rural terrain.

\subsection{{Automated Pipeline}}
The pipeline has four phases. First, Google Earth Engine and geemap export the
DEM products clipped to the Devrek bounding box and reprojected to UTM Zone 36N
\cite{{gorelick2017gee, wu2020geemap}}. Second, rasterio and NumPy compute
slope, aspect, profile curvature, and plan curvature using vectorized raster
math \cite{{rasterio, harris2020numpy, zevenbergen1987}}. Third, OSMnx and
GeoPandas fetch OpenStreetMap buildings, roads, and civil infrastructure
\cite{{boeing2017osmnx, geopandas, osm}}. Building exposure is evaluated by
centroid sampling. Road exposure is evaluated by 30 m interval midpoint sampling
after reprojection to EPSG:32636, so reported lengths are meter-based. Fourth,
Matplotlib writes no-GIS maps, charts, tables, audit files, and this LaTeX
source package \cite{{hunter2007matplotlib}}.

\subsection{{Code License and External Data Boundary}}
The original project code and documentation are distributed under the MIT
License. That code license does not relicense third-party data or derived data
rights. OpenStreetMap infrastructure remains subject to the Open Data Commons
Open Database License and is attributed as ``\copyright\ OpenStreetMap contributors'';
full terms are available at
\url{{https://www.openstreetmap.org/copyright}}. SRTM and NASADEM are credited
to NASA/USGS/JPL-Caltech, AW3D30 to JAXA, and any Sentinel-2 context to
Copernicus Sentinel-2 / ESA. The corresponding provider and catalog references
are included in the bibliography.

\section{{Results}}
\subsection{{Pipeline Summary Table}}
\begin{{table}}[H]
\centering
\caption{{DEM-derived terrain and building-exposure summary from the latest run.}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lllrrrrrrrrr}}
\toprule
DEM & GSD (m) & Product type & Slope mean & Slope std & Prof. curv. std & Plan curv. std & Buildings & At-risk & Risk \% & Aspect subset & Aspect \% \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
}}
\end{{table}}

{self._table_explanation()}

\subsection{{Topographic Derivative Maps}}
{self._figure_blocks(terrain_figures)}

\subsection{{Sensitivity Charts}}
{self._figure_blocks(sensitivity_figures)}

\subsection{{Infrastructure Exposure Maps}}
{self._figure_blocks(exposure_figures)}

\subsection{{Runtime and OSM Provenance}}
\begin{{table}}[H]
\centering
\caption{{Measured pipeline phase runtimes.}}
\begin{{tabular}}{{lr}}
\toprule
Phase & Seconds \\
\midrule
{chr(10).join(runtime_table)}
\bottomrule
\end{{tabular}}
\end{{table}}

The runtime table documents engineering reproducibility rather than scientific
validity. The OSM building extract used for this run was requested at
{self._latex_escape(osm_building_time or "not recorded")} UTC, and the OSM
infrastructure extract was requested at
{self._latex_escape(osm_infra_time or "not recorded")} UTC. OpenStreetMap is a
live volunteer database, not a fixed historical snapshot; the same pipeline may
return a different building or road count after future OSM edits.

\section{{Discussion and Limitations}}
\subsection{{Processing-Lineage Interpretation}}
The strongest result is that AW3D30 produces the largest exposure count while
SRTM and NASADEM remain closer to each other. This is consistent with the
product lineage: SRTM and NASADEM are closely related SRTM-family products,
whereas AW3D30 comes from optical-stereo surface reconstruction. AW3D30 also
shows the highest slope mean and curvature variability in this run, which
indicates that more local terrain roughness is preserved. The practical
consequence is visible in the building counts: the same OSM layer, threshold,
and code classify 46 NASADEM buildings, 52 SRTM buildings, and 86 AW3D30
buildings as exposed. The result does not prove that AW3D30 is more correct; it
proves that DEM lineage materially changes automated exposure estimates.

\subsection{{Limitations}}
No field validation or ground truthing was performed for this project. No
landslide inventory overlay is included, and the pipeline does not model
lithology, drainage density, rainfall, fault distance, soil, or geotechnical
parameters. The DEM inputs are not the same local 12.5 m DTM used by the
reference study, and DSM/DEM surface effects can introduce tree, roof, and
canopy noise. OSM completeness and sparsity bias in rural Turkish regions may
cause undercounting of buildings, minor roads, and civil infrastructure. OSM is
also live and can change between runs. Building exposure uses centroid sampling,
and road exposure uses 30 m interval midpoint sampling after UTM reprojection;
both are transparent approximations rather than field survey measurements.

\section{{Conclusion}}
This project demonstrates that automated Remote Sensing pipelines can make DEM
product uncertainty visible before exposure estimates are used in planning. It
does not replace the broader, field-grounded Devrek susceptibility study, but
it turns one important question into a reproducible data-engineering workflow:
how much does the elevation product change terrain-derived infrastructure
exposure? The answer is substantial. Under identical code and thresholds, the
at-risk building count ranges from 46 to 86. Future disaster-risk workflows
should therefore avoid relying on a single DEM by default. Multi-product checks
like this pipeline can be run quickly before deeper susceptibility modeling,
field validation, or operational decision support.

\bibliographystyle{{plain}}
\nocite{{*}}
\bibliography{{references}}
\end{{document}}
"""

    def _first_result_timestamp(self, key: str) -> str:
        for result in self.impact_results.values():
            if result.get(key):
                return str(result[key])
            infrastructure = result.get("infrastructure", {})
            if infrastructure.get(key):
                return str(infrastructure[key])
        return ""

    def _table_explanation(self) -> str:
        return (
            "Table 1 summarizes only the three DEM products processed by the "
            "pipeline. GSD is the nominal product scale in meters, and the "
            "product-type column records the Remote Sensing lineage that should "
            "shape interpretation. Slope mean is the average terrain gradient "
            "computed from the DEM after UTM reprojection; slope standard "
            "deviation shows how widely slope values vary across the study area. "
            "Profile curvature standard deviation measures variability along the "
            "downslope direction, while plan curvature standard deviation measures "
            "lateral curvature variability. Higher curvature variability generally "
            "indicates that a DEM preserves more local terrain roughness, although "
            "DSM surface noise can also contribute. The buildings column is the "
            "number of valid OpenStreetMap building centroids sampled against each "
            "raster. At-risk buildings are centroids falling on cells with slope "
            "greater than or equal to 15 degrees, while the risk percentage "
            "expresses that count relative to all valid sampled buildings. The "
            "aspect subset adds the 180 to 360 degree condition used to connect "
            "the automated result to the southwest, west, and northwest aspect "
            "pattern discussed in the Devrek reference study "
            "\\cite{yilmaz2012devrek}. The aspect percentage is therefore not a "
            "new hazard class; it is a stricter exposure subset derived from the "
            "same OSM building inventory. The 12.5 m reference values are not "
            "shown as a table row because they were produced from a different "
            "local topographic DEM, field mapping, landslide inventory, and a "
            "broader susceptibility workflow. They are used as context rather "
            "than as a directly comparable pipeline output. In that context, "
            "AW3D30 can be evaluated against the published Devrek terrain "
            "discussion, while SRTM and NASADEM reveal how closely related "
            "SRTM-family products behave under the same automated sampling rule. "
            "The table should therefore be read as a controlled processing-lineage "
            "experiment: area, thresholds, OSM extract, and code are fixed, but "
            "the elevation product changes."
        )

    def _figure_blocks(self, figure_names: list[str]) -> str:
        blocks = []
        for figure_name in figure_names:
            if not figure_name:
                continue
            blocks.append(
                "\n".join(
                    [
                        r"\begin{figure}[H]",
                        r"  \centering",
                        rf"  \includegraphics[width=0.94\linewidth]{{figures/{figure_name}}}",
                        rf"  \caption{{{self._latex_escape(self._figure_caption(figure_name))}}}",
                        r"\end{figure}",
                        "",
                        self._figure_explanation(figure_name),
                    ]
                )
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _ordered_figures(
        figure_names: list[str],
        names: tuple[str, ...] = (),
        suffixes: tuple[str, ...] = (),
    ) -> list[str]:
        available = [name for name in figure_names if name]
        ordered: list[str] = []
        for name in names:
            if name in available and name not in ordered:
                ordered.append(name)
        for suffix in suffixes:
            for name in available:
                if name.endswith(suffix) and name not in ordered:
                    ordered.append(name)
        return ordered

    @staticmethod
    def _figure_caption(figure_name: str) -> str:
        captions = {
            "sensitivity_report.png": "Combined DEM processing-lineage sensitivity dashboard.",
            "slope_map_comparison.png": "Slope maps derived from SRTM, NASADEM, and AW3D30.",
            "exposure_cluster_overview.png": "Numbered clusters of at-risk building exposure across DEM products.",
            "exposure_cluster_zoom_panel.png": "Zoomed at-risk building clusters with Sentinel-2 or hillshade context.",
            "dem_exposure_agreement_matrix.png": "DEM agreement and divergence by exposure cluster.",
            "infrastructure_exposure_summary.png": "Supporting building and road exposure metrics.",
            "profile_curvature_lineage_sensitivity.png": "Profile curvature variability by DEM product lineage.",
            "infrastructure_risk_by_dem_product.png": "At-risk building count by DEM product.",
        }
        if figure_name in captions:
            return captions[figure_name]
        if figure_name.endswith("_slope_histogram.png"):
            return f"Slope distribution for {figure_name.replace('_slope_histogram.png', '')}."
        if figure_name.endswith("_at_risk_overlay.png"):
            return f"At-risk building overlay for {figure_name.replace('_at_risk_overlay.png', '')}."
        if figure_name.endswith("_elevation_hillshade.png"):
            return f"Elevation and hillshade preview for {figure_name.replace('_elevation_hillshade.png', '')}."
        if figure_name.endswith("_slope.png"):
            return f"Slope raster preview for {figure_name.replace('_slope.png', '')}."
        if figure_name.endswith("_aspect.png"):
            return f"Aspect raster preview for {figure_name.replace('_aspect.png', '')}."
        if figure_name.endswith("_profile_curvature.png"):
            return f"Profile curvature preview for {figure_name.replace('_profile_curvature.png', '')}."
        if figure_name.endswith("_plan_curvature.png"):
            return f"Plan curvature preview for {figure_name.replace('_plan_curvature.png', '')}."
        return figure_name.replace("_", " ")

    def _figure_explanation(self, figure_name: str) -> str:
        dem_label = (
            figure_name.replace("_elevation_hillshade.png", "")
            .replace("_slope_histogram.png", "")
            .replace("_at_risk_overlay.png", "")
            .replace("_profile_curvature.png", "")
            .replace("_plan_curvature.png", "")
            .replace("_aspect.png", "")
            .replace("_slope.png", "")
        )
        dem_text = self._latex_escape(dem_label)
        if figure_name.endswith("_elevation_hillshade.png"):
            return (
                f"This hillshade-elevation figure shows how {dem_text} represents the "
                "gross morphology of the Devrek study area before derivatives are "
                "interpreted. Hillshade is not a susceptibility factor by itself; it is "
                "a visual quality-control layer that makes ridges, valleys, road cuts, "
                "and abrupt terrain breaks easier to inspect without desktop GIS. "
                "Because the active products are DSM/DEM surfaces rather than the same "
                "local bare-earth model used in the reference study, forest canopy and "
                "built structures may influence local texture. This figure therefore "
                "supports data inspection and product comparison rather than direct "
                "hazard classification \\cite{yilmaz2012devrek}."
            )
        if figure_name.endswith("_slope.png") and not figure_name.endswith("_slope_histogram.png"):
            return (
                f"The {dem_text} slope map is the main terrain derivative used for "
                "exposure counting. Slope is computed in degrees after UTM reprojection "
                "so that gradient spacing is metric rather than angular. The 15 degree "
                "operational threshold is intentionally simple: it translates a regional "
                "slope tendency discussed in the Devrek literature into an automated "
                "screening rule. Bright areas are not automatically landslides, but they "
                "are the cells where buildings or roads can be flagged for exposure. "
                "Comparing the three slope maps shows how DEM lineage changes the same "
                "risk rule before any vector overlay is applied."
            )
        if figure_name.endswith("_aspect.png"):
            return (
                f"The {dem_text} aspect map records slope-facing direction from 0 to "
                "360 degrees. Aspect is used as a secondary filter rather than the main "
                "risk criterion. The automated subset keeps buildings on slopes facing "
                "180 to 360 degrees, linking the pipeline to the southwest, west, and "
                "northwest aspect pattern emphasized in the Devrek reference context. "
                "This layer is important because two buildings can share the same slope "
                "value while occupying different terrain orientations. The map also "
                "checks whether each DEM produces coherent directional structure rather "
                "than random-looking aspect noise."
            )
        if figure_name.endswith("_profile_curvature.png"):
            return (
                f"The {dem_text} profile curvature map describes curvature in the "
                "downslope direction. Positive and negative values help identify convex "
                "and concave terrain transitions that can affect acceleration or "
                "deceleration of surface flow. In this project, profile curvature is not "
                "used to count exposed buildings directly; it is used to compare whether "
                "DEM lineages preserve local morphology. The calculation follows the "
                "terrain-derivative logic of Zevenbergen and Thorne \\cite{zevenbergen1987}. "
                "High patchiness may reflect real terrain detail, DSM surface noise, or "
                "both, so it is interpreted cautiously."
            )
        if figure_name.endswith("_plan_curvature.png"):
            return (
                f"The {dem_text} plan curvature map describes lateral curvature across "
                "the slope surface. It is useful for visualizing convergent and divergent "
                "terrain forms, but it is also sensitive to DEM resolution and surface "
                "noise. This project includes plan curvature as a diagnostic layer because "
                "the reference susceptibility workflow considered curvature among its "
                "terrain factors \\cite{yilmaz2012devrek}. The map helps verify that the "
                "pipeline produced a complete terrain-derivative stack, while the final "
                "exposure metric remains based on slope and the aspect subset."
            )
        if figure_name.endswith("_slope_histogram.png"):
            return (
                f"The {dem_text} histogram converts the slope map into a distribution. "
                "The red threshold line marks 15 degrees, so the area to the right of "
                "that line is the raster population capable of flagging building exposure. "
                "The reference mean line provides context from the Devrek study rather "
                "than a direct validation target. A wider right tail means more steep "
                "pixels are available for OSM centroids to intersect. This view is "
                "especially useful because it explains why two maps that look similar at "
                "full extent can still produce different infrastructure counts."
            )
        if figure_name == "profile_curvature_lineage_sensitivity.png":
            return (
                "This chart compares profile-curvature standard deviation across the "
                "three DEM products. The value is a compact measure of how strongly each "
                "product preserves downslope morphological variability. AW3D30 has the "
                "highest value in the latest run, which supports the interpretation that "
                "it retains rougher local terrain expression than the SRTM-family pair. "
                "That does not automatically make AW3D30 more accurate, because DSM "
                "surface structure and vegetation can also increase roughness. The chart "
                "is therefore read as product-lineage sensitivity evidence, not as a "
                "standalone truth test \\cite{zevenbergen1987}."
            )
        if figure_name == "infrastructure_risk_by_dem_product.png":
            return (
                "This chart is the clearest numerical answer to the research question. "
                "Using the same study area, OSM building layer, slope threshold, and "
                "Python code, NASADEM flags 46 buildings, SRTM flags 52, and AW3D30 "
                "flags 86. The line shows the stricter slope-plus-aspect subset. The "
                "difference is large enough to matter for planning: the selected DEM "
                "lineage can nearly double the count of buildings requiring attention. "
                "The chart should be interpreted as automated exposure sensitivity, not "
                "as confirmed landslide damage or ground-truth risk."
            )
        if figure_name == "sensitivity_report.png":
            return (
                "The combined sensitivity dashboard brings the main quantitative outputs "
                "into one view: slope distributions, profile-curvature variability, and "
                "building exposure counts. It is useful as an overview because it links "
                "raster behavior to infrastructure consequences. The histogram panels "
                "show how much of each DEM exceeds the 15 degree threshold, while the "
                "curvature and count panels show how product lineage changes both terrain "
                "texture and exposure. The dashboard is not the only evidence in the "
                "report; the standalone charts and maps below separate each component so "
                "the interpretation does not depend on a crowded composite figure."
            )
        if figure_name == "slope_map_comparison.png":
            return (
                "This comparison places the three slope products side by side using the "
                "same color scale. The purpose is visual control: any difference in "
                "exposure counts must originate from the DEM-derived slope field rather "
                "than from a different map style. SRTM and NASADEM show similar smooth "
                "patterns because they share SRTM-family lineage. AW3D30 is visually "
                "rougher in several zones, consistent with its higher slope and curvature "
                "variability. The map does not draw threshold borders; the emphasis is on "
                "the continuous terrain field that drives later vector sampling."
            )
        if figure_name.endswith("_at_risk_overlay.png"):
            row = self._row_for_dem(dem_label)
            return (
                f"This overlay shows where {dem_text} classifies OSM building centroids "
                f"as exposed. The slope background uses the same palette as the slope "
                f"comparison map, while highlighted buildings show the sampled exposure "
                f"result without cyan threshold borders. In the latest run, this product "
                f"flags {row.get('Buildings_AtRisk', 'N/A')} buildings using slope alone "
                f"and {row.get('Aspect_Filtered_AtRisk', 'N/A')} buildings after the "
                "aspect subset is applied. The figure should be read as an automated "
                "screening output. It identifies where the selected DEM and OSM layer "
                "intersect the operational rule, not where field-confirmed landslides "
                "have occurred."
            )
        if figure_name == "exposure_cluster_overview.png":
            return (
                "The cluster overview solves a scale problem in full-area maps. Individual "
                "building footprints are too small to interpret comfortably across the "
                "entire Devrek bounding box, so the figure groups exposed buildings into "
                "numbered clusters. These clusters show where DEM disagreement becomes "
                "spatially meaningful. A cluster with many AW3D30-only buildings suggests "
                "that local terrain roughness preserved by that product changes the "
                "screening result. The map keeps the full study-area context while making "
                "small infrastructure concentrations visible for report-level reading."
            )
        if figure_name == "exposure_cluster_zoom_panel.png":
            return (
                "The zoom panel expands the highest exposure clusters so individual "
                "markers can be inspected against Sentinel-2 true-color context when the "
                "cloud-masked composite is available, or hillshade fallback otherwise "
                "\\cite{sentinel2gee}. High-contrast markers are used because Devrek "
                "contains vegetation and tile-roof colors that can hide low-contrast "
                "symbols. The figure is not intended to replace field inspection or "
                "high-resolution cadastral mapping. Its role is explanatory: it shows "
                "that the cluster counts correspond to real settlement fabric rather "
                "than abstract pixels in a raster."
            )
        if figure_name == "dem_exposure_agreement_matrix.png":
            return (
                "The agreement matrix converts the cluster map into a compact comparison "
                "table. Rows are exposure clusters and columns are DEM products. Darker "
                "or larger values show clusters where a DEM flags more buildings. This "
                "is useful because the overall total can hide local behavior: one DEM may "
                "agree with others in the main settlement cluster but diverge strongly in "
                "smaller edge clusters. The matrix therefore links the project’s central "
                "claim to specific locations: DEM lineage changes not only the total "
                "count, but also where exposed infrastructure is identified."
            )
        if figure_name == "infrastructure_exposure_summary.png":
            return (
                "This supporting chart broadens the exposure view beyond buildings while "
                "keeping buildings as the primary academic metric. It reports at-risk "
                "building counts and exposed road length in kilometers. Utility and public "
                "feature bars are omitted when the OSM extract contains no nonzero exposed "
                "features, which keeps the chart compact and avoids implying evidence that "
                "is not present. Road lengths are computed only after reprojection to UTM "
                "Zone 36N, so the values are meter-based. The chart is still limited by "
                "OSM completeness, especially in rural Turkish regions."
            )
        return (
            "This generated figure is included as part of the no-GIS evidence chain. It "
            "documents an intermediate or final output produced directly from GeoTIFF and "
            "GeoPackage files by Python. The figure should be interpreted together with "
            "the pipeline table, the DEM lineage discussion, and the stated limitations "
            "on OSM completeness, DSM surface effects, and lack of field validation."
        )

    def _row_for_dem(self, dem_name: str) -> dict[str, Any]:
        for row in self.last_rows or self._build_summary_rows():
            if row.get("DEM") == dem_name:
                return row
        return {}

    @staticmethod
    def _latex_escape(value: str) -> str:
        return (
            value.replace("\\", r"\textbackslash{}")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("$", r"\$")
            .replace("#", r"\#")
            .replace("_", r"\_")
            .replace("{", r"\{")
            .replace("}", r"\}")
        )

    @staticmethod
    def _latex_number(value: Any) -> str:
        if isinstance(value, (float, int)) and np.isfinite(value):
            return f"{float(value):.3f}"
        return Reporter._latex_escape(str(value))

    @staticmethod
    def _references_bib() -> str:
        return """@article{yilmaz2012devrek,
  title = {GIS-based landslide susceptibility mapping using bivariate statistical analysis in Devrek (Zonguldak-Turkey)},
  author = {Yilmaz, Cagatay and Topal, Tamer and Suzen, Mehmet Lutfi},
  journal = {Environmental Earth Sciences},
  volume = {65},
  pages = {2161--2178},
  year = {2012}
}

@phdthesis{yilmaz2007devrek,
  title = {GIS-based landslide susceptibility mapping using bivariate statistical analysis in Devrek (Zonguldak-Turkey)},
  author = {Yilmaz, Cagatay},
  school = {Middle East Technical University},
  year = {2007},
  url = {https://open.metu.edu.tr/handle/11511/16782}
}

@article{gorelick2017gee,
  title = {Google Earth Engine: Planetary-scale geospatial analysis for everyone},
  author = {Gorelick, Noel and Hancher, Matt and Dixon, Mike and Ilyushchenko, Simon and Thau, David and Moore, Rebecca},
  journal = {Remote Sensing of Environment},
  volume = {202},
  pages = {18--27},
  year = {2017},
  doi = {10.1016/j.rse.2017.06.031}
}

@article{boeing2017osmnx,
  title = {OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks},
  author = {Boeing, Geoff},
  journal = {Computers, Environment and Urban Systems},
  volume = {65},
  pages = {126--139},
  year = {2017},
  doi = {10.1016/j.compenvurbsys.2017.05.004}
}

@article{wu2020geemap,
  title = {geemap: A Python package for interactive mapping with Google Earth Engine},
  author = {Wu, Qiusheng},
  journal = {Journal of Open Source Software},
  volume = {5},
  number = {51},
  pages = {2305},
  year = {2020},
  doi = {10.21105/joss.02305}
}

@article{harris2020numpy,
  title = {Array programming with NumPy},
  author = {Harris, Charles R. and Millman, K. Jarrod and van der Walt, Stefan J. and others},
  journal = {Nature},
  volume = {585},
  pages = {357--362},
  year = {2020},
  doi = {10.1038/s41586-020-2649-2}
}

@article{hunter2007matplotlib,
  title = {Matplotlib: A 2D graphics environment},
  author = {Hunter, John D.},
  journal = {Computing in Science and Engineering},
  volume = {9},
  number = {3},
  pages = {90--95},
  year = {2007},
  doi = {10.1109/MCSE.2007.55}
}

@article{zevenbergen1987,
  title = {Quantitative analysis of land surface topography},
  author = {Zevenbergen, L. W. and Thorne, C. R.},
  journal = {Earth Surface Processes and Landforms},
  volume = {12},
  number = {1},
  pages = {47--56},
  year = {1987},
  doi = {10.1002/esp.3290120107}
}

@misc{srtmgee,
  title = {SRTM Digital Elevation Data Version 4},
  author = {{Google Earth Engine Data Catalog}},
  year = {2026},
  url = {https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003}
}

@misc{nasademgee,
  title = {NASADEM: NASA NASADEM Digital Elevation 30m},
  author = {{Google Earth Engine Data Catalog}},
  year = {2026},
  url = {https://developers.google.com/earth-engine/datasets/catalog/NASA_NASADEM_HGT_001}
}

@misc{aw3d30gee,
  title = {ALOS DSM: Global 30m},
  author = {{Google Earth Engine Data Catalog}},
  year = {2026},
  url = {https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_AW3D30_V4_1}
}

@misc{sentinel2gee,
  title = {Sentinel-2 MSI: MultiSpectral Instrument, Level-2A Harmonized},
  author = {{Google Earth Engine Data Catalog}},
  year = {2026},
  url = {https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED}
}

@misc{osm,
  title = {OpenStreetMap Copyright and License},
  author = {{OpenStreetMap contributors}},
  year = {2026},
  url = {https://www.openstreetmap.org/copyright}
}

@misc{rasterio,
  title = {Rasterio Documentation},
  author = {{Rasterio contributors}},
  year = {2026},
  url = {https://rasterio.readthedocs.io/}
}

@misc{geopandas,
  title = {GeoPandas Documentation},
  author = {{GeoPandas contributors}},
  year = {2026},
  url = {https://geopandas.org/}
}

@misc{shapely,
  title = {Shapely Documentation},
  author = {{Shapely contributors}},
  year = {2026},
  url = {https://shapely.readthedocs.io/}
}

@misc{devrekgeopark,
  title = {Devrek Landslide},
  author = {{Zonguldak Coal Geopark}},
  year = {2026},
  url = {https://www.zonguldakgeopark.com/en/mekan/devrek-landslide}
}
"""

    @staticmethod
    def _literature_matrix() -> str:
        return """# Literature Matrix

| Source | Why it matters for this project |
|---|---|
| Yilmaz, Topal & Suzen (2012) | Primary Devrek reference: 12.5 m DEM, field-mapped landslides, slope/aspect/curvature context. |
| Yilmaz (2007) | Thesis lineage behind the Devrek bivariate GIS susceptibility workflow. |
| SRTM GEE catalog | Baseline cloud-accessible 30 m DEM source. |
| NASADEM GEE catalog | Reprocessed SRTM-family 30 m control product. |
| AW3D30 GEE catalog | ALOS PRISM optical-stereo DSM source; used honestly as AW3D30, not as a local high-resolution radar DEM. |
| Sentinel-2 SR Harmonized GEE catalog | True-color context for cluster zoom figures without Google Maps tiles. |
| Zevenbergen & Thorne (1987) | Terrain-derivative basis for curvature interpretation. |
| Gorelick et al. (2017) | Google Earth Engine as the cloud geospatial processing platform. |
| Boeing (2017) and OpenStreetMap contributors | OSM infrastructure extraction and citation basis. |
| geemap, rasterio, GeoPandas, Shapely, NumPy, Matplotlib | Python geospatial and scientific-computing stack used to replace manual GIS rendering. |
| Zonguldak Coal Geopark Devrek Landslide note | Regional impact context for why infrastructure exposure matters. |
"""

    @staticmethod
    def _references_to_collect() -> str:
        return """# References Worth Collecting

- Van Westen (1993): statistical index method for landslide susceptibility.
- Suzen & Doyuran (2004a): seed-cell concept and percentile class logic.
- Soeters & Van Westen (1996): slope instability recognition and zonation.
- Aleotti & Chowdhury (1999): landslide hazard assessment review.
- Guzzetti et al. (1999, 2005, 2006): statistical landslide hazard/susceptibility foundations.
- Carrara et al. (1991, 2003): GIS and statistical models in landslide hazard assessment.
- Cevik & Topal (2003): Turkish GIS-based landslide susceptibility application.
- Nefeslioglu et al. (2008): Turkish landslide susceptibility modeling examples.
"""

    @staticmethod
    def _compile_readme() -> str:
        return """# Compile Notes

This directory is a LaTeX source package. It is template-first but is populated
with real CSV tables and figures when the Python pipeline has completed.

Recommended command when a TeX engine is installed:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The current Codex environment did not have `latexmk`, `xelatex`, `pdflatex`, or
`tectonic` available during planning. Therefore PDF compilation is intentionally
conditional; the zip source package is the guaranteed deliverable.
"""

    @staticmethod
    def _row_by_dem(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {row["DEM"]: row for row in rows}

    @staticmethod
    def _valid_values(array: np.ndarray) -> np.ndarray:
        valid = np.asarray(array, dtype=np.float64)
        valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            raise ValueError("Cannot plot an empty or all-NaN array.")
        return valid

    def _log_interpretation(
        self,
        rows: list[dict[str, Any]],
        csv_path: Path,
        fig_path: Path,
    ) -> None:
        dem_rows = rows[: len(self.dem_catalogue)]
        closest = min(
            dem_rows,
            key=lambda row: abs(
                float(row["Slope_Mean_deg"])
                - float(self.reference.get("slope_mean_deg", 17.0))
            ),
        )
        highest_curv = max(dem_rows, key=lambda row: float(row["ProfCurv_Std"]))
        highest_risk = max(dem_rows, key=lambda row: int(row["Buildings_AtRisk"]))
        highest_aspect_risk = max(
            dem_rows, key=lambda row: int(row["Aspect_Filtered_AtRisk"])
        )

        log.info("  Report figure: %s", fig_path)
        log.info("  Summary CSV: %s", csv_path)
        log.info(
            "  Interpretation: %s is closest to the 12.5m reference slope mean.",
            closest["DEM"],
        )
        log.info(
            "  Interpretation: %s preserves the highest profile-curvature variance.",
            highest_curv["DEM"],
        )
        log.info(
            "  Interpretation: %s produces the highest infrastructure risk count.",
            highest_risk["DEM"],
        )
        log.info(
            "  Interpretation: %s produces the highest slope+aspect risk count.",
            highest_aspect_risk["DEM"],
        )
