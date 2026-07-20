"""
core/visualizer.py
==================
Creates no-GIS PNG previews from the raster and vector outputs.

The visualizer is intentionally separate from reporter.py:
  - reporter.py produces tables and the compact academic sensitivity figure;
  - visualizer.py renders map products for report use and verification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("visualizer")


class Visualizer:
    """
    Render full-study-area map previews without a GIS desktop application.
    """

    def __init__(
        self,
        dem_catalogue: dict[str, dict[str, Any]],
        dem_paths: dict[str, Path],
        topo_layers: dict[str, dict[str, Any]],
        impact_results: dict[str, dict[str, Any]],
        output_dir: Path,
        report_dir: Path,
        bbox: dict[str, float] | None = None,
    ) -> None:
        self.dem_catalogue = dem_catalogue
        self.dem_paths = {key: Path(value) for key, value in dem_paths.items()}
        self.topo_layers = topo_layers
        self.impact_results = impact_results
        self.output_dir = Path(output_dir)
        self.report_dir = Path(report_dir)
        self.bbox = bbox
        self.report_maps_dir = self.output_dir / "report"
        self.verification_dir = self.output_dir / "verification"
        self.report_maps_dir.mkdir(parents=True, exist_ok=True)
        self.verification_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate(self) -> dict[str, Any]:
        """
        Generate report-facing and verification-facing figures.
        """
        report_figures: list[Path] = []
        verification_figures: list[Path] = []

        report_figures.append(self._plot_slope_comparison())

        for dem_name in self.dem_catalogue:
            verification_figures.append(self._plot_elevation_hillshade(dem_name))
            verification_figures.append(
                self._plot_layer(
                    dem_name,
                    layer_key="slope",
                    title=f"{dem_name} - slope",
                    cmap="magma",
                    output_path=self.verification_dir / f"{dem_name}_slope.png",
                    label="Slope (deg)",
                    vmin=0,
                    vmax=65,
                )
            )
            verification_figures.append(
                self._plot_layer(
                    dem_name,
                    layer_key="aspect",
                    title=f"{dem_name} - aspect",
                    cmap="twilight",
                    output_path=self.verification_dir / f"{dem_name}_aspect.png",
                    label="Aspect (deg)",
                    vmin=0,
                    vmax=360,
                    mask_negative=True,
                )
            )
            verification_figures.append(
                self._plot_layer(
                    dem_name,
                    layer_key="profile_curvature",
                    title=f"{dem_name} - profile curvature",
                    cmap="coolwarm",
                    output_path=self.verification_dir
                    / f"{dem_name}_profile_curvature.png",
                    label="Profile curvature (rad/m)",
                    robust=True,
                )
            )
            verification_figures.append(
                self._plot_layer(
                    dem_name,
                    layer_key="plan_curvature",
                    title=f"{dem_name} - plan curvature",
                    cmap="coolwarm",
                    output_path=self.verification_dir / f"{dem_name}_plan_curvature.png",
                    label="Plan curvature (rad/m)",
                    robust=True,
                )
            )
            report_figures.append(self._plot_at_risk_overlay(dem_name))

        cluster_info = self._build_exposure_clusters()
        cluster_csv = self._write_cluster_summary(cluster_info)
        infra_csv = self._write_infrastructure_summary()
        report_figures.append(self._plot_exposure_cluster_overview(cluster_info))
        report_figures.append(self._plot_cluster_zoom_panel(cluster_info))
        report_figures.append(self._plot_dem_exposure_agreement_matrix(cluster_info))
        report_figures.append(self._plot_infrastructure_summary())
        report_figures.extend(self._plot_standalone_sensitivity_charts())

        index_path = self._write_index(report_figures, verification_figures)
        log.info("  Map index: %s", index_path)
        return {
            "report_figures": report_figures,
            "verification_figures": verification_figures,
            "index": index_path,
            "tables": [cluster_csv, infra_csv],
            "cluster_csv": cluster_csv,
            "infrastructure_csv": infra_csv,
        }

    def _plot_slope_comparison(self) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        dem_names = list(self.dem_catalogue)
        fig, axes = plt.subplots(1, len(dem_names), figsize=(5.6 * len(dem_names), 5.2))
        if len(dem_names) == 1:
            axes = [axes]

        for ax, dem_name in zip(axes, dem_names):
            data, profile, extent = self._layer_data(dem_name, "slope")
            image = ax.imshow(data, extent=extent, origin="upper", cmap="magma", vmin=0, vmax=65)
            ax.set_title(f"{dem_name}\nSlope field")
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
            fig.colorbar(image, ax=ax, shrink=0.78, label="Slope (deg)")

        fig.suptitle("Slope Factor Maps Used for Infrastructure Exposure", fontweight="bold")
        fig.tight_layout()
        out_path = self.report_maps_dir / "slope_map_comparison.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        log.info("  Report map: %s", out_path.name)
        return out_path

    def _plot_elevation_hillshade(self, dem_name: str) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        elev, profile, extent = self._read_raster(self.dem_paths[dem_name])
        shade = self._hillshade(elev, cellsize=abs(profile["transform"].a))

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(shade, extent=extent, origin="upper", cmap="gray")
        image = ax.imshow(elev, extent=extent, origin="upper", cmap="terrain", alpha=0.42)
        ax.set_title(f"{dem_name} - elevation with hillshade")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        fig.colorbar(image, ax=ax, shrink=0.82, label="Elevation (m)")
        fig.tight_layout()

        out_path = self.verification_dir / f"{dem_name}_elevation_hillshade.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        log.info("  Verification map: %s", out_path.name)
        return out_path

    def _plot_layer(
        self,
        dem_name: str,
        layer_key: str,
        title: str,
        cmap: str,
        output_path: Path,
        label: str,
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = False,
        mask_negative: bool = False,
    ) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        data, profile, extent = self._layer_data(dem_name, layer_key)
        if mask_negative:
            data = np.where(data < 0, np.nan, data)
        if robust:
            vmin, vmax = self._robust_limits(data)

        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(
            data,
            extent=extent,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        fig.colorbar(image, ax=ax, shrink=0.82, label=label)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        log.info("  Verification map: %s", output_path.name)
        return output_path

    def _plot_at_risk_overlay(self, dem_name: str) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        slope, profile, extent = self._layer_data(dem_name, "slope")
        out_path = self.report_maps_dir / f"{dem_name}_at_risk_overlay.png"

        fig, ax = plt.subplots(figsize=(8.5, 7.2))
        image = ax.imshow(slope, extent=extent, origin="upper", cmap="magma", vmin=0, vmax=65)

        result = self.impact_results.get(dem_name, {})
        vector_path = result.get("at_risk_path")
        if vector_path and Path(vector_path).exists():
            try:
                import geopandas as gpd

                gdf = gpd.read_file(vector_path)
                if not gdf.empty:
                    gdf = gdf.to_crs(profile["crs"])
                    gdf.plot(ax=ax, facecolor="#fff2a8", edgecolor="#111111", linewidth=0.45, alpha=0.92)
                    gdf.centroid.plot(ax=ax, color="#ffffff", markersize=12, alpha=0.95, edgecolor="#111111")
            except Exception as exc:  # pragma: no cover - visual fallback only
                log.warning("Could not overlay at-risk buildings for %s: %s", dem_name, exc)

        ax.set_title(
            f"{dem_name} - at-risk buildings\n"
            f"slope-only={result.get('at_risk', 0)}, "
            f"slope+aspect={result.get('aspect_filtered_at_risk', 0)}"
        )
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        fig.colorbar(image, ax=ax, shrink=0.82, label="Slope (deg)")
        self._add_public_attribution(fig, osm=True)
        fig.tight_layout(rect=[0, 0.035, 1, 1])
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        log.info("  Report map: %s", out_path.name)
        return out_path

    def _plot_exposure_cluster_overview(self, cluster_info: dict[str, Any]) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        dem_name = next(iter(self.dem_catalogue))
        elev, profile, extent = self._read_raster(self.dem_paths[dem_name])
        shade = self._hillshade(elev, cellsize=abs(profile["transform"].a))
        points = cluster_info["points"]
        clusters = cluster_info["clusters"]

        fig, ax = plt.subplots(figsize=(9.2, 7.5))
        ax.imshow(shade, extent=extent, origin="upper", cmap="gray", alpha=0.95)
        colors = self._dem_colors()
        for name in self.dem_catalogue:
            subset = [item for item in points if item["dem"] == name]
            if subset:
                ax.scatter(
                    [item["x"] for item in subset],
                    [item["y"] for item in subset],
                    s=14,
                    color=colors[name],
                    label=name,
                    alpha=0.82,
                    edgecolors="white",
                    linewidths=0.25,
                )
        for idx, cluster in enumerate(clusters, start=1):
            circle = Circle(
                (cluster["x"], cluster["y"]),
                cluster["radius_m"],
                fill=False,
                edgecolor="#111111",
                linewidth=1.25,
            )
            ax.add_patch(circle)
            ax.text(
                cluster["x"],
                cluster["y"],
                str(idx),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
                bbox={"boxstyle": "circle,pad=0.25", "facecolor": "#111111", "edgecolor": "white"},
            )
        ax.set_title("At-Risk Infrastructure Clusters Across DEM Products")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="lower right", fontsize=8, framealpha=0.88)
        self._add_public_attribution(fig, osm=True)
        fig.tight_layout(rect=[0, 0.035, 1, 1])
        out_path = self.report_maps_dir / "exposure_cluster_overview.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        log.info("  Report map: %s", out_path.name)
        return out_path

    def _plot_cluster_zoom_panel(self, cluster_info: dict[str, Any]) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        clusters = cluster_info["clusters"][:6]
        points = cluster_info["points"]
        if not clusters:
            return self._write_empty_report_figure(
                "exposure_cluster_zoom_panel.png",
                "No at-risk building clusters were available for zoom panels.",
            )

        rgb_context = self._sentinel_context_path()
        fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.8))
        axes_flat = axes.ravel()
        colors = self._zoom_dem_colors()

        for ax, cluster in zip(axes_flat, clusters):
            half_width = max(cluster["radius_m"] * 1.35, 260.0)
            window = [
                cluster["x"] - half_width,
                cluster["x"] + half_width,
                cluster["y"] - half_width,
                cluster["y"] + half_width,
            ]
            self._draw_context_background(ax, window, rgb_context)
            for name in self.dem_catalogue:
                subset = [
                    item
                    for item in points
                    if item["dem"] == name
                    and window[0] <= item["x"] <= window[1]
                    and window[2] <= item["y"] <= window[3]
                ]
                if subset:
                    ax.scatter(
                        [item["x"] for item in subset],
                        [item["y"] for item in subset],
                        s=36,
                        color=colors[name],
                        label=name,
                        alpha=0.9,
                        edgecolors="white",
                        linewidths=1.1,
                    )
            ax.set_xlim(window[0], window[1])
            ax.set_ylim(window[2], window[3])
            ax.set_title(
                f"Cluster {cluster['rank']} | union={cluster['union_count']} | "
                f"AW3D30={cluster['counts'].get('ALOS_AW3D30_30m', 0)}"
            )
            ax.set_xticks([])
            ax.set_yticks([])

        for ax in axes_flat[len(clusters):]:
            ax.axis("off")

        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.suptitle("Zoomed Exposure Clusters with Sentinel-2 or Hillshade Context", fontweight="bold")
        self._add_public_attribution(fig, osm=True, sentinel=rgb_context is not None)
        fig.tight_layout(rect=[0, 0.07, 1, 0.96])
        out_path = self.report_maps_dir / "exposure_cluster_zoom_panel.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        log.info("  Report map: %s", out_path.name)
        return out_path

    def _plot_dem_exposure_agreement_matrix(self, cluster_info: dict[str, Any]) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        clusters = cluster_info["clusters"][:6]
        dem_names = list(self.dem_catalogue)
        if not clusters:
            return self._write_empty_report_figure(
                "dem_exposure_agreement_matrix.png",
                "No cluster-level DEM agreement matrix could be computed.",
            )

        matrix = np.array(
            [[cluster["counts"].get(dem_name, 0) for dem_name in dem_names] for cluster in clusters],
            dtype=float,
        )
        fig, ax = plt.subplots(figsize=(9.4, 5.6))
        image = ax.imshow(matrix, cmap="YlOrRd")
        ax.set_xticks(np.arange(len(dem_names)))
        ax.set_xticklabels(dem_names, rotation=18, ha="right")
        ax.set_yticks(np.arange(len(clusters)))
        ax.set_yticklabels([f"Cluster {cluster['rank']}" for cluster in clusters])
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, f"{int(matrix[row, col])}", ha="center", va="center", color="#111111")
        fig.colorbar(image, ax=ax, shrink=0.82, label="At-risk building count")
        ax.set_title("DEM Agreement and Divergence by Exposure Cluster")
        self._add_public_attribution(fig, osm=True)
        fig.tight_layout(rect=[0, 0.035, 1, 1])
        out_path = self.report_maps_dir / "dem_exposure_agreement_matrix.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        log.info("  Report map: %s", out_path.name)
        return out_path

    def _plot_infrastructure_summary(self) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        dem_names = list(self.dem_catalogue)
        buildings = [self.impact_results[name].get("at_risk", 0) for name in dem_names]
        roads = [
            self.impact_results[name].get("infrastructure", {}).get("roads_at_risk_m", 0.0) / 1000.0
            for name in dem_names
        ]
        utilities = [
            sum(
                self.impact_results[name].get("infrastructure", {}).get(key, 0)
                for key in (
                    "power_at_risk",
                    "water_utility_at_risk",
                    "public_facility_at_risk",
                )
            )
            for name in dem_names
        ]

        x = np.arange(len(dem_names))
        fig, ax = plt.subplots(figsize=(10.2, 5.8))
        include_utilities = any(value > 0 for value in utilities)
        if include_utilities:
            building_x = x - 0.25
            road_x = x
            utility_x = x + 0.25
            width = 0.25
        else:
            building_x = x - 0.18
            road_x = x + 0.18
            utility_x = None
            width = 0.34

        ax.bar(building_x, buildings, width=width, label="Buildings", color="#4b6f8f")
        ax2 = ax.twinx()
        ax2.bar(road_x, roads, width=width, label="Road length (km)", color="#c4843d")
        if include_utilities and utility_x is not None:
            ax.bar(utility_x, utilities, width=width, label="Utility/public features", color="#4f8f5b")
        ax.set_xticks(x)
        ax.set_xticklabels(dem_names, rotation=12)
        ax.set_ylabel("Feature count")
        ax2.set_ylabel("Exposed road length (km)")
        ax.set_title("Supporting Civil Infrastructure Exposure Metrics")
        handles_1, labels_1 = ax.get_legend_handles_labels()
        handles_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left")
        ax.grid(axis="y", alpha=0.2)
        self._add_public_attribution(fig, osm=True)
        fig.tight_layout(rect=[0, 0.035, 1, 1])
        out_path = self.report_maps_dir / "infrastructure_exposure_summary.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        log.info("  Report map: %s", out_path.name)
        return out_path

    def _plot_standalone_sensitivity_charts(self) -> list[Path]:
        outputs: list[Path] = []
        for dem_name in self.dem_catalogue:
            outputs.append(self._plot_slope_histogram(dem_name))
        outputs.append(self._plot_profile_curvature_chart())
        outputs.append(self._plot_infrastructure_risk_chart())
        return outputs

    def _plot_slope_histogram(self, dem_name: str) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        slope, _, _ = self._layer_data(dem_name, "slope")
        valid = slope[np.isfinite(slope)]
        result = self.impact_results.get(dem_name, {})
        out_path = self.report_maps_dir / f"{dem_name}_slope_histogram.png"

        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        ax.hist(valid, bins=60, color=self._dem_colors().get(dem_name, "#4b6f8f"), alpha=0.82)
        ax.axvline(15.0, color="#b00020", linestyle="--", linewidth=1.4, label="15 deg threshold")
        ax.axvline(17.0, color="#214f9c", linestyle=":", linewidth=1.8, label="Reference mean context")
        ax.set_title(f"{dem_name} Slope Distribution")
        ax.set_xlabel("Slope (deg)")
        ax.set_ylabel("Pixel count")
        ax.text(
            0.98,
            0.95,
            f"mean={np.nanmean(valid):.2f}\nstd={np.nanstd(valid):.2f}\nat-risk={result.get('at_risk', 0)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.82},
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
        self._add_public_attribution(fig, osm=True)
        fig.tight_layout(rect=[0, 0.035, 1, 1])
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def _plot_profile_curvature_chart(self) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        dem_names = list(self.dem_catalogue)
        values = []
        for dem_name in dem_names:
            data, _, _ = self._layer_data(dem_name, "profile_curvature")
            values.append(float(np.nanstd(data)))

        out_path = self.report_maps_dir / "profile_curvature_lineage_sensitivity.png"
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        ax.bar(dem_names, values, color=[self._dem_colors().get(name, "#4b6f8f") for name in dem_names])
        ax.axhline(0.003, color="#222222", linestyle="--", linewidth=1.2, label="Reference context 0.003")
        ax.set_title("Profile Curvature Processing-Lineage Sensitivity")
        ax.set_ylabel("Standard deviation (rad/m)")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def _plot_infrastructure_risk_chart(self) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        dem_names = list(self.dem_catalogue)
        counts = [self.impact_results[name].get("at_risk", 0) for name in dem_names]
        aspect_counts = [
            self.impact_results[name].get("aspect_filtered_at_risk", 0) for name in dem_names
        ]
        pct = [self.impact_results[name].get("pct_at_risk", 0.0) for name in dem_names]
        x = np.arange(len(dem_names))
        out_path = self.report_maps_dir / "infrastructure_risk_by_dem_product.png"

        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        bars = ax.bar(x, counts, color="#4b6f8f", alpha=0.9, label="At-risk buildings")
        ax.plot(x, aspect_counts, color="#111111", marker="o", linewidth=1.8, label="Slope+aspect subset")
        for idx, value in enumerate(pct):
            ax.text(idx, counts[idx] + 2, f"{value:.1f}%", ha="center", fontsize=9)
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(dem_names, rotation=12)
        ax.set_ylabel("Building count")
        ax.set_title("Infrastructure at Landslide Risk by DEM Product")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(fontsize=9)
        self._add_public_attribution(fig, osm=True)
        fig.tight_layout(rect=[0, 0.035, 1, 1])
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def _layer_data(
        self, dem_name: str, layer_key: str
    ) -> tuple[np.ndarray, dict[str, Any], list[float]]:
        path = Path(self.topo_layers[dem_name][layer_key])
        return self._read_raster(path)

    @staticmethod
    def _read_raster(path: Path) -> tuple[np.ndarray, dict[str, Any], list[float]]:
        # pyrefly: ignore [missing-import]
        import rasterio

        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            profile = src.profile.copy()
            transform = src.transform
            width = src.width
            height = src.height
            left = transform.c
            right = transform.c + transform.a * width
            top = transform.f
            bottom = transform.f + transform.e * height
            extent = [min(left, right), max(left, right), min(bottom, top), max(bottom, top)]
        return data, profile, extent

    @staticmethod
    def _read_rgb_raster(path: Path) -> tuple[np.ndarray, list[float]]:
        # pyrefly: ignore [missing-import]
        import rasterio

        with rasterio.open(path) as src:
            data = src.read([1, 2, 3]).astype(np.float32)
            transform = src.transform
            width = src.width
            height = src.height
            left = transform.c
            right = transform.c + transform.a * width
            top = transform.f
            bottom = transform.f + transform.e * height
            extent = [min(left, right), max(left, right), min(bottom, top), max(bottom, top)]

        rgb = np.moveaxis(data, 0, -1)
        valid = rgb[np.isfinite(rgb)]
        if valid.size:
            lo, hi = np.percentile(valid, [2, 98])
            if hi > lo:
                rgb = (rgb - lo) / (hi - lo)
        rgb = np.clip(rgb, 0, 1)
        return rgb, extent

    @staticmethod
    def _hillshade(
        elev: np.ndarray,
        cellsize: float,
        azimuth: float = 315.0,
        altitude: float = 45.0,
    ) -> np.ndarray:
        filled = np.asarray(elev, dtype=np.float64)
        if np.isnan(filled).any():
            median = np.nanmedian(filled)
            filled = np.where(np.isnan(filled), median, filled)
        dy, dx = np.gradient(filled, cellsize, cellsize)
        slope = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))
        aspect = np.arctan2(-dx, dy)
        az = np.deg2rad(azimuth)
        alt = np.deg2rad(altitude)
        shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
        return np.clip((shaded + 1.0) / 2.0, 0.0, 1.0)

    @staticmethod
    def _robust_limits(data: np.ndarray) -> tuple[float, float]:
        valid = data[np.isfinite(data)]
        if valid.size == 0:
            return -1.0, 1.0
        lower, upper = np.percentile(valid, [2, 98])
        bound = max(abs(float(lower)), abs(float(upper)))
        if bound == 0:
            bound = 1.0
        return -bound, bound

    def _write_index(self, report_figures: list[Path], verification_figures: list[Path]) -> Path:
        index_path = self.report_dir / "map_index.md"
        lines = [
            "# No-GIS Map Index",
            "",
            "These PNG files are generated directly by Python from GeoTIFF and GeoPackage outputs.",
            "Report figures are intended for the academic report; verification figures are layer checks.",
            "",
            "## Report Figures",
            "",
        ]
        for path in report_figures:
            lines.append(f"- `{path.relative_to(self.report_dir.parent).as_posix()}`")
        lines.extend(["", "## Verification Figures", ""])
        for path in verification_figures:
            lines.append(f"- `{path.relative_to(self.report_dir.parent).as_posix()}`")
        lines.append("")
        index_path.write_text("\n".join(lines), encoding="utf-8")
        return index_path

    def _build_exposure_clusters(self) -> dict[str, Any]:
        points = self._at_risk_points()
        clusters = self._cluster_points(points)
        return {"points": points, "clusters": clusters}

    def _at_risk_points(self) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        first_dem = next(iter(self.dem_catalogue))
        _, profile, _ = self._layer_data(first_dem, "slope")
        try:
            import geopandas as gpd
        except ImportError:  # pragma: no cover
            return points

        for dem_name, result in self.impact_results.items():
            path = result.get("at_risk_path")
            if not path or not Path(path).exists():
                continue
            gdf = gpd.read_file(path)
            if gdf.empty:
                continue
            gdf = gdf.to_crs(profile["crs"])
            for geom in gdf.geometry:
                centroid = geom.centroid
                points.append(
                    {
                        "dem": dem_name,
                        "x": float(centroid.x),
                        "y": float(centroid.y),
                        "key": f"{round(centroid.x, 1)}:{round(centroid.y, 1)}",
                    }
                )
        return points

    def _cluster_points(
        self,
        points: list[dict[str, Any]],
        radius_m: float = 150.0,
        max_clusters: int = 6,
    ) -> list[dict[str, Any]]:
        if not points:
            return []

        coords = np.array([[item["x"], item["y"]] for item in points], dtype=float)
        remaining = set(range(len(points)))
        raw_clusters: list[list[int]] = []
        while remaining:
            seed = remaining.pop()
            cluster = {seed}
            queue = [seed]
            while queue:
                current = queue.pop()
                distances = np.sqrt(((coords - coords[current]) ** 2).sum(axis=1))
                neighbors = {idx for idx in remaining if distances[idx] <= radius_m}
                remaining -= neighbors
                cluster |= neighbors
                queue.extend(neighbors)
            raw_clusters.append(sorted(cluster))

        clusters: list[dict[str, Any]] = []
        dem_names = list(self.dem_catalogue)
        for members in raw_clusters:
            member_points = [points[idx] for idx in members]
            xs = np.array([item["x"] for item in member_points])
            ys = np.array([item["y"] for item in member_points])
            counts = {
                dem_name: len({item["key"] for item in member_points if item["dem"] == dem_name})
                for dem_name in dem_names
            }
            clusters.append(
                {
                    "x": float(xs.mean()),
                    "y": float(ys.mean()),
                    "radius_m": max(radius_m, float(np.sqrt(((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2).max())) + 70.0),
                    "counts": counts,
                    "union_count": len({item["key"] for item in member_points}),
                }
            )

        clusters.sort(
            key=lambda item: (
                item["counts"].get("ALOS_AW3D30_30m", 0),
                item["union_count"],
            ),
            reverse=True,
        )
        for rank, cluster in enumerate(clusters[:max_clusters], start=1):
            cluster["rank"] = rank
        return clusters[:max_clusters]

    def _write_cluster_summary(self, cluster_info: dict[str, Any]) -> Path:
        import csv

        path = self.report_dir / "cluster_exposure_summary.csv"
        dem_names = list(self.dem_catalogue)
        fieldnames = ["Cluster", "Union_AtRisk_Buildings", "Easting", "Northing"] + dem_names
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for cluster in cluster_info["clusters"]:
                row = {
                    "Cluster": cluster["rank"],
                    "Union_AtRisk_Buildings": cluster["union_count"],
                    "Easting": f"{cluster['x']:.1f}",
                    "Northing": f"{cluster['y']:.1f}",
                }
                row.update(cluster["counts"])
                writer.writerow(row)
        return path

    def _write_infrastructure_summary(self) -> Path:
        import csv

        path = self.report_dir / "infrastructure_exposure_summary.csv"
        fieldnames = [
            "DEM",
            "Buildings_AtRisk",
            "Buildings_Aspect_Filtered",
            "Roads_AtRisk_m",
            "Roads_Aspect_Filtered_m",
            "Power_AtRisk",
            "Water_Utility_AtRisk",
            "Public_Facility_AtRisk",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for dem_name in self.dem_catalogue:
                result = self.impact_results.get(dem_name, {})
                infra = result.get("infrastructure", {})
                writer.writerow(
                    {
                        "DEM": dem_name,
                        "Buildings_AtRisk": result.get("at_risk", 0),
                        "Buildings_Aspect_Filtered": result.get("aspect_filtered_at_risk", 0),
                        "Roads_AtRisk_m": f"{infra.get('roads_at_risk_m', 0.0):.1f}",
                        "Roads_Aspect_Filtered_m": f"{infra.get('roads_aspect_filtered_m', 0.0):.1f}",
                        "Power_AtRisk": infra.get("power_at_risk", 0),
                        "Water_Utility_AtRisk": infra.get("water_utility_at_risk", 0),
                        "Public_Facility_AtRisk": infra.get("public_facility_at_risk", 0),
                    }
                )
        return path

    def _sentinel_context_path(self) -> Path | None:
        if not self.bbox:
            return None
        out_path = self.report_maps_dir / "sentinel2_true_color_context.tif"
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        try:
            # pyrefly: ignore [missing-import]
            import ee
            # pyrefly: ignore [missing-import]
            import geemap

            try:
                ee.Initialize(project="geoe431-hazar")
            except Exception:
                ee.Initialize()

            roi = ee.Geometry.Rectangle(
                [self.bbox["west"], self.bbox["south"], self.bbox["east"], self.bbox["north"]]
            )
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(roi)
                .filterDate("2024-06-01", "2024-09-30")
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 35))
                .map(self._mask_sentinel2_clouds)
            )
            image = collection.median().clip(roi).reproject(crs="EPSG:32636", scale=10)
            geemap.ee_export_image(
                image,
                filename=str(out_path),
                scale=10,
                region=roi,
                file_per_band=False,
            )
            if out_path.exists() and out_path.stat().st_size > 0:
                log.info("  Sentinel-2 context image: %s", out_path.name)
                return out_path
        except Exception as exc:  # pragma: no cover - network/auth fallback
            log.warning("  Sentinel-2 context fetch failed; using hillshade fallback: %s", exc)
        return None

    @staticmethod
    def _mask_sentinel2_clouds(image: Any) -> Any:
        """Mask Sentinel-2 opaque cloud, cirrus, cloud shadow, snow, and cloud classes."""
        qa = image.select("QA60")
        opaque_cloud_bit = 1 << 10
        cirrus_bit = 1 << 11
        qa_mask = qa.bitwiseAnd(opaque_cloud_bit).eq(0).And(
            qa.bitwiseAnd(cirrus_bit).eq(0)
        )

        scl = image.select("SCL")
        scl_mask = (
            scl.neq(3)
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )
        return image.updateMask(qa_mask).updateMask(scl_mask).select(["B4", "B3", "B2"])

    def _draw_context_background(self, ax: Any, window: list[float], rgb_context: Path | None) -> None:
        if rgb_context and rgb_context.exists():
            try:
                rgb, extent = self._read_rgb_raster(rgb_context)
                ax.imshow(rgb, extent=extent, origin="upper")
                return
            except Exception as exc:  # pragma: no cover
                log.warning("  Could not render Sentinel-2 context: %s", exc)

        dem_name = next(iter(self.dem_catalogue))
        elev, profile, extent = self._read_raster(self.dem_paths[dem_name])
        shade = self._hillshade(elev, cellsize=abs(profile["transform"].a))
        ax.imshow(shade, extent=extent, origin="upper", cmap="gray")

    def _write_empty_report_figure(self, filename: str, message: str) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        ax.axis("off")
        out_path = self.report_maps_dir / filename
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    @staticmethod
    def _add_public_attribution(
        fig: Any,
        *,
        osm: bool = False,
        sentinel: bool = False,
    ) -> None:
        """Add compact source notes only to public figures that use those data."""
        notes = []
        if osm:
            notes.append("© OpenStreetMap contributors")
        if sentinel:
            notes.append("Copernicus Sentinel-2 / ESA")
        if notes:
            fig.text(
                0.99,
                0.01,
                " | ".join(notes),
                ha="right",
                va="bottom",
                fontsize=7,
                color="#444444",
            )

    @staticmethod
    def _dem_colors() -> dict[str, str]:
        return {
            "SRTM_30m": "#2f6f9f",
            "NASADEM_30m": "#b45f3c",
            "ALOS_AW3D30_30m": "#4f8f5b",
        }

    @staticmethod
    def _zoom_dem_colors() -> dict[str, str]:
        return {
            "SRTM_30m": "#ff00ff",
            "NASADEM_30m": "#00e5ff",
            "ALOS_AW3D30_30m": "#ffff00",
        }
