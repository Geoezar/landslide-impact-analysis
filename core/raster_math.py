"""
core/raster_math.py
===================
Computes topographic derivative layers (Slope, Aspect, Profile Curvature,
Plan Curvature) from a single DEM GeoTIFF.

Physical / methodological rationale
-------------------------------------
Slope  = arctan(√(dZ/dx² + dZ/dy²))
  The most critical parameter in every landslide model (used by 97.9% of
  studies; Süzen & Kaya 2012, Fig.3). In Devrek, landslides cluster at
  5°–17° on Çaycuma Formation slopes (Yilmaz et al. 2012).

Aspect = arctan2(dZ/dy, dZ/dx)  → 0°–360° azimuth
  SW/W/NW aspects (180°–315°) dominate Devrek landslide inventory
  (Yilmaz et al. 2012). We also reclassify into 8 octants for comparison.

Profile Curvature  (∂²Z/∂s²  along slope direction)
  Controls flow acceleration and erosion rate.
  CRITICAL TEST: Süzen & Kaya (2012) found profile & plan curvature
  "statistically insignificant at 25m grid spacing." This project tests
  whether DEM processing lineage can change curvature variance
  relative to the 12.5m Devrek reference.

Plan Curvature  (∂²Z/∂n²  perpendicular to slope)
  Controls flow convergence/divergence → soil moisture accumulation.
  Values for Devrek from Yilmaz et al. (2012): -1.8 to +1.5 rad/m.

All calculations use richdem for hydrologically-correct terrain analysis
(as specified in Sistem Mimarisi) with rasterio for I/O.
"""

import logging
import numpy as np
from pathlib import Path

log = logging.getLogger("raster_math")


class RasterMath:
    """
    Computes and saves topographic derivative layers for a given DEM.

    Parameters
    ----------
    dem_path  : Path   – input GeoTIFF (elevation in metres, UTM CRS)
    output_dir: Path   – directory where derivative GeoTIFFs will be saved
    """

    def __init__(self, dem_path: Path, output_dir: Path):
        self.dem_path   = Path(dem_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._dem_name  = self.dem_path.stem   # e.g. "SRTM_30m"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all(self) -> dict:
        """
        Computes all derivative layers and saves them as GeoTIFF.

        Returns
        -------
        dict  {layer_name: Path}
            e.g. {"slope": Path(...), "aspect": Path(...), ...}
        """
        log.info("[%s] Loading DEM ...", self._dem_name)
        elev, profile = self._load_dem()

        log.info("[%s] Computing Slope ...", self._dem_name)
        slope  = self._compute_slope(elev, profile)
        slope_path = self._save(slope, profile, "slope", unit="degrees")

        log.info("[%s] Computing Aspect ...", self._dem_name)
        aspect = self._compute_aspect(elev, profile, slope)
        aspect_path = self._save(aspect, profile, "aspect", unit="degrees")

        log.info("[%s] Computing Curvatures ...", self._dem_name)
        prof_curv, plan_curv = self._compute_curvature(elev, profile)
        prof_path = self._save(prof_curv, profile, "profile_curvature",
                               unit="rad_per_m")
        plan_path = self._save(plan_curv, profile, "plan_curvature",
                               unit="rad_per_m")

        # ── Statistical summary (key for academic report) ────────────
        self._log_stats(slope, aspect, prof_curv, plan_curv)

        return {
            "slope"            : slope_path,
            "aspect"           : aspect_path,
            "profile_curvature": prof_path,
            "plan_curvature"   : plan_path,
            # Also carry arrays in memory for the vector_analyzer
            "_arrays": {
                "slope"            : slope,
                "aspect"           : aspect,
                "profile_curvature": prof_curv,
                "plan_curvature"   : plan_curv,
            },
            "_profile": profile,   # rasterio profile for spatial ops
        }

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_dem(self) -> tuple:
        """
        Opens the GeoTIFF and returns (elevation_array, rasterio_profile).
        NoData pixels are set to NaN for safe numpy operations.
        """
        import rasterio
        with rasterio.open(self.dem_path) as src:
            profile = src.profile.copy()
            elev    = src.read(1).astype(np.float32)
            nodata  = src.nodata
            if nodata is not None:
                elev[elev == nodata] = np.nan
            log.info(
                "  [%s] Shape=%s | CRS=%s | Pixel size=%s m",
                self._dem_name, elev.shape,
                src.crs, abs(src.transform.a),
            )
        return elev, profile

    def _save(self, array: np.ndarray, profile: dict,
              layer_name: str, unit: str) -> Path:
        """Saves a float32 array as GeoTIFF, returns the path."""
        import rasterio
        out_path = self.output_dir / f"{self._dem_name}_{layer_name}.tif"
        out_profile = profile.copy()
        out_profile.update(dtype="float32", count=1, nodata=np.nan)
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(array.astype(np.float32), 1)
        log.info("  Saved: %s  [unit=%s]", out_path.name, unit)
        return out_path

    # ------------------------------------------------------------------
    # Slope
    # ------------------------------------------------------------------

    def _compute_slope(self, elev: np.ndarray, profile: dict) -> np.ndarray:
        """
        Calculates slope in degrees using richdem.TerrainAttribute.

        richdem implements Horn's (1981) finite-difference method:
            dZ/dx = ((z3+2z6+z9) - (z1+2z4+z7)) / (8 * cellsize)
        This is the same algorithm used in ArcGIS and QGIS, ensuring
        our results are directly comparable to Yilmaz et al. (2012).

        Validation target (from paper):
            mean = 17°, std = 10°, range = 0°–63°
        """
        try:
            import richdem as rd
            cellsize = abs(profile["transform"].a)   # pixel width in metres
            rda      = rd.rdarray(elev, no_data=np.nan)
            rda.geotransform = [
                profile["transform"].c, cellsize, 0,
                profile["transform"].f, 0, -cellsize,
            ]
            slope_rad = rd.TerrainAttribute(rda, attrib="slope_riserun")
            slope_deg = np.degrees(np.arctan(np.array(slope_rad)))
            log.info("  Slope computed via richdem (Horn 1981).")
            return slope_deg.astype(np.float32)

        except ImportError:
            log.warning("richdem not available - falling back to numpy gradient.")
            return self._slope_numpy(elev, profile)

    def _slope_numpy(self, elev: np.ndarray, profile: dict) -> np.ndarray:
        """
        Numpy fallback: central-difference gradient.
        Less accurate at edges but sufficient for exploratory analysis.
        """
        cellsize = abs(profile["transform"].a)
        dy, dx   = np.gradient(elev, cellsize, cellsize)
        slope    = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        log.info("  Slope computed via numpy gradient (fallback).")
        return slope.astype(np.float32)

    # ------------------------------------------------------------------
    # Aspect
    # ------------------------------------------------------------------

    def _compute_aspect(
        self,
        elev: np.ndarray,
        profile: dict,
        slope: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calculates aspect (0°–360° azimuth, -1 for flat).

        Flat cells (slope < 0.01°) are set to -1, matching the convention
        in Yilmaz et al. (2012) where aspect mean=200°, std=112°.

        NW-facing slopes (270°–360°) are the highest-risk aspect class
        in the Devrek inventory → we will flag these in the reporter.
        """
        try:
            import richdem as rd
            cellsize = abs(profile["transform"].a)
            rda      = rd.rdarray(elev, no_data=np.nan)
            rda.geotransform = [
                profile["transform"].c, cellsize, 0,
                profile["transform"].f, 0, -cellsize,
            ]
            aspect = np.array(rd.TerrainAttribute(rda, attrib="aspect"))
            log.info("  Aspect computed via richdem.")
        except ImportError:
            aspect = self._aspect_numpy(elev, profile)

        # Mark flat areas as -1 (no aspect), matching Yilmaz et al. (2012).
        # Reuse the already computed slope when available to avoid a second
        # full-raster gradient pass per DEM.
        flat_source = slope if slope is not None else self._slope_numpy(elev, profile)
        aspect[flat_source < 0.01] = -1.0
        return aspect.astype(np.float32)

    def _aspect_numpy(self, elev: np.ndarray, profile: dict) -> np.ndarray:
        """Numpy fallback for aspect."""
        cellsize = abs(profile["transform"].a)
        dy, dx   = np.gradient(elev, cellsize, cellsize)
        # arctan2 gives math convention; convert to geographic azimuth
        aspect   = np.degrees(np.arctan2(dy, -dx))
        aspect   = 90.0 - aspect
        aspect[aspect < 0]   += 360.0
        aspect[aspect > 360] -= 360.0
        log.info("  Aspect computed via numpy (fallback).")
        return aspect.astype(np.float32)

    # ------------------------------------------------------------------
    # Curvature (the critical test vs Süzen & Kaya 2012)
    # ------------------------------------------------------------------

    def _compute_curvature(
        self, elev: np.ndarray, profile: dict
    ) -> tuple:
        """
        Computes Profile and Plan Curvature via 3×3 polynomial fit
        (Zevenbergen & Thorne 1987) – the standard used in GIS literature.

        This is the KEY scientific test of our pipeline:
        ┌──────────────────────────────────────────────────────────────┐
        │ Süzen & Kaya (2012) finding:                                 │
        │   Profile & plan curvatures were statistically INSIGNIFICANT │
        │   at 25m grid spacing. We test product-lineage effects.       │
        └──────────────────────────────────────────────────────────────┘

        Expected output range from Yilmaz et al. (2012) using 12.5m DEM:
            Profile curvature : -0.04 to +0.03 rad/m  (std ≈ 0.003)
            Plan curvature    : -1.8  to +1.5  rad/m  (std ≈ 0.02)

        If a cloud DEM product produces std values significantly different
        from these, the result is product-lineage sensitivity evidence.
        """
        cellsize = abs(profile["transform"].a)
        rows, cols = elev.shape

        prof_curv = np.full((rows, cols), np.nan, dtype=np.float32)
        plan_curv = np.full((rows, cols), np.nan, dtype=np.float32)

        # Pad to allow 3×3 windows at borders
        elev_pad = np.pad(elev, 1, mode="edge")

        log.info("  Computing curvature at %dx%d grid (GSD=%.0fm) - vectorized ...",
                 rows, cols, cellsize)

        # ── Vectorized Zevenbergen & Thorne (1987) ──────────────────────
        # Extract all 3×3 neighbours as 2D arrays simultaneously.
        # This avoids a Python loop over ~2M pixels (ALOS case) which
        # would take ~10 minutes; the vectorized version runs in seconds.
        #
        # Neighbourhood layout (p = padded elevation array):
        #   z1 z2 z3      p[0:-2, 0:-2]  p[0:-2, 1:-1]  p[0:-2, 2:]
        #   z4 z5 z6  →   p[1:-1, 0:-2]  p[1:-1, 1:-1]  p[1:-1, 2:]
        #   z7 z8 z9      p[2:,   0:-2]  p[2:,   1:-1]  p[2:,   2:]

        z1 = elev_pad[0:-2, 0:-2]
        z2 = elev_pad[0:-2, 1:-1]
        z3 = elev_pad[0:-2, 2:  ]
        z4 = elev_pad[1:-1, 0:-2]
        z5 = elev_pad[1:-1, 1:-1]   # centre (== elev)
        z6 = elev_pad[1:-1, 2:  ]
        z7 = elev_pad[2:,   0:-2]
        z8 = elev_pad[2:,   1:-1]
        z9 = elev_pad[2:,   2:  ]

        cs2 = cellsize ** 2
        cs2x = 2.0 * cellsize

        # Polynomial surface coefficients (Zevenbergen & Thorne notation)
        D = ((z4 + z6) / 2.0 - z5) / cs2          # ∂²Z/∂x²
        E = ((z2 + z8) / 2.0 - z5) / cs2          # ∂²Z/∂y²
        F = (-z1 + z3 + z7 - z9) / (4.0 * cs2)   # ∂²Z/∂x∂y
        G = (-z4 + z6) / cs2x                     # ∂Z/∂x
        H = (z2 - z8)  / cs2x                     # ∂Z/∂y

        denom_sq = G**2 + H**2

        # Mask: flat cells and NaN centres
        flat_mask = (denom_sq == 0) | np.isnan(z5)

        # Safe division: set denominator to 1 where masked (result masked later)
        denom_safe = np.where(flat_mask, 1.0, denom_sq)

        # Profile curvature  (concave up = negative → accelerates flow)
        pc = (
            -2.0 * (D * G**2 + E * H**2 + F * G * H)
            / (denom_safe * np.sqrt(1.0 + denom_safe))
        )

        # Plan curvature  (negative = converging flow → higher pore pressure)
        lc = (
            2.0 * (D * H**2 + E * G**2 - F * G * H)
            / denom_safe
        )

        # Apply mask
        pc[flat_mask] = np.nan
        lc[flat_mask] = np.nan

        prof_curv[:, :] = pc.astype(np.float32)
        plan_curv[:, :] = lc.astype(np.float32)

        log.info("  Curvature computed (Zevenbergen & Thorne 1987, vectorized).")
        return prof_curv, plan_curv

    # ------------------------------------------------------------------
    # Statistical summary (academic report input)
    # ------------------------------------------------------------------

    def _log_stats(self, slope, aspect, prof_curv, plan_curv):
        """
        Logs descriptive statistics for comparison against ground truth.

        Reference (Yilmaz et al. 2012 – 12.5m DEM):
            Slope  : mean=17°, std=10°, range 0°–63°
            Aspect : mean=200°, std=112°
            Prof.Curv: mean≈0, std≈0.003 rad/m, range -0.04 to +0.03
            Plan Curv: mean≈0, std≈0.02 rad/m, range -1.8 to +1.5
        """
        def stats(arr, name):
            valid = arr[~np.isnan(arr)]
            log.info(
                "  %-22s mean=%6.3f  std=%6.3f  "
                "min=%7.3f  max=%7.3f",
                name, valid.mean(), valid.std(),
                valid.min(), valid.max(),
            )

        log.info("-" * 65)
        log.info("  [%s] STATISTICAL SUMMARY", self._dem_name)
        log.info("-" * 65)
        stats(slope,     "Slope (deg)")
        stats(aspect[aspect >= 0], "Aspect (deg) [excl. flat]")
        stats(prof_curv, "Profile Curvature (rad/m)")
        stats(plan_curv, "Plan Curvature    (rad/m)")

        # ── Validation flag: compare slope mean to Yilmaz et al. ────
        valid_slope = slope[~np.isnan(slope)]
        ref_mean, ref_std = 17.0, 10.0
        deviation = abs(valid_slope.mean() - ref_mean)
        if deviation < 3.0:
            log.info(
                "  [OK] Slope mean (%.1f deg) within 3 deg of Yilmaz et al."
                " (2012) reference (%.1f deg)", valid_slope.mean(), ref_mean
            )
        else:
            log.warning(
                "  [!] Slope mean (%.1f deg) deviates %.1f deg from reference "
                "(%.1f deg) - check bbox or reproject settings.",
                valid_slope.mean(), deviation, ref_mean
            )

        # ── Curvature resolution sensitivity test ───────────────────
        valid_pc = prof_curv[~np.isnan(prof_curv)]
        log.info(
            "  Profile Curv std=%.5f  "
            "(Suzen & Kaya 2012 reference at 12.5m: ~0.003)",
            valid_pc.std()
        )
        if valid_pc.std() < 0.001:
            log.info(
                "  [OK] Very low curvature std supports Suzen & Kaya "
                "(2012) finding: curvature insignificant at this GSD."
            )
        log.info("-" * 65)
