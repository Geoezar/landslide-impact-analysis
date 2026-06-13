"""
core/data_fetcher.py
====================
Downloads Digital Elevation Models from Google Earth Engine to local GeoTIFF
files for offline topographic analysis.

Active catalogue rationale:
  SRTM_30m    - baseline 30m SRTM C-band InSAR DSM.
  NASADEM_30m - reprocessed SRTM 30m DEM using auxiliary ASTER GDEM,
                ICESat GLAS, and PRISM inputs.
  ALOS_AW3D30_30m - cloud-accessible ALOS PRISM AW3D30 optical-stereo DSM.

The previous ASTER GED asset was removed because it is an emissivity/LST
product, not a defensible optical-stereo DEM input for this analysis.
"""

import logging
from pathlib import Path

log = logging.getLogger("data_fetcher")


class DataFetcher:
    """
    Fetches DEMs from Google Earth Engine and exports them as GeoTIFF.

    Parameters
    ----------
    bbox:
        {"west", "south", "east", "north"} in WGS84 decimal degrees.
    dem_catalogue:
        Catalogue of DEMs defined in main.py.
    output_dir:
        Directory where GeoTIFF files will be saved.
    """

    def __init__(self, bbox: dict, dem_catalogue: dict, output_dir: Path):
        self.bbox = bbox
        self.catalogue = dem_catalogue
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ee = None

    def fetch_all(self) -> dict:
        """
        Download each DEM in the catalogue.

        Returns
        -------
        dict
            {dem_name: Path} mapping to downloaded GeoTIFF files.
        """
        self._init_gee()
        results = {}
        for dem_name, cfg in self.catalogue.items():
            log.info(
                "Fetching %s (%s, %dm GSD) ...",
                dem_name,
                cfg["sensor_type"],
                cfg["gsd_m"],
            )
            path = self._fetch_single(dem_name, cfg)
            results[dem_name] = path
            log.info("  saved: %s", path)
        return results

    def _init_gee(self) -> None:
        """
        Authenticates and initializes the Earth Engine API.
        """
        try:
            import ee
            self._ee = ee
        except ImportError as exc:
            raise ImportError(
                "earthengine-api not installed. Run: uv pip install earthengine-api"
            ) from exc

        try:
            ee.Initialize(project="geoe431-hazar")
            log.info("GEE initialised with existing credentials.")
        except Exception:
            log.warning("No cached credentials found. Launching browser auth ...")
            ee.Authenticate()
            ee.Initialize(project="geoe431-hazar")
            log.info("GEE authentication successful.")

    def _build_roi(self):
        """Convert bbox dict to an ee.Geometry.Rectangle."""
        ee = self._ee
        return ee.Geometry.Rectangle(
            [
                self.bbox["west"],
                self.bbox["south"],
                self.bbox["east"],
                self.bbox["north"],
            ]
        )

    def _fetch_single(self, dem_name: str, cfg: dict) -> Path:
        """
        Downloads a single DEM asset for the study area.

        The image is clipped to the Devrek ROI and reprojected to UTM Zone 36N
        before download. Slope and curvature require metric spacing; geographic
        coordinates would corrupt gradient magnitudes.
        """
        ee = self._ee
        roi = self._build_roi()

        image = self._load_image(cfg)
        band = cfg.get("band", "elevation")
        image = image.select(band).rename("elevation")
        image = image.clip(roi)

        scale = cfg["gsd_m"]
        image = image.reproject(crs="EPSG:32636", scale=scale)

        out_path = self.output_dir / f"{dem_name}.tif"
        self._download(image, out_path, scale, roi)
        return out_path

    def _load_image(self, cfg: dict) -> object:
        """Load an EE Image or ImageCollection asset."""
        ee = self._ee
        asset = cfg["asset"]

        if cfg.get("is_collection", False):
            log.debug("Loading as ee.ImageCollection: %s", asset)
            return ee.ImageCollection(asset).mosaic()

        log.debug("Loaded as ee.Image: %s", asset)
        return ee.Image(asset)

    def _download(self, image, out_path: Path, scale: int, roi) -> None:
        """
        Uses geemap.ee_export_image for direct local download, with a URL
        fallback if geemap is unavailable.
        """
        if out_path.exists():
            log.info("  Overwriting existing file: %s", out_path.name)
            out_path.unlink()

        try:
            import geemap

            geemap.ee_export_image(
                image,
                filename=str(out_path),
                scale=scale,
                region=roi,
                file_per_band=False,
            )
            log.info("  Downloaded via geemap: %s", out_path.name)

        except ImportError:
            log.warning("geemap not found. Falling back to URL download ...")
            self._download_via_url(image, out_path, scale, roi)

        except Exception as exc:
            log.error("Download failed for %s: %s", out_path.name, exc)
            raise

    def _download_via_url(self, image, out_path: Path, scale: int, roi) -> None:
        """Fallback download for small GeoTIFF exports."""
        import requests

        url = image.getDownloadURL(
            {
                "scale": scale,
                "region": roi,
                "format": "GEO_TIFF",
                "bands": ["elevation"],
            }
        )
        log.info("  Downloading from URL ...")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        with out_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                handle.write(chunk)
        log.info("  Saved: %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)

    def validate(self, dem_paths: dict) -> bool:
        """
        Opens each GeoTIFF and checks basic integrity.
        """
        try:
            import numpy as np
            import rasterio
        except ImportError:
            log.warning("rasterio/numpy not available for validation.")
            return True

        all_ok = True
        for dem_name, path in dem_paths.items():
            if not path.exists():
                log.error("MISSING: %s", path)
                all_ok = False
                continue
            with rasterio.open(path) as src:
                data = src.read(1, masked=True)
                log.info(
                    "  [%s] CRS=%s | Shape=%s | Elev: min=%.0fm max=%.0fm",
                    dem_name,
                    src.crs,
                    src.shape,
                    float(np.ma.min(data)),
                    float(np.ma.max(data)),
                )
        return all_ok
