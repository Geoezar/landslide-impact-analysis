from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from core.data_fetcher import DataFetcher


def _write_raster(path: Path, data: np.ndarray, nodata: float = -9999.0) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32636",
        transform=from_origin(400000, 4560000, 30, 30),
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


def _fetcher(tmp_path: Path) -> DataFetcher:
    return DataFetcher(bbox={}, dem_catalogue={}, output_dir=tmp_path)


def test_valid_raster_is_accepted(tmp_path):
    path = _write_raster(tmp_path / "valid.tif", np.array([[10, 20], [30, 40]]))

    assert _fetcher(tmp_path).validate({"valid": path}) is True


def test_corrupted_raster_is_rejected(tmp_path):
    path = tmp_path / "corrupted.tif"
    path.write_bytes(b"not-a-geotiff")

    with pytest.raises(ValueError, match="valid readable raster"):
        _fetcher(tmp_path).validate({"corrupted": path})


def test_all_nodata_raster_is_rejected(tmp_path):
    path = _write_raster(tmp_path / "nodata.tif", np.full((2, 2), -9999.0))

    with pytest.raises(ValueError, match="no finite, non-NoData pixels"):
        _fetcher(tmp_path).validate({"nodata": path})


def test_partial_download_never_becomes_final_tiff(tmp_path, monkeypatch):
    fetcher = _fetcher(tmp_path)
    final_path = tmp_path / "dem.tif"

    def write_partial_then_fail(_image, target_path, _scale, _roi):
        target_path.write_bytes(b"partial")
        raise RuntimeError("interrupted download")

    monkeypatch.setattr(fetcher, "_export_to_path", write_partial_then_fail)

    with pytest.raises(RuntimeError, match="interrupted download"):
        fetcher._download(object(), final_path, 30, object())

    assert not final_path.exists()
    assert not (tmp_path / "dem.part.tif").exists()


def test_previous_valid_final_survives_failed_replacement(tmp_path, monkeypatch):
    fetcher = _fetcher(tmp_path)
    final_path = _write_raster(tmp_path / "dem.tif", np.array([[1, 2], [3, 4]]))
    original_bytes = final_path.read_bytes()

    def write_corrupted_replacement(_image, target_path, _scale, _roi):
        target_path.write_bytes(b"corrupted replacement")

    monkeypatch.setattr(fetcher, "_export_to_path", write_corrupted_replacement)

    with pytest.raises(ValueError, match="valid readable raster"):
        fetcher._download(object(), final_path, 30, object())

    assert final_path.read_bytes() == original_bytes
    assert not (tmp_path / "dem.part.tif").exists()
