import numpy as np

from core.raster_math import RasterMath


def test_compute_aspect_uses_precomputed_slope_for_flat_mask(tmp_path):
    rm = RasterMath(dem_path=tmp_path / "SRTM_30m.tif", output_dir=tmp_path)
    elev = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    slope = np.array([[0.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    profile = {"transform": type("Transform", (), {"a": 30.0})()}

    aspect = rm._compute_aspect(elev, profile, slope=slope)

    assert aspect[0, 0] == -1.0
    assert np.all(aspect[np.array([[False, True], [True, True]])] >= 0.0)
