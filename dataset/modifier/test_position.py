import os
import numpy as np
import rasterio as rio
import pygeodesy
import rasterio.crs
from rasterio.windows import Window
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

import matplotlib.pyplot as plt


def test_position(tile_path, sentinel_path):
    sentinel = rio.open(sentinel_path)
    tile = rio.open(tile_path)

    window2 = sentinel.read(1, window=from_bounds(
        round(tile.bounds.left, 1),
        round(tile.bounds.bottom, 1),
        round(tile.bounds.right, 1),
        round(tile.bounds.top, 1),
        sentinel.transform
    ))

    return window2
