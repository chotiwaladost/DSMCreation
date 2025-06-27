from PIL import Image
from itertools import product

import os
import rasterio as rio
import numpy as np
from osgeo import gdal
import subprocess


def cut(tilepath, tilename, cutting_length):
    out_path = tilepath
    output_filename = tilename

    tile_size_x = cutting_length
    tile_size_y = cutting_length

    ds = gdal.Open(os.path.join(tilepath, tilename))
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize

    # Complete new structure to cut the raster, previous one was not generating the files.
    c = 0
    for i in range(0, xsize, tile_size_x):
        for j in range(0, ysize, tile_size_y):
            com_string_list = ["gdal_translate", "-of", "GTIFF", "-srcwin", str(i), str(j), str(tile_size_x),
                               str(tile_size_y),
                               os.path.join(tilepath, tilename),
                               os.path.join(out_path,f"cut_{output_filename.replace('.tif', '')}_{c}.tif")]

            print(os.path.join(tilepath, tilename),
                  os.path.join(out_path, f"cut_{output_filename.replace('.tif', '')}_{c}.tif"))
            try:
                result = subprocess.run(com_string_list, capture_output=True, text=True, check=True)
                #print(f"DEBUG: Successfully processed tile {c}. STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

            except subprocess.CalledProcessError as e:
                print(f"Error creating tile {c}:")
                print(f"Command: {e.cmd}")
                print(f"Return Code: {e.returncode}")
                print(f"Stdout: {e.stdout}")
                print(f"Stderr: {e.stderr}")
            c += 1

    ds = None  # Added New line to close the ds and be able to remove it on next line

    os.remove(os.path.join(tilepath, tilename))
