import os
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import sys
import zipfile  # <-- ADD THIS LINE
import zlib  

sys.path.append(os.getcwd())
script_directory = os.path.dirname(os.path.abspath(__file__))
#print(script_directory)

from dataset.helper.dataset_helper import (
    split)

def detection(paths):
    c = 0
    outlier_file = open(os.path.join(script_directory, "outliers_unchecked.txt"), "w+")

    for path in paths:
        files = os.listdir(path)
        t = len(files)
        n = 0

        for file in files:
            c += 1
            # In resolute_detection.py, inside the loop where you process files
           
            print(f"DEBUG: Attempting to load file: {file}")  # <-- ADD THIS
            try:
                data_frame = np.load(os.path.join(path, file), allow_pickle=True)
                #data_frame = np.load(file_pat, allow_pickle=True)
                red = data_frame["red"]
                green = data_frame["green"]
                blue = data_frame["blue"]
                nir = data_frame["nir"]
                dom = data_frame["dom"]

            except (zipfile.BadZipFile, zlib.error) as e:  # <-- CATCH THE SPECIFIC ERRORS
                print(f"  ERROR: Corrupted or invalid .npz file: {file}")
                print(f"  Details: {e}")
                # 'continue' will skip this broken file and proceed to the next one
                continue
            except Exception as e:
                print(f"  ERROR: An unexpected error occurred with file {file}: {e}")
                continue

            

           
            isOk = True

            if blue.shape[0] != blue.shape[1] or green.shape[0] != green.shape[1] or\
                    red.shape[0] != red.shape[1] or nir.shape[0] != nir.shape[1] or\
                    dom.shape[0] != dom.shape[1]:
                isOk = False
            if np.sum(dom < -12) / (len(dom) * len(dom)) > 0.2:
                isOk = False
            elif np.any(dom < -50):
                isOk = False
            elif np.max(dom) < 0:
                isOk = False
            elif np.min(dom) > 10:
                isOk = False
            elif abs(np.min(dom) - np.max(dom)) > 400:
                isOk = False
            elif np.all(blue == 0) or np.all(green == 0) or np.all(red == 0) or np.all(nir == 0):
                isOk = False

            if not isOk:
                n += 1
                outlier_file.write(os.path.join(path, file) + "\n")
                print("[" + str(c) + "/" + str(t) + "]::(" + str(n) + ") Error with " + file)

    outlier_file.close()


if __name__ == '__main__':
    train_folder = split["train"][1]
    validation_folder = split["validation"][1]
    test_folder= split["test"][1]

    detection(
        [
            train_folder,
            validation_folder,
            test_folder
        ]
    )
