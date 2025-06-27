import urllib.request as ulib
import os
import pandas as pd

from helper.dataset_helper import (
    dom_path
) # This is where files will be downloaded

# Path to the CSV file listing data to download
csv_file_path = "/home/julian/scripts/probe/DSMCreation-1_Baseline_Unet/dataset/meta/789_cities.csv"

try:
    selected_data_pd = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"ERROR: CSV file not found at {csv_file_path}")
    exit()
except pd.errors.EmptyDataError:
    print(f"ERROR: CSV file at {csv_file_path} is empty.")
    exit()
except Exception as e:
    print(f"ERROR: Could not read CSV {csv_file_path}. Error: {e}")
    exit()


def download_adapted(tile_name_with_extension, save_directory):
    BASE_DOWNLOAD_URL = "https://www.opengeodata.nrw.de/produkte/geobasis/hm/ndom50_tiff/ndom50_tiff"

    download_url = f"{BASE_DOWNLOAD_URL}/{tile_name_with_extension}"
    file_save_path = os.path.join(save_directory, tile_name_with_extension)

    # This print statement is good for seeing what's being attempted *before* the download
    print(f"Attempting to download: {download_url}")
    print(f"Saving to: {file_save_path}")

    try:
        ulib.urlretrieve(download_url, file_save_path)
        print(f"Successfully downloaded: {file_save_path} ✅")
    except ulib.error.HTTPError as e: # Corrected
        print(f"HTTP ERROR for {tile_name_with_extension}: {e.code} {e.reason} (URL: {download_url}) ❌")
    except ulib.error.URLError as e:   # Corrected
        print(f"URL ERROR for {tile_name_with_extension}: {e.reason} (URL: {download_url}) ❌")
    except Exception as e:
        print(f"An unexpected error occurred while downloading {tile_name_with_extension} (URL: {download_url}): {e} ❌")

if __name__ == "__main__": # Make sure this block is present to run the loop
    if not os.path.exists(dom_path):
        print(f"ERROR: Download directory '{dom_path}' does not exist. Please create it.")
        exit()
    if not os.access(dom_path, os.W_OK):
        print(f"ERROR: No write permission to download directory '{dom_path}'. Please check permissions.")
        exit()

    if "Kachelname" not in selected_data_pd.columns:
        print(f"ERROR: Column 'Kachelname' not found in the CSV file at {csv_file_path}. Please check the CSV header.")
        exit()

    print(f"Starting download process. Files will be saved in '{dom_path}'.")
    for kachel_name in selected_data_pd["Kachelname"]:
        if pd.isna(kachel_name) or str(kachel_name).strip() == "":
            print(f"Skipping empty or invalid Kachelname.")
            continue

        file_to_download = str(kachel_name).strip() + ".tif"
        download_adapted(file_to_download, dom_path)

    print("Download script finished. 🎉")





