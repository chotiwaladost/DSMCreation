import os
import numpy as np
import zipfile
import zlib
from tqdm import tqdm

# --- USER: CONFIGURE THESE PATHS ---
NPZ_DIRECTORY = "/data/output/complete/validation"  # PLEASE VERIFY THIS PATH
OUTLIER_FILE_PATH = "/home/julian/scripts/DSM_tiles_complete/dataset/correction/outliers_checked_stayed.txt"
CORRUPTED_FILES_LOG = "corrupted_files_validation.txt"


# --- END CONFIGURATION ---

def find_corrupted_files_thorough():
    """
    Scans .npz files and attempts to read all expected arrays to find deep corruption.
    """
    print("--- Starting THOROUGH Dataset Verification ---")

    # Step 1: Get the list of files to check (same logic as before)
    try:
        with open(OUTLIER_FILE_PATH, 'r') as f:
            outliers = [os.path.basename(line.rstrip()) for line in f]
        print(f"Loaded {len(outliers)} outliers to exclude.")
    except FileNotFoundError:
        print(f"WARNING: Outlier file not found at {OUTLIER_FILE_PATH}. Checking all files.")
        outliers = []

    files_to_check = []
    for file in os.listdir(NPZ_DIRECTORY):
        if file.endswith(".npz") and file not in outliers:
            files_to_check.append(os.path.join(NPZ_DIRECTORY, file))

    if not files_to_check:
        print("No .npz files found to check in the specified directory.")
        return

    print(f"Found {len(files_to_check)} files to verify.")

    # Step 2: Loop through and try to load ALL expected arrays from each file
    corrupted_files = []
    for file_path in tqdm(files_to_check, desc="Thoroughly verifying files"):
        try:
            with np.load(file_path) as data:
                # --- THIS IS THE KEY CHANGE ---
                # We now try to access each array to force decompression.
                # This will trigger the zlib.error if any of them are corrupt.
                _ = data["red"]
                _ = data["green"]
                _ = data["blue"]
                _ = data["nir"]
                _ = data["dom"]
                # --- END OF KEY CHANGE ---
        except (zipfile.BadZipFile, zlib.error, KeyError, ValueError) as e:
            print(f"\n  -> Found corrupted file: {file_path}")
            print(f"     Reason: {type(e).__name__} - {e}")
            corrupted_files.append(file_path)

    # Step 3: Report the results (same as before)
    print("\n--- Verification Complete ---")
    if corrupted_files:
        print(f"Found {len(corrupted_files)} corrupted files.")
        with open(CORRUPTED_FILES_LOG, 'w') as f:
            for path in corrupted_files:
                f.write(f"{path}\n")
        print(f"A list of these corrupted files has been saved to: {CORRUPTED_FILES_LOG}")
    else:
        print("Great! No corrupted .npz files were found after a thorough check.")


if __name__ == '__main__':
    find_corrupted_files_thorough()