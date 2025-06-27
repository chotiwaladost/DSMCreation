import os.path
import math
import statistics as s
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.serialization  # Make sure to import this
from tqdm.auto import tqdm as prog
import rasterio
from PIL import Image
import sys
sys.path.append(os.getcwd())


import torchvision.transforms.functional as tf

from network.provider.pytorchtools import EarlyStopping  # Or the actual module where EarlyStopping is defined

torch.serialization.add_safe_globals([EarlyStopping])

from provider.dataset_provider import (
    get_loader
)
from helper.network_helper import (
    num_workers,
    pin_memory,
    device
)

import provider.pytorchtools as pytorchtools

from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError
)

from torchmetrics.image import StructuralSimilarityIndexMeasure

from metrics.zncc import zncc
from metrics.ssim import custom_ssim

from unet_fanned.model_v1 import UNET_FANNED

# import torchvision.transforms.functional as tf


sys.path.append(os.getcwd())
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# import shutup
# shutup.please()


def test(amount, model_path, test_data_path, path_original_raster, raster_folder):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    unet = UNET_FANNED(in_channels=4, out_channels=1)
    # state_dictionary_from_file = torch.load(model_path, map_location=DEVICE)

    state_dictionary_from_file = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # unet.load_state_dict(state_dictionary_from_file)
    unet.load_state_dict(state_dictionary_from_file['model_state_dict'])
    unet.to(DEVICE)

    unet.eval()
    torch.no_grad()

    loader = get_loader(test_data_path, 1, num_workers, pin_memory, amount=amount, shuffle=False)
    c = 0

    #if not os.path.exists(
     #       r"C:\Users\renec\OneDrive\Documents\SEForALL\Local\OBI Tool\V1_DSM_creation\network\results_L1Loss_Adam_UNET_FANNED_v1_2\results"):
      #  os.mkdir(
       #     r"C:\Users\renec\OneDrive\Documents\SEForALL\Local\OBI Tool\V1_DSM_creation\network\results_L1Loss_Adam_UNET_FANNED_v1_2\results")

    mae = MeanAbsoluteError().to(device)
    mse = MeanSquaredError().to(device)
    ssim = StructuralSimilarityIndexMeasure(kernel_size=(5, 5)).to(device)

    walking_mae = 0

    running_mae = []
    running_mse = []
    running_ssim = []
    running_zncc = []
    running_median = []

    loop = prog(loader)

    for (data, target, src_path) in loop:
        data = data.to(device)

        data[data < 0] = 0
        target[target < 0] = 0

        prediction = unet(data)
        prediction[prediction < 0] = 0

        target = target.unsqueeze(1).to(device)

        prediction = tf.center_crop(prediction, [500, 500])
        target = tf.center_crop(target, [500, 500])

        prediction = prediction.contiguous()
        target = target.contiguous()

        #prediction = prediction.squeeze(0).squeeze(0).detach().cpu()
        target = target.squeeze(0).squeeze(0).detach().cpu()
        #output_folder = r"C:\Users\renec\OneDrive\Documents\SEForALL\Local\OBI Tool\V1_DSM_creation\Düsseldorf_Comparison\infered_DSM"

        prediction = prediction.squeeze(0).squeeze(0).detach().cpu().numpy()

        #np.save(os.path.basename(src_path[0]), prediction)
        #filename = os.path.splitext(os.path.basename(src_path[0]))[0] + ".npy"

        # Save file
        #save_path = os.path.join(output_folder, filename)
        #np.save(save_path, prediction)

        #print(f"Saved prediction to {save_path}")



        print(src_path[0])

        filename = os.path.basename(src_path[0])  # e.g. 'ndom50_32344_5675_1_nw_2023_3~SENTINEL2X_20230515-000000-000_L3A_T32ULB_C_V1-2.npz'
        base = filename.split('~')[0]
        parts = base.split('_')
        parts[-1] = str(int(parts[-1]) - 1)
        modified_base = '_'.join(parts)
        raster_name = f"cut_transformed_{modified_base}.tif"

        # Step 3: Target folder where the corresponding raster is located
        #raster_folder = r"C:\Users\renec\OneDrive\Documents\SEForALL\Local\OBI Tool\V1_DSM_creation\Düsseldorf_Comparison\True nDSM"
        raster_path = os.path.join(path_original_raster, raster_name)

        # Step 5: Open the original raster and write the new data
        #output_folder = r"C:\Users\renec\OneDrive\Documents\SEForALL\Local\OBI Tool\V1_DSM_creation\Düsseldorf_Comparison\infered_DSM"
        #os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(raster_folder, raster_name)

        with rasterio.open(raster_path) as src:
            profile = src.profile
            profile.update(dtype=prediction.dtype, count=1)

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(prediction, 1)

        print(f"Raster saved to: {output_path}")


if __name__ == '__main__':

    #test(0,
     #    r"C:\Users\renec\OneDrive\Documents\SEForALL\Local\OBI Tool\V1_DSM_creation\network\results_L1Loss_Adam_UNET_FANNED_v1\model_epoch15.pt",
      #   r"C:\Users\renec\OneDrive\Documents\SEForALL\Local\OBI Tool\V1_DSM_creation\Düsseldorf_Comparison\npz_files",
       #  r"C:\Users\renec\OneDrive\Documents\SEForALL\Local\OBI Tool\V1_DSM_creation\Düsseldorf_Comparison\True nDSM"
    #)

    base_path = "/data/inference_test/V7_Yadgiri"

    test(0,
    "/home/julian/scripts/DSM_tiles_complete/results_L1Loss_Adam_UNET_FANNED_v1/model_epoch26.pt",
     os.path.join(base_path, "npz_files"),
     os.path.join(base_path, "dummy_nDSM"),
     os.path.join(base_path, "inferred_nDSM")
    )
