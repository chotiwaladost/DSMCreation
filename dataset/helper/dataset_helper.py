import os

sen_path = "/data/nrw_data/sennrw/S2_L3A_WASP/files/32/U/"  # Path to the sentinel data. The subfolder of sennrw then contains LB,LC,MB,LC
dom_path = "/data/nDSM/"  # Path to the ndsm data. The subfolder of domnrw then contains all .tif files of the ndsm
meta_file = open("/home/julian/scripts/DSM_tiles_complete/dataset/meta/DSM_data.csv")
# Path to the meta file. It contains the timestamps of the ndsm data frames
sen_example = "/data/nrw_data/sennrw/S2_L3A_WASP/files/32/U/LB/2021/SENTINEL2X_20210515-000000-000_L3A_T32ULB_C_V1-2/"\
	"SENTINEL2X_20210515-000000-000_L3A_T32ULB_C_V1-2_FRC_B2.tif" # Path to a completely random sentinel tile
output_path = "/data/output/complete/"  # Path of the output directory of the .npz files produces by the matching

positiontest_LB = os.path.join(sen_path, "LB", "positiontest")
positiontest_LC = os.path.join(sen_path, "LC", "positiontest")
positiontest_MB = os.path.join(sen_path, "MB", "positiontest")
positiontest_MC = os.path.join(sen_path, "MC", "positiontest")

split = {
    'train': (0.7, "/data/output/complete/training"),
    'validation': (0.2, "/data/output/complete/validation"),
    'test': (0.1, "/data/output/complete/testing")
}  # Split of training, validation and test and their directories

size_in = 500
size_out = 500

cutting_length = 500  # The ndsm have a size of 2000x2000 pixels, too big for most neural networks. The cutter
# script generates smaller tiles with the size of cutting_length x cutting_length.

upsampling_multiplier_sentinel = 20  # Upsample factor from sentinel to ndsm, to not modify this if using our data
upsampling_technique = 0 # Upsampling technique code. 3 = cubic, 1 = bilinear, 0 = nearest neighbour



