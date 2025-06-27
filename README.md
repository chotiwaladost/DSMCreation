The work follows a similar approach with some small variations:
All of the available nDSM files for NRW were not used, but just half of the data for training, validation and testing.
As an aditional step, there are a couple of scripts that take the infered output (in npz format) and transform it to a .TIFF format, which was useful for
our pourpuse: To estimate solar rooftop potential using open-source satellite data: Sentinel-2.
It is also important to note that as now the model is highly computational intense, it was run using IBM infrastructure with 64GB of dedicated GPU for the training,
inference and validation.

To set-up the script, follow what is indicated in: https://github.com/KonstiDE/DSMCreation/tree/V1_Baseline_Unet

It is also very useful to familiarize first with their paper: https://ieeexplore.ieee.org/document/10189905

