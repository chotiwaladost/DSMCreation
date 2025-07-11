import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import torchvision.transforms.functional as tf

sys.path.append(os.getcwd())

import shutup
shutup.please()

from network.helper.network_helper import (
    device
)

from unet_fanned.model_v1 import UNET_FANNED

from ptflops import get_model_complexity_info

def estimate():
    torch.cuda.empty_cache()

    model = UNET_FANNED(in_channels=4, out_channels=1).to(device)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))

    macs, params = get_model_complexity_info(model, (4, 512, 512), as_strings=True,
                                             print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == '__main__':
    estimate()
