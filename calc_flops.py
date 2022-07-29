import argparse
import copy
import os
import os.path as osp
import time
import warnings

import torch
from numbers import Number
from typing import Any, Callable, List, Optional, Union
from numpy import prod
import numpy as np
from fvcore.nn import FlopCountAnalysis



def rfft_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the rfft/rfftn operator.
    """
    input_shape = inputs[0].type().sizes()
    B, H, W, C = input_shape
    N = H * W
    flops = N * C * np.ceil(np.log2(N))
    return flops

def calc_hornet_flops(model, img_size=224, show_details=False):
    with torch.no_grad():
        x = torch.randn(1, 3, img_size, img_size).cuda()
        fca1 = FlopCountAnalysis(model, x)
        handlers = {
            'aten::fft_rfft2': rfft_flop_jit,
            'aten::fft_irfft2': rfft_flop_jit,
        }
        fca1.set_op_handle(**handlers)
        flops1 = fca1.total()
        if show_details:
            print(fca1.by_module())
        print("#### GFLOPs: {}".format(flops1 / 1e9))
    return flops1 / 1e9
