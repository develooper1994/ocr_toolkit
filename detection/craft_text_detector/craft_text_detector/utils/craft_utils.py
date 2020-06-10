# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch.backends import cudnn

""" auxilary functions """


# unwarp corodinates


def warp_coord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0] / out[2], out[1] / out[2]])


""" end of auxilary functions """

def adjust_result_coordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


def device_selection_helper_pytorch(device):
    is_device = torch.cuda.is_available()
    if device is None or device == 'gpu':
        assert is_device, "!!!CUDA is not available!!!"  # Double check ;)
        device_object = torch.device('cuda')
        cudnn.enabled = True
        cudnn.benchmark = False
    # elif device == 'auto':
    #     if num_device == 1:
    #         device_object = mx.gpu(num_device - 1) if mx.context.num_gpus() > 0 else mx.cpu(num_device - 1)
    #     else:
    #         device_object = [mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu() for i in range(num_device)]
    #         # device = [mx.gpu(i) for i in range(num_device)] if mx.context.num_gpus() > 0 else [mx.cpu(i) for i in
    #         #                                                                                    range(num_device)]
    elif device == 'cpu':
        device_object = torch.device('cpu')
    elif device == 'auto':
        device_object = None  # default for Pytorch. Use all.
    else:
        # If it isn't a string.
        print("Assuming device is a mxnet ctx or device object or queue. Exp: mx.gpu(0)")
        device_object = device
    return device_object
