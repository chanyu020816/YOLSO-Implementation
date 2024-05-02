import os
import numpy as np
import csv

import torch
from tqdm import tqdm
import config

def get_paddings_size(model_config: list) -> int:
    """
    Compute required padding size from model config

    Parameters:
        model_config (list): config of model, specified as
            # [from, number, module, args]
            [
                [-1, 1, 'Conv', [64, 7, 1, 0]],
                [-1, 1, 'MaxPool', [2, 2, 0]],
                ...
            ]
    """
    max_pool_num = 0
    network_arc = {
        'conv_kernel_size': [],
        'conv_prev_pool': []
    }

    for module in model_config:
        module_type = module[2]
        if module_type == 'Conv':
            kernel_size = module[3][1]
            number = module[1]
            for _ in range(number):
                network_arc['conv_kernel_size'].append(kernel_size)
                network_arc['conv_prev_pool'].append(max_pool_num)
        elif module_type == 'Residual':
            pass
        elif module_type == 'MaxPool':
            max_pool_num += 1

    result = sum(
        (kernel-1) / 2 * (2 ** prev_pool)
        for kernel, prev_pool in zip(network_arc['conv_kernel_size'], network_arc['conv_prev_pool'])
    )
    return int(result)

def mean_average_precision(
        pred_boxes: list, true_boxes: list, iou_threshold: float = 0.5,
        box_format: str = 'center', num_classes: int = 10
    ):
    """
    Compute mean average precision

    Parameters:
        pred_boxes (list): list of predicted boxes
        true_boxes (list): list of true boxes
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (string): format of predicted boxes ('center' or 'top_left')
        num_classes (int): number of classes
    """
    average_precisions = []
    epsilon = 1e-6

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == '__main__':
    # assert get_paddings_size(model_configs['origin_config']) == 33, "error in origin config"
    pass