import os
import numpy as np
import pandas as pd
import csv

import torch
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
from model.yolso_dataset import *
import model.config as cfg

def intersection_over_union(boxes1: list, boxes2: list, box_format: str = "center"):
    """
    Calculate intersection over union score

    Parameters:
        boxes1: boxes1
        boxes2: boxes2
        box_format: center (x, y, w, h) or corners (x1, y1, x2, y2)
    Return:
        IOU score
    """

    if box_format == "center":
        box1_x1 = boxes1[..., 0:1] - boxes1[..., 2:3] / 2
        box1_y1 = boxes1[..., 1:2] - boxes1[..., 3:4] / 2
        box1_x2 = boxes1[..., 0:1] + boxes1[..., 2:3] / 2
        box1_y2 = boxes1[..., 1:2] + boxes1[..., 3:4] / 2
        box2_x1 = boxes2[..., 0:1] - boxes2[..., 2:3] / 2
        box2_y1 = boxes2[..., 1:2] - boxes2[..., 3:4] / 2
        box2_x2 = boxes2[..., 0:1] + boxes2[..., 2:3] / 2
        box2_y2 = boxes2[..., 1:2] + boxes2[..., 3:4] / 2
    elif box_format == "corners":
        box1_x1 = boxes1[..., 0:1]
        box1_y1 = boxes1[..., 1:2]
        box1_x2 = boxes1[..., 2:3]
        box1_y2 = boxes1[..., 3:4]  # (N, 1)
        box2_x1 = boxes2[..., 0:1]
        box2_y1 = boxes2[..., 1:2]
        box2_x2 = boxes2[..., 2:3]
        box2_y2 = boxes2[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def non_max_suppression(bboxes, iou_threshold, conf_threshold, box_format="center") -> list:
    bboxes = [box for box in bboxes if box[1] > conf_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0] or
            intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def mean_average_precision(
    pred_bboxes: list, true_bboxes: list, iou_threshold: float = 0.5, num_classes: int = 20, box_format="center"
    ):
    """
    Calculate mean average precision

    Parameters:
        pred_bboxes: predicted boxes
        true_bboxes: true boxes
        iou_threshold: iou threshold
        num_classes: number of classes
        box_format: center (x, y, w, h) or corners (x1, y1, x2, y2)
    Return:
        MAP score
    """
    average_precisions = []

    epsilon = 1e-7

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_bboxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_bboxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

def get_symbol_size(csv_file: str, label_dir: str, image_size: int) -> int:
    """
    Read all symbol size from csv and label files, compute the average size.

    Parameters:
        csv_file (str): path to csv file
        label_dir (str): path to labels folder
        image_size (int): image size

    Return:
        required average symbol size (int)
    """
    file_names = pd.read_csv(csv_file)["text"]
    symbol_sizes = []
    for file in tqdm(file_names[:10]):
        if os.path.exists(os.path.join(label_dir, file)):
            with open(os.path.join(label_dir, file), "r") as f:
                for label in f.readlines():
                    class_label, x, y, width, height = [
                        float(x) if float(x) != int(float(x)) else int(x)
                        for x in label.replace("\n", "").split()
                    ]
                    symbol_sizes.append(width * image_size) if width == height else None
    assert len(symbol_sizes) != 0, "no bounding box detected."

    return int(np.ceil(sum(symbol_sizes) / len(symbol_sizes)))

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
    Return:
        required padding size (int)
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
    checkpoint = torch.load(checkpoint_file, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def get_dataloaders():

    train_dataset = YOLSODataset(
        csv_file=cfg.TRAIN_CSV_FILE,
        img_dir=cfg.TRAIN_IMG_DIR,
        label_dir=cfg.TRAIN_LABEL_DIR,
        image_size=cfg.IMAGE_SIZE,
        symbol_size=cfg.SYMBOL_SIZE,
        padding_size = cfg.PADDING_SIZE,
        num_classes = cfg.NUM_CLASSES,
        transform=cfg.TRANSFORM,
    )
    val_dataset = YOLSODataset(
        csv_file=cfg.VAL_CSV_FILE,
        img_dir=cfg.VAL_IMG_DIR,
        label_dir=cfg.VAL_LABEL_DIR,
        image_size=cfg.IMAGE_SIZE,
        symbol_size=cfg.SYMBOL_SIZE,
        padding_size=cfg.PADDING_SIZE,
        num_classes=cfg.NUM_CLASSES,
        transform=cfg.TRANSFORM,
    )
    test_dataset = YOLSODataset(
        csv_file=cfg.TEST_CSV_FILE,
        img_dir=cfg.TEST_IMG_DIR,
        label_dir=cfg.TEST_LABEL_DIR,
        image_size=cfg.IMAGE_SIZE,
        symbol_size=cfg.SYMBOL_SIZE,
        padding_size=cfg.PADDING_SIZE,
        num_classes=cfg.NUM_CLASSES,
        transform=cfg.TRANSFORM,
    )

    train_loader = DataLoader(
        train_dataset,
        num_workers=cfg.NUM_WORKERS,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=cfg.NUM_WORKERS,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=cfg.NUM_WORKERS,
        batch_size=1,
        shuffle=True,
        drop_last=True
    )
    return train_loader, val_loader, test_loader

def viz_model(image, output):
    image = image.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image_bgr = image[:, :, ::-1]
    img = (image_bgr * 255).astype(np.uint8)
    output = output.cpu().numpy()

    ori_img = img.copy()
    # top, bot, left, right = get_exclude_position(image_size = cfg.IMAGE_SIZE, padding_size=cfg.PADDING_SIZE)
    cv2.imwrite(f'./demo/test_ori_img.jpg', ori_img)
    print(img.shape)

def get_exclude_position(image_size, padding_size):
    """
    :return: top, bottom, left, right coordinates of excluded box
    """
    top = padding_size
    bot = image_size - padding_size
    left = padding_size
    right = image_size - padding_size
    return top, bot, left, right

if __name__ == '__main__':
    #assert get_paddings_size(cfg.model_configs['origin_config']) == 33, "error in origin config"
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders()
    for batch_idx, (image, target) in enumerate(train_dataloader):
        # Print the size of the first batch
        print("Batch Index:", batch_idx)
        print("image Shape:", image.shape)
        print("Target Shape:", target.shape)

        break