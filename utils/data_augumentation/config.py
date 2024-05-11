import os
import albumentations as A

IMAGE_PATH = os.path.join("..", "..", "data", "train", "images")
LABELS_PATH = os.path.join("..", "..", "data", "train", "labels")
OUTPUT_PATH = os.path.join("..", "..", "data", "aug_train")
AUG_NAME_PREFIX = 'aug'
AUG_SIZE = 15
TRANSFORMS = A.Compose([
    # A.RandomCrop(width=550, height=550),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.RGBShift(p=0.5),
    A.RandomCropFromBorders(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.GaussianBlur(p=0.5)
], bbox_params=A.BboxParams(format='yolo'))