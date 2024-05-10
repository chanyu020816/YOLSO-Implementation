import os
import albumentations as A

IMAGE_PATH = '../../test/test_data/test/images/'
LABELS_PATH = '../../test/test_data/test/labels'
OUTPUT_PATH = '../../test/test_data/test/aug'
AUG_NAME_PREFIX = 'aug'
AUG_SIZE = 20
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