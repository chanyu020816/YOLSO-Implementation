import torch
import os.path

DATASET = os.path.join('../data')
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

IN_CHANNELS = 3
NUM_WORKERS = 4
BATCH_SIZE = 16
# BUILD_CONFIG = default_config
IMAGE_SIZE = 480
NUM_CLASSES = 8
LEARNING_RATE = 1e-4
LR_EXP_DECAY_FACTOR = 0.95
WEIGHT_DECAY = 1e-4
EPOCHS = 100
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
SCALES = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
SERIES_DIR = os.path.join(DATASET, "1d_series")
IMG_DIR = os.path.join(DATASET, "images")
LABEL_DIR = os.path.join(DATASET, "labels")
TEST_INTERVAL = 1

"""
[from, number, module, args]
Conv args: [filters, kernel_size, stride, padding]
MaxPool args: [kernel_size, stride, padding]
"""
origin_config = [
    [-1, 1, 'Conv', [64, 7, 1, 0]],
    [-1, 1, 'MaxPool', [2, 2, 0]],
    [-1, 1, 'Conv', [64, 3, 1, 0]],
    [-1, 1, 'MaxPool', [2, 2, 0]],
    [-1, 1, 'Conv', [128, 3, 1, 0]],
    [-1, 1, 'MaxPool', [2, 2, 0]],
    [-1, 1, 'Conv', [256, 3, 1, 0]],
    [-1, 1, 'MaxPool', [2, 2, 0]],
    [-1, 1, 'Conv', [512, 3, 1, 0]],
    [-1, 1, 'Conv', [(NUM_CLASSES + 3), 1, 1, 0]],
]

thin_config = [
    [-1, 1, 'Conv', [32, 3, 1, 0]],
    [-1, 1, 'Conv', [64, 3, 1, 0]],
]

model_configs = {
    'origin_config': origin_config,
    'thin': thin_config,
}