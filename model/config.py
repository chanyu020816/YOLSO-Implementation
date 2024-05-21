from PIL import Image
from torch import cuda
from torch.backends import mps
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize
import os.path

if cuda.is_available():
    DEVICE = 'cuda'
elif mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

DATASET = os.path.join('.', 'data')
IN_CHANNELS = 3
NUM_WORKERS = 4
BATCH_SIZE = 2
# BUILD_CONFIG = default_config
IMAGE_SIZE = 480
SYMBOL_SIZE = 16
PADDING_SIZE = 33
NUM_CLASSES = 8
LEARNING_RATE = 1e-4
LR_EXP_DECAY_FACTOR = 0.95
WEIGHT_DECAY = 1e-4
EPOCHS = 20
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
SCALES = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
SERIES_DIR = os.path.join(DATASET, "1d_series")
TRAIN_IMG_DIR = os.path.join(DATASET, "train", "images")
TRAIN_LABEL_DIR = os.path.join(DATASET, "train", "labels")
TRAIN_CSV_FILE = os.path.join(DATASET, "train.csv")
VAL_IMG_DIR = os.path.join(DATASET, "val", "images")
VAL_LABEL_DIR = os.path.join(DATASET, "val", "labels")
VAL_CSV_FILE = os.path.join(DATASET, "val.csv")
TEST_IMG_DIR = os.path.join(DATASET, "test", "images")
TEST_LABEL_DIR = os.path.join(DATASET, "test", "labels")
TEST_CSV_FILE = os.path.join(DATASET, "test.csv")
TEST_INTERVAL = 1


class CustomTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = self.transforms(image)
        return image, boxes

TRANSFORM = CustomTransform(Compose([
    ToTensor(),  # 将图像转换为张量，并且会自动将图像数据类型转换为 float32
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
]))
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
    # [-1, 1, 'Conv', [(NUM_CLASSES + 3), 1, 1, 0]],
    [-1, 1, 'V1Prediction']
]

thin_config = [
    [-1, 1, 'Conv', [32, 3, 1, 0]],
    [-1, 1, 'Conv', [64, 3, 1, 0]],
]

model_configs = {
    'origin_config': origin_config,
    'thin': thin_config,
}