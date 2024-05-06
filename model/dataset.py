import os
import numpy as np
from pandas import read_csv
from PIL import Image
import torch
from torch.utils.data import Dataset
import utils
import config

class YOLSODataset(Dataset):
    def __init__(
            self, csv_file: str, img_dir: str, label_dir: str,
            image_size: int, symbol_size: int, padding_size:int,
            num_classes: int, transform=None
    ):
        """

        :param csv_file:
        :param img_dir:
        :param label_dir:
        :param image_size:
        :param symbol_size:
        :param padding_size:
        :param num_classes:
        :param transform:

        """
        self.annotations = read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.symbol_size = symbol_size
        self.padding_size = padding_size
        self.num_classes = num_classes
        self.transform = transform
        self.output_grid_size = self._get_output_grid_size()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # read image file
        image_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(image_path)

        # read label file
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # convert boxes to required target format
        exc_top, exc_bottom, exc_left, exc_right = self._get_exclude_position()
        target = torch.zeros(
            (self.output_grid_size, self.output_grid_size, self.num_classes + 3)
        )

        for box in boxes:
            class_label, x_norm, y_norm, width_norm, height_norm = box.tolist()

            x, y, width, height = np.dot([x_norm, y_norm, width_norm, height_norm], self.image_size)

            # remove boundary annotations
            if x-width < exc_left or x+width > exc_right or y-height < exc_top or y+height > exc_bottom:
                continue
            class_label = int(class_label)
            cell_size = self.symbol_size
            # i, j represents the cell row and cell column
            grid_norm = self.padding_size / self.image_size

            i, j = (
                int(self.output_grid_size*(y-self.padding_size)/self.image_size),
                int(self.output_grid_size * (x - self.padding_size) / self.image_size),
                )
            x_cell, y_cell = (
                self.output_grid_size*(x-self.padding_size)/self.image_size - j,
                self.output_grid_size*(y-self.padding_size)/self.image_size - i
            )

            # if no object already found in cell i, j, set to 1
            if target[i, j, self.num_classes] == 0:
                target[i, j, self.num_classes] = 1

                box_coord = torch.tensor(
                    [x_cell, y_cell]
                )
                target[i, j, (self.num_classes + 1):(self.num_classes+3)] = box_coord
                target[i, j, class_label] = 1
                # e.g. target[i, j, ] = [0, 0, ..., 1, 0, ..., 1, 0.3, 0.7]
        return image, target

    def _get_output_grid_size(self) -> int:
        """
        :return: number of output grid width and height
        """
        return int((self.image_size - 2 * self.padding_size) / self.symbol_size)

    def _get_exclude_position(self):
        """

        :return:
        """
        top = self.padding_size
        bot = self.image_size - self.padding_size
        left = self.padding_size
        right = self.image_size - self.padding_size
        return top, bot, left, right

    def test(self):
        img, lab = self.__getitem__(0)

if __name__ == '__main__':
    datasett = YOLSODataset("../test/train.csv", "../test/train/images", "../test/train/labels", 480, 16, 30, 30)
    datasett.test()