import os
import numpy as np
from pandas import read_csv
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset

class YOLSODataset(Dataset):
    def __init__(
            self, csv_file: str, img_dir: str, label_dir: str,
            image_size: int, symbol_size: int, padding_size:int,
            num_classes: int, transform=None
    ):
        """

        :param csv_file: path to csv file which contains the image filename and label filename
        :param img_dir: path to image directory
        :param label_dir: path to label directory
        :param image_size: image size
        :param symbol_size: symbol size
        :param padding_size: padding size
        :param num_classes: number of classes
        :param transform: transform to be applied on a sample

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
            (self.output_grid_size, self.output_grid_size, self.num_classes + 4)
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
            relative_size = width / self.symbol_size
            # if no object already found in cell i, j, set to 1
            if target[i, j, self.num_classes] == 0:
                target[i, j, self.num_classes] = 1

                box_coord = torch.tensor(
                    [x_cell, y_cell, relative_size]
                )
                target[i, j, (self.num_classes + 1):(self.num_classes+4)] = box_coord
                target[i, j, class_label] = 1
                # e.g. target[i, j, ] = [0, 0, ..., 1, 0, ..., 1, 0.3, 0.7, 1.5]
        return torch.tensor(np.array(image)).permute(0, 1, 2), target

    def _get_output_grid_size(self) -> int:
        """
        :return: number of output grid width and height
        """
        return int((self.image_size - 2 * self.padding_size) / self.symbol_size)

    def _get_exclude_position(self):
        """
        :return: top, bottom, left, right coordinates of excluded box
        """
        top = self.padding_size
        bot = self.image_size - self.padding_size
        left = self.padding_size
        right = self.image_size - self.padding_size
        return top, bot, left, right

    def test(self, index):
        img, lab = self.__getitem__(index)
        lab_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])

        self._vis_lab_loc(img, lab, lab_path, index)

    def _vis_lab_loc(self, img, lab, lab_path, index):
        img = img.cpu().numpy()
        img = img.astype(np.uint8)  # Ensure the datatype is uint8
        img = np.transpose(img, (1, 2, 0))  # Transpose to HWC format
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ori_img = img.copy()
        top, bot, left, right = self._get_exclude_position()
        cv2.imwrite(f'../demo/ori_img{index}.jpg', ori_img)
        boxes = []
        with open(lab_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])

        for box in boxes:
            class_label, x, y, width, height = box

            cv2.rectangle(
                ori_img,
                (int((x-width / 2) * self.image_size), int((y-height / 2) * self.image_size)),
                (int((x+width / 2) * self.image_size), int((y+height / 2) * self.image_size)),
                (0, 0, 255), 1
            )
        cv2.rectangle(ori_img, (left, top), (right, bot), (0, 255, 255), 2)
        cv2.imwrite(f'../demo/ori_lab_img{index}.jpg', ori_img)


        cv2.rectangle(img, (left, top), (right, bot), (0, 255, 255), 2)
        for i in range(lab.shape[0]):
            for j in range(lab.shape[1]):
                anno = lab[i, j, ...]
                # grid with annotation
                if any(anno) != 0:
                    class_id = (anno[0:self.num_classes] != 0).nonzero()[0].item()
                    center = (
                        int((anno[self.num_classes + 1]+ j) * self.image_size / self.output_grid_size + self.padding_size),
                        int((anno[self.num_classes + 2]+ i) * self.image_size / self.output_grid_size + self.padding_size)
                    )
                    size = anno[self.num_classes + 3] * self.symbol_size
                    box_tl = (int(center[0] - size / 2), int(center[1] - size / 2))
                    box_br = (int(center[0] + size / 2), int(center[1] + size / 2))
                    cv2.circle(img, center, 3, (0, 0, 255), -1)
                    cv2.rectangle(img, box_tl, box_br, (0, 255, 0), 1)
                    cv2.putText(
                        img, f'{class_id}', (center[0] - 5, center[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                    )
        cv2.imwrite(f'../demo/train_lab_img{index}.jpg', img)

if __name__ == '__main__':
    datasett = YOLSODataset(
        "../data/train.csv",
        "../data/train/images",
        "../data/train/labels",
        480,
        16,
        33,
        8
    )
    datasett.test(3)