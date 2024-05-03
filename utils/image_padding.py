import cv2
import numpy as np
import os


class ImagePadding:

    def __init__(self, image_name, image, size: int):
        # self.img_path = img_path
        self.image_name = image_name
        self.image = image
        self.Size = size
        self.oriH, self.oriW, _ = self.image.shape
        self.numH = (self.oriH // self.Size) + 1
        self.numW = (self.oriW // self.Size) + 1
        self.newH = ((self.oriH // self.Size) + 1) * self.Size
        self.padH = int((self.newH - self.oriH) / 2)
        self.newW = ((self.oriW // self.Size) + 1) * self.Size
        self.padW = int((self.newW - self.oriW) / 2)
        self.new_image = self.generate_image()

    def generate_image(self):
        new_image = np.zeros((self.newH, self.newW, 3), dtype=np.uint8)
        new_image[
            self.padH:(self.oriH + self.padH),
            self.padW:(self.oriW + self.padW),
            0:3
        ] = self.image

        return new_image

    def get_image_padding(self, h, w):
        if self.numH == 1 or self.numW == 1:
            if self.numH == 1:
                if self.numW == 1:
                    padxmin = self.padW
                    padymin = self.padH
                    padxmax = self.Size - self.padW
                    padymax = self.Size - self.padH
                else:
                    if w == 0:
                        padxmin = self.padW
                        padymin = self.padH
                        padxmax = self.Size
                        padymax = self.Size - self.padH
                    elif w == self.numW - 1:
                        padxmin = 0
                        padymin = self.padH
                        padxmax = self.Size - self.padW
                        padymax = self.Size - self.padH
                    else:
                        padxmin = 0
                        padymin = self.padH
                        padxmax = self.Size
                        padymax = self.Size - self.padH
            else:
                if h == 0:
                    padxmin = self.padW
                    padymin = self.padH
                    padxmax = self.Size - self.padW
                    padymax = self.Size
                elif h == self.numH - 1:
                    padxmin = self.padW
                    padymin = 0
                    padxmax = self.Size - self.padW
                    padymax = self.Size - self.padH
                else:
                    padxmin = self.padW
                    padymin = 0
                    padxmax = self.Size - self.padW
                    padymax = self.Size
        elif h == 0:
            if w == self.numW - 1:
                padxmin = 0
                padymin = self.padH
                padxmax = self.Size - self.padW
                padymax = self.Size
            elif w == 0:
                padxmin = self.padW
                padymin = self.padH
                padxmax = self.Size
                padymax = self.Size
            else:
                padxmin = 0
                padymin = self.padH
                padxmax = self.Size
                padymax = self.Size
        elif h == self.numH - 1:
            if w == self.numW - 1:
                padxmin = 0
                padymin = 0
                padxmax = self.Size - self.padW
                padymax = self.Size - self.padH
            elif w == 0:
                padxmin = self.padW
                padymin = 0
                padxmax = self.Size
                padymax = self.Size - self.padH
            else:
                padxmin = 0
                padymin = 0
                padxmax = self.Size
                padymax = self.Size - self.padH
        else:
            if w == 0:
                padxmin = self.padW
                padymin = 0
                padxmax = self.Size
                padymax = self.Size
            elif w == self.numW - 1:
                padxmin = 0
                padymin = 0
                padxmax = self.Size - self.padW
                padymax = self.Size
            else:
                padxmin = 0
                padymin = 0
                padxmax = self.Size
                padymax = self.Size
        return [padxmin, padymin, padxmax, padymax]

    def split_image(self):
        image_names = []
        images = []
        positions = []
        paddings = []

        for h in range(self.numH):
            for w in range(self.numW):
                image_names.append(f'{self.image_name}_{h}_{w}.jpg')
                h_start = h * self.Size
                w_start = w * self.Size
                img = self.new_image[h_start:(h_start + self.Size), w_start:(w_start + self.Size), ]
                images.append(img)
                padding = self.get_image_padding(h, w)
                position = [h, w]
                positions.append(position)
                paddings.append(padding)

        return {
            'image_names': image_names, 'images': images, 'positions': positions,
            'paddings': paddings, 'padH': self.padH, 'padW': self.padW
        }

    def save_image(self, image_folder_path):
        for h in range(self.numH):
            for w in range(self.numW):
                path = os.path.join(image_folder_path, f'test_h{int(h)}_w{int(w)}.jpg')
                h_start = h * self.Size
                w_start = w * self.Size
                img = self.new_image[h_start:(h_start + self.Size), w_start:(w_start + self.Size), ]
                cv2.imwrite(path, img)
        return

    def save_annotation(self, label_folder_path):
        pass