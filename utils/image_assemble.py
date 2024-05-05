import os
import argparse
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from image_padding import ImagePadding

colors = [
    (255, 0, 0),    # 红色
    (0, 255, 0),    # 绿色
    (0, 0, 255),    # 蓝色
    (255, 255, 0),  # 黄色
    (255, 0, 255),  # 紫色
    (0, 255, 255),  # 青色
    (128, 0, 0),    # 深红色
    (0, 128, 0)     # 深绿色
]

def predict_image(img_path, confidence_threshold = 0.01, split_size = 480, image_folder="./", label_folder="./"):
    ori_image = cv2.imread(img_path)
    ori_h, ori_w = ori_image.shape[:2]
    image_name = os.path.basename(img_path)
    image_name = image_name[:-4]

    image_padding = ImagePadding(image_name, ori_image, split_size)
    image_dict = image_padding.split_image()
    padW = image_dict['padW']
    padH = image_dict['padH']

    total_annotations = []
    for image, position in zip(image_dict['images'], image_dict["positions"]):
        pred = model.predict(image)
        image_annotations = []
        img_h, img_w = image.shape[:2]
        for p in pred:
            for bbox in p.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = bbox
                class_id = int(class_id)
                color = colors[class_id] if class_id < len(colors) else colors[0]

                if score < confidence_threshold: continue
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

                # position on split image
                x = ((x1 + x2) / 2) / img_w
                y = ((y1 + y2) / 2) / img_h
                w = abs(x2 - x1) / img_w
                h = abs(y2 - y1) / img_h
                image_annotations.append([class_id, x, y, w, h])

                # reposition to full image
                x1 = int(x1 + position[1] * split_size - padW)
                x2 = int(x2 + position[1] * split_size - padW)
                y1 = int(y1 + position[0] * split_size - padH)
                y2 = int(y2 + position[0] * split_size - padH)
                label = f'{score:.2f}'
                cv2.rectangle(ori_image, (x1, y1), (x2, y2), color, 1)
                cv2.putText(
                    ori_image, label, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1
                )

                x = ((x1 + x2) / 2) / ori_w
                y = ((y1 + y2) / 2) / ori_h
                w = abs(x2 - x1) / ori_w
                h = abs(y2 - y1) / ori_h

                total_annotations.append([class_id, x, y, w, h])

        cv2.imwrite(os.path.join(image_folder, f'{image_name}_h{position[0]}_w{position[1]}.jpg'), image)
        with open(os.path.join(label_folder, f'{image_name}_h{position[0]}_w{position[1]}.txt' ), 'w') as file:
            for anno in image_annotations:
                class_id, x, y, w, h = anno
                file.write(f'{class_id} {x} {y} {w} {h}\n')

    cv2.imwrite(os.path.join(image_folder, f'{image_name}_pred.jpg'), ori_image)
    output_path = os.path.join(label_folder, f'{image_name}_pred.txt')
    with open(output_path, 'w') as file:
        for anno in total_annotations:
            class_id, x, y, w, h = anno
            file.write(f'{class_id} {x} {y} {w} {h}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./yolov8n.pt', help='path to model')
    parser.add_argument('--img_path', type=str, help='path to image')
    parser.add_argument('--confidence_threshold', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--split_size', type=int, default=480, help='split size')
    parser.add_argument('--image_folder', type=str, default='./', help='path to save image')
    parser.add_argument('--label_folder', type=str, default='./', help='path to save label')

    args = parser.parse_args()
    # assert args.img_path.exists(), "image path does not exist"

    model = YOLO(args.model)
    predict_image(
        args.img_path,
        args.confidence_threshold,
        args.split_size,
        args.image_folder,
        args.label_folder
    )

