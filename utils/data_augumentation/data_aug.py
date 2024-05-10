import os
import cv2
import yaml
import albumentations as A
from tqdm import tqdm
import config as cfg

def run_aug():
    images = os.listdir(cfg.IMAGE_PATH)
    prefix = cfg.AUG_NAME_PREFIX
    aug_size = int(cfg.AUG_SIZE)
    transforms = cfg.TRANSFORMS
    os.makedirs(os.path.join(cfg.OUTPUT_PATH, 'images'), exist_ok=True)
    print(os.path.join(cfg.OUTPUT_PATH, 'images'))
    os.makedirs(os.path.join(cfg.OUTPUT_PATH, 'labels'), exist_ok=True)

    for img_name in tqdm(images):
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(cfg.IMAGE_PATH, img_name)
        label_path = os.path.join(cfg.LABELS_PATH, f"{base_name}.txt")

        if os.path.exists(img_path) and os.path.exists(label_path):
            image = cv2.imread(img_path)
            height, width, _ = image.shape
            bboxes = read_yolo_labels(label_path, width, height)
            if bboxes:
                for i in range(aug_size):
                    out_img = os.path.join(cfg.OUTPUT_PATH, 'images', f'{base_name}_{prefix}{i}.jpg')
                    out_label = os.path.join(cfg.OUTPUT_PATH, 'labels', f'{base_name}_{prefix}{i}.txt')
                    image_aug(image, bboxes, transforms, out_img, out_label)

def read_yolo_labels(label_path, image_width, image_height):
    bboxes = []
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().split()
            if len(data) == 5:  # YOLO format: class x_center y_center width height
                class_id = int(data[0])
                x_center, y_center, width, height = map(float, data[1:])
                bboxes.append([x_center, y_center, width, height, class_id])
    return bboxes

def image_aug(image, bboxes, transforms_fn, out_image_name, out_label_path):
    results = transforms_fn(image=image, bboxes=bboxes)
    aug_image, aug_bboxes = results['image'], results['bboxes']

    cv2.imwrite(out_image_name, aug_image)
    with open(out_label_path, 'w') as file:
        for bbox in aug_bboxes:
            bbox = [bbox[4], bbox[0], bbox[1], bbox[2], bbox[3]]
            file.write(' '.join(map(str, bbox)) + '\n')

if __name__ == '__main__':
    run_aug()