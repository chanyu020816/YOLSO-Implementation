import os
import csv

from tqdm import tqdm

def split_data(data_folder: str, train_size: float, val_size: float, test_size: float) -> None:
    pass

def datadir2csv(folder_path: str) -> None:
    assert os.path.isdir(folder_path), "The folder path is not valid"
    image_folder = os.path.join(folder_path, 'images')
    label_folder = os.path.join(folder_path, 'labels')
    assert os.path.exists(image_folder), "Image folder not found"
    assert os.path.exists(label_folder), "Labels folder doesn't exist"

    image_files = os.listdir(image_folder)
    # label_files = os.listdir(label_folder)
    folder_path_split = folder_path.split(os.path.sep)
    csv_name = folder_path_split[-1] if folder_path_split[-1] != '' else folder_path_split[-2]
    csv_path = os.path.join(*folder_path_split[:-1], f'{csv_name}.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for image_file in tqdm(image_files):
            label_file = image_file.split('.')[0] + '.txt'
            label_path = os.path.join(label_folder, label_file)
            if os.path.exists(label_path):
                writer.writerow({'image': image_file, 'text': label_file})
