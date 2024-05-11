import os
import shutil
import csv
import random
from tqdm import tqdm

def split_data(data_folder: str, train_size: float, val_size: float, test_size: float) -> None:
    """
    Split data into train and val sets
    :param data_folder: path to folder which contains images and labels subfolder
    :param train_size: size of train set (in percent or number of data)
    :param val_size: size of validation set (in percent or number of data)
    :param test_size: size of test set (in percent or number of data)
    """
    assert os.path.exists(data_folder), "Data folder doesn't exist"
    image_folder = os.path.join(data_folder, 'images')
    label_folder = os.path.join(data_folder, 'labels')
    assert os.path.exists(image_folder), "Image folder does not exist"
    assert os.path.exists(label_folder), "Label folder does not exist"

    image_files = [os.path.splitext(f)[0] for f in os.listdir(image_folder) if
                   os.path.isfile(os.path.join(image_folder, f))]
    label_files = [os.path.splitext(f)[0] for f in os.listdir(label_folder) if
                   os.path.isfile(os.path.join(label_folder, f))]

    # Find common filenames between image and label files
    common_files = set(image_files).intersection(label_files)
    common_files = list(common_files)
    random.shuffle(common_files)
    image_files = [f"{file}.jpg" for file in common_files]
    label_files = [f"{file}.txt" for file in common_files]
    files_len = len(common_files)
    train_size = int(train_size * files_len) if type(train_size) == float else train_size
    val_size = int(val_size * files_len) if type(val_size) == float else val_size
    test_size = int(test_size * files_len) if type(test_size) == float else test_size

    train_images = image_files[:train_size]
    val_images = image_files[train_size:train_size + val_size]
    test_images = image_files[train_size + val_size:]
    train_labels = label_files[:train_size]
    val_labels = label_files[train_size:train_size + val_size]
    test_labels = label_files[train_size + val_size:]

    move_files(train_images, image_folder, os.path.join(data_folder, 'train', 'images'))
    move_files(train_labels, label_folder, os.path.join(data_folder, 'train', 'labels'))
    move_files(val_images, image_folder, os.path.join(data_folder, 'val', 'images'))
    move_files(val_labels, label_folder, os.path.join(data_folder, 'val', 'labels'))
    move_files(test_images, image_folder, os.path.join(data_folder, 'test', 'images'))
    move_files(test_labels, label_folder, os.path.join(data_folder, 'test', 'labels'))

    datadir2csv(os.path.join(data_folder, 'train'))
    datadir2csv(os.path.join(data_folder, 'val'))
    datadir2csv(os.path.join(data_folder, 'test'))

def move_files(files, src, dest):
    """
    Move files from source_folder to destination_folder
    :param files: list of files to move
    :param src: source folder
    :param dest: destination folder
    """
    os.makedirs(dest, exist_ok=True)
    for file in files:
        shutil.copy(os.path.join(src, file), os.path.join(dest, file))

def datadir2csv(folder_path: str) -> None:
    """
    Convert data folder to csv file
    :param folder_path:
    """
    assert os.path.isdir(folder_path), "The folder path is not valid"
    image_folder = os.path.join(folder_path, 'images')
    label_folder = os.path.join(folder_path, 'labels')
    assert os.path.exists(image_folder), "Image folder not found"
    assert os.path.exists(label_folder), "Labels folder doesn't exist"

    image_files = os.listdir(image_folder)
    print(len(image_files))
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


if __name__ == '__main__':
    #folder_path = "../data/"
    #split_data(folder_path, 0.75, 0.15, 0.10)\

    datadir2csv("../data/aug_train")
    datadir2csv("../data/aug_val")