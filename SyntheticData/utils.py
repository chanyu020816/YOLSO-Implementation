import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import albumentations as A

flip = A.ReplayCompose([
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3)
])

transform = A.ReplayCompose([
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    # A.RandomBrightnessContrast(p=0.3),
    # A.RandomGamma(p=0.3),
    # A.RGBShift(p=0.3),
    A.GaussianBlur(p=0.3)
])

def add_symbol2background(background, symbol, x_offset, y_offset):
    symbol_h, symbol_w = symbol.shape[:2]
    symbol_gray = cv2.cvtColor(symbol, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(symbol_gray, 150, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.merge([mask, mask, mask])

    roi = background[y_offset:y_offset+symbol_h, x_offset:x_offset+symbol_w]
    foreground = cv2.bitwise_and(symbol, mask)
    background_masked = cv2.bitwise_and(roi, cv2.bitwise_not(mask))
    result = cv2.add(foreground, background_masked)

    background[y_offset:y_offset+symbol_h, x_offset:x_offset+symbol_w] = result

    return background

def add_foreground2background(bg, fg):
    foreground_gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(foreground_gray, 100, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.merge([mask, mask, mask])

    foreground_part = cv2.bitwise_and(fg, mask)
    background_part = cv2.bitwise_and(bg, cv2.bitwise_not(mask))
    result = cv2.add(foreground_part, background_part)

    return result

def create_synthetic_bgfg(background, foreground, nums):
    image_num = len(background)
    synthetic_background = []
    synthetic_foreground = []

    for n in range(nums):
        bg = np.zeros((480, 480, 3), dtype=np.uint8)
        fg = np.zeros((480, 480, 3), dtype=np.uint8)
        for i in range(3):
            for j in range(3):
                index = random.randint(0, image_num - 1)
                slice_background = background[index]
                result = flip(image=slice_background)
                slice_background = result['image']
                replay = result['replay']
                slice_foreground = foreground[index]
                result = A.ReplayCompose.replay(replay, image=slice_foreground)
                slice_foreground = result['image']
                h, w = slice_background.shape[:2]
                h_start = random.randint(0, h - 160)
                w_start = random.randint(0, w - 160)
                bg[i*160:(i+1)*160, j*160:(j+1)*160, :] = slice_background[h_start:h_start + 160, w_start:w_start + 160, :]
                fg[i*160:(i+1)*160, j*160:(j+1)*160, :] = slice_foreground[h_start:h_start + 160, w_start:w_start + 160, :]
        full_results = transform(image=bg)
        synthetic_background.append(full_results['image'])
        cv2.imwrite(f'{n}bg.jpg', full_results['image'])
        full_replay = full_results['replay']
        full_results = A.ReplayCompose.replay(full_replay, image=fg)
        synthetic_foreground.append(full_results['image'])
        cv2.imwrite(f'{n}fg.jpg', full_results['image'])

    return synthetic_background, synthetic_foreground

def create_synthetic_image(background, foreground, symbols):
    image_index = random.randint(0, len(background) - 1)
    image = background[image_index]
    fg = foreground[image_index]
    new_image = image.copy()
    labels = ''
    for i in range(int(480 / 40)):
        for j in range(int(480 / 40)):
            sym_index = random.randint(0, len(symbols) - 1)
            symbol = symbols[sym_index]
            h, w = symbol.shape[:2]
            x_offset = i * 40 + random.randint(0, 40 - w)
            y_offset = j * 40 + random.randint(0, 40 - h)
            new_image = add_symbol2background(new_image, symbol, x_offset, y_offset)
            x_center = x_offset + (w / 2)
            x = x_center / 480
            y_center = y_offset + (h / 2)
            y = y_center / 480
            h = h / 480
            w = w / 480
            labels += f'{sym_index} {x} {y} {w} {h}\n'
    new_image = add_foreground2background(new_image, fg)
    return new_image, labels

def get_symbols(folder_path):
    symbols = []
    # symbols_shape = []
    for i in range(1, 9):
        image = cv2.imread(os.path.join(folder_path, f'L{i}.png'))
        # h, w = image.shape[:2]
        symbols.append(image)
        # symbols_shape.append((h, w))
        
    return symbols

def get_images(background_folder_path, foreground_folder_path):
    files = os.listdir(background_folder_path)
    background = []
    foreground = []
    # images_shape = []
    for f in files:
        background_path = os.path.join(background_folder_path, f)
        if background_path.endswith('.png') or background_path.endswith('.jpg'):
            background_image = cv2.imread(background_path)
            foreground_path = os.path.join(foreground_folder_path, f)
            if os.path.exists(foreground_path):
                background.append(background_image)
                foreground_image = cv2.imread(foreground_path)
                foreground.append(foreground_image)
    return background, foreground

if __name__ == '__main__':
    background, foreground = get_images("./background", "./foreground")
    symbols = get_symbols("./symbols")
    background, foreground = create_synthetic_bgfg(background, foreground,3)

    for i in range(4):
        syn_images, syn_labels = create_synthetic_image(background, foreground, symbols)
        cv2.imwrite(f'syn_{i}.png', syn_images)
        with open(f'syn_{i}.txt', 'w') as f:
            f.write(syn_labels)
