import os
import cv2
import random
from tqdm import tqdm 

def add_symbol_to_background(background, symbol, symbol_shape, x_offset, y_offset):
    symbol_h, symbol_w = symbol_shape
    symbol_gray = cv2.cvtColor(symbol, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(symbol_gray, 254, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.merge([mask, mask, mask])

    roi = background[y_offset:y_offset+symbol_h, x_offset:x_offset+symbol_w]
    foreground = cv2.bitwise_and(symbol, mask)
    background_masked = cv2.bitwise_and(roi, cv2.bitwise_not(mask))
    result = cv2.add(foreground, background_masked)

    background[y_offset:y_offset+symbol_h, x_offset:x_offset+symbol_w] = result

    return background

def synthetic_bg(background_images):
    nums = images.shape[0]
    
def read_symbols(folder_path):
    
    symbols = []
    symbols_shape = []
    for i in range(1, 9):
        files = [f'L{i}.png']