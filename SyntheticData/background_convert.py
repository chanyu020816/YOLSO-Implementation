import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--background_dir', '-bg', type=str, default='./background/')
    parser.add_argument('--foreground_dir', '-fg', type=str, default='./foreground/')
    parser.add_argument('--threshold', '-t', type=int, default=110, help='Gray value threshold')

    args = parser.parse_args()
    
    bg_imgs = os.listdir(args.background_dir)
    
    for img in tqdm(bg_imgs):
        if not img.endswith('.jpg') | img.endswith('.png'): continue
        bg_path = os.path.join(args.background_dir, img)
        fg_path = os.path.join(args.foreground_dir, img)
        if os.path.exists(fg_path):
            # print(f"Foreground image: {img} already exists")
            continue
        bg = cv2.imread(bg_path)
        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        mask = bg_gray > args.threshold
        fg = np.full_like(bg, 255)

        # Copy the original image colors where the mask is False
        fg[~mask] = bg[~mask]

        cv2.imwrite(fg_path, fg)

