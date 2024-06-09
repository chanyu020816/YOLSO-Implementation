import cv2
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--background_dir', '-bg', type=str, default='./background/')
    parser.add_argument('--foreground_dir', '-fg', type=str, default='./foreground/')
    parser.add_argument('--threshold', '-t', type=int, default=110, help='Gray value threshold')

    args = parser.parse_args()
    
    bg_imgs = os.listdir(args.background_dir)
    
    for img in tqdm(bg_imgs):
        fg_path = os.path.join(args.background_dir, img)