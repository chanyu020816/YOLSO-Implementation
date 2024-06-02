import os
import cv2
import random
import argparse
from tqdm import tqdm 
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--background_dir', '-bg', type=str, default='./background/')
    parser.add_argument('--symbol_dir', '-fg', type=str, default='./symbol/')
    parser.add_argument('--output_dir', '-o', type=str, default='./synthetic_data/')
    parser.add_argument('--synthetic_nums', '-n', type=int, default=10, help='Number of synthetic images')
    parser.add_argument('--synthetic_bg_nums', '-bgn', type=int, default=3, help='Number of synthetic background')
    args = parser.parse_args()

    background, foreground = utils.get_images("./background", "./foreground")
    symbols = utils.get_symbols("./symbols")

    if args.synthetic_bg_nums != 0:
        background, foreground = utils.create_synthetic_bgfg(background, foreground, args.synthetic_bg_nums)

    for i in tqdm(range(args.synthetic_nums)):
        syn_images, syn_labels = utils.create_synthetic_image(background, foreground, symbols)
        cv2.imwrite(os.path.join(args.output_dir, f'syn_{i}.png'), syn_images)
        with open(os.path.join(args.output_dir, f'syn_{i}.txt'), 'w') as f:
            f.write(syn_labels)