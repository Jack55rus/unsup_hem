import numpy as np
from tap import Tap
from pathlib import Path
from typing import List, Optional, Sequence
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


class ArgumentParser(Tap):
    input_img_dir: Path
    bin_pred_dir_1: Path
    bin_pred_dir_2: Optional[Path] = None
    bin_pred_dir_3: Optional[Path] = None
    output_dir: Path

    # def configure(self) -> None:
    #     self.add_argument("-in", "--input_img_dir")
    #     self.add_argument("-out", "--output_dir")
    #     self.add_argument("-p1", "--bin_pred_dir_1")
    #     self.add_argument("-p2", "--bin_pred_dir_2")
    #     self.add_argument("-p3", "--bin_pred_dir_3")


def colorize_bin_image(bin_img: np.ndarray, channel: int) -> np.ndarray:
    bin_img_col = np.where(bin_img > 250, 255, 0)  # remove compression artifacts
    for i in range(3):
        if i != channel:
            bin_img_col[..., i] = 0
            bin_img_col[..., i] = 0
    return bin_img_col


def visualize_preds(input_img_dir: Path, bin_pred_dir_1: Path, output_dir: Path,
                    bin_pred_dir_2: Optional[Path] = None, bin_pred_dir_3: Optional[Path] = None):
    output_dir.mkdir(exist_ok=True, parents=True)
    orig_images = sorted([file.name for file in input_img_dir.glob('*.jpg')])
    pred_files_1 = sorted([file.name for file in bin_pred_dir_1.glob('*.jpg')])
    pred_dirs = [bin_pred_dir_1, bin_pred_dir_2, bin_pred_dir_3]
    assert pred_files_1 == orig_images, 'filenames do not match in the prediction and original folders'
    if bin_pred_dir_2:
        pred_files_2 = sorted([file.name for file in bin_pred_dir_2.glob('*.jpg')])
        assert pred_files_1 == pred_files_2, 'filenames do not match in the prediction folders'
    if bin_pred_dir_3:
        pred_files_3 = sorted([file.name for file in bin_pred_dir_3.glob('*.jpg')])
        assert pred_files_1 == pred_files_3, 'filenames do not match in the prediction folders'
    for fname in tqdm(orig_images, desc='Visualizing predictions'):
        img = cv2.imread(str(input_img_dir / fname), cv2.IMREAD_COLOR)  # read as 3 channels
        channels = 0
        bin_preds = []
        pred_1_bin = cv2.imread(str(bin_pred_dir_1 / fname), cv2.IMREAD_COLOR)
        pred_1_bin = colorize_bin_image(pred_1_bin, channel=channels)
        bin_preds.append(pred_1_bin)
        if bin_pred_dir_2:
            channels += 1
            pred_2_bin = cv2.imread(str(bin_pred_dir_2 / fname), cv2.IMREAD_COLOR)
            pred_2_bin = colorize_bin_image(pred_2_bin, channel=channels)
            bin_preds.append(pred_2_bin)
        if bin_pred_dir_3:
            channels += 1
            pred_3_bin = cv2.imread(str(bin_pred_dir_3 / fname), cv2.IMREAD_COLOR)
            pred_3_bin = colorize_bin_image(pred_3_bin, channel=channels)
            bin_preds.append(pred_3_bin)
        fig, ax = plt.subplots(1, len(bin_preds) + 1, figsize=(60, 20))
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("Input image", fontsize=30)
        for j in range(len(bin_preds)):
            ax[j+1].imshow(img, cmap="gray")
            ax[j+1].imshow(bin_preds[j], alpha=0.2)
            ax[j+1].set_title(f"Prediction {pred_dirs[j].name}", fontsize=30)
        plt.savefig(output_dir / fname)
        fig.clf()
        plt.close()

if __name__ == '__main__':
    args = ArgumentParser().parse_args()
    print(args)
    visualize_preds(input_img_dir=args.input_img_dir, output_dir=args.output_dir, bin_pred_dir_1=args.bin_pred_dir_1,
                    bin_pred_dir_2=args.bin_pred_dir_2, bin_pred_dir_3=args.bin_pred_dir_3)
