from submission_config import Params

from utils import UNet
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from datetime import date
from skimage import measure
from scipy.ndimage.morphology import binary_erosion


def filter_pred_by_area_np(pred: torch.Tensor) -> torch.Tensor:
    # remove tiny detections
    if np.sum(pred) == 0:
        return pred
    pred[pred > 0] = 1
    mask = measure.label(pred)
    objects = measure.regionprops(mask)
    object_ids_to_remove = []
    for obj in objects:
        if obj.area < Params.smallest_size:
            object_ids_to_remove.append(obj.label)
    mask[np.isin(mask, object_ids_to_remove)] = 0
    mask[mask != 0] = 255
    return mask


def remove_extra_areas(image: np.ndarray, erosion_window: int) -> np.ndarray:
    # remove potential artifacts in the image by removing small regions
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    binary = binary_erosion(binary, structure=np.ones(shape=(erosion_window, erosion_window))).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, 0, 255, cv2.FILLED)
    return mask / 255


if __name__ == '__main__':
    current_time = datetime.now().strftime("%H_%M_%S")
    today = date.today()
    d1 = today.strftime("%d_%m_%Y")
    save_time = f"{d1}_{current_time}"

    model = UNet(out_classes=1, unsup_dim=Params.unsup_emb).double().to(Params.device)
    model.load_state_dict(torch.load(Params.checkpoint_path)['state_dict'], strict=True)
    model.eval()

    if Params.submission_pred_dir:
        pred_save_dir = Params.submission_pred_dir / save_time
        pred_save_dir.mkdir(exist_ok=True, parents=True)
    if Params.submission_vis_dir:
        vis_dir = Params.submission_vis_dir / save_time
        vis_dir.mkdir(exist_ok=True, parents=True)

    img_paths = list(Params.submission_img_dir.glob('*.jpg'))
    if Params.num_files_to_infer != -1:
        img_paths = sorted(list(Params.submission_img_dir.glob('*.jpg')))[:Params.num_files_to_infer]

    submission_dict = {'ID': [], 'Value': []}

    with torch.no_grad():
        for fname in tqdm(img_paths, desc='Inference'):
            image = cv2.imread(str(fname), 0)  # 512x512
            mask = remove_extra_areas(image, erosion_window=Params.erosion_window)
            image = torch.from_numpy(image / 255).to(Params.device)  # normalize
            image = torch.unsqueeze(image, 0)  # 1x512x512
            image = torch.unsqueeze(image, 0).double()  # 1x1x512x512

            prediction = model(image, None, None)
            prediction = ((torch.sigmoid(prediction) > Params.threshold) * 255).type(torch.uint8)

            prediction = prediction[0, 0, ...]
            prediction = np.array(prediction.cpu())
            prediction = (prediction * mask).astype(np.uint8)
            prediction = filter_pred_by_area_np(prediction)

            for x in range(prediction.shape[1]):
                for y in range(prediction.shape[0]):
                    submission_dict['ID'].append(f'{fname.stem}_{x}_{y}')
                    submission_dict['Value'].append(int(prediction[x][y]))

            if Params.submission_pred_dir:
                plt.imsave(pred_save_dir / f'{fname.name}', prediction, cmap='gray')
            # vis purpose only
            if Params.submission_vis_dir:
                image = (image * 255).type(torch.uint8)
                image = image[0, 0, ...]
                image = np.array(image.cpu())
                xy = np.hstack((image, prediction))
                plt.imsave(vis_dir / f'{fname.name}', xy, cmap='gray')

    if Params.csv_path:
        Params.csv_path.mkdir(exist_ok=True, parents=True)
        save_path = Params.csv_path / f'{Params.csv_name}_{save_time}.csv'
        print('Generating submission csv')
        df = pd.DataFrame(submission_dict)
        df.to_csv(save_path, index=False)
