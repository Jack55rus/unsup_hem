from submission_config import Params

from utils import UNet
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from datetime import date

def generate_csv(submission_pred_dir: Path, save_path: Path):
    submission_dict = {'ID': [], 'Value': []}
    fnames = sorted(list(submission_pred_dir.glob('*.jpg')))
    for fname in tqdm(fnames, desc='Generating CSV'):
        if '026725' in fname.stem:
            continue
        image = cv2.imread(str(fname), 0)  # 512x512
        image = np.where(image > 250, 1, 0)  # make sure it's binary
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                submission_dict['ID'].append(f'{fname.stem}_{x}_{y}')
                submission_dict['Value'].append(int(image[x][y]))
    df = pd.DataFrame(submission_dict)
    df.to_csv(save_path, index=False)

if __name__ == '__main__':
    current_time = datetime.now().strftime("%H_%M_%S")
    today = date.today()
    d1 = today.strftime("%d_%m_%Y")
    save_time = f"{d1}_{current_time}"

    model = UNet(out_classes=1, unsup_dim=Params.unsup_emb).double().to(Params.device)
    model.load_state_dict(torch.load(Params.checkpoint_path)['state_dict'], strict=True)
    model.eval()
    pred_save_dir = Params.submission_pred_dir / save_time
    pred_save_dir.mkdir(exist_ok=True, parents=True)
    vis_dir = Params.submission_vis_dir / save_time
    vis_dir.mkdir(exist_ok=True, parents=True)

    img_paths = list(Params.submission_img_dir.glob('*.jpg'))
    if Params.num_files_to_infer != -1:
        img_paths = np.random.choice(img_paths, size=Params.num_files_to_infer, replace=False)

    with torch.no_grad():
        for fname in tqdm(img_paths, desc='Inference'):
            image = cv2.imread(str(fname), 0)  # 512x512
            image = torch.from_numpy(image / 255).to(Params.device)  # normalize
            image = torch.unsqueeze(image, 0)  # 1x512x512
            image = torch.unsqueeze(image, 0).double()  # 1x1x512x512
            y = model(image, None, None)
            y = ((torch.sigmoid(y) > Params.threshold) * 255).type(torch.uint8)
            # print(torch.unique(y))
            y = y[0, 0, ...]
            y = np.array(y.cpu())
            plt.imsave(pred_save_dir / f'{fname.name}', y, cmap='gray')
            # vis purpose only
            image = (image * 255).type(torch.uint8)
            image = image[0, 0, ...]
            image = np.array(image.cpu())
            xy = np.hstack((image, y))
            plt.imsave(vis_dir / f'{fname.name}', xy, cmap='gray')
    Params.csv_path.mkdir(exist_ok=True, parents=True)
    save_path = Params.csv_path / f'{Params.csv_name}_{save_time}.csv'
    generate_csv(submission_pred_dir=pred_save_dir, save_path=save_path)