import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm


class SSLDataset(Dataset):
    def __init__(self, auto_mask_dir: Path, img_root_dir: Path, transforms: Optional = None):
        self.auto_mask_dir = auto_mask_dir
        self.img_root_dir = img_root_dir
        self.sup_fnames, self.unsup_fnames = self.partition_fnames()
        self.num_sup_files = len(self.sup_fnames)
        self.num_unsup_files = len(self.unsup_fnames)
        self.transforms = transforms

    def partition_fnames(self) -> Tuple[List[str], List[str]]:
        all_fnames = [x.name for x in self.img_root_dir.glob('*.jpg') if x.is_file()]
        sup_fnames = [x.name for x in self.auto_mask_dir.glob('*.jpg') if x.is_file()]  # fnames for which masks are available
        unsup_fnames = list(set(all_fnames).difference(set(sup_fnames)))  # all other files used for the unsup part
        return sup_fnames, unsup_fnames

    def __len__(self):
        return max(self.num_sup_files, self.num_unsup_files)

    def apply_augs(self, sup_image: np.ndarray, mask: np.ndarray,
                   unsup_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sup_image = self.transforms.space_invariant(image=sup_image)['image']
        unsup_image_1 = self.transforms.space_variant(image=unsup_image)['image']
        space_variant_transform = self.transforms.space_variant(image=sup_image, mask=mask)
        sup_image, mask = space_variant_transform['image'], space_variant_transform['mask']
        unsup_image_2 = self.transforms.space_invariant(image=unsup_image_1)['image']
        return sup_image, mask, unsup_image_1, unsup_image_2

    def __getitem__(self, idx):
        if self.num_unsup_files >= self.num_sup_files:  # if there are more unsup files than sup ones (almost 100% prob)
            idx_unsup = idx
            idx_sup = idx % self.num_sup_files
        else:
            idx_sup = idx
            idx_unsup = idx % self.num_unsup_files
        unsup_image = cv2.imread(str(self.img_root_dir / self.unsup_fnames[idx_unsup]), 0)
        sup_image = cv2.imread(str(self.img_root_dir / self.sup_fnames[idx_sup]), 0)
        mask = cv2.imread(str(self.auto_mask_dir / self.sup_fnames[idx_sup]), 0)
        if self.transforms is not None:
            sup_image, mask, unsup_image_1, unsup_image_2 = self.apply_augs(sup_image, mask, unsup_image)
            sup_image, mask, unsup_image_1, unsup_image_2 = torch.from_numpy(sup_image / 255), torch.from_numpy(mask / 255), \
                torch.from_numpy(unsup_image_1 / 255), torch.from_numpy(unsup_image_2 / 255)
            sup_image, mask, unsup_image_1, unsup_image_2 = torch.unsqueeze(sup_image, 0), torch.unsqueeze(mask, 0), \
                torch.unsqueeze(unsup_image_1, 0), torch.unsqueeze(unsup_image_2, 0)
            return sup_image.double(), mask.double(), unsup_image_1.double(), unsup_image_2.double()
        sup_image, mask, unsup_image = torch.from_numpy(sup_image / 255), torch.from_numpy(mask / 255), \
            torch.from_numpy(unsup_image / 255)
        sup_image, mask, unsup_image= torch.unsqueeze(sup_image, 0), torch.unsqueeze(mask, 0), \
            torch.unsqueeze(unsup_image, 0)
        return sup_image.double(), mask.double(), unsup_image.double()


class Augs:
    def __init__(self):
        self.space_invariant = A.Compose([
            A.GaussNoise(var_limit=(0.1, 10), mean=0, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.8, 1.0), p=0.3),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.8),
            A.MotionBlur(blur_limit=7, p=0.05),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
            A.MedianBlur(blur_limit=5, p=0.3),
        ])
        self.space_variant = A.Compose([
            A.ElasticTransform(alpha=20, sigma=50, alpha_affine=50, p=0.5),
            A.GridDistortion(num_steps=10, distort_limit=0.5, p=0.3),
            A.Perspective(scale=(0.01, 0.04), keep_size=True, p=0.1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=(-270, 270), p=0.6),
            A.RandomScale(scale_limit=(-0.19, 0.19), p=0.5),
            A.Resize(height=512, width=512, always_apply=True)
        ])

def train_iter(model: nn.Module, train_loader: DataLoader, optimizer, loss_sup, loss_unsup, device: str,
               print_interval: int, img_save_dir: Path, weights_save_dir: Path, train_params, log_dir: Path,
               save_interval: int):
    model.train()
    loss_dict = {'segm_loss': [], 'ssl_loss': [], 'total_loss': []}
    log_dict = {'segm_loss': [], 'ssl_loss': []}
    iters = []
    for it, (sup_image, mask, unsup_image_1, unsup_image_2) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        sup_image, mask, unsup_image_1, unsup_image_2 = sup_image.to(device), mask.to(device), unsup_image_1.to(device), unsup_image_2.to(device)
        sup_y, unsup_y1, unsup_y2 = model(sup_image, unsup_image_1, unsup_image_2)
        segm_loss = loss_sup(sup_y, mask)
        ssl_loss = loss_unsup(unsup_y1, unsup_y2, target=torch.ones(size=(unsup_y2.shape[0],)).to(device))
        total_loss = segm_loss + ssl_loss
        total_loss.backward()
        loss_dict['segm_loss'].append(segm_loss.item())
        loss_dict['ssl_loss'].append(ssl_loss.item())
        loss_dict['total_loss'].append(total_loss.item())
        optimizer.step()
        if it % print_interval == 0 and it != 0:
            inference(model=model, x=unsup_image_1, save_path=img_save_dir, iteration=it)
            iters.append(it)
            log_dict['segm_loss'].append(np.mean(loss_dict['segm_loss'][-print_interval:]))
            log_dict['ssl_loss'].append(np.mean(loss_dict['ssl_loss'][-print_interval:]))
            print(f"Segm loss: {log_dict['segm_loss'][-1]}, SSL loss: {log_dict['ssl_loss'][-1]}, "
                  f"total: {log_dict['segm_loss'][-1] + log_dict['ssl_loss'][-1]}")
            plt.plot(iters, log_dict['segm_loss'], label='segm loss')
            plt.plot(iters, log_dict['ssl_loss'], label='ssl_loss')
            plt.savefig(log_dir / 'loss.jpg')
        if it % save_interval == 0 and it != 0:
            save_weights(model=model, train_params=train_params, segm_loss=segm_loss.item(), ce_loss=ssl_loss.item(),
                         save_dir=weights_save_dir, it=it, optimizer=optimizer)
    loss_dict['segm_loss'] = sum(loss_dict['segm_loss']) / len(loss_dict['segm_loss'])
    loss_dict['ssl_loss'] = sum(loss_dict['ssl_loss']) / len(loss_dict['ssl_loss'])
    loss_dict['total_loss'] = sum(loss_dict['total_loss']) / len(loss_dict['total_loss'])
    return loss_dict


def save_weights(model, train_params, segm_loss, ce_loss, save_dir, it, optimizer):
    save_dict = {
        'state_dict': model.state_dict(),
        'opt_dict': optimizer.state_dict(),
        'params': train_params,
        'curr_iter': it,
        'LR': optimizer.param_groups[0]['lr'],
        'segm_loss': segm_loss,
        'ce_loss': ce_loss,
        'total_loss': segm_loss + ce_loss
    }
    torch.save(save_dict, save_dir / f'weights_{it}.pth')


@torch.no_grad()
def inference(model: nn.Module, x: torch.Tensor, save_path: Path, iteration: int):
    # visualize output for the first image in a batch
    model.eval()
    x = x[0:1, ...]
    y = model(x, None, None)
    y = ((torch.sigmoid(y) > 0.5) * 255).type(torch.uint8)
    y = y[0, 0, ...]
    y = np.array(y.cpu())
    x = (x * 255).type(torch.uint8)
    x = x[0, 0, ...]
    x = np.array(x.cpu())
    xy = np.hstack((x, y))
    plt.imsave(save_path / f'{iteration}.jpg', xy, cmap='gray')
    model.train()


def calculate_iou(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    # takes two non-binary roi images and calculates iou
    inter = np.sum(np.logical_and(y_pred, y_true))
    union = np.sum(y_pred) + np.sum(y_true) - inter
    return inter / union


# code for the network
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels - out_channels, in_channels - out_channels, kernel_size=2,
                                                stride=2)
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, out_classes=2, unsup_dim: int = 32, up_sample_mode='conv_transpose'):
        super(UNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Downsampling Pat
        self.down_conv1 = DownBlock(1, 32) # 1, 32
        self.down_conv2 = DownBlock(32, 64)  # 32, 64
        self.down_conv3 = DownBlock(64, 128)  # 64, 128
        self.down_conv4 = DownBlock(128, 256)  # 128, 256
        # Bottleneck
        self.double_conv = DoubleConv(256, 512)  # 256, 512
        # Upsampling Path
        self.up_conv4 = UpBlock(256 + 512, 256, self.up_sample_mode)  # 256 + 512, 256
        self.up_conv3 = UpBlock(128 + 256, 128, self.up_sample_mode)  # 128 + 256, 128
        self.up_conv2 = UpBlock(64 + 128, 64, self.up_sample_mode)  # 64 + 128, 64
        self.up_conv1 = UpBlock(64 + 32, 32, self.up_sample_mode)  # 64 + 32, 32
        # Final Convolution
        self.conv_last = nn.Conv2d(32, out_classes, kernel_size=1)  # 32
        ## unsup branch
        self.gap = nn.MaxPool2d(kernel_size=32)
        self.ssl_feats = nn.Sequential(
                    nn.Linear(in_features=512, out_features=256),  # 512, 256
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(in_features=256, out_features=unsup_dim) # 256, unsup_dim
                )

    def forward_sup(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x

    def forward_unsup(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.gap(x)
        x = self.ssl_feats(x.view(x.shape[0], -1))
        return x

    def forward(self, sup_x, unsup_x1, unsup_x2):
        sup_y = self.forward_sup(sup_x)
        if unsup_x1 is not None and unsup_x2 is not None:
            unsup_y1 = self.forward_unsup(unsup_x1)
            unsup_y2 = self.forward_unsup(unsup_x2)
            return sup_y, unsup_y1, unsup_y2
        return sup_y
