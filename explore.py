from scipy.ndimage import distance_transform_bf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from skimage.measure import label, regionprops

def extract_main_content(image: np.ndarray) -> np.ndarray:
    # this func extracts the largest contour in the image,
    # based on the assumption that the brain is the largest object in the image
    # Convert image to grayscale if it's 3-ch
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply threshold to convert to binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = binary_dilation(binary, structure=np.ones(shape=(7, 7))).astype(np.uint8)
    # Find contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Create a mask image
    mask = np.zeros_like(gray)
    # Draw the largest contour on the mask image
    cv2.drawContours(mask, contours, 0, 255, cv2.FILLED)
    mask = binary_erosion(mask, structure=np.ones(shape=(7, 7))).astype(np.uint8) * 255
    result = cv2.bitwise_and(gray, mask)
    brain = np.where(result > 100, 0, result)
    return brain


def generate_brains(in_root: Path, out_root: Path):
    for img_path in tqdm(in_root.glob('*.jpg'), desc='Generating possible ROIs'):
        image = cv2.imread(str(img_path))
        brain = extract_main_content(image)
        out_name = str(out_root / img_path.name)
        cv2.imwrite(out_name, brain)

def generate_vector(brain_img: np.ndarray) -> Tuple[np.ndarray, float]:
    # generate a feature vector of a sample by combining a histogram and taking 95th percentile
    brain_only = brain_img[brain_img > 0]
    hist = np.histogram(brain_only, bins=4, density=False, range=(0, 100))[0]  # take values only
    hist = hist / np.sum(hist)  # normalize
    perc_95 = np.percentile(brain_only, q=95) / 100  # account for possible outliers
    return hist, perc_95


def clusterize(in_root: Path) -> Tuple[List[str], np.ndarray, Any]:
    fnames = []
    feats = []
    for i, img_path in tqdm(enumerate(in_root.glob('*.jpg'))):
        # take some of the images
        if i < 100:  # 100 is an arbitrary number, you may pick a larger number
            image = cv2.imread(str(img_path), 0)
            if np.sum(image > 0) < 3_500:
                continue  # skip very small images if present
            fnames.append(img_path.name)
            hist, perc_95 = generate_vector(image)
            feat = hist.tolist()[2:]
            feat.extend([perc_95])
            feats.append(feat)
    feats = np.array(feats)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=20).fit(feats)
    return fnames, feats, kmeans

def choose_closest_to_center(feats: np.ndarray, kmeans: Any, top_k: int = 15) -> Tuple[List[int], List[int]]:
    # get indices of the top k closest samples to the respective cluster centroids
    centers = kmeans.cluster_centers_
    center_dist_0 = np.linalg.norm(np.tile(centers[0], (feats.shape[0], 1)) - feats, axis=1)
    center_dist_1 = np.linalg.norm(np.tile(centers[1], (feats.shape[0], 1)) - feats, axis=1)
    top_closest_0_inds = np.argsort(center_dist_0)[:top_k]
    top_closest_1_inds = np.argsort(center_dist_1)[:top_k]
    return top_closest_0_inds, top_closest_1_inds

def calculate_iou(brain_1: np.ndarray, brain_2: np.ndarray) -> float:
    # takes two non-binary roi images and calculates iou
    brain_1_bin = np.where(brain_1 > 0, 1, 0)
    brain_2_bin = np.where(brain_2 > 0, 1, 0)
    inter = np.sum(np.logical_and(brain_1_bin, brain_2_bin))
    union = np.sum(brain_1_bin) + np.sum(brain_2_bin) - inter
    return inter / union

def remove_small_regions(bin_array: np.ndarray, min_area: int):
    labeled_objs = label(bin_array)
    output_arr = labeled_objs.copy()
    for lbl in np.unique(labeled_objs):
        area = np.sum(labeled_objs == lbl)
        if area < min_area:
            output_arr = np.where(output_arr == lbl, 0, output_arr)
    return np.where(output_arr > 0, 1, 0)

def get_distributions_from_suspicious(brain: np.ndarray, suspicious_region_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    brain_susp_region = brain[suspicious_region_mask == 1]
    brain_without_susp_region = brain[suspicious_region_mask == 0]
    susp_region_dist = np.histogram(brain_susp_region, bins=10, range=(0, 100), density=True)[0]
    susp_region_mean = np.mean(brain_susp_region)
    brain_without_susp_dist = np.histogram(brain_without_susp_region, bins=10, range=(0, 100), density=True)[0]
    brain_without_susp_mean = np.mean(brain_without_susp_region)
    return brain_without_susp_dist, susp_region_dist, susp_region_mean, brain_without_susp_mean

def is_pathology(mean_1: float, mean_2: float, thresh_mean: float) -> bool:
    M = abs(mean_1 - mean_2)
    return M > thresh_mean

def find_suspicious_region(brain_1: np.ndarray, brain_2: np.ndarray,
                           thresh_low: int = 40, thresh_high: int = 80,
                           struct_1: int = 3, struct_2: int = 5, min_area: int = 100) -> np.ndarray:
    brain_1_bin = np.where(brain_1 > 0, 1, 0)
    brain_2_bin = np.where(brain_2 > 0, 1, 0)
    inter_region = (np.logical_and(brain_1_bin, brain_2_bin) * 255).astype(np.uint8)
    # for both brains remove area that is not common
    brain_1 = cv2.bitwise_and(brain_1, inter_region)
    brain_2 = cv2.bitwise_and(brain_2, inter_region)
    diff = np.abs(brain_1.astype(np.float64) - brain_2.astype(np.float64)).astype(np.uint8)
    diff_bin = np.where((diff > thresh_low) & (diff < thresh_high), 255, 0)
    diff_bin_eroded = binary_erosion(diff_bin, structure=np.ones((struct_1, struct_1))).astype(diff_bin.dtype)
    diff_bin_dilated = binary_dilation(diff_bin_eroded, structure=np.ones((struct_2, struct_2))).astype(diff_bin.dtype)
    suspicious_region_mask = remove_small_regions(bin_array=diff_bin_dilated, min_area=min_area)
    return suspicious_region_mask

def refine_final_mask(region: np.ndarray):
    # makes binary masks more appealing by applying convex Hull
    contours, hierarchy = cv2.findContours((region*255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(region)
    final_mask = np.ascontiguousarray(final_mask, dtype=np.uint8)
    # calculate points for each contour
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(final_mask, [hull], 0, 255, cv2.FILLED)
    return final_mask

def compare_closest_sample(in_root: Path, out_root: Path, fnames: List[str],
                           top_closest_0_inds: List[int], top_closest_1_inds: List[int],
                            thresh_low: int = 40, thresh_high: int = 80,
                           struct_1: int = 3, struct_2: int = 5, min_area: int = 100,
                           iou_threshold: float = 0.9):
    for i, ind_0 in tqdm(enumerate(top_closest_0_inds), desc='Comparing clusters'):
        brain_0 = cv2.imread(str(in_root / fnames[ind_0]), 0)
        for j, ind_1 in enumerate(top_closest_1_inds):
            brain_1 = cv2.imread(str(in_root / fnames[ind_1]), 0)
            iou = calculate_iou(brain_1=brain_0, brain_2=brain_1)
            if iou < iou_threshold or fnames[ind_0] == fnames[ind_1]:
                continue  # skip self-check and those samples that didn't pass the threshold
            # segment suspicious region by abs diff between two samples
            susp_region = find_suspicious_region(brain_1=brain_0, brain_2=brain_1, thresh_low=thresh_low, thresh_high=thresh_high,
                                                 struct_1=struct_1, struct_2=struct_2, min_area=min_area)
            brain_0_without_susp_dist, brain_0_susp_region_dist, susp_region_mean_0, brain_without_susp_mean_0 = \
                get_distributions_from_suspicious(brain=brain_0, suspicious_region_mask=susp_region)
            brain_1_without_susp_dist, brain_1_susp_region_dist, susp_region_mean_1, brain_without_susp_mean_1 = \
                get_distributions_from_suspicious(brain=brain_1, suspicious_region_mask=susp_region)

            is_brain_0_hem = is_pathology(susp_region_mean_0,
                                          brain_without_susp_mean_0, thresh_mean=40)
            is_brain_1_hem = is_pathology(susp_region_mean_1,
                                          brain_without_susp_mean_1, thresh_mean=40)
            susp_region = refine_final_mask(susp_region)
            if is_brain_0_hem:
                cv2.imwrite(str(out_root / fnames[ind_0]), susp_region)
            if is_brain_1_hem:
                cv2.imwrite(str(out_root / fnames[ind_1]), susp_region)


if __name__ == '__main__':
    # todo: change paths
    in_root = Path('data/competition/competition')
    out_root = Path('data/brains')
    out_root.mkdir(exist_ok=True, parents=True)
    generate_brains(in_root=in_root, out_root=out_root)

    in_root = out_root
    out_root = Path('data/pseudo')
    out_root.mkdir(exist_ok=True, parents=True)
    fnames, feats, kmeans = clusterize(in_root=in_root)
    top_closest_0_inds, top_closest_1_inds = choose_closest_to_center(feats=feats, kmeans=kmeans, top_k=50)
    compare_closest_sample(in_root=in_root, out_root=out_root, fnames=fnames, top_closest_0_inds=top_closest_0_inds,
                           top_closest_1_inds=top_closest_1_inds, iou_threshold=.80, thresh_low=30, thresh_high=70,
                           struct_1=3, struct_2=11, min_area=800)
