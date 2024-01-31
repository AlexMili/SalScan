# this file contains all the transformation (data augmentation) used for evaluating models

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


def grey_normalize(img: np.array) -> np.array:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    return img


# Kao Zhang (2019) in IIP_TwoS_Saliency
# Implementations by https://github.com/zhangkao/IIP_TwoS_Saliency
class Padding:
    def __init__(self, shape_r=480, shape_c=640, channels=3, normalize=False):
        self.channels = channels
        self.shape_r = shape_r
        self.shape_c = shape_c
        self.normalize = normalize

    def __call__(self, img):
        img_padded = np.zeros((self.shape_r, self.shape_c, self.channels), dtype=np.uint8)
        if self.channels == 1:
            img_padded = np.zeros((self.shape_r, self.shape_c), dtype=np.uint8)

        original_shape = img.shape
        rows_rate = original_shape[0] / self.shape_r
        cols_rate = original_shape[1] / self.shape_c

        if rows_rate > cols_rate:
            new_cols = (original_shape[1] * self.shape_r) // original_shape[0]
            img = cv2.resize(img, (new_cols, self.shape_r))
            if new_cols > self.shape_c:
                new_cols = self.shape_c
            img_padded[
                :,
                ((img_padded.shape[1] - new_cols) // 2) : (
                    (img_padded.shape[1] - new_cols) // 2 + new_cols
                ),
            ] = img
        else:
            new_rows = (original_shape[0] * self.shape_c) // original_shape[1]
            img = cv2.resize(img, (self.shape_c, new_rows))
            if new_rows > self.shape_r:
                new_rows = self.shape_r
            img_padded[
                ((img_padded.shape[0] - new_rows) // 2) : (
                    (img_padded.shape[0] - new_rows) // 2 + new_rows
                ),
                :,
            ] = img

        if self.normalize is True:
            img_padded = np.divide(img_padded, 255)
        return img_padded


# Kao Zhang (2019) in IIP_TwoS_Saliency
# Implementations by https://github.com/zhangkao/IIP_TwoS_Saliency
def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols), np.uint8)
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0] * factor_scale_r))
        c = int(np.round(coord[1] * factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1

    return out


# Kao Zhang (2019) in IIP_TwoS_Saliency
# Implementations by https://github.com/zhangkao/IIP_TwoS_Saliency
def padding_fixation(img, shape_r=480, shape_c=640):
    img_padded = np.zeros((shape_r, shape_c), np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = resize_fixation(img, rows=shape_r, cols=new_cols)
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[
            :,
            ((img_padded.shape[1] - new_cols) // 2) : (
                (img_padded.shape[1] - new_cols) // 2 + new_cols
            ),
        ] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=shape_c)
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[
            ((img_padded.shape[0] - new_rows) // 2) : (
                (img_padded.shape[0] - new_rows) // 2 + new_rows
            ),
            :,
        ] = img

    return img_padded


class Preprocess_STSANet:
    "class used for STSANet stimuli preprocessing"

    def __init__(self, image_height, image_width):
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_height, image_width)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, img: np.array) -> np.array:
        # Transforms image from np to PIL and then back to np to comply with STSANet augmentation
        img = img.astype(np.uint8)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)
        return np.array(img)


class Preprocess_UniSal:
    "class used for UniSal stimuli preprocessing"
    preproc_cfg = {"rgb_mean": (0.485, 0.456, 0.406), "rgb_std": (0.229, 0.224, 0.225)}

    def __init__(self, out_size, common_aspect_ratio: list = None):
        if out_size is not None and common_aspect_ratio is not None:
            raise ValueError(
                "You can either set the out_size or compute the ideal aspect ratio"
            )

        if common_aspect_ratio is not None:
            out_size = self.get_optimal_out_size(common_aspect_ratio)
        transformations = []
        transformations.append(transforms.ToPILImage())
        transformations.append(transforms.Resize(out_size, interpolation=Image.LANCZOS))
        transformations.append(transforms.ToTensor())
        if "rgb_mean" in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg["rgb_mean"], self.preproc_cfg["rgb_std"]
                )
            )
        self.transform = transforms.Compose(transformations)

    def __call__(self, img: np.array) -> np.array:
        # Transforms image from np to PIL and then back to np to comply with STSANet augmentation
        img = img.astype(np.uint8)

        img = self.transform(img)
        return img

    def get_optimal_out_size(self, img_size):
        """_summary_

        Args:
            img_size (Union[List, Tuple]): height and width of a 'common' aspect ratio within your dataset

        Returns:
            out_size (Tuple): the most idoneus height and width for your dataset
        """
        ar = img_size[0] / img_size[1]
        min_prod = 100
        max_prod = 120
        ar_array = []
        size_array = []
        for n1 in range(7, 14):
            for n2 in range(7, 14):
                if min_prod <= n1 * n2 <= max_prod:
                    this_ar = n1 / n2
                    this_ar_ratio = min((ar, this_ar)) / max((ar, this_ar))
                    ar_array.append(this_ar_ratio)
                    size_array.append((n1, n2))

        max_ar_ratio_idx = np.argmax(np.array(ar_array)).item()
        bn_size = size_array[max_ar_ratio_idx]
        out_size = tuple(r * 32 for r in bn_size)
        return out_size
