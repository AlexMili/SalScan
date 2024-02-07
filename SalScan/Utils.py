import glob
import hashlib
import json
import logging
import os
import re
import tempfile
import warnings
from typing import Any, Optional, Union

import colorlog
import cv2
import numpy as np

from SalScan.Metric.Saliency import EPSILON

# Types of float that can be used to store an image
DTYPES = {16: np.float16, 32: np.float32, 64: np.float64}
CVTYPES = {16: cv2.CV_16UC1, 32: cv2.CV_32F, 64: cv2.CV_64F}


def _get_console_handler() -> Any:
    console_handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(fg_black)s%(bg_white)s%(name)s %(asctime)s%(reset)-4s"
        " %(log_color)s[%(levelname)s]: %(log_color)s%(message)s",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red, bg_white",
        },
        style="%",
    )
    console_handler.setFormatter(formatter)
    return console_handler


def get_logger(logger_name: str, logger_level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)
    logger.addHandler(_get_console_handler())
    logger.propagate = False
    return logger


def has_files(path: str, pattern) -> bool:
    r = re.compile(pattern)
    files = glob.glob(os.path.join(path, "*"))
    files_found = len(list(filter(r.search, files)))

    if files_found == 0:
        return False
    else:
        return True


def fixmap_2_salmap(
    fix_map: np.ndarray, ref_size: int = 256, ref_sigma: int = 8
) -> np.ndarray:
    """This function takes as input a fixation_map [0, 1] and returns a
        blurred saliency map. Default ref_size and ref_sigma were determined
        heuristically to generate a saliency map as closer as possible
        to the ones of MIT1003 starting from the corresponding fixation maps.
        The two arguments ref_size and ref_sigma determine the kernel size
        and sigma: specifically value of sigma is influenced by the ref_size
        and the value of kernel_size is influenced by both ref_size and ref_sigma.

    Args:
        fix_map (np.ndarray): fixation map [0, 1]
        ref_size (int): default image size
        ref_sigma (int): default kernel size
    """
    if np.min(fix_map) < 0:
        raise ValueError("Your array has negative values")
    if np.max(fix_map) > 1:
        raise ValueError("Your array has values exceeding 1")

    # The ref_size shouldn't be modified because changing it
    # would alter the way sigma adapts to different image sizes
    # Furthermore it would even alter the way kernel_size is adapted
    # to different image sizes. ref_size = 256 is the result of
    # heuristical experiments whose scope was to recreate MIT1003
    # sal_maps given the corresponding fix_maps with the latter
    # ones at different sizes (height and width)

    if ref_size != 256:
        warnings.warn(
            "\n[WARNING] ref_size is not set to 256, resulting"
            "sal_map and metrics will be influenced"
        )
    # The ref_sigma shouldn't be modified because changing it
    # would alter the way kernel_size adapts to different image sizes
    # ref_sigma = 8 is the result of heuristical experiments whose
    # scope was to recreate MIT1003 sal_maps given the corresponding
    # fix_maps with the latter ones at different sizes (height and width)

    if ref_sigma != 8:
        warnings.warn(
            "\n[WARNING] ref_sigma is not set to 8, resulting"
            "sal_map and metrics will be influenced"
        )

    # the sigma is computed relative to the lower dimension to avoid
    # scenarios where an highly rectangular image is fed to the function
    # and fixations blurs become too large.

    min_dimension = np.min([fix_map.shape[0], fix_map.shape[1]])
    sigma = ref_sigma * (min_dimension / ref_size)
    # the below formula suggested here: https://stackoverflow.com/a/62002971
    kernel_size = 2 * int(3 * sigma) + 1

    sal_map = cv2.GaussianBlur(
        fix_map, (kernel_size, kernel_size), sigma, borderType=cv2.INTER_NEAREST
    )

    sal_map = cv2.normalize(
        sal_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    return sal_map


def normalize(x, method="standard", axis=None):
    """Normalize an array.

    Args:
        x (np.array): the array to normalize.
        method (str, opt): method used for normalisation.
            Can be either "standard", "range" or "sum".
            Default is standard.
        axis (int, opt): axis along which the normalisation has to be done.
            Default is None.

    Returns:
        res (np.array): the normalized array.
    """
    x = np.array(x, copy=False)

    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]

        if method == "standard":
            res = x - np.mean(y, axis=1).reshape(shape)
            res /= np.std(y, axis=1).reshape(shape)
        elif method == "range":
            res = x - np.min(y, axis=1).reshape(shape)
            res /= (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == "sum":
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')

    else:
        if method == "standard":
            res = (x - np.mean(x)) / np.std(x)
        elif method == "range":
            res = (x - np.min(x)) / (np.max(x) - np.min(x) + EPSILON)
        elif method == "sum":
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')

    return res


def generate_tmp_filename(filename_ext: str = "jpg", filename_length: int = 10) -> str:
    tmp_dir = tempfile.gettempdir()
    tmp_dir = os.path.join(tmp_dir, "salscan_data")

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)

    filename = hashlib.md5(os.urandom(filename_length)).hexdigest()

    return os.path.join(tmp_dir, f"{filename}.{filename_ext}")


def is_json_serializable(variable: Any) -> bool:
    try:
        json.dumps(variable)
        return True
    except TypeError:
        return False


def gauss_c(
    x: np.ndarray, y: np.ndarray, sigma: Union[float, int], center: np.ndarray
) -> np.ndarray:
    # For formula see:
    # https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    xc = center[0, 0]
    yc = center[0, 1]

    # Only divided by 2sigma^2 because we want a gaussian like a circle instead of a disk
    exponent = (np.square(x - xc) + np.square(y - yc)) / (2 * np.power(sigma, 2))
    val = np.exp(-exponent)
    return val


def gauss2d(img: np.ndarray, sigma: Union[float, int], center: np.ndarray) -> np.ndarray:
    # From https://fr.mathworks.com/matlabcentral/answers/13020-2d-gaussian-function
    gsize = img.shape
    b = np.arange(0, gsize[1])
    a = np.arange(0, gsize[0])
    (C, R) = np.meshgrid(b, a)
    mat = gauss_c(C, R, sigma, center)
    return mat


def load_video(path: str) -> Optional[cv2.VideoCapture]:
    if os.path.exists(path) is False:
        raise FileExistsError(f"'{path}' does not exist")

    video_stream = cv2.VideoCapture(path)

    if video_stream.isOpened() is False:
        raise ValueError(f"Failed to load video file '{path}'")

    # Read a frame to see if everything is working fine
    flag, frame = video_stream.read()

    if flag is False or flag is None:
        raise ValueError("Can't acces frames in video stream")

    if frame is None:
        raise ValueError("Tested frame is empty")

    # Reset to frame 0
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return video_stream


def release_video(video_stream: Optional[cv2.VideoCapture]) -> None:
    if video_stream is not None and video_stream.isOpened() is True:
        video_stream.release()
