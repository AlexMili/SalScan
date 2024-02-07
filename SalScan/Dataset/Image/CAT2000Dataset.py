# -*- coding: utf-8 -*-

import glob
import os

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

from SalScan.Dataset.AbstractDataset import AbstractDataset, check_structure_path
from SalScan.Utils import get_logger

logger = get_logger(__name__)


class CAT2000Dataset(AbstractDataset):
    """
    CAT2000 Dataset class for image saliency evaluation.

    It allows to download the dataset automatically and it handles the retrieval of
    stimuli, saliency maps and fixation maps from the specified folder.

    Parameters:
        path (str): Root path where the dataset is stored or will be downloaded.
        download (bool): If True, the dataset will be downloaded to the specified
                        `path`.
        training_path (str): Path to the training CAT2000 dataset located outside of
                            SalScan. For example, given a CAT2000_training dataset,
                            `training_path` would be set for example to
                            `path/to/CAT2000_training/train`. This way all the files
                            contained in your training directory will not be retrieved by
                            this dataset. Notice, this works only if did not rename
                            image filenames within your custom dataset.

    Attributes:
        stimuli: Pandas dataframe composed by the following information: "id",
                "stimulus", "salmap", "fixmap", "label". These information will
                be inserted with the `populate` method.
        stimuli_shape: A tuple of int indicating the number of stimuli in the current
                        dataset
        name: "CAT2000"
        has_label: Set to True since CAT2000 folder is organized in such a way that
                    the parent folder of each stimulus can be meant as its label.
        training_images (str): When `training_path` is specified this attribute will
                                be a python `set` containing the filenames of all the
                                images located within `training_path`. If `training_path`
                                is not specified, `training_images` is set to `None`

    Reference:
        Borji, A., & Itti, L. (2015). CAT2000: A Large Scale Fixation Dataset for
        Boosting Saliency Research. [arXiv preprint arXiv:1505.03581]
    """

    _structure = {
        "fixations": [
            {
                "dir": ["FIXATIONLOCS"],
                "subdirs": [
                    "Affective",
                    "Art",
                    "BlackWhite",
                    "Cartoon",
                    "Fractal",
                    "Indoor",
                    "Inverted",
                    "Jumbled",
                    "LineDrawing",
                    "LowResolution",
                    "Noisy",
                    "Object",
                    "OutdoorManMade",
                    "OutdoorNatural",
                    "Pattern",
                    "Random",
                    "Satelite",
                    "Sketch",
                    "Social",
                ],
                "pattern": "[0-9]{3}.mat$",
            }
        ],
        "stimuli": [
            {
                "dir": ["Stimuli"],
                "subdirs": [
                    "Affective",
                    "Art",
                    "BlackWhite",
                    "Cartoon",
                    "Fractal",
                    "Indoor",
                    "Inverted",
                    "Jumbled",
                    "LineDrawing",
                    "LowResolution",
                    "Noisy",
                    "Object",
                    "OutdoorManMade",
                    "OutdoorNatural",
                    "Pattern",
                    "Random",
                    "Satelite",
                    "Sketch",
                    "Social",
                ],
                "pattern": "[0-9]{3}.jpg$",
            }
        ],
        "salmap": [
            {
                "dir": ["FIXATIONMAPS"],
                "subdirs": [
                    "Affective",
                    "Art",
                    "BlackWhite",
                    "Cartoon",
                    "Fractal",
                    "Indoor",
                    "Inverted",
                    "Jumbled",
                    "LineDrawing",
                    "LowResolution",
                    "Noisy",
                    "Object",
                    "OutdoorManMade",
                    "OutdoorNatural",
                    "Pattern",
                    "Random",
                    "Satelite",
                    "Sketch",
                    "Social",
                ],
                "pattern": "[0-9]{3}.jpg$",
            }
        ],
    }

    name = "CAT2000"
    has_label = True

    def __init__(
        self, path: str, download: bool = False, training_path: str = None
    ) -> None:
        super().__init__()

        self.root_path = path
        if download is True:
            self._download()

        check_structure_path(os.path.join(self.root_path, "trainSet"), self._structure)
        # Initilises self.training_images that will be overwritten by the
        # self.collect_training_images_method
        self.training_images = None
        if training_path is not None:
            self.collect_training_images(path=training_path)

    def populate(self) -> None:
        """
        Populates a dataframe with the paths of stimuli, fixation maps and saliency maps
        and the label corresponding to the image folder (i.e. "art", "cartoon",
        "satelite").
        """

        self.stimuli: pd.DataFrame = self._default_stimuli_values
        stim_search = os.path.join(self.root_path, "trainSet", "Stimuli", "**", "*.jpg")

        pbar_items = tqdm(glob.glob(stim_search))
        for stim in pbar_items:
            if self.training_images and os.path.basename(stim) in self.training_images:
                continue
            pbar_items.set_description(f"Dataset {self.name} - Populate stimuli")

            stim_filename = os.path.basename(stim)
            stim_dirname = os.path.basename(os.path.dirname(stim))
            self.stimuli = pd.concat(
                [
                    self.stimuli,
                    pd.DataFrame(
                        [
                            {
                                "id": self.stimuli.shape[0] + 1,
                                "stimulus": stim,
                                "salmap": os.path.join(
                                    self.root_path,
                                    "trainSet",
                                    "FIXATIONMAPS",
                                    stim_dirname,
                                    stim_filename,
                                ),
                                "fixmap": os.path.join(
                                    self.root_path,
                                    "trainSet",
                                    "FIXATIONLOCS",
                                    stim_dirname,
                                    stim_filename.replace(".jpg", ".mat"),
                                ),
                                "label": stim_dirname,
                            }
                        ]
                    ),
                ]
            )

        # Reset index because all index should be to 0
        self.stimuli = self.stimuli.reset_index(drop=True)

    def _load_stimulus(self, path: str) -> np.array:
        """
        Loads a stimulus image from the given path.

        Parameter:
            path (str): Path to the stimulus image file.

        Returns:
            np.array: The loaded stimulus image.
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_saliencymap(self, path: str) -> np.array:
        """
        Loads a saliency map from the given path.

        Parameter:
            path (str): Path to the saliency map file.

        Returns:
            np.array: The loaded saliency map.
        """
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

    def _load_fixationmap(self, path: str) -> np.array:
        """
        Loads a fixation map from the given path.

        Parameter:
            path (str): Path to the fixation map file.

        Returns:
            np.array: The loaded fixation map.
        """
        return sio.loadmat(path)["fixLocs"].astype(int)

    def __getitem__(self, idx):
        """
        Retrieves stimuli, saliency maps and fixation maps and the corresponding label.

        Parameter:
            idx (int): Index of the paths to retrieve.

        Returns:
            tuple: A tuple containing the path, stimulus, saliency map, fixation map, and
                    label of the data point.
        """
        stim_path = self.stimuli.loc[idx, "stimulus"]
        stimulus = self._load_stimulus(stim_path)
        salmap = self._load_saliencymap(self.stimuli.loc[idx, "salmap"])
        fixmap = self._load_fixationmap(self.stimuli.loc[idx, "fixmap"])
        label = self.stimuli.loc[idx, "label"]

        salmap = salmap / 255
        return stim_path, stimulus, salmap, fixmap, label

    def _download(self):
        """
        Downloads the CAT2000 dataset in your machine.

        If the dataset is not present at the specified root path, this method will
        download it. Requires 'unar' installed in the system for unarchiving the dataset.
        """
        logger.warning(
            "\n 🔴 You need to have unar installed in your terminal "
            "(https://theunarchiver.com/command-line)"
        )
        if os.path.isdir(os.path.join(self.root_path, "trainSet")):
            logger.warning(
                f"\n 🔴 The output directory {self.root_path} is already present on "
                "your disk and the dataset won't be downloaded. \n"
                f"In order to silence this warning message set the {self.name} "
                "parameter download=False \n"
            )
        else:
            os.makedirs(os.path.join(self.root_path, "trainSet"))
            self._download_dataset("http://saliency.mit.edu/trainSet.zip", self.root_path)
