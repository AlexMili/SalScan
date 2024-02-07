# -*- coding: utf-8 -*-

"""MIT (Judd's) Dataset."""

import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
from tqdm import tqdm

from SalScan.Dataset.AbstractDataset import AbstractDataset, check_structure_path
from SalScan.Utils import get_logger

logger = get_logger(__name__)


class MIT1003Dataset(AbstractDataset):
    """
    MIT1003 Dataset class for image saliency evaluation.

    It allows to download the dataset automatically and it handles the retrieval of
    stimuli, saliency maps and fixation maps from the specified folder.

    Parameters:
        path (str): Root path where the dataset is stored or will be downloaded.
        download (bool): If True, the dataset will be downloaded to the specified
                        `path`.
        training_path (str): Path to the training MIT1003 dataset located outside of
                            SalScan. For example, given a MIT1003_training dataset,
                            `training_path` would be set for example to
                            `path/to/MIT1003_training/train`. This way all the files
                            contained in your training directory will not be retrieved by
                            this dataset. Notice, this works only if did not rename
                            image filenames within your custom dataset.

    Attributes:
        stimuli: Pandas dataframe composed by the following information: "id",
                "stimulus", "salmap", "fixmap", "label". These information will
                be inserted with the `populate` method.
        stimuli_shape: A tuple of int indicating the number of stimuli in the current
                        dataset
        name: "MIT1003"
        has_label: Set to False since MIT1003 folder structure does not allow
                    considering parent folders as labels.
        training_images (str): When `training_path` is specified this attribute will
                                be a python `set` containing the filenames of all the
                                images located within `training_path`. If `training_path`
                                is not specified, `training_images` is set to `None`

    Reference:
        Wang, W., Shen, J., Xie, J., Cheng, M.-M., Ling, H., & Borji, A. (2021).
        Revisiting Video Saliency Prediction in the Deep Learning Era. IEEE Transactions
        on Pattern Analysis and Machine Intelligence.
    """

    _structure = {
        "fixations": [
            {
                "dir": ["DATA"],
                "subdirs": [
                    "CNG",
                    "ajs",
                    "emb",
                    "ems",
                    "ff",
                    "hp",
                    "jcw",
                    "jw",
                    "kae",
                    "krl",
                    "po",
                    "tmj",
                    "tu",
                    "ya",
                    "zb",
                ],
                "pattern": "(.*).mat$",
            }
        ],
        "stimuli": [{"dir": ["ALLSTIMULI"], "pattern": "(.*).jpeg$"}],
        "salmap": [{"dir": ["ALLFIXATIONMAPS"], "pattern": "(.*)_fixMap.jpg$"}],
        "fixmap": [{"dir": ["ALLFIXATIONMAPS"], "pattern": "(.*)_fixPts.jpg$"}],
    }

    name = "MIT1003"
    has_label = False

    # In this experiment Screen size is 1280x1024 px (section 2.1 of Reference) and
    # the viewing distance is 75cm.
    # If we take screen dimensions which are 19 inch, we obtain 48x48cm.
    # Which corresponds to 34px for 1 degree of visual angle
    # Reference:
    # Tilke Judd, Krista Ehinger, Fredo Durand, Antonio Torralba. Learning to Predict
    # where Humans Look [ICCV 2009]

    def __init__(
        self, path: str, download: bool = False, training_path: str = None
    ) -> None:
        super().__init__()

        self.root_path = path
        if download is True:
            self._download()
        check_structure_path(self.root_path, self._structure)
        # Initilises self.training_images that will be overwritten by the
        # self.collect_training_images_method
        self.training_images = None
        if training_path is not None:
            self.collect_training_images(path=training_path)

    def populate(self) -> None:
        """
        Populates a dataframe with the paths of stimuli, fixation maps and saliency maps.
        """

        self.stimuli: pd.DataFrame = self._default_stimuli_values
        self.fixations: pd.DataFrame = self._default_fixations_values

        stim_search = os.path.join(self.root_path, "ALLSTIMULI", "*.jpeg")
        pbar_items = tqdm(glob.glob(stim_search))

        salmap_path = os.path.join(self.root_path, "ALLFIXATIONMAPS")
        # folder in which we are going to store all the observations
        all_obs_path = os.path.join(self.root_path, "ALL_DATA")
        if os.path.exists(all_obs_path) is False:
            os.mkdir(all_obs_path)
            logger.info("Creating labels from '.mat' files..")

        fixation_path = os.path.join(self.root_path, "DATA")
        people = glob.glob(f"{fixation_path}/*/")
        observations = os.listdir(people[0])
        observations = [obs for obs in observations if obs.endswith(".mat")]

        # if directory is empty, creates ALL_DATA folder
        if len(os.listdir(all_obs_path)) == 0:
            pbar = tqdm(observations)
            for file_name in pbar:
                file_basename = file_name.split(".mat")[0]
                file_basename = f"{file_basename}_fixMap.jpg"
                salmap_file = os.path.join(salmap_path, file_basename)
                width, height = Image.open(salmap_file).size
                array_file = np.zeros((height, width))

                for person in people:
                    file_path = os.path.join(person, file_name)
                    try:
                        coord = sio.loadmat(file_path)[file_name.split(".mat")[0][:63]]
                    except KeyError:
                        continue
                    # path to array of coordinates within mat file
                    # coords are x and y

                    arr_coord = coord["DATA"][0][0][0][0][2]

                    # removing obs with negative values (blinks)
                    mask = np.any(arr_coord < 0, axis=1)
                    arr_coord = arr_coord[~mask]
                    # removing gaze outside the image (not <= for 0 indexing)
                    mask = (arr_coord[:, 1] < height) & (arr_coord[:, 0] < width)
                    arr_coord = arr_coord[mask].astype(int)

                    # setting the coordinates to 1 (https://stackoverflow.com/a/16396203)
                    # coords are x and y!
                    array_file[arr_coord[:, 1], arr_coord[:, 0]] = 1

                dump_path = os.path.join(
                    all_obs_path, f'{file_name.split(".mat")[0]}.json'
                )

                y, x = np.where(array_file == 1)
                # fixations need to be casted to int to avoid np.int64
                # not-serializable error with json dump
                fixations = [[int(y[i]), int(x[i])] for i in range(y.size)]

                json_file = {"img_size": [height, width], "fixations": fixations}
                with open(dump_path, "w") as f:
                    json.dump(json_file, f)

        for stim in pbar_items:
            if self.training_images and os.path.basename(stim) in self.training_images:
                continue
            pbar_items.set_description(f"Dataset {self.name} - Populate stimuli")

            stim_filename = os.path.basename(stim)

            self.stimuli = pd.concat(
                [
                    self.stimuli,
                    pd.DataFrame(
                        [
                            {
                                "id": self.stimuli.shape[0] + 1,
                                "stimulus": os.path.join(
                                    self.root_path, "ALLSTIMULI", stim_filename
                                ),
                                "salmap": os.path.join(
                                    self.root_path,
                                    "ALLFIXATIONMAPS",
                                    stim_filename.replace(".jpeg", "_fixMap.jpg"),
                                ),
                                "fixmap": os.path.join(
                                    self.root_path,
                                    "ALL_DATA",
                                    stim_filename.replace(".jpeg", ".json"),
                                ),
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
        with open(path, "r") as f:
            file = json.load(f)
        height, width = file["img_size"]
        fixations = np.array(file["fixations"])
        fixmap = np.zeros((height, width))
        fixmap[fixations[:, 0], fixations[:, 1]] = 1
        return fixmap

    def __getitem__(self, idx):
        """
        Retrieves stimuli, saliency maps and fixation maps and the corresponding label.
        The last element of the return tuple is a an empty placeholder (None) for the
        label.

        Parameter:
            idx (int): Index of the paths to retrieve.

        Returns:
            tuple: A tuple containing the path, stimulus, saliency map, fixation map, and
                    None (placeholder for label).
        """
        stim_path = self.stimuli.loc[idx, "stimulus"]
        stimulus = self._load_stimulus(stim_path)
        salmap = self._load_saliencymap(self.stimuli.loc[idx, "salmap"])
        fixmap = self._load_fixationmap(self.stimuli.loc[idx, "fixmap"])
        salmap = salmap / 255
        return stim_path, stimulus, salmap, fixmap, None

    def _download(self):
        """
        Downloads the MIT1003 dataset on your machine.

        If the dataset is not present at the specified root path, this method will
        download it. Requires 'unar' installed in the system for unarchiving the dataset.
        """
        logger.warning(
            "\n 🔴 You need to have unar installed in your terminal "
            "(https://theunarchiver.com/command-line)"
        )
        if os.path.isdir(self.root_path):
            logger.warning(
                f"\n 🔴 The output directory {self.root_path} is already present on "
                "your disk and the dataset won't be downloaded. \n"
                f"In order to silence this warning message set the {self.name} "
                "parameter download=False \n"
            )
        else:
            os.mkdir(self.root_path)
            self._download_dataset(
                "http://people.csail.mit.edu/tjudd/WherePeopleLook/ALLSTIMULI.zip",
                self.root_path,
            )
            self._download_dataset(
                "http://people.csail.mit.edu/tjudd/WherePeopleLook/DATA.zip",
                self.root_path,
            )
            self._download_dataset(
                "http://people.csail.mit.edu/tjudd/WherePeopleLook/ALLFIXATIONMAPS.zip",
                self.root_path,
            )
