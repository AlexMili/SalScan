# -*- coding: utf-8 -*-

"""Module containing Datasets abstract class."""
import glob
import os
import shutil
import subprocess
import zipfile
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import gdown
import numpy as np
import pandas as pd
import requests

from SalScan.Utils import get_logger, has_files

logger = get_logger(__name__)

Stimulus = namedtuple("Stimulus", "id stimulus salmap fixmap label")
Fixations = namedtuple("Fixations", "id stimulus part_id eye position_in_file fixations")


def check_structure_path(path: str, struct: Dict, pattern: Optional[str] = None) -> bool:
    """Check directories and sub-directories structure and containing files.

    Walks through root directory and sub-directories to ensure all encountered files
    follow the given patterns. This function is recursive.

    Args:
        struct: A dict containing the structure of the directory. Example:
            ```
            [
                "dir": ["array", "of", "dirs"],
                "pattern": "pattern to apply to files in dir or subdirs",
                "subdirs":[
                    {
                        "dir": ["array", "of", "dirs"],
                        "pattern": "Override parent dir pattern if defined"
                    },
                    {
                        "dir": ["array", "of", "dirs"],
                        "subdirs": [
                            { ... }
                        ]
                        "pattern": "Override parent dir pattern if defined"
                    }
                ]
            ]
            ```

            or

            ```
            [
                "dir": ["array", "of", "dirs"],
                "pattern": "pattern to apply to files in dir or subdirs",
                "subdirs": ["first_subdir", "second_subdir", "third subdir"]
            ]
            ```

        path: A string containing the root path to check.
        pattern: A string regular expression to apply to the current files/subdirectory to check
            if all files are present.

    Returns:
        True if succeeded, raises an error otherwise.

    Raises:
        FileNotFoundError: A file or a directory defined in the structure does not exist.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} root path does not exist !")

    for elm in struct:
        for current_dir in struct[elm]:
            dirs = os.path.join(path, *current_dir["dir"])

            if not os.path.exists(dirs):
                raise FileNotFoundError(f"{dirs} does not exist in current dataset")

            # Slect if there is a pattern to check
            if "pattern" in current_dir and current_dir["pattern"] is not None:
                current_pattern = current_dir["pattern"]
            elif pattern is not None:
                current_pattern = pattern
            else:
                # If no pattern is provided, only the dir structure is checked
                pass

            files_found = has_files(dirs, current_pattern)

            if "subdirs" in current_dir and len(current_dir["subdirs"]) > 0:
                for sub in current_dir["subdirs"]:
                    if isinstance(sub, str):
                        subdir = os.path.join(path, *current_dir["dir"], sub)

                        if not os.path.exists(subdir):
                            raise FileNotFoundError(
                                "{} does not exist in current dataset".format(subdir)
                            )

                        if not has_files(subdir, current_pattern):
                            raise FileNotFoundError(
                                "Missing files in Dataset ({})".format(subdir)
                            )
                    elif isinstance(sub, []):
                        sub_path = os.path.join(path, *current_dir["dir"], sub)
                        check_structure_path(
                            struct=sub, path=sub_path, pattern=current_pattern
                        )

            elif not files_found:
                raise FileNotFoundError(f"Missing files in Dataset ({dirs})")

    return True


class AbstractDataset(ABC):
    """Datasets abstract class.

    This class cannot be instantiated and must be inherited from a child class. It's
    purpose is to define a structure and some basic functions in order to implement any
    dataset class.

    Parameters:
        path (str): Root path where the dataset is stored or will be downloaded.
        download (bool): If True, the dataset will be downloaded to the specified
                        `path`.
        training_path (str): Path to the training custom dataset located outside of
                            SalScan. For example, given a custom training dataset,
                            `training_path` would be set for example to
                            `path/to/custom_training_dataset/train`.
                            This way all the files contained in your training directory
                            will not be retrieved by this dataset. Notice, this works
                            only if did not rename image filenames within your custom
                            dataset.

    Attributes:
        name: A string indicating the name of the dataset.
        has_label: It describes if the stimuli are organized within the folder such
                    that each stimulus parent folder can be considered as its label.
        stimuli: Pandas dataframe composed by the following information: "id",
                "stimulus", "salmap", "fixmap", "label". These information will
                be inserted with the `populate` method.
        stimuli_shape: A tuple of int indicating the number of stimuli in the current
                        dataset
        fixations_shape: A tuple of int indicating the number of fixations in the current
                        dataset
        training_images (str): When `training_path` is specified this attribute will
                                be a python `set` containing the filenames of all the
                                images located within `training_path`. If `training_path`
                                is not specified, `training_images` is set to `None`
        type: A string indicating the type of the dataset. i.e. "saliency"
        structure (Optional): Dict containing dataset's folder structure.
    """

    ### Class attributes ###
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    def _structure(self) -> Dict:
        raise NotImplementedError

    @property
    @abstractmethod
    def has_label(self) -> bool:
        return NotImplementedError

    _default_stimuli_values: pd.DataFrame = pd.DataFrame(
        columns=["id", "stimulus", "salmap", "fixmap", "label"]
    )
    _default_fixations_values: pd.DataFrame = pd.DataFrame(
        columns=["id", "stimulus", "part_id", "eye", "position_in_file", "fixations"]
    )

    ### Instance attributes ###

    # A tuple of int indicating the number of stimuli in the current dataset
    @property
    def stimuli_shape(self) -> Tuple[int, int]:
        return self.stimuli.shape

    # A tuple of int indicating the number of fixations data in the current dataset
    @property
    def fixations_shape(self) -> Tuple[int, int]:
        return self.fixations.shape

    def __init__(
        self, path: str = None, download: bool = False, training_path: str = None
    ):
        ### Instance attributes ###

        #: A Dataframe where all stimuli-related data are stored.
        self.stimuli: pd.DataFrame = self._default_stimuli_values

        #: A Dataframe where all raw-fixations-related data are stored.
        self.fixations: pd.DataFrame = self._default_fixations_values

        #: Root path of the dataset.
        self.root_path: str = path

        # Set of images used for training that will be discarded from evaluation.
        # When specified in dataset arg, this attribute will be overwritten by
        # the collect_training_images method.
        self.training_images = None

    @abstractmethod
    def populate(self, force: bool = False) -> None:
        """Populate stimuli and fixations Dataframes.

        This method parses the stimulus and fixations folders and creates
        a new row for each element it finds in the two corresponding DataFrames.
        This method must be defined in each child class.

        Args:
            force: Build again dataset internal stimuli and fixations references.
        """

        raise NotImplementedError

    @abstractmethod
    def _load_stimulus(self, path: str) -> np.array:
        """Load a stimulus.

        Loads stimulus given its path. This method must be defined in subclass to implement specific behavior.

        Args:
            path: Stimulus path.

        Returns:
            The stimulus RGB image normalized between 0 and 1.
        """
        raise NotImplementedError

    def _load_saliencymap(self, path: str) -> np.array:
        """Load a saliency map.

        Loads saliency map given its path. This method must be defined in subclass to implement specific behavior.

        Args:
            path: Saliency map path.

        Returns:
            The saliency map RGB image normalized between 0 and 1.
        """
        raise NotImplementedError

    def _load_fixationmap(self, path: str) -> np.array:
        """Loads a fixation map for a given stimulus.

        Loads fixation map given its path.
        This method must be defined in subclass to implement specific behavior.

        Args:
            path: fixation map path.

        Returns:
            The fixation map with values equal to 0 or 1.
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        stim_path = self.stimuli.loc[idx, "stimulus"]
        stimulus = self._load_stimulus(stim_path)
        salmap = self._load_saliencymap(self.stimuli.loc[idx, "salmap"])
        fixmap = self._load_fixationmap(self.stimuli.loc[idx, "fixmap"])
        label = self.stimuli.loc[idx, "label"]

        return stim_path, stimulus, salmap, fixmap, label

    def __len__(self):
        return len(self.stimuli)

    def _download_dataset(self, url: str, output_dir: str) -> None:
        local_filename = os.path.basename(url)

        logger.info(f" 🟢 Downloading {url} into {output_dir}")
        if "drive.google" in url:
            gdown.download_folder(url=url, output=output_dir, quiet=False)
        else:
            path_file = os.path.join(output_dir, local_filename)
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(path_file, "wb") as f:
                    shutil.copyfileobj(r.raw, f, length=16 * 1024 * 1024)

        # unrar files
        rar_files = glob.glob(os.path.join(output_dir, "*.rar"))
        if len(rar_files) != 0:
            curr_dir = os.getcwd()
            os.chdir(output_dir)
            for rar_file in rar_files:
                subprocess.run(f"unar {os.path.basename(rar_file)}", shell=True)
                os.remove(rar_file)
            os.chdir(curr_dir)

        # unzip files
        zip_files = glob.glob(os.path.join(output_dir, "*.zip"))
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(zip_file)

    def _download(self):
        raise NotImplementedError

    def collect_training_images(self, path: str) -> Set[str]:
        """Collects all the path of the filenames present within `path`. This method
            overwrites the `training_images` attribute to contain all the filenames
            present within `path` so that the dataset will not retrieve them.

        Args:
            path (str): Directory outside SalScan pointing toward to the training data
                        folder: i.e. `~/path/to/custom_dataset/train`.
        """
        directory = Path(path)
        if directory.exists is False:
            raise FileNotFoundError(
                f"The path {directory} does not exist in your" " system."
            )
        all_images = directory.glob("**/*")
        all_images = [file for file in all_images if file.is_file()]
        all_images = [file.name for file in all_images]
        logger.info(
            f"INFO: {len(all_images)} training images found within your curtom dataset "
            f"training path: {path}. All these images will not be retrieved by the "
            f"{self.name} dataset."
        )
        self.training_images = set(all_images)


class VirtualDataset(AbstractDataset):
    structure = {}
    name = ""
    type = ""
    ONE_DEGREE = 0
    SCREEN_SIZE = (0, 0)

    def __init__(self, path: str) -> None:
        """Inits Toronto Dataset attributes."""

        super().__init__()
        self.root_path = path

    def _populate(self, force: bool = False) -> None:
        pass
