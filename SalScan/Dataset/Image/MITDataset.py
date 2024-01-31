# -*- coding: utf-8 -*-

"""MIT (Judd's) Dataset."""

import glob
import os
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..AbstractDataset import AbstractDataset, check_structure_path


class MITDataset(AbstractDataset):
    """
    MIT (Judd's) Dataset class.

    Adapts methods created by `AbstractDataset` class to MIT (Judd's) dataset.
    Reference: Tilke Judd, Krista Ehinger, Fredo Durand, Antonio Torralba.
    Learning to Predict where Humans Look [ICCV 2009]

    Attributes:
        name: A string indicating the name of the dataset.
        type: A string indicating the type of the dataset. For saliency evaluation set it to "saliency".
        ONE_DEGREE: An integer representing the number of pixels contained in one degree of visual angle.
        SCREEN_SIZE: Screen's size during the experiment used to build the dataset.
    """

    structure = {
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
                "pattern": "(.*).csv$",
            }
        ],
        "stimuli": [{"dir": ["ALLSTIMULI"], "pattern": "(.*).jpeg$"}],
        "salmap": [{"dir": ["ALLFIXATIONMAPS"], "pattern": "(.*)_fixMap.jpg$"}],
        "fixmap": [{"dir": ["ALLFIXATIONMAPS"], "pattern": "(.*)_fixPts.jpg$"}],
    }

    name = "MIT"
    type = "saliency"

    # In this experiment Screen size is 1280x1024 px (section 2.1 of Reference) and the viewing distance is 75cm.
    # If we take screen dimensions which are 19 inch, we obtain 48x48cm.
    # Which corresponds to 34px for 1 degree of visual angle
    # Reference:
    # Tilke Judd, Krista Ehinger, Fredo Durand, Antonio Torralba. Learning to Predict where Humans Look [ICCV 2009]
    ONE_DEGREE = 34

    SCREEN_SIZE = (1024, 768)

    def __init__(self, path: str) -> None:
        """Inits MIT Dataset attributes."""

        super().__init__()

        self.root_path = path

        check_structure_path(self.root_path, self.structure)

    def _populate(self, force: Optional[bool] = False) -> None:
        """Populate stimuli and fixations Dataframes."""

        if not self.is_populated or force:
            self.stimuli: pd.DataFrame = self._default_stimuli_values
            self.fixations: pd.DataFrame = self._default_fixations_values

            stim_search = os.path.join(self.root_path, "ALLSTIMULI", "*.jpeg")
            pbar_items = tqdm(glob.glob(stim_search))
            for stim in pbar_items:
                pbar_items.set_description(f"Dataset {self.name} - Populate stimuli")

                stim_filename = os.path.basename(stim)

                self.stimuli = self.stimuli.append(
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
                            "ALLFIXATIONMAPS",
                            stim_filename.replace(".jpeg", "_fixPts.jpg"),
                        ),
                    },
                    ignore_index=True,
                )

            fix_search = os.path.join(self.root_path, "DATA", "**", "*.csv")

            pbar_items = tqdm(glob.glob(fix_search))
            for filename in pbar_items:
                pbar_items.set_description(f"Dataset {self.name} - Populate fixations")

                user = os.path.split(os.path.dirname(filename))[-1]
                stim = os.path.splitext(os.path.basename(filename))[0] + ".jpeg"

                self.fixations = self.fixations.append(
                    {
                        "id": self.fixations.shape[0] + 1,
                        "stimulus": os.path.join(self.root_path, "ALLSTIMULI", stim),
                        "part_id": user,
                        "eye": "Unknown",
                        "position_in_file": 0,
                        "fixations": filename,
                    },
                    ignore_index=True,
                )

            self.is_populated = True

    def _load_stimulus(self, path: str) -> np.array:
        """Load a stimulus."""
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def _load_saliencymap(self, path: str) -> np.array:
        """Load a saliency map."""
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2GRAY)

    def _load_fixationmap(self, path):
        """Loads a fixation map for a given stimulus."""
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2GRAY)
