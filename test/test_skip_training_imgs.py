import glob
import os
import shutil

import pytest

from SalScan.Dataset.Image.CAT2000Dataset import CAT2000Dataset
from SalScan.Dataset.Image.MIT1003Dataset import MIT1003Dataset

DATASET_FOLDER = os.path.join(os.path.expanduser("~"), "datasets")


@pytest.fixture
def temp_dir_mit():
    # Create a temporary directory
    tmp_dir = "tmp_mit_1003"
    os.makedirs(tmp_dir, exist_ok=True)
    # Provide the temp directory path to the test
    yield tmp_dir
    # Cleanup after the test is done
    shutil.rmtree(tmp_dir)


def test_discard_mit_training_images(temp_dir_mit):
    dirpath: str = os.path.join(DATASET_FOLDER, "mit1003")
    if os.path.exists(dirpath) is False:
        raise FileNotFoundError(f"Folder {dirpath} does not exist in your system.")

    # Ensure that MIT1003 is present by downloading it
    mit1003 = MIT1003Dataset(path=dirpath, download=True)

    mit_stimuli_path = os.path.join(dirpath, "ALLSTIMULI")
    mit_stimuli = glob.glob(os.path.join(mit_stimuli_path, "*"))
    assert len(mit_stimuli) == 1003

    # Move every 20th file to the temporary directory
    for idx, file in enumerate(mit_stimuli):
        if idx % 20 == 0:
            shutil.copyfile(file, os.path.join("tmp_mit_1003", os.path.basename(file)))

    # Initialize datasets
    mit1003_val = MIT1003Dataset(path=dirpath, training_path="tmp_mit_1003")

    # Populate datasets
    mit1003.populate()
    mit1003_val.populate()

    # Assertions
    assert len(mit1003) == 1003
    # 952 is the expected count after discarding training images present within
    # tmp_mit_1003
    assert len(mit1003_val) == 952


@pytest.fixture
def temp_dir_cat():
    # Create a temporary directory
    tmp_dir = "tmp_cat_2000"
    os.makedirs(tmp_dir, exist_ok=True)
    # Provide the temp directory path to the test
    yield tmp_dir
    # Cleanup after the test is done
    shutil.rmtree(tmp_dir)


def test_discard_cat_training_images(temp_dir_cat):
    dirpath: str = os.path.join(DATASET_FOLDER, "CAT2000")
    if os.path.exists(dirpath) is False:
        raise FileNotFoundError(f"Folder {dirpath} does not exist in your system.")
    # Use the temp_dir from the fixture
    cat2000 = CAT2000Dataset(path=dirpath, download=True)
    cat_stimuli_path = os.path.join(dirpath, "trainSet", "Stimuli")
    cat_stimuli = glob.glob(os.path.join(cat_stimuli_path, "**", "*"))
    cat_stimuli = [stim for stim in cat_stimuli if os.path.isfile(stim)]

    # Move every 20th file to the temporary directory
    for idx, file in enumerate(cat_stimuli):
        if idx % 20 == 0:
            par_dir = os.path.basename(os.path.dirname(file))
            par_dir = os.path.join("tmp_cat_2000", par_dir)
            if os.path.exists(par_dir) is False:
                os.mkdir(par_dir)
            shutil.copyfile(file, os.path.join(par_dir, os.path.basename(file)))

    # Initialize datasets

    cat2000_val = CAT2000Dataset(path=dirpath, training_path="tmp_cat_2000")

    # Populate datasets
    cat2000.populate()
    cat2000_val.populate()

    # Assertions
    assert len(cat2000) == 2000
    # Assuming 953 is the expected count after exclusion
    assert len(cat2000_val) == 1900
