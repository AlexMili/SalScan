import os
import shutil

from SalScan.Dataset.Image.CAT2000Dataset import CAT2000Dataset
from SalScan.Dataset.Image.MIT1003Dataset import MIT1003Dataset
from SalScan.Dataset.Video.DHF1K import DHF1KDataset

DATASET_PATH = os.path.join(os.path.expanduser("~"), "datasets")


def test_download_MIT():
    path_mit = os.path.join(DATASET_PATH, "MIT1003")

    if os.path.exists(path_mit) is True:
        shutil.rmtree(path_mit)

    dataset = MIT1003Dataset(path_mit, download=True)
    dataset.populate()

    assert dataset.stimuli.empty is False


def test_download_CAT():
    path_cat = os.path.join(DATASET_PATH, "CAT2000")

    if os.path.exists(path_cat) is True:
        shutil.rmtree(path_cat)

    dataset = CAT2000Dataset(path_cat, download=True)
    dataset.populate()

    assert dataset.stimuli.empty is False


def test_downloadDHF1K():
    path_dhf1k = os.path.join(DATASET_PATH, "DHF1K")

    if os.path.exists(path_dhf1k) is True:
        shutil.rmtree(path_dhf1k)

    dataset = DHF1KDataset(path_dhf1k, download=True)
    dataset.populate()

    assert dataset.stimuli.empty is False


if __name__ == "__main__":
    test_downloadDHF1K()
