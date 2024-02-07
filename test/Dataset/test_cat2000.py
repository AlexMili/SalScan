import os

import pytest

from SalScan.Dataset.Image.CAT2000Dataset import CAT2000Dataset


@pytest.mark.skipif(os.name != "posix")
@pytest.fixture(params=["Datasets/CAT2000"])
def dataset(request):
    return CAT2000Dataset(path=request.param)


def test_populate(dataset):
    dataset.populate()

    assert dataset.database["stimulus"].shape[0] == 2000


def test_get_stimulus(dataset):
    dataset.populate()

    _ = dataset.get_stimulus(11)


def test_iter_data(dataset):
    dataset.populate()

    for item in dataset:
        t = dataset.get_fixationmap()
        assert t.shape[:2] == dataset.get_stimulus().shape[:2]

        break
