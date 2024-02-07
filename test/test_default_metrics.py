import os

from SalScan.Dataset.Image.MIT1003Dataset import MIT1003Dataset
from SalScan.Metric.Saliency import AUC_JUDD, CC, IG, KLD, NSS, SIM, sAUC
from SalScan.Model.Saliency.UniformModel import UniformModel
from SalScan.Session.Saliency.ImageSession import ImageSaliencySession
from SalScan.Session.Saliency.VideoSession import VideoSaliencySession

DATASET_FOLDER = os.path.join(os.path.expanduser("~"), "datasets")


def test_default_metrics_img():
    default_metrics = (AUC_JUDD, sAUC, NSS, CC, KLD, SIM, IG)

    model = UniformModel
    dataset = MIT1003Dataset(os.path.join(DATASET_FOLDER, "MIT1003"))
    session_args = {"dataset": dataset, "model": model, "params": {}}
    session = ImageSaliencySession(session_args)
    session_default_metrics = tuple(session.metrics)
    assert session_default_metrics == default_metrics


def test_default_metrics_video():
    default_metrics = (AUC_JUDD, sAUC, NSS, CC, KLD, SIM)

    model = UniformModel
    dataset = MIT1003Dataset(os.path.join(DATASET_FOLDER, "MIT1003"))
    session_args = {"dataset": dataset, "model": model, "params": {}}
    session = VideoSaliencySession(session_args)
    session_default_metrics = tuple(session.metrics)
    assert session_default_metrics == default_metrics
