import os
import shutil

import pandas as pd
import pytest

from SalScan.Dataset.Image.MIT1003Dataset import MIT1003Dataset
from SalScan.Dataset.Video.DHF1K import DHF1KDataset
from SalScan.Evaluate import ImageSaliencyEvaluation, VideoSaliencyEvaluation
from SalScan.Metric.Saliency import AUC_JUDD_fast
from SalScan.Model.Saliency.CenterBias import CenterBias

DATASET_FOLDER = os.path.join(os.path.expanduser("~"), "datasets")


@pytest.fixture
def temp_dir():
    # Create a temporary directory
    temp_path = "tmp"
    os.makedirs(temp_path, exist_ok=True)
    yield temp_path  # Provide the temp directory path to the test
    # Cleanup after the test is done
    shutil.rmtree(temp_path)


def test_add_metrics_image(temp_dir):
    # CREATE A NEW METRIC, in this it is a copy of AUC_JUDD_fast
    # named "new_metric"
    def new_metric(gen_saliency_map, gt_fixation_map):
        return AUC_JUDD_fast(gen_saliency_map, gt_fixation_map)

    new_metric_name = new_metric.__name__

    model = CenterBias
    dataset = MIT1003Dataset
    image_dict = {
        "models": [
            {
                "model": model,
                "params": {"dimension": (200, 200)},
            },
        ],
        "datasets": [
            dataset(os.path.join(DATASET_FOLDER, "MIT1003")),
        ],
    }
    # Store a copy of the ImageSaliencyClass in a variable
    # to add a metric among the class attribe "available metrics"

    ImageSaliencyEval_copy = ImageSaliencyEvaluation
    ImageSaliencyEval_copy.available_metrics[new_metric_name] = 1

    evaluate = ImageSaliencyEval_copy(
        evaluate_dict=image_dict,
        logs_folder=temp_dir,
        ranking_metrics=[AUC_JUDD_fast, new_metric],
        debug=True,
    )

    evaluate.compute_ranking()
    path_exp = os.path.join(
        temp_dir, dataset.name.lower(), model.name.lower(), "params_setting_1", "val.csv"
    )
    experiment_results = pd.read_csv(path_exp, usecols=["metric", "dataset"])
    print(experiment_results)  # noqa
    # checking that the new metric has been added to the resulting csv file
    assert new_metric_name in experiment_results.loc[:, "metric"].tolist()
    # since the NEW_METRIC is a copy of AUC_JUDD_fast, we are asserting that
    # the results are the same as for AUC_JUDD_fast
    assert experiment_results.loc[0, "dataset"] == experiment_results.loc[1, "dataset"]


def test_add_metrics_video(temp_dir):
    # CREATE A NEW METRIC, in this it is a copy of AUC_JUDD_fast
    # named "new_metric"
    def new_metric(gen_saliency_map, gt_fixation_map):
        return AUC_JUDD_fast(gen_saliency_map, gt_fixation_map)

    new_metric_name = new_metric.__name__

    model = CenterBias
    dataset = DHF1KDataset
    video_list = [
        {
            "model": CenterBias,
            "params": {
                "dimension": (200, 200),
            },
            "dataset": dataset(os.path.join(DATASET_FOLDER, "DHF1K"), sequence_length=1),
        },
    ]
    # Store a copy of the ImageSaliencyClass in a variable
    # to add a metric among the class attribe "available metrics"
    VideoSaliencyEval_copy = VideoSaliencyEvaluation
    VideoSaliencyEval_copy.available_metrics[new_metric_name] = 1

    evaluate = VideoSaliencyEval_copy(
        evaluate_list=video_list,
        logs_folder=temp_dir,
        ranking_metrics=[AUC_JUDD_fast, new_metric],
        debug=True,
    )

    evaluate.compute_ranking()
    path_exp = os.path.join(
        temp_dir, dataset.name.lower(), model.name.lower(), "params_setting_1", "val.csv"
    )
    experiment_results = pd.read_csv(path_exp, usecols=["metric", "dataset"])
    # checking that the new metric has been added to the resulting csv file
    assert new_metric_name in experiment_results.loc[:, "metric"].tolist()
    # since the NEW_METRIC is a copy of AUC_JUDD_fast, we are asserting that
    # the results are the same as for AUC_JUDD_fast
    assert experiment_results.loc[0, "dataset"] == experiment_results.loc[1, "dataset"]
