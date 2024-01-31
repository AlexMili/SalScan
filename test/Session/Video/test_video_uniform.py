import os

# Models
from SalScan.Model.Saliency.UniformModel import UniformModel

from SalScan.Evaluate import VideoSaliencyEvaluation

##### IMPORT DATASETS #####
from SalScan.Dataset.Video.DHF1K import DHF1KDataset


if __name__ == "__main__":
    dataset_folder = os.path.join(os.path.expanduser("~"), "datasets")

    video_dict = [
        {
            "model": UniformModel,
            "params": {
                "n_fix": None,
            },
            "dataset": DHF1KDataset(os.path.join(dataset_folder, "DHF1K")),
        },
    ]

    evaluate = VideoSaliencyEvaluation(
        evaluate_list=video_dict,
        logs_folder="val_logs",
    )
    missing = evaluate.compute_ranking()
