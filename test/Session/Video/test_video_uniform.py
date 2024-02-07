import os

# #### IMPORT DATASETS #####
from SalScan.Dataset.Video.DHF1K import DHF1KDataset
from SalScan.Evaluate import VideoSaliencyEvaluation

# Models
from SalScan.Model.Saliency.UniformModel import UniformModel

if __name__ == "__main__":
    dhf1k = DHF1KDataset(os.path.join(os.path.expanduser("~"), "datasets", "DHF1K"))

    video_dict = [
        {
            "model": UniformModel,
            "params": {"n_fix": None},
            "dataset": dhf1k,
        },
    ]

    evaluate = VideoSaliencyEvaluation(
        evaluate_list=video_dict,
        logs_folder="val_logs",
    )
    missing = evaluate.compute_ranking()
