import os

# #### IMPORT DATASETS #####
from SalScan.Dataset.Video.DHF1K import DHF1KDataset
from SalScan.Evaluate import VideoSaliencyEvaluation

# #### IMPORT MODELS #####
from SalScan.Model.Saliency.STSANet import STSANetwork

# #### IMPORT TRANFORM #####
from SalScan.Transforms import Preprocess_STSANet

if __name__ == "__main__":
    dataset_folder = os.path.join(os.path.expanduser("~"), "datasets")

    video_dict = [
        {
            "model": STSANetwork,
            "params": {
                "path_to_weights": "Model/Weights/DHF1k.pth",
                "image_height": 224,
                "image_width": 384,
            },
            "dataset": DHF1KDataset(
                "datasets/DHF1K",
                sequence_length=32,
                sequence_target=16,
                to_array=True,
                transform_stimuli=Preprocess_STSANet(384, 224),
            ),
        },
    ]

    evaluate = VideoSaliencyEvaluation(
        evaluate_list=video_dict,
        logs_folder="val_logs",
    )
    missing = evaluate.compute_ranking()
