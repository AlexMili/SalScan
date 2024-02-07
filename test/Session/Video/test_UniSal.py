import os

# Datasets
from SalScan.Dataset.Video.DHF1K import DHF1KDataset

# Session
from SalScan.Evaluate import VideoSaliencyEvaluation

# Models
from SalScan.Model.Saliency.UniSal import UniSal
from SalScan.Transforms import Preprocess_UniSal

path2dhf = os.path.join("datasets", "DHF1K")

if __name__ == "__main__":
    dhf1k_kwargs = {
        "root_path": path2dhf,
        "sequence_length": 6,
        "many_to_many": True,
        "to_array": True,
        "transform_stimuli": Preprocess_UniSal(out_size=(224, 384)),
        "evaluation_fps": 6,
    }
    video_list = [
        {
            "model": UniSal,
            "params": {
                "mobilnet_weights": "paths/to/weights/mobilenet_v2.pth.tar",
                "decoder_weights": "paths/to/weights/weights_best.pth",
                "sequence_length": 6,
            },
            "dataset": DHF1KDataset(**dhf1k_kwargs),
        },
        {
            "model": UniSal,
            "params": {
                "mobilnet_weights": "paths/to/weights/mobilenet_v2.pth.tar",
                "decoder_weights": "paths/to/weights/weights_best.pth",
                "sequence_length": 24,
            },
            "dataset": DHF1KDataset(**dhf1k_kwargs),
        },
    ]

    evaluate = VideoSaliencyEvaluation(
        evaluate_list=video_list,
        logs_folder="val_logs",
        debug=True,
    )
    missing = evaluate.compute_ranking()
