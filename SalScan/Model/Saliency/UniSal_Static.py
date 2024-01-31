import os

import numpy as np
import torch

from SalScan.Model.AbstractModel import AbstractModel
from SalScan.Model.Saliency.UniSal import UNISAL
from SalScan.Transforms import Preprocess_UniSal
from SalScan.Utils import normalize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class UniSal_Static(AbstractModel):
    """
    Unified Image and Video Saliency Modeling (UniSal) for image saliency prediction.
    This model generates a saliency map based on an input stimulus.

    Parameters:
        options (dict): Configuration options to set `mobilnet_weights`,
                        `decoder_weights` and`sequence_length`.

    Attributes:
        name (str): 'UniSal_static'.
        type (str): Type of the model, 'saliency'.
        params_to_store (list): List of attributes of the model stored in the
                                evaluation logs by
                                `SalScan.Evaluate.ImageSaliencyEvaluation` or by
                                `SalScan.Evaluate.VideoSaliencyEvaluation`. Includes
                                `mobilnet_weights`, `decoder_weights`, and
                                `sequence_length`.
        model: UniSal Pytorch model.
        forward_kwargs (dict): Dictionary containing parameters used by the `forward`
                                method of UniSal.
        device (bool): Device used to perform the prediction. Can be either `cpu` or `gpu`
                        depending on `torch.cuda.is_available()`
        transform (Callable): Function used to preprocess stimuli to the format used by
                                UniSal: `SalScan.Transforms.Preprocess_UniSal`
    """

    name = "UniSal_static"
    type = "saliency"
    wrapper = "python"
    params_to_store = ["mobilnet_weights", "decoder_weights"]

    def __init__(self, options):
        super().__init__()
        self._config = {**options}
        # weight are in unisal/training_runs/pretrained_unisal/weights_best.pth
        mobilnet_weights = self._config.get("mobilnet_weights", "")
        decoder_weights = self._config.get("decoder_weights", "")

        if os.path.exists(mobilnet_weights) is False:
            raise FileNotFoundError(
                "\n ❌ 'mobilnet_weights' either an empty or non valid directory."
            )
        if os.path.exists(decoder_weights) is False:
            raise FileNotFoundError(
                "\n ❌ 'decoder_weights' either an empty or non valid directory."
            )

        self.device = DEVICE
        self.transform = Preprocess_UniSal(out_size=(224, 384))
        self.model_cfg["mobilnet_weights"] = mobilnet_weights
        self.model = UNISAL(**self.model_cfg).to(self.device)
        self.model.load_state_dict(
            torch.load(decoder_weights, map_location=self.device), strict=False
        )
        self.forward_kwargs = {
            "source": "SALICON",
            "target_size": (360, 640),
            "static": True,
        }
        self.model.eval()

    def run(self, img: np.array, params) -> np.ndarray:
        """
        Generates the saliency map.

        Args:
            img (np.array): The input stimulus.
            params (dict): model's run parameters
        """
        img = self.transform(img=img)[None, None, ...]
        imgs = img.to(self.device)
        smap = self.model(imgs, **self.forward_kwargs)
        # size (1, seq_len, 1, heigth, width) -> (6, 224, 384) -> list

        # Postprocess prediction
        smap = smap.exp()
        smap = torch.squeeze(smap)
        smap = smap.detach().cpu().numpy()
        smap = normalize(smap, method="range")

        return smap
