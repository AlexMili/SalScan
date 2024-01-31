import numpy as np

from SalScan.Model.AbstractModel import AbstractModel
from SalScan.Utils import gauss2d


class CenterBias(AbstractModel):
    """
    This model generates a saliency map in the center of the image assuming that viewers
    are more likely to focus towards the center of stimuli. The model uses a Gaussian
    distribution whose size (diameter) can be customized.

    Parameters:
        options (dict): Configuration options to set the diameter of the Gaussian bias.

    Attributes:
        diameter: diameter size as percentage of the smaller side of the image: i.e. if
                the lower side is 400 pixels and diameter is set to 0.1, then the
                diameter will be set to 40 pixels.
        params_to_store (list): List of attributes of the model stored in the
                                evaluation logs by
                                `SalScan.Evaluate.ImageSaliencyEvaluation` and
                                `SalScan.Evaluate.VideoSaliencyEvaluation`. For this model
                                includes `diameter`.
        name (str): The name of the model, 'CenterBias'.
        type (str): The type of the model, 'saliency'.
        wrapper (str): The wrapper type, 'python'.
    """

    name = "CenterBias"
    type = "saliency"
    wrapper = "python"
    params_to_store = ["diameter"]

    def __init__(self, options):
        super().__init__()

        self._config = {**options}
        self.diameter = self._config.get("diameter", 0.15)

    def run(self, img, params) -> np.ndarray:
        """
        Generates the saliency map.

        Args:
            img (np.array): The input stimulus.
            params (dict): model's run parameters
        """

        center_coord = np.array([[img.shape[1] * 0.5, img.shape[0] * 0.5]])
        # the diameter is based on the size of the lower dimension
        min_dim = np.argmin([img.shape[0], img.shape[1]])
        diameter = img.shape[min_dim] * self.diameter
        return gauss2d(np.zeros_like(img), sigma=diameter, center=center_coord)
