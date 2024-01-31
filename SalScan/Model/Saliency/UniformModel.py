import numpy as np

from SalScan.Model.AbstractModel import AbstractModel
from SalScan.Utils import fixmap_2_salmap


class UniformModel(AbstractModel):
    """
    This model generates a saliency map by uniformly distributing a specified number of
    fixation points across the image. This simplistic approach serves as a baseline model
    in saliency modeling.

    Parameters:
        options (dict): Configuration options to set the number of fixations used to
                        generate the saliency map.

    Attributes:
        n_fix: Number of fixations used to generate the uniform saliency map. When set to
                `None`, `SalScan.Session.Saliency.ImageSession` and
                `SalScan.Session.Saliency.VideoSession` set this value to the total
                amount of ground truth fixations.
        params_to_store (list): List of attributes of the model stored in the
                                evaluation logs by
                                `SalScan.Evaluate.ImageSaliencyEvaluation` and
                                `SalScan.Evaluate.VideoSaliencyEvaluation`. For this model
                                includes it includes `n_fix`.
        name (str): The name of the model, `CenterBiasStochastic'.
        type (str): The type of the model, 'saliency'.
        wrapper (str): The wrapper type, 'python'.
    """

    name = "UniformModel"
    type = "saliency_baseline"
    wrapper = "python"
    params_to_store = ["n_fix"]

    def __init__(self, options):
        super().__init__()
        self._config = {**options}
        self.n_fix = self._config.get("n_fix", None)

    def run(self, img, params) -> np.ndarray:
        """
        Generates the saliency map.

        Args:
            img (np.array): The input stimulus.
            params (dict): model's run parameters
        """
        height, width = img.shape[:2]
        n_fix = params["n_fix"]
        fix_uniform = np.zeros((height, width))
        flatten = fix_uniform.flatten()
        # indexes (pixels) are randomly selected
        coordinates = np.random.choice(len(flatten), size=n_fix)
        flatten[coordinates] = 1

        fix_uniform = flatten.reshape(height, width)
        sal_uniform = fixmap_2_salmap(fix_uniform)

        return sal_uniform
