import numpy as np

from SalScan.Model.AbstractModel import AbstractModel
from SalScan.Utils import fixmap_2_salmap


class CenterBiasStochastic(AbstractModel):
    """
    This class generates a saliency map using a stochastic approach based on the center
    bias principle. This approach generates more realistic saliency maps compared to
    traditional fixation-independent center bias models.

    Parameters:
        options (dict): Configuration options to set the diameter of the Gaussian bias
                        and the number of fixations used to generate the saliency map.

    Attributes:
        diameter: Diameter size as percentage of the smaller side of the image: i.e. if
                the lower side is 400 pixels and diameter is set to 0.1, then the
                diameter will be set to 40 pixels.
        n_fix: Number of fixations used to simulate the stochastic center biased map. When
                set to None, `SalScan.Session.Saliency.ImageSession` and
                `SalScan.Session.Saliency.VideoSession` set this value to the total
                amount of ground truth fixations.
        params_to_store (list): List of attributes of the model stored in the
                                evaluation logs by
                                `SalScan.Evaluate.ImageSaliencyEvaluation` and
                                `SalScan.Evaluate.VideoSaliencyEvaluation`. For this model
                                includes it includes `diameter` and `n_fix`.
        name (str): The name of the model, 'CenterBiasStochastic'.
        type (str): The type of the model, 'saliency'.
        wrapper (str): The wrapper type, 'python'.
    """

    name = "CenterBiasStochastic"
    type = "saliency_baseline"
    wrapper = "python"
    params_to_store = ["n_fix", "diameter"]

    def __init__(self, options):
        super().__init__()
        self._config = {**options}
        self.n_fix = self._config.get("n_fix", None)
        self.diameter = self._config.get("diameter", 0.15)

    def run(self, img, params) -> np.ndarray:
        """
        Generates the saliency map.

        Args:
            img (np.array): The input stimulus.
            params (dict): model's run parameters
        """
        height, width = img.shape[:2]
        n_fix = params["n_fix"]
        min_dimension = np.min([img.shape[0], img.shape[1]])
        diameter = min_dimension * self.diameter

        fix_map = np.zeros((height, width))
        center = fix_map.shape[0] // 2, fix_map.shape[1] // 2
        # extracting n_fix times a random radius (in pixels)
        obs = np.abs(np.random.normal(loc=0, scale=diameter, size=n_fix))
        # randomly setting angles
        angles = np.random.randint(low=0, high=360, size=n_fix)

        # to get random fixations given a radius and an angle, we computing
        # by using the sine and the cosine: for instance:
        # x_random_point = center_x + (cosine(random_angle) * random_radius)

        new_x = ((center[1]) + np.cos(np.deg2rad(angles)) * obs).astype(int)
        new_y = ((center[0]) + np.sin(np.deg2rad(angles)) * obs).astype(int)

        # here we clip the result from 0 to dim-1 to avoid index errors
        new_x = np.clip(new_x, 0, width - 1)
        new_y = np.clip(new_y, 0, height - 1)

        fix_map[new_y, new_x] = 1
        salmap = fixmap_2_salmap(fix_map)

        return salmap
