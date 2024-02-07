import numpy as np
import onnxruntime as ort

from SalScan.Model.AbstractModel import AbstractModel
from SalScan.Transforms import Preprocess_UniSal
from SalScan.Utils import normalize


class UniSal_Static_onnx(AbstractModel):
    name = "Unisal_static_onnx"
    type = "saliency"
    wrapper = "python"
    params_to_store = ["path_to_weigths"]

    def __init__(self, options):
        """Initialises deep model onnx

        Args:
            options (dict): a dictionaire containing the params_name: params_value pairs.
            options["weights"] (str): Path to the model. Defaults at SalScan/Model/Weights/
        """

        super().__init__()

        self._config = {**options}
        self.path_to_weigths = self._config.get("path_to_weigths", None)
        self.session = ort.InferenceSession(self.path_to_weigths)
        self.transform = Preprocess_UniSal(out_size=(224, 384))

    def run(self, img, params):
        """Generate saliency map given a stimulus

        Args:
                img (np.array): input stimulus

        Returns:
                saliency (np.array): saliency map computed on the image.
                        Saliency map is normalized between 0 and 1.
        """
        x = self.transform(img)[None, None, ...].numpy()
        onnx_input = {"input": x}
        out = self.session.run(None, onnx_input)[0]
        out = np.exp(out)
        out = np.squeeze(out)
        out = normalize(out, method="range")

        return out
