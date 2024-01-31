import copy
import glob
import json
import os
from typing import Callable, List

import numpy as np
import pandas as pd

from SalScan.Metric.Saliency import AUC_JUDD, AUC_JUDD_fast, SIM, sAUC, NSS, IG, CC, KLD
from SalScan.Model.Saliency.CenterBias import CenterBias
from SalScan.Session.Saliency.ImageSession import ImageSaliencySession
from SalScan.Session.Saliency.VideoSession import VideoSaliencySession
from SalScan.Utils import get_logger

logger = get_logger(__name__)


class ImageSaliencyEvaluation:
    """
    This class evaluates image saliency models against image saliency datasets.
    It uses a variety of metrics to assess the performance of models and ranks
    them accordingly. Users can specify custom metrics and their evaluation criteria
    (higher or lower scores being better) in the available_metrics dictionary.

    Parameters:
        logs_folder (str): The path where to store the logs of the evaluation: the
                            evaluation metrics of each model and the final ranking of
                            models.
        evaluate_dict (dict): Evaluation dict containing models, parameters and datasets.
                                See example below.
        ranking_metrics (list): Saliency metrics used for evaluation. The earlier the
                                index of a metric within this list, the more the metric
                                will be important in creating the ranking. Available
                                values are specified in the class attribute
                                `available_metrics`.
        debug: (bool): When set to True, the evaluation will only run on the first 2
                        stimuli, since a full-evaluation would take between hours and
                        days which is not suitable for unit-testing.

    Attributes:
        available_metrics (dict): A dictionary listing all the metrics available for
                                    evaluating image saliency. Each metric is associated
                                    with a value (1 or 0) indicating whether a higher (1)
                                    or lower (0) score is better.

    Examples:
        >>> evaluate_dict = {
            "models": [
                {
                    "model": UniSal_Static,
                    "params": {
                        "mobilnet_weights": "path/to/weights.pht",
                        "decoder_weights": "path/to/custom_weights_1.pth",
                    },
                },
                {
                    "model": UniSal_Static,
                    "params": {
                        "mobilnet_weights": "path/to/weights.pht",
                        "decoder_weights": "path/to/custom_weights_2.pth",
                    },
                },
            ],
            "datasets": [
                CAT2000Dataset(os.path.join(dataset_folder, "CAT2000"), download=True),
                MIT1003Dataset(os.path.join(dataset_folder, "MIT1003"), download=True)
            ],
        }

        >>> evaluate = ImageSaliencyEvaluation(
            evaluate_dict=image_dict,
            ranking_metrics=[AUC_JUDD, sAUC, SIM, NSS, CC, KLD]
            )
        >>> evaluate.compute_ranking()

        This operation will store a csv file named "ranking_image.csv" in the current
        working directory.
        The file will look like this:

        | dataset  | ranking | model         | params                                                                                         | AUC_JUDD_fast | sAUC | SIM   | NSS  | CC   | KLD  |
        |----------|---------|---------------|------------------------------------------------------------------------------------------------|---------------|------|-------|------|------|------|
        | cat2000  | 1       | unisal_static | {'mobilnet_weights': 'path/to/weights.pht', 'decoder_weights': 'path/to/custom_weights_1.pth'} | 0.856         | 0.65 | 0.558 | 0.695| 0.463| 0.752|
        | cat2000  | 2       | unisal_static | {'path_to_weights': 'path/to/weights.pht', 'decoder_weights': 'path/to/custom_weights_2.pth'}  | 0.821         | 0.62 | 0.587 | 0.643| 0.567| 0.721|
        | mit1003  | 1       | unisal_static | {'mobilnet_weights': 'path/to/weights.pht', 'decoder_weights': 'path/to/custom_weights_2.pth'} | 0.842         | 0.622| 0.652 | 1.3  | 0.745| 0.544|
        | mit1003  | 2       | unisal_static | {'mobilnet_weights': 'path/to/weights.pht', 'decoder_weights': 'path/to/custom_weights_1.pth'} | 0.794         | 0.621| 0.689 | 1.21 | 0.733| 0.602|
    """

    available_metrics = {
        "AUC_JUDD": 1,
        "sAUC": 1,
        "NSS": 1,
        "CC": 1,
        "KLD": 0,
        "SIM": 1,
        "AUC_JUDD_fast": 1,
        "IG": 1,
    }

    def __init__(
        self,
        logs_folder: str = os.path.join(os.getcwd(), "val_logs"),
        evaluate_dict: dict = None,
        ranking_metrics: List[Callable[[np.ndarray, np.ndarray], np.ndarray]] = [
            AUC_JUDD,
            sAUC,
            NSS,
            CC,
            KLD,
            SIM,
            IG,
        ],
        debug: bool = False,
    ):
        self.evaluate_dict = evaluate_dict.copy()

        self.logs_folder = logs_folder

        self.debug = debug

        # initialise variable used in class methods
        self._missing_evaluation = None
        for metric in ranking_metrics:
            if metric.__name__ not in self.available_metrics.keys():
                raise ValueError("ranking metrics must be part of available metrics")

        self.ranking_metrics = ranking_metrics

        # if single values are passed as values in within the params of the
        # evaluate dict, these are casted to list. for instance key: value
        # will be converted to key: [value]. This is needed because to use
        # correctly itertools.product
        self._cast_values_tolist()

    def _cast_values_tolist(self):
        """
        Converts single values in the evaluate_dict parameters to lists.
        """
        for idx, model in enumerate(self.evaluate_dict["models"]):
            for key, value in model["params"].items():
                if isinstance(value, tuple) is True:
                    self.evaluate_dict["models"][idx]["params"][key] = list(value)

    def _check_logs(self):
        """
        Checks for missing evaluations by examining the evaluation logs. Determines which
        combinations of datasets, models, and parameters have not yet been evaluated and
        returns them.

        Returns:
            List[Dict]: A list of dictionaries, each representing a missing evaluation
                        combination.
        """

        missing_evaluation = []
        for dataset_class in self.evaluate_dict["datasets"]:
            dataset_class_name = dataset_class.name.lower()
            for pos_idx, model in enumerate(self.evaluate_dict["models"]):
                model_name = model["model"].name.lower()

                if sorted(model["params"].keys()) != sorted(
                    model["model"].params_to_store
                ):
                    raise ValueError(
                        f"The parameters of {model['model'].name}, specified at position"
                        f"{pos_idx+1} in the evaluation dict contain fewer/extra"
                        f"parameters compared to {model['model'].name}.params_to_store"
                        "(class attribute)"
                    )

                # we compute all the combinations of the session
                # parameters including the implicit default values
                exp_path = os.path.join(
                    self.logs_folder, dataset_class_name, model_name, "*"
                )

                already_stored = False
                for exp in glob.glob(exp_path):
                    with open(os.path.join(exp, "params.json"), "r") as f:
                        params = json.load(f)
                        if model["params"] == params:
                            already_stored = True
                            break

                if already_stored is False:
                    exp_to_compute = {
                        "model": model["model"],
                        # we create a deepcopy of the dataset because
                        # the parameters of the dataset are changed in the
                        # run_evaluation() method by setting the value directly
                        # to the attribute. Without using deepcopy each 'dataset'
                        # value in the run_evaluation would reference to the same
                        # object, which would imply a wrong session setting.
                        "dataset": copy.deepcopy(dataset_class),
                        "params": model["params"],
                    }

                    missing_evaluation.append(exp_to_compute)

        return missing_evaluation

    def compute_ranking(self):
        """
        Computes and stores the ranking of models with respect to each dataset. This
        involves running missing evaluations, if any, and then ranking the models based
        on the specified metrics. The final result is a csv file called
        `ranking_image.csv` which is stored in the current working directory.
        """
        self._missing_evaluation = self._check_logs()

        if self._missing_evaluation != []:
            self._run_evaluation()

        # contains a list of booleans describing whether a given metrics
        # is ascending or descending
        metrics_ascending = []
        for metric in self.ranking_metrics:
            metric_value = self.available_metrics[metric.__name__]
            if metric_value == 1:
                order = False
            else:
                order = True
            metrics_ascending.append(order)

        metric_names = [metric.__name__ for metric in self.ranking_metrics]
        columns = ["dataset", "ranking", "model", "params"] + metric_names
        overall_ranking = pd.DataFrame(columns=columns)

        for dataset in self.evaluate_dict["datasets"]:
            dataset_ranking = pd.DataFrame(columns=columns)
            dataset = dataset.name.lower()
            for model in self.evaluate_dict["models"]:
                model_name = model["model"].name.lower()
                model_params = model["params"]
                experiments = glob.glob(
                    os.path.join(self.logs_folder, dataset, model_name, "*")
                )
                for experiment in experiments:
                    with open(os.path.join(experiment, "params.json")) as f:
                        params = json.load(f)

                    if params != model_params:
                        continue

                    # the following csv file has two colums (metric name and
                    # metric value) and n metric rows
                    evaluation = pd.read_csv(
                        os.path.join(experiment, "val.csv"),
                        usecols=["metric", "dataset"],
                    )
                    # using only selected metrics
                    evaluation = evaluation.loc[evaluation["metric"].isin(metric_names)]
                    # transposing make the df of two rows: (metric name and metric
                    # value) and n_metrics columns
                    evaluation = evaluation.transpose()
                    # reindex order of evaluation metrics to the order of ranking_metrics
                    evaluation = (
                        # rename columns from int (0, 1, 2, 3) to metrics
                        # that are present in the first row (index=0)
                        evaluation.rename(columns=evaluation.iloc[0])
                        # after renaming the column like the first row
                        # the first row is can be discarded as duplicate
                        .drop(evaluation.index[0]).reset_index(drop=True)
                    )

                    evaluation["dataset"] = dataset
                    evaluation["model"] = model_name
                    # evaluation params must be set to list to make it a hashable
                    evaluation["params"] = [params]

                    # before concatenating we align the order of metrics stored
                    # in a file with the order specified to initialise the class.
                    # This operation makes sure that the columns concatenate on
                    # axis=0 are actually the same metric.
                    evaluation = evaluation.reindex(columns=columns)

                    dataset_ranking = pd.concat([dataset_ranking, evaluation])

            # making params from unhashable type list to hashable type str
            dataset_ranking["params"] = dataset_ranking["params"].astype(str)
            dataset_ranking.sort_values(
                by=metric_names, ascending=metrics_ascending, inplace=True
            )
            dataset_ranking["ranking"] = range(1, len(dataset_ranking) + 1)

            overall_ranking = pd.concat([overall_ranking, dataset_ranking])

        overall_ranking.reset_index(drop=True, inplace=True)

        overall_ranking[metric_names] = (
            overall_ranking[metric_names].astype(float).round(4)
        )

        overall_ranking.reset_index(drop=True, inplace=True)

        logger.info("✅ Stored results in ranking_image.csv")

        overall_ranking.to_csv("ranking_image.csv", sep=",")

    def _run_evaluation(self):
        """
        Runs the evaluation for each missing combination of dataset, model, and
        parameters. This process includes setting up the evaluation session, populating
        datasets, and executing the evaluation.
        """
        # If baseline model is not specified, it is set to default Center Bias with
        # diameter of 0.15
        if "baseline_model" not in self.evaluate_dict.keys():
            logger.warning(
                "\n WARN: Baseline model not specified in evaluation dict, "
                "baseline model is set to Center Bias with diameter: 0.15"
            )
        baseline = self.evaluate_dict.get("baseline_model", {})
        baseline_model = baseline.get("model", CenterBias)
        baseline_parameters = baseline.get("parameters", {"diameter": 0.15})

        for idx, combination in enumerate(self._missing_evaluation):
            logger.info(
                "\n\n"
                "++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
                f"+++++++++ RUNNING EVALUTATION ON SESSION {idx+1}/"
                f"{len(self._missing_evaluation)} ++++++++\n"
                "++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
            )
            model = combination["model"]
            dataset = combination["dataset"]
            params = combination["params"]

            evaluation_metrics = []
            for metric in self.ranking_metrics:
                evaluation_metrics.append(metric)

            session_args = {
                "logs_directory": self.logs_folder,
                "dataset": dataset,
                "model": model,
                "baseline_model": baseline_model,
                "metrics": evaluation_metrics,
                "params": {
                    "model": params,
                    "baseline_model": baseline_parameters,
                },
                "debug": self.debug,
            }

            dataset.populate()

            session = ImageSaliencySession(session_args)
            session.evaluate()


class VideoSaliencyEvaluation:
    """
    This class evaluates video saliency models against video saliency datasets.
    It uses a variety of metrics to assess the performance of models and ranks
    them accordingly. Users can specify custom metrics and their evaluation criteria
    (higher or lower scores being better) in the available_metrics dictionary.

    Parameters:
        logs_folder (str): The path where to store the logs of the evaluation: the
                            evaluation metrics of each model and the final ranking of
                            models.
        evaluate_list (list): Evaluation list containing models, parameters and datasets.
        ranking_metrics (list): Saliency metrics used for evaluation. The earlier the
                                index of a metric within this list, the more the metric
                                will be important in creating the ranking. Available
                                values are specified in the class attribute
                                `available_metrics`.
        debug: (bool): When set to True, the evaluation will run only on the
                        first 2 stimuli, since a full-evaluation would take between
                        hours and days which is not suitable for unit-testing.

    Attributes:
        available_metrics (dict): A dictionary listing all the metrics available for
                                    evaluating image saliency. Each metric is associated
                                    with a value (1 or 0) indicating whether a higher (1)
                                    or lower (0) score is better.

    Examples:
        >>> dhf1k_kwargs_1 = {
            "root_path": "path/to/dhf1k",
            "sequence_length": 6,
            "many_to_many": True,
            "to_array": True,
            "transform_stimuli": Preprocess_UniSal(out_size=(224, 384)),
            "evaluation_fps": 6,
        }

        >>> dhf1k_kwargs_2 = {
            "root_path": "path/to/dhf1k",
            "sequence_length": 24,
            "many_to_many": True,
            "to_array": True,
            "transform_stimuli": Preprocess_UniSal(out_size=(224, 384)),
            "evaluation_fps": 6,
        }

        >>> video_list = [
                {
                    "model": UniSal,
                    "params": {
                        "mobilnet_weights": "path/to/weights.pht",
                        "decoder_weights": "path/to/custom_weights_1.pth",
                        "sequence_length": 6,
                    },
                    "dataset": DHF1KDataset(**dhf1k_kwargs),
                },
                {
                    "model": UniSal,
                    "params": {
                        "mobilnet_weights": "path/to/weights.pht",
                        "decoder_weights": "path/to/custom_weights_1.pth",
                        "sequence_length": 24,
                    },
                    "dataset": DHF1KDataset(**dhf1k_kwargs),
                },
        ]
        >>> evaluate = VideoSaliencyEvaluation(
                evaluate_dict=video_list,
                ranking_metrics=[AUC_JUDD, sAUC, SIM, NSS, CC, KLD]
            )
        >>> evaluate.compute_ranking()

        This operation will store a csv file named "ranking_video.csv" in the current
        working directory.
        The file will look like this:

        | dataset | ranking | model  | params                                                                                                                | AUC_JUDD| sAUC |  NSS  |  CC   |  KLD  | SIM   |
        |---------|---------|--------|-----------------------------------------------------------------------------------------------------------------------|---------|------|-------|-------|-------|-------|
        | dhf1k   | 1       | unisal | {'mobilnet_weights': 'path/to/weights.pht', 'decoder_weights': 'path/to/custom_weights_1.pth', 'sequence_length': 6}  | 0.882   | 0.62 | 4.234 | 0.623 | 1.028 | 0.42  |
        | dhf1k   | 2       | unisal | {'mobilnet_weights': 'path/to/weights.pht', 'decoder_weights': 'path/to/custom_weights_1.pth', 'sequence_length': 24} | 0.876   | 0.65 | 3.95  | 0.589 | 1.094 | 0.405 |
    """

    available_metrics = {
        "AUC_JUDD": 1,
        "sAUC": 1,
        "NSS": 1,
        "CC": 1,
        "KLD": 0,
        "SIM": 1,
        "AUC_JUDD_fast": 1,
    }

    def __init__(
        self,
        logs_folder: str = os.path.join(os.getcwd(), "val_logs"),
        evaluate_list: list = None,
        ranking_metrics: List[Callable[[np.ndarray, np.ndarray], np.ndarray]] = [
            AUC_JUDD,
            sAUC,
            NSS,
            CC,
            KLD,
            SIM,
        ],
        debug: bool = False,
    ):
        self.evaluate_list = evaluate_list.copy()

        self.logs_folder = logs_folder

        self.debug = debug

        # initialise variable used in class methods
        self._missing_evaluation = None
        for metric in ranking_metrics:
            if metric.__name__ not in self.available_metrics.keys():
                raise ValueError("ranking metrics must be part of available metrics")

        self.ranking_metrics = ranking_metrics

        # if single values are passed as values in within the params of the
        # evaluate dict, these are casted to list. for instance key: value
        # will be converted to key: [value]. This is needed because to use
        # correctly itertools.product
        self._cast_values_tolist()

    def _cast_values_tolist(self):
        """
        Converts single values in the evaluate_list parameters to lists.
        """

        for idx, model in enumerate(self.evaluate_list):
            for key, value in model["params"].items():
                if isinstance(value, tuple) is True:
                    self.evaluate_list[idx]["params"][key] = list(value)

    def _check_logs(self):
        """
        Checks for missing evaluations by examining the evaluation logs. Determines which
        combinations of datasets, models, and parameters have not yet been evaluated and
        returns them.

        Returns:
            List[Dict]: A list of dictionaries, each representing a missing evaluation
                        combination.
        """

        missing_evaluation = []
        for pos_idx, items in enumerate(self.evaluate_list):
            model = items["model"]
            params = items["params"]
            dataset = items["dataset"]
            dataset_class_name = dataset.name.lower()
            model_name = model.name.lower()

            if sorted(params.keys()) != sorted(model.params_to_store):
                raise ValueError(
                    f"The parameters of {model.name}, specified at position {pos_idx+1}"
                    "in the evaluation dict contain fewer/extra parameters compared to"
                    f"{model.name}.params_to_store (class attribute)"
                )

            # we compute all the combinations of the session
            # parameters including the implicit default values
            exp_path = os.path.join(self.logs_folder, dataset_class_name, model_name, "*")

            already_stored = False
            for exp in glob.glob(exp_path):
                with open(os.path.join(exp, "params.json"), "r") as f:
                    exp_params = json.load(f)
                    if exp_params == params:
                        already_stored = True
                        break

            if already_stored is False:
                exp_to_compute = {
                    "model": model,
                    # we create a deepcopy of the dataset because
                    # the parameters of the dataset are changed in the
                    # run_evaluation() method by setting the value directly
                    # to the attribute. Without using deepcopy each 'dataset'
                    # value in the run_evaluation would reference to the same
                    # object, which would imply a wrong session setting.
                    "dataset": copy.deepcopy(dataset),
                    "params": params,
                }

                missing_evaluation.append(exp_to_compute)

        return missing_evaluation

    def compute_ranking(self):
        """
        Computes and stores the ranking of models with respect to each dataset. This
        involves running missing evaluations, if any, and then ranking the models based
        on the specified metrics. The final result is a csv file called
        `ranking_video.csv` which is stored in the current working directory.
        """
        self._missing_evaluation = self._check_logs()

        if self._missing_evaluation != []:
            self._run_evaluation()

        # contains a list of booleans describing whether a given metrics
        # is ascending or descending
        metrics_ascending = []
        for metric in self.ranking_metrics:
            metric_value = self.available_metrics[metric.__name__]
            if metric_value == 1:
                order = False
            else:
                order = True
            metrics_ascending.append(order)

        metric_names = [metric.__name__ for metric in self.ranking_metrics]
        columns = ["dataset", "ranking", "model", "params"] + metric_names
        overall_ranking = pd.DataFrame(columns=columns)
        dataset_ranking = pd.DataFrame(columns=columns)

        for items in self.evaluate_list:
            model = items["model"]
            model_params = items["params"]
            dataset = items["dataset"]
            model_name = model.name.lower()
            dataset = dataset.name.lower()

            experiments = glob.glob(
                os.path.join(self.logs_folder, dataset, model_name, "*")
            )
            for experiment in experiments:
                with open(os.path.join(experiment, "params.json")) as f:
                    params = json.load(f)

                if params != model_params:
                    continue

                # the following csv file has two colums (metric name and
                # metric value) and n metric rows
                evaluation = pd.read_csv(
                    os.path.join(experiment, "val.csv"),
                    usecols=["metric", "dataset"],
                )
                # using only selected metrics
                evaluation = evaluation.loc[evaluation["metric"].isin(metric_names)]
                # transposing make the df of two rows: (metric name and metric
                # value) and n_metrics columns
                evaluation = evaluation.transpose()
                # reindex order of evaluation metrics to the order of ranking_metrics
                evaluation = (
                    # rename columns from int (0, 1, 2, 3) to metrics
                    # that are present in the first row (index=0)
                    evaluation.rename(columns=evaluation.iloc[0])
                    # after renaming the column like the first row
                    # the first row is can be discarded as duplicate
                    .drop(evaluation.index[0]).reset_index(drop=True)
                )

                evaluation["dataset"] = dataset
                evaluation["model"] = model_name
                # evaluation params must be set to list to make it a hashable
                evaluation["params"] = [params]

                # before concatenating we align the order of metrics stored
                # in a file with the order specified to initialise the class.
                # This operation makes sure that the columns concatenate on
                # axis=0 are actually the same metric.
                evaluation = evaluation.reindex(columns=columns)

                dataset_ranking = pd.concat([dataset_ranking, evaluation])

        # making params from unhashable type list to hashable type str
        dataset_ranking["params"] = dataset_ranking["params"].astype(str)
        dataset_ranking.sort_values(
            by=metric_names, ascending=metrics_ascending, inplace=True
        )
        dataset_ranking["ranking"] = range(1, len(dataset_ranking) + 1)

        overall_ranking = pd.concat([overall_ranking, dataset_ranking])

        overall_ranking.reset_index(drop=True, inplace=True)

        overall_ranking[metric_names] = (
            overall_ranking[metric_names].astype(float).round(4)
        )

        overall_ranking.reset_index(drop=True, inplace=True)

        logger.info(f"✅ Stored results in ranking_video.csv")

        overall_ranking.to_csv(f"ranking_video.csv", sep=",")

    def _run_evaluation(self):
        """
        Runs the evaluation for each missing combination of dataset, model, and
        parameters. This process includes setting up the evaluation session, populating
        datasets, and executing the evaluation.
        """
        for idx, combination in enumerate(self._missing_evaluation):
            logger.info(
                "\n\n"
                "++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
                f"+++++++++ RUNNING EVALUTATION ON SESSION {idx+1}/"
                f"{len(self._missing_evaluation)} ++++++++\n"
                "++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
            )
            model = combination["model"]
            dataset = combination["dataset"]
            params = combination["params"]

            evaluation_metrics = []
            for metric in self.ranking_metrics:
                evaluation_metrics.append(metric)

            session_args = {
                "logs_directory": self.logs_folder,
                "dataset": dataset,
                "model": model,
                "metrics": evaluation_metrics,
                "params": {"model": params},
                "debug": self.debug,
            }

            dataset.populate()
            session = VideoSaliencySession(session_args)
            session.evaluate()
