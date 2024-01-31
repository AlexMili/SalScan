"""Definition of Metric class, used to handle all the metrics and their computation.
Based on implementation by Erwan DAVID (IPI, LS2N, Nantes, France), 2018
"""

import numpy as np

# Global variable
EPSILON = np.finfo("float").eps


# Matthias Kümmerer (2016) in pysaliency
# Implementations by https://github.com/matthias-k/pysaliency/blob/master/pysaliency/metrics.py


def probabilistic_image_based_kl_divergence(
    logp1: np.ndarray,
    logp2: np.ndarray,
    log_regularization: float = 0.0,
    quotient_regularization: float = 0.0,
) -> float:
    """
    Computes the Kullback-Leibler divergence between two probability distributions
    represented by their log probabilities.

    This function calculates the divergence where logp1 and logp2 are the log
    probabilities of the two distributions. Regularization can be applied through
    log_regularization and quotient_regularization parameters to avoid numerical
    instabilities.

    Parameters:
    logp1 (np.ndarray): Log probabilities of the first distribution.
    logp2 (np.ndarray): Log probabilities of the second distribution.
    log_regularization (float): Regularization term added to the logarithm to avoid
                                log(0).
    quotient_regularization (float): Regularization term added to the quotient to avoid
                                    division by zero.

    Returns:
    float: The Kullback-Leibler divergence between the two distributions.
    """

    if log_regularization or quotient_regularization:
        return (
            np.exp(logp2)
            * np.log(
                log_regularization
                + np.exp(logp2) / (np.exp(logp1) + quotient_regularization)
            )
        ).sum()
    else:
        return (np.exp(logp2) * (logp2 - logp1)).sum()


def image_based_kl_divergence(
    saliency_map_1: np.ndarray,
    saliency_map_2: np.ndarray,
    minimum_value: float = 1e-20,
    log_regularization: float = 0.0,
    quotient_regularization: float = 0.0,
) -> float:
    """
    Computes the image-based Kullback-Leibler divergence between two saliency maps.

    This function first converts the saliency maps into probability density functions and
    then computes the KL divergence.

    Parameters:
    saliency_map_1 (np.ndarray): The first saliency map.
    saliency_map_2 (np.ndarray): The second saliency map.
    minimum_value (float): A small value added to the saliency maps to avoid log(0).
    log_regularization (float): Regularization term for the logarithm in KL computation.
    quotient_regularization (float): Regularization term for the quotient in KL
                                        computation.

    Returns:
    float: The Kullback-Leibler divergence between the two saliency maps.
    """
    log_density_1 = np.log(
        convert_saliency_map_to_density(saliency_map_1, minimum_value=minimum_value)
    )
    log_density_2 = np.log(
        convert_saliency_map_to_density(saliency_map_2, minimum_value=minimum_value)
    )

    return probabilistic_image_based_kl_divergence(
        log_density_1,
        log_density_2,
        log_regularization=log_regularization,
        quotient_regularization=quotient_regularization,
    )


# Matthias Kümmerer (2016) in pysaliency
# Implementations by https://github.com/matthias-k/pysaliency/blob/master/pysaliency/metrics.py


def KLD(saliency_map_1: np.ndarray, saliency_map_2: np.ndarray) -> float:
    """
    Computes the image-based KL divergence between two saliency maps using predefined
    hyperparameters.

    This function is a specific case of image_based_kl_divergence, using the same
    hyperparameters as in the Tuebingen/MIT Saliency Benchmark.

    Parameters:
    saliency_map_1 (np.ndarray): The first saliency map.
    saliency_map_2 (np.ndarray): The second saliency map.

    Returns:
    float: The Kullback-Leibler divergence between the two saliency maps, computed with
    benchmark-specific hyperparameters.
    """
    return image_based_kl_divergence(
        saliency_map_1,
        saliency_map_2,
        minimum_value=EPSILON,
        log_regularization=2.2204e-16,
        quotient_regularization=2.2204e-16,
    )


# Matthias Kümmerer (2016) in pysaliency
# Implementations by https://github.com/matthias-k/pysaliency/blob/master/pysaliency/metrics.py


def NSS(saliency_map: np.ndarray, fixation_map: np.ndarray) -> float:
    """
    Computes the Normalized Scanpath Saliency (NSS) score for a given saliency map and a
    fixation map.

    The NSS score is a commonly used metric in visual attention models to evaluate the
    correspondence between predicted and actual human fixations. It normalizes the
    saliency map and then calculates the average saliency at fixation points.

    Parameters:
    saliency_map (np.ndarray): The saliency map being evaluated. It should be a 2D array
                                representing the predicted visual saliency.
    fixation_map (np.ndarray): The fixation map, typically a binary map where '1'
                                represents the presence of a human fixation and '0'
                                represents its absence.

    Returns:
    float: The NSS score. Higher values indicate better correspondence between the
    saliency map and the actual human fixations.
    """
    ys, xs = np.where(fixation_map == 1)
    xs = np.asarray(xs, dtype=int)
    ys = np.asarray(ys, dtype=int)

    mean = saliency_map.mean()
    std = saliency_map.std()

    value = saliency_map[ys, xs].copy()
    value -= mean

    if std:
        value /= std

    return np.mean(value)


# Matthias Kümmerer (2016) in pysaliency
# Implementations by https://github.com/matthias-k/pysaliency/blob/master/pysaliency/metrics.py


def CC(saliency_map_1: np.ndarray, saliency_map_2: np.ndarray) -> float:
    """
    Computes the Correlation Coefficient (CC) between two saliency maps.

    This function evaluates the similarity between two saliency maps by calculating the
    Pearson correlation coefficient. It normalizes the saliency maps before computing the
    correlation to ensure a fair comparison. The normalization subtracts the mean and
    divides by the standard deviation. If a map has a standard deviation of zero
    (constant map), it is handled specifically to avoid division by zero.

    Parameters:
    saliency_map_1 (np.ndarray): The first saliency map to be compared. It should be a
                                2D array.
    saliency_map_2 (np.ndarray): The second saliency map to be compared. It should also
                                be a 2D array.

    Returns:
    float: The correlation coefficient between the two saliency maps. The value ranges
    from -1 to 1, where 1 indicates perfect positive correlation, -1 indicates perfect
    negative correlation, and 0 indicates no correlation.
    """

    def normalize(saliency_map):
        saliency_map -= saliency_map.mean()
        std = saliency_map.std()

        if std:
            saliency_map /= std

        return saliency_map, std == 0

    smap1, constant1 = normalize(saliency_map_1.copy())
    smap2, constant2 = normalize(saliency_map_2.copy())

    if constant1 and not constant2:
        return 0.0
    else:
        return np.corrcoef(smap1.flatten(), smap2.flatten())[0, 1]


def convert_saliency_map_to_density(
    saliency_map: np.ndarray, minimum_value: float = 0.0, eps: float = EPSILON
) -> np.ndarray:
    """
    Converts a saliency map to a probability density function.

    This function normalizes a given saliency map such that its values sum up to 1,
    turning it into a probability density function.

    Parameters:
    saliency_map (np.ndarray): The saliency map to be converted. It should be a 2D array
                                representing visual saliency.
    minimum_value (float): A minimum value to be added to the saliency map to avoid
                            negative values.
    eps (float): A small constant (epsilon) used to ensure numerical stability,
                    especially to avoid division by zero. It's used to replace any values
                    in the saliency map that are lower than eps.
    Returns:
    np.ndarray: The normalized saliency map, now representing a probability density
                function.
    """
    if saliency_map.min() < 0:
        saliency_map = saliency_map - saliency_map.min()
        saliency_map = saliency_map + minimum_value

    saliency_map = np.where(saliency_map < eps, eps, saliency_map)
    saliency_map_sum = saliency_map.sum()

    # softmax style: np.sum(saliency_map) = 1
    if saliency_map_sum:
        saliency_map = saliency_map / saliency_map_sum
    else:
        saliency_map[:] = 1.0
        saliency_map /= saliency_map.sum()

    return saliency_map


# Matthias Kümmerer (2016) in pysaliency
# Implementations by https://github.com/matthias-k/pysaliency/blob/master/pysaliency/metrics.py


def SIM(saliency_map_1: np.ndarray, saliency_map_2: np.ndarray) -> float:
    """
    Computes the similarity metric between two saliency maps.

    This function calculates the similarity (SIM) metric, a common measure used to
    compare two saliency maps in terms of their spatial distribution. The function
    first converts each saliency map into a probability density function and then
    computes the similarity by summing the minimum value at each corresponding location
    in the two density maps.
    Parameters:
    saliency_map_1 (np.ndarray): The first saliency map to be compared. It should be a 2D
                                array representing visual saliency.
    saliency_map_2 (np.ndarray): The second saliency map to be compared. It should also
                                be a 2D array.

    Returns:
    float: The SIM score between the two saliency maps. This score ranges from 0 to 1,
    where higher values indicate greater similarity.
    """
    density_1 = convert_saliency_map_to_density(saliency_map_1, minimum_value=0.0)
    density_2 = convert_saliency_map_to_density(saliency_map_2, minimum_value=0.0)

    return np.min([density_1, density_2], axis=0).sum()


# By Akis Linardos (2018) in saliency-2018-videosalgan
# https://github.com/imatge-upc/saliency-2018-videosalgan/blob/master/metric_calculation/salience_metrics.py#L212


def IG(
    baseline_map: np.ndarray, gen_saliency_map: np.ndarray, gt_fixation_map: np.ndarray
) -> float:
    """
    Computes the Information Gain (IG) between a generated saliency map and a baseline
    saliency map with respect to a ground truth fixation map.

    Information Gain is a metric used to evaluate the effectiveness of a saliency model
    by comparing its predictions (gen_saliency_map) to those of a baseline model
    (baseline_map). It measures how much more (or less) information the generated
    saliency map provides about where people look compared to the baseline.

    Parameters:
    baseline_map (np.ndarray): The baseline saliency map, typically representing a simple
                                or average saliency model. It should be a 2D array.
    gen_saliency_map (np.ndarray): The generated saliency map from the model being
                                    evaluated. It should also be a 2D array.
    gt_fixation_map (np.ndarray): The ground truth fixation map, usually a binary map
                                    indicating actual human fixation points.

    Returns:
    float: The average Information Gain at the fixation points. Higher values indicate
    that the generated saliency map provides more information about human fixations
    compared to the baseline.
    """
    s_map = gen_saliency_map / (np.sum(gen_saliency_map) * 1.0 + EPSILON)
    baseline_map = baseline_map / (np.sum(baseline_map) * 1.0 + EPSILON)

    # for all places where gt=1, calculate info gain
    temp = []
    x, y = np.where(gt_fixation_map == 1)
    for i in zip(x, y):
        temp.append(
            np.log2(EPSILON + s_map[i[0], i[1]])
            - np.log2(EPSILON + baseline_map[i[0], i[1]])
        )

    return np.mean(temp)


def AUC_JUDD_fast(gen_saliency_map: np.ndarray, gt_fixation_map: np.ndarray) -> float:
    """
    Computes a fast approximation of the AUC (Area Under the Curve) metric for saliency
    models, specifically the Judd variant.

    This function is an optimized version of the AUC_JUDD metric, designed to handle
    cases with a large number of fixation points efficiently. Unlike the original
    AUC_JUDD, which uses all unique saliency values at fixation points as thresholds,
    this function uses fixed thresholds at 0.01 intervals between 0 and 1. This approach
    significantly reduces computation time, especially when the fixation map contains
    thousands of fixations. This metric, adapted by Alessandro Mondin, is an
    update of:
    https://github.com/tarunsharma1/saliency_metrics/blob/master/salience_metrics.py

    Parameters:
    gen_saliency_map (np.ndarray): The saliency map generated by the model. It should be
                                    a continuous and normalized 2D array.
    gt_fixation_map (np.ndarray): The ground truth fixation map, typically a binary map
                                    where '1' indicates a fixation and '0' indicates no
                                    fixation.

    Returns:
    float: The AUC score calculated using the Judd method. This score ranges from 0 to 1,
            where higher values indicate better model performance.
    """
    # ground truth is discrete, s_map is continous and normalized
    # thresholds are calculated from the saliency map, only at places
    # where fixations are present

    # num fixations is no. of saliency map values at gt >0
    thresholds = [thresh * 0.1 for thresh in range(1, 100)]
    num_fixations = np.sum(gt_fixation_map)

    area = [(0.0, 0.0)]
    for thresh in thresholds:
        # in the saliency map, keep only those pixels with values above threshold
        temp = np.zeros(gen_saliency_map.shape)
        temp[gen_saliency_map >= thresh] = 1.0

        num_overlap = np.where(np.add(temp, gt_fixation_map) == 2)[0].shape[0]
        # tp is the true positives rate
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap
        # with gt / total number of non fixated pixels this becomes nan when
        # gt is full of fixations..this won't happen

        # fp is the false positives rate
        fp = (np.sum(temp) - num_overlap) / (
            (np.shape(gt_fixation_map)[0] * np.shape(gt_fixation_map)[1]) - num_fixations
        )

        area.append((round(tp, 4), round(fp, 4)))

    area.append((1.0, 1.0))

    # here below we sort by increasing true positive rate in order
    # to correctly compute the underlying area with np.trapz
    area.sort(key=lambda x: x[0])
    tp_list = [x[0] for x in area]
    fp_list = [x[1] for x in area]
    return np.trapz(np.array(tp_list), np.array(fp_list))


# By Akis Linardos (2018) in saliency-2018-videosalgan
# https://github.com/imatge-upc/saliency-2018-videosalgan/blob/master/metric_calculation/salience_metrics.py#L35


def AUC_JUDD(gen_saliency_map: np.ndarray, gt_fixation_map: np.ndarray) -> np.ndarray:
    """
    Computes the Area Under the Curve (AUC) metric for saliency models using the Judd
    method.

    This function evaluates the performance of a saliency model by comparing its
    generated saliency map with a ground truth fixation map. It calculates the AUC score
    by considering all unique values in the saliency map at the locations of the
    fixations as thresholds. For each threshold, it determines true positive and false
    positive rates, which are then used to calculate the AUC.

    Parameters:
    gen_saliency_map (np.ndarray): The saliency map generated by the model. It is a
                                    continuous and normalized 2D array.
    gt_fixation_map (np.ndarray): The ground truth fixation map, usually a binary map
                                    where '1' indicates a fixation and '0' indicates no
                                    fixation.

    Returns:
    float: The resulting AUC score according to the Judd method. This score ranges from 0
    to 1, where higher values indicate better model performance.
    """
    # ground truth is discrete, s_map is continous and normalized
    # thresholds are calculated from the saliency map, only at places
    # where fixations are present

    # num fixations is no. of saliency map values at gt >0

    thresholds = gen_saliency_map[gt_fixation_map > 0].tolist()
    num_fixations = len(thresholds)

    area = [(0.0, 0.0)]
    for thresh in thresholds:
        # in the saliency map, keep only those pixels with values above threshold
        temp = np.zeros(gen_saliency_map.shape)
        temp[gen_saliency_map >= thresh] = 1.0

        num_overlap = np.where(np.add(temp, gt_fixation_map) == 2)[0].shape[0]
        # tp is the true positives rate
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap
        # with gt / total number of non fixated pixels this becomes nan when
        # gt is full of fixations..this won't happen

        # fp is the false positives rate
        fp = (np.sum(temp) - num_overlap) / (
            (np.shape(gt_fixation_map)[0] * np.shape(gt_fixation_map)[1]) - num_fixations
        )

        area.append((round(tp, 4), round(fp, 4)))

    area.append((1.0, 1.0))

    # here below we sort by increasing true positive rate in order
    # to correctly compute the underlying area with np.trapz
    area.sort(key=lambda x: x[0])
    tp_list = [x[0] for x in area]
    fp_list = [x[1] for x in area]
    return np.trapz(np.array(tp_list), np.array(fp_list))


def sAUC(
    gen_saliency_map,
    gt_fixation_map,
    thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    diameter=0.65,
):
    """
    Computes the Shuffled AUC (sAUC) metric for saliency models, with a focus on
    balancing central fixations. Unlike the original sAUC implementation that samples
    n fixation maps at a time in order to get a representation of center bias, this
    version adds an equal amount of false positives for every true positive within the
    center. This approach balances central fixations against external ones, providing a
    more equitable evaluation of saliency maps.

    Parameters:
    gen_saliency_map (np.array): The saliency map generated by the model.
    gt_fixation_map (np.array): The ground truth fixation map.
    thresholds (list): Thresholds for the metric calculation.
    diameter (float): Proportional diameter for central fixation balancing.

    Returns:
    float: The resulting Shuffled AUC metric score, where a higher score indicates better
            model performance.
    """
    # num fixations is no. of saliency map values at gt >0
    num_fixations = np.sum(gt_fixation_map)

    thresholds = sorted(set(thresholds))

    height, width = gen_saliency_map.shape[0], gen_saliency_map.shape[1]
    center_y, center_x = height // 2, width // 2
    min_dimension = np.min([height, width])
    diameter = min_dimension * diameter

    area = [(0.0, 0.0)]

    for thresh in thresholds:
        # in the saliency map, keep only those pixels with values above threshold
        temp = np.zeros(gen_saliency_map.shape)
        temp[gen_saliency_map >= thresh] = 1.0

        # tp are computed in the same way as AUC Judd
        # tp stands for true positive, and here below tp_y and tp_x
        # are 2 1d array representing the coordinates of true positives
        # w.r.t. the corresponsing axis
        tp_y, tp_x = np.where(np.add(temp, gt_fixation_map) == 2)
        num_overlap = tp_x.size
        tp = num_overlap / (num_fixations * 1.0)

        # here we add a number of fp equals to the number of observations
        # included in the center of the picture
        if tp_x.size != 0:
            obs_within_circle = (tp_x - center_x) ** 2 + (tp_y - center_y) ** 2 < (
                diameter / 2
            ) ** 2
            fp = np.sum(obs_within_circle) / num_fixations
        else:
            fp = 0

        area.append((round(tp, 4), round(fp, 4)))

    area.append((1.0, 1.0))

    # here below we sort by increasing true positive rate in order
    # to correctly compute the underlying area with np.trapz
    area.sort(key=lambda x: x[0])
    tp_list = [x[0] for x in area]
    fp_list = [x[1] for x in area]

    return np.trapz(np.array(tp_list), np.array(fp_list))
