import numpy as np

from SalScan.Metric.Saliency import AUC_JUDD
from SalScan.Utils import normalize


# https://github.com/imatge-upc/saliency-2018-videosalgan/blob/master/metric_calculation/salience_metrics.py
def AUC_JUDD_SALIENCE_METRICS(s_map, gt):
    # ground truth is discrete, s_map is continous and normalized
    s_map = normalize_map(s_map)
    # thresholds are calculated from the salience map, only at places where fixations are present
    thresholds = []
    for i in range(0, gt.shape[0]):
        for k in range(0, gt.shape[1]):
            if gt[i][k] > 0:
                thresholds.append(s_map[i][k])

    num_fixations = np.sum(gt)
    # num fixations is no. of salience map values at gt >0

    thresholds = sorted(set(thresholds))

    # fp_list = []
    # tp_list = []
    area = []
    area.append((0.0, 0.0))
    for thresh in thresholds:
        # in the salience map, keep only those pixels with values above threshold
        temp = np.zeros(s_map.shape)
        temp[s_map >= thresh] = 1.0
        assert (
            np.max(gt) <= 1
        ), "something is wrong with ground truth..not discretized properly max value > 1"
        assert (
            np.max(s_map) <= 1
        ), "something is wrong with salience map..not normalized properly max value > 1"
        num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap with gt / total number of non fixated pixels
        # this becomes nan when gt is full of fixations..this won't happen
        fp = (np.sum(temp) - num_overlap) / (
            (np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations
        )

        area.append((round(tp, 4), round(fp, 4)))
        # tp_list.append(tp)
        # fp_list.append(fp)

    # tp_list.reverse()
    # fp_list.reverse()
    area.append((1.0, 1.0))
    # tp_list.append(1.0)
    # fp_list.append(1.0)
    # print tp_list
    area.sort(key=lambda x: x[0])
    tp_list = [x[0] for x in area]
    fp_list = [x[1] for x in area]
    return np.trapz(np.array(tp_list), np.array(fp_list))


def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    norm_s_map = (s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map)))
    return norm_s_map


# https://github.com/rdroste/unisal/blob/3fc15be2f66fba581bd406eb4039f11043d9292b/unisal/salience_metrics.py
def AUC_JUDD_UNISAL(s_map, gt):
    # ground truth is discrete, s_map is continous and normalized
    s_map = normalize_map(s_map)
    assert np.max(gt) == 1.0, "Ground truth not discretized properly max value > 1.0"
    assert np.max(s_map) == 1.0, "Salience map not normalized properly max value > 1.0"

    # thresholds are calculated from the salience map,
    # only at places where fixations are present
    thresholds = s_map[gt > 0].tolist()

    num_fixations = len(thresholds)
    # num fixations is no. of salience map values at gt >0

    thresholds = sorted(set(thresholds))

    area = []
    area.append((0.0, 0.0))
    for thresh in thresholds:
        # in the salience map,
        # keep only those pixels with values above threshold
        temp = s_map >= thresh
        num_overlap = np.sum(np.logical_and(temp, gt))
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap
        # with gt / total number of non fixated pixels
        # this becomes nan when gt is full of fixations..this won't happen
        fp = (np.sum(temp) - num_overlap) / (np.prod(gt.shape[:2]) - num_fixations)

        area.append((round(tp, 4), round(fp, 4)))

    area.append((1.0, 1.0))
    area.sort(key=lambda x: x[0])
    tp_list, fp_list = list(zip(*area))
    return np.trapz(np.array(tp_list), np.array(fp_list))


def test_auc_unisal_vs_salience_metrics():
    salmap = np.random.rand(640, 640)
    salmap = normalize(salmap, method="range")
    fixmap = np.zeros((640, 640))
    num_points = 100
    rows = np.random.randint(0, fixmap.shape[0], size=num_points)
    cols = np.random.randint(0, fixmap.shape[1], size=num_points)

    # Set the chosen locations to 1
    fixmap[rows, cols] = 1
    assert AUC_JUDD_UNISAL(salmap, fixmap) == AUC_JUDD_SALIENCE_METRICS(salmap, fixmap)
