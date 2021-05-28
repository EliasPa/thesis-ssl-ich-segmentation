import numpy as np
from enum import Enum

class SegmentationMetricMode(Enum):

    """

    Different modes for segmentation metrics Dice and IoU.

    This attempts to address the issue with undefined scores when
    both the prediction and target are empty sets for a specific class.

    """
    EMPTY_SET_IGNORE = "ignore_empty_sets"
    EMPTY_SET_ONES = "ignore_empty_sets"
    DEFAULT = "default"

def dice(prediction, target, num_classes, segmentation_mode=SegmentationMetricMode.DEFAULT):
    intersections = []
    bottoms = []

    prediction = prediction.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    for i in range(num_classes):
        p_match = prediction == i
        t_match = target == i
        
        b1 = (p_match & p_match) * 1.0
        b2 = (t_match & t_match) * 1.0
        
        intersection = (p_match & t_match).sum() * 1.0
        bottom = (b1 + b2).sum()

        if segmentation_mode == SegmentationMetricMode.EMPTY_SET_IGNORE:
            # Completely ignore the class score if denominator is zero.
            if bottom == 0:
                intersection = np.nan
                bottom = np.nan
        elif segmentation_mode == SegmentationMetricMode.EMPTY_SET_ONES:
            # Yield a score of 1 if both the prediction and target are empty sets
            if bottom == 0:
                intersection = 1.0
                bottom = 1.0
        elif segmentation_mode == SegmentationMetricMode.DEFAULT:
            pass

        intersections.append(intersection)
        bottoms.append(bottom)

    epsilon = 0.001
    intersections = np.array(intersections)
    bottoms = np.array(bottoms)

    return 2*intersections.sum() / (bottoms.sum() + epsilon), 2*intersections, bottoms

def iou(prediction, target, num_classes, segmentation_mode=SegmentationMetricMode.DEFAULT):
    intersections = []
    unions = []

    prediction = prediction.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    for i in range(num_classes):
        p_match = prediction == i
        t_match = target == i
        intersection = (p_match & t_match).sum() * 1.0
        union = (p_match | t_match).sum() * 1.0

        if segmentation_mode == SegmentationMetricMode.EMPTY_SET_IGNORE:
            # Completely ignore the class score if denominator is zero.
            if union == 0:
                intersection = np.nan
                union = np.nan
        elif segmentation_mode == SegmentationMetricMode.EMPTY_SET_ONES:
            # Yield a score of 1 if both the prediction and target are empty sets
            if union == 0:
                intersection = 1.0
                union = 1.0
        elif segmentation_mode == SegmentationMetricMode.DEFAULT:
            pass

        intersections.append(intersection)
        unions.append(union)

    epsilon = 0.001
    intersections = np.array(intersections)
    unions = np.array(unions)

    return intersections.sum() / (unions.sum() + epsilon), intersections, unions


def get_statistics_from_cms(cms):
    n_classes = cms.shape[1]

    all_stats = {}
    for cm in cms:
        cm_stats = get_statistics_from_cm(cm, n_classes)
        for key in cm_stats.keys():
            
            if isinstance(cm_stats[key], np.ndarray):
                if key not in all_stats:
                    all_stats[key] = cm_stats[key][np.newaxis, ...]
                else:
                    all_stats[key] = np.vstack((all_stats[key], cm_stats[key]))

    return all_stats

def get_statistics_from_cm(cm, n_classes):
    """
    Calculate statistics from confusion matrix
    """
    precisions = np.zeros((n_classes,))
    recalls = np.zeros((n_classes,))
    f1 = np.zeros((n_classes,))
    specificities = np.zeros((n_classes,))
    sensitivities = np.zeros((n_classes,))
    epsilon = 0.001

    tps = np.zeros((n_classes,))
    tns = np.zeros((n_classes,))
    fps = np.zeros((n_classes,))
    fns = np.zeros((n_classes,))

    for c in range(0,n_classes):
        tp = cm[c,c]
        up_left = cm[:c,:c].sum()
        up_right = cm[:c,c+1:].sum()
        down_left = cm[c+1:,:c].sum()
        down_right = cm[c+1:,c+1:].sum()

        tn = up_left + up_right + down_left + down_right

        fp = cm[c,:].sum() - tp 
        fn = cm[:,c].sum() - tp
        
        precisions[c] = tp / (tp + fp + epsilon)
        recalls[c] = tp / (tp + fn + epsilon)
        f1[c] = 2*(precisions[c]*recalls[c]) / (precisions[c] + recalls[c] + epsilon)
        specificities[c] = tn / (tn + fp + epsilon)
        sensitivities[c] = tp / (tp + fn + epsilon)

        tps[c] = tp
        tns[c] = tn
        fps[c] = fp
        fns[c] = fn

    stats = {
        "precisions": precisions,
        "recalls": recalls,
        "f1": f1,
        "specificities": specificities,
        "fpr": 1 - specificities,
        "sensitivities": sensitivities,
        "base_stats": (tps, tns, fps, fns)
    }

    return stats
