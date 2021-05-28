from .test_metrics import dice, iou, SegmentationMetricMode
import numpy as np
from enum import Enum

class SegmentationMetricReductionMode(Enum):

    EMPTY_SET_IGNORE_BINARY_NO_BG = "empty_set_ignore_binary_no_bg"
    EMPTY_SET_IGNORE_MULTICLASS_NO_BG = "empty_set_ignore_multiclass_no_bg"
    NONE = "none"

class PerformanceMeter(object):

    def __init__(self, scoring_function):
        self.scoring_function = scoring_function

    def get_final_metric(self):
        pass

class SegmentationMeter(PerformanceMeter):

    def __init__(self, n_classes, scoring_function, mode, reduction_mode):
        super(SegmentationMeter, self).__init__(scoring_function=scoring_function)
        self.top = 0
        self.bottom = 0
        self.epsilon = 0.000001
        self.n_classes = n_classes
        self.mode = mode
        self.reduction_mode = reduction_mode

    def update(self, prediction, target):

        if self.reduction_mode != SegmentationMetricReductionMode.NONE:
            _, t, b = self.scoring_function(prediction, target, num_classes=self.n_classes, segmentation_mode=self.mode)
        else:
            _, t, b, _ = self.scoring_function(prediction, target)

        if self.reduction_mode == SegmentationMetricReductionMode.EMPTY_SET_IGNORE_BINARY_NO_BG:
            self.top += t[1] if not np.isnan(b[1:])[0] else 0
            self.bottom += b[1] if not np.isnan(b[1:])[0] else 0
        elif self.reduction_mode == SegmentationMetricReductionMode.EMPTY_SET_IGNORE_MULTICLASS_NO_BG:
            self.top += t[1:][~np.isnan(t[1:])].sum() # don't include first class (bg)
            self.bottom += b[1:][~np.isnan(b[1:])].sum() # don't include first class (bg)
        elif self.reduction_mode == SegmentationMetricReductionMode.NONE:
            self.top += t.item()
            self.bottom += b.item()
        else:
            raise Exception(f'Segmentation metric mode {self.reduction_mode} not implemented yet.')

    def get_final_metric(self):
        return self.top / (self.bottom + self.epsilon)

class DiceMeter(SegmentationMeter):

    def __init__(self, n_classes, reduction_mode, mode=SegmentationMetricMode.EMPTY_SET_IGNORE):
        super(DiceMeter, self).__init__(
            n_classes=n_classes,
            scoring_function=dice,
            mode=mode,
            reduction_mode=reduction_mode
        )

class LegacySegmentationMeter(SegmentationMeter):

    def __init__(self, n_classes, scoring_function):
        super(LegacySegmentationMeter, self).__init__(
            n_classes=n_classes,
            scoring_function=scoring_function,
            mode=None,
            reduction_mode=SegmentationMetricReductionMode.NONE
        )

class IoUMeter(SegmentationMeter):

    def __init__(self, n_classes, reduction_mode, mode=SegmentationMetricMode.EMPTY_SET_IGNORE):
        super(IoUMeter, self).__init__(
            n_classes=n_classes,
            scoring_function=iou,
            mode=mode,
            reduction_mode=reduction_mode
        )
