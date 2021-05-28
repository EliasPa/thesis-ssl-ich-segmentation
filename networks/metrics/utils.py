import math
import statistics
import numpy as np
from collections import defaultdict
import json

class StatisticEvaluator(object):

    def __init__(self):
        self.statistics_dict = defaultdict(list)
        self.results = None
        self.ID_key = 'ID'

    def update(self, test_metrics, ID, extract_from_list_metrics=True):
        self.statistics_dict[self.ID_key].append(ID) # Keep track of ID (for downstream analysis)
        for key, value in test_metrics.items():
            if isinstance(value, (np.ndarray, list)) and extract_from_list_metrics:
                assert(not isinstance(value, np.ndarray))

                # BG and any-blood recall, precision etc.
                if len(value) == 1: 
                    metric_value = value[0] # Just take the blood's value
                    if isinstance(metric_value, (list)):
                        metric_value = metric_value[0] # Just take the blood's value
                    
                    if not isinstance(metric_value, (dict)):
                        self.statistics_dict[key].append(metric_value)
                if len(value) == 2: 
                    metric_value = value[1] # Just take the blood's value
                    self.statistics_dict[key].append(metric_value)
            elif isinstance(value, (float, int)):
                self.statistics_dict[key].append(value) # Any-blood-dice, iou etc.
            elif isinstance(value, (dict)):
                sub_dict = {}
                for sub_key, sub_value in value.items():
                    sub_dict[f'{key}_{sub_key}'] = sub_value

                self.update(sub_dict, extract_from_list_metrics)

    def calculate_final_statistics(self):
        result_dict = defaultdict(dict)
        for key, score_list in self.statistics_dict.items():
            print(score_list)
            print(key)
            if len(score_list) != 0:

                if key != self.ID_key:
                    # Regular statistics
                    if len(score_list) > 1:
                        result_dict[key]['mean'] = statistics.mean(score_list)
                        result_dict[key]['stdev'] = statistics.stdev(score_list)
                        result_dict[key]['variance'] = statistics.variance(score_list)
                        result_dict[key]['N'] = len(score_list)
                    else:
                        result_dict[key]['mean'] = score_list[0]
                        result_dict[key]['stdev'] = 0
                        result_dict[key]['variance'] = 0
                        result_dict[key]['N'] = len(score_list)

                # Raw data and IDs
                result_dict[key]['raw_data'] = score_list
            
        self.results = result_dict
        return result_dict

    def save_as_json(self, save_path):
        if self.results is None:
            self.calculate_final_statistics()

        with open(save_path, 'w') as f:
            json.dump(self.results, f)
