import pandas as pd
import numpy as np
import statistics
from glob import glob
import os
from pathlib import Path
import argparse
import torch
from tqdm import tqdm

def init_arguments():
    args_parser = argparse.ArgumentParser()
    
    args_parser.add_argument('-p', '--file-path',
        type=str,
        required=False,
        help='Path to stats file without the * at the end'
    )
    args_parser.add_argument('-m', '--models-path',
        type=str,
        help='Path to models',
        default='../experiment_runs/'
    )
    args_parser.add_argument('-n', '--experiment-name', type=str, required=False)
    args_parser.add_argument('-t', '--threshold', default=0.005, type=float, required=False, help='Values below are considered outlier and are removed from new stats.')

    args = args_parser.parse_args()
    return args

def process(file_path, threshold):
    file_names = glob(file_path + '*.json')

    for file_name in file_names:
        basename = os.path.basename(file_name)
        print(basename)
        new_basename = f'modified_{basename}'

        try:
            data_table = pd.read_json(file_name)
            data_dict = data_table.to_dict()
            property_names = data_table.columns.values
            row_names = list(data_table.index)

            nz_mask = np.array(data_table['any_blood_dice']['raw_data']) > threshold
            for metric_name in property_names:
                if metric_name != 'ID':
                    for row_name in row_names:
                        raw_data = np.array(data_table[metric_name]['raw_data'])
                        thresholded_data = raw_data[nz_mask]
                        
                        new_key = f'thresholded_{row_name}'
                        if row_name == 'mean':
                            data_dict[metric_name][new_key] = statistics.mean(thresholded_data)
                        elif row_name == 'stdev':
                            data_dict[metric_name][new_key] = statistics.stdev(thresholded_data)
                        elif row_name == 'variance':
                            data_dict[metric_name][new_key] = statistics.variance(thresholded_data)
                        elif row_name == 'N':
                            data_dict[metric_name][new_key] = nz_mask.astype(int).sum()
                        elif row_name == 'raw_data':
                            data_dict[metric_name][new_key] = thresholded_data.tolist()

            data_table = pd.DataFrame.from_dict(data_dict)
            new_dir = os.path.join(file_path, 'modified')
            Path(new_dir).mkdir(parents=False, exist_ok=True)
            save_path_name = os.path.join(new_dir, new_basename)
            data_table.to_json(save_path_name)
            
            print(f'Successfully processed {new_basename}. Saved modified stats to {save_path_name}\n')
        except Exception as error:
            print(f'Error! {error}')
            print('')

def create_table(file_path):
    print(file_path)
    file_names = glob(file_path + '*.json')
    
    metrics_to_latex = {
        'any_blood_dice': 'Dice',
        'any_blood_iou': 'IoU',
        'pixel_specificities': 'TPR (pixel)',
        'pixel_sensitivities': 'TNR (pixel)',
        'slice_specificities': 'TPR (slice)',
        'slice_sensitivities': 'TNR (slice)'
    }

    rows = []
    print(file_names)
    for i, file_name in enumerate(file_names):
        basename = os.path.basename(file_name)
        data_table = pd.read_json(file_name)

        row_data = {}
        basename_without_ending = basename.split('.')[:-1]
        basename_without_ending = ''.join(basename_without_ending)
        basename_without_ending = basename_without_ending.replace('_', ' ')
        row_data['Model'] = basename_without_ending
        for metric in metrics_to_latex.keys():
            mean = data_table[metric]['thresholded_mean']
            stdev = data_table[metric]['thresholded_stdev']
            row_data[metrics_to_latex[metric]] = '${0:.3f} (\\pm {1:.3f})$'.format(mean, stdev)
        rows.append(row_data)

    data = pd.DataFrame(rows)
    tex_name = 'table.tex'
    save_path_tex_name = os.path.join(file_path, tex_name)
    data.to_latex(save_path_tex_name, index=False, escape=False)

def create_subtables(file_path, args):
    print(file_path)
    file_names = glob(file_path + '*.json')

    metrics_to_latex = {
        'any_blood_dice': 'Dice',
        'any_blood_iou': 'IoU',
        'pixel_specificities': 'TPR (pixel)',
        'pixel_sensitivities': 'TNR (pixel)',
        'slice_specificities': 'TPR (slice)',
        'slice_sensitivities': 'TNR (slice)'
    }

    rows = []
    print(file_names)
    for file_name in tqdm(file_names):
        basename = os.path.basename(file_name)
        data_table = pd.read_json(file_name)

        model_ID = data_table['ID']['raw_data'][0]
        assert(isinstance(model_ID, (str)))
        model_path = os.path.join(args.models_path, args.experiment_name, 'models', model_ID)
        state_dict = torch.load(model_path)
        hps = state_dict['hps']

        row_data = {}
        basename_without_ending = basename.split('.')[:-1]
        basename_without_ending = ''.join(basename_without_ending)
        basename_without_ending = basename_without_ending.replace('_', ' ')
        row_data['Model'] = basename_without_ending
        for metric in metrics_to_latex.keys():
            mean = data_table[metric]['thresholded_mean']
            stdev = data_table[metric]['thresholded_stdev']
            row_data[metrics_to_latex[metric]] = '${0:.3f} (\\pm {1:.3f})$'.format(mean, stdev)

        row_data['Consistency loss type'] = hps.consistency_loss_type
        row_data['Supervised augmentation'] = hps.augmentation_type if hasattr(hps, 'augmentation_type') else '-'
        row_data['Consistency augmentation'] = hps.consistency_augmentation_type if hasattr(hps, 'consistency_augmentation_type') else '-'
        row_data['$\lambda$'] = hps.lambd if hasattr(hps, 'lambd') else '-'

        rows.append(row_data)

    data = pd.DataFrame(rows)
    tex_name = 'subtables.tex'
    save_path_tex_name = os.path.join(file_path, tex_name)
    data.to_latex(save_path_tex_name, index=False, escape=False)

def main(args):
    if args.file_path is None:
        assert(args.experiment_name is not None)
        file_path = f'../experiment_runs/{args.experiment_name}/evaluation/statistics/'
    else:
        assert(args.file_path is not None)
        file_path = args.file_path

    print(f'Beginning to process {file_path}')
    process(file_path, threshold=args.threshold)

    file_path = os.path.join(file_path, 'modified/')
    create_table(file_path)
    create_subtables(file_path, args)
    print('Done!')

if __name__ == '__main__':
    args = init_arguments()
    main(args)
