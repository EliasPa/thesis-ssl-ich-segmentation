import numpy as np
import os
import shutil
from tqdm import tqdm
import random
import argparse
import analyze_split
import pandas as pd

def bool_arg(arg):
    assert(arg.lower() in ('true', 'false'))
    return arg.lower() == 'true'

def init_arguments():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('-dir', '--directory', type=str, required=True)
    args_parser.add_argument('--dry-run', default=True, type=bool_arg)
    args_parser.add_argument('--manual', default=False, type=bool_arg)
    args = args_parser.parse_args()
    return args

def move_file(source_directory, target_directory, file_name, args):
        source_path = os.path.join(source_directory, file_name)
        target_path = os.path.join(target_directory, file_name)

        print(source_path, target_path)
        if args.dry_run:
            print("dry")
            pass
        else:
            shutil.move(source_path, target_path)

def move_files(source_directory, target_directory, files_to_move, args):
    for file_name in tqdm(files_to_move):
        move_file(source_directory, target_directory, file_name, args)

    print(f'\nMoved {len(files_to_move)} files.')

def move(ids, source_directory, target_directory, args):

    ids_to_move=[]
    for sample in ids:
        s_str = str(sample)
        if len(s_str) == 2:
            s_str = '0'+s_str

        s_str = s_str+'.nii'
        ids_to_move.append(s_str)

    print(f'IDs to move: {ids_to_move}')

    move_files(
        source_directory=source_directory,
        target_directory=target_directory,
        files_to_move=ids_to_move,
        args=args
    )
def main(args):

    seed = 123
    np.random.seed(seed)
    random.seed(seed)

    patients = pd.read_csv('../data/hemorrhage_diagnosis_raw_ct.csv')
    test_dir = f'{args.directory}/test/'
    train_dir = f'{args.directory}/train/'
    validation_dir = f'{args.directory}/validation/'

    intravents, subarachs, subdurals = analyze_split.analyze_set(train_dir, patients)

    intravents_to_test = np.random.choice(intravents, 1, replace=False)
    subarachs_to_test = np.random.choice(subarachs, 1, replace=False)
    
    intravents = intravents[np.isin(intravents, intravents_to_test, invert=True)]
    intravents = intravents[np.isin(intravents, subarachs_to_test, invert=True)]
    subarachs = subarachs[np.isin(subarachs, subarachs_to_test, invert=True)]
    subarachs = subarachs[np.isin(subarachs, intravents_to_test, invert=True)]

    if args.manual: # nasty hack
        intravents_to_test = [85]
        subarachs_to_test = [92]

    augment_test = np.concatenate((intravents_to_test, subarachs_to_test), axis=0)
    move(augment_test, train_dir, test_dir, args)

    intravents_to_validation = np.random.choice(intravents, 1, replace=False)
    subarachs_to_validation = np.random.choice(subarachs, 1, replace=False)
    subdurals_to_validation = np.random.choice(subdurals, 1, replace=False)

    if args.manual: # nasty hack
        intravents_to_validation = [94]
        subarachs_to_validation = [93]
        subdurals_to_validation = [51]

    augment_val = np.concatenate((intravents_to_validation, subarachs_to_validation, subdurals_to_validation), axis=0)
    move(augment_val, train_dir, validation_dir, args)

    print('Done.')


if __name__ == '__main__':
    args = init_arguments()
    main(args)
