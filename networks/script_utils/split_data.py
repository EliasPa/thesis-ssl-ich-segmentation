"""
Randomizes a split (train, val, test).
"""

import numpy as np
import os
from shutil import copyfile
from tqdm import tqdm
import pandas as pd
import random
import argparse

np.random.seed(123)
random.seed(123)

#patients = pd.read_csv('../data/hemorrhage_diagnosis_raw_ct.csv')
#patients_with_some_hematoma = np.array([49, 50, 51, 52, 53, 58, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
# 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 97])
#negative_samples = np.array([54,  55,  56,  57,  95,  96,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
#                    110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
#                    128, 129, 130])
#
#assert(np.alltrue(np.isin(patients_with_some_hematoma, negative_samples, invert=True)))
#
#all_patient_ids = patients.PatientNumber.unique()
#print("All patients: \n", all_patient_ids)
#
#n_val = 3
#n_test = 5
#n_negatives_in_val_and_test = 5
#
#k = 5
#
#validation_folds = []
#test_folds = []
#train_folds = []
#
#test_fold = np.random.choice(patients_with_some_hematoma, n_test, replace=False) # Same test data for all folds for now
#test_negative_samples = np.random.choice(negative_samples, n_negatives_in_val_and_test, replace=False)
#print(f"test negative samples {test_negative_samples}")
#post_test_selection_negatives = negative_samples[np.isin(negative_samples, test_negative_samples, invert=True)]
#print(f"negative samples left after test {post_test_selection_negatives}")
#test_fold = np.concatenate((test_fold, test_negative_samples), axis=0)
#
#post_test_selection_patients = patients_with_some_hematoma[np.isin(patients_with_some_hematoma, test_fold, invert=True)]
#validation_fold = np.random.choice(post_test_selection_patients, n_val, replace=False)
#val_negative_samples = np.random.choice(post_test_selection_negatives, n_negatives_in_val_and_test, replace=False)
#print(f"val negative samples {val_negative_samples}")
#post_val_selection_negatives = post_test_selection_negatives[np.isin(post_test_selection_negatives, val_negative_samples, invert=True)]
#print(f"negative samples left after val {post_val_selection_negatives}")
#validation_fold = np.concatenate((validation_fold, val_negative_samples), axis=0)
#
#already_reserved_data = np.concatenate((validation_fold, test_fold), axis=0)
#train_fold = all_patient_ids[np.isin(all_patient_ids, already_reserved_data, invert=True)]
#
#print("Splits created!")
#
#print("Validation: \n", validation_fold)
#print("Test: \n", test_fold)
#print("Train: \n", train_fold)
#
#assert(np.intersect1d(validation_fold, test_fold).size == 0)
#assert(np.intersect1d(validation_fold, train_fold).size == 0)
#assert(np.intersect1d(test_fold, train_fold).size == 0)

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def copy_files_in_fold(source_directory, target_directory, fold):
    for patient_number in tqdm(fold):
        p_str = f'{patient_number}'
        if len(p_str) == 2:
            p_str = '0'+p_str
        
        old_file = f'{source_directory}/{p_str}.nii'
        new_file = f'{target_directory}/{p_str}.nii'

        print(f'copying {old_file} to {new_file}')
        copyfile(old_file, new_file)

def copy_files_in_folds(base_directory, validation_fold, test_fold, train_fold):
    
    i = 0
    directory = f'{base_directory}/fold_{i}'

    create_dir(directory)

    val_directory = f'{directory}/validation'
    test_directory = f'{directory}/test'
    train_directory = f'{directory}/train'

    create_dir(val_directory)
    create_dir(test_directory)
    create_dir(train_directory)

    print("val test:")
    print(validation_fold, test_fold)
    copy_files_in_fold(base_directory, val_directory, validation_fold)
    copy_files_in_fold(base_directory, test_directory, test_fold)
    copy_files_in_fold(base_directory, train_directory, train_fold)

def create_folds():
    np.random.seed(123)
    random.seed(123)

    patients = pd.read_csv('../data/hemorrhage_diagnosis_raw_ct.csv')
    N_negatives_val_test = 5

    patients_grouped = patients.groupby(['PatientNumber']).agg({
        'Intraventricular': 'sum',
        'Intraparenchymal': 'sum',
        'Subarachnoid': 'sum',
        'Epidural': 'sum',
        'Subdural': 'sum',
        'No_Hemorrhage': 'sum',
        'SliceNumber': 'max'
        }).reset_index()

    patients_with_hemorrhage = patients_grouped[patients_grouped['No_Hemorrhage'] != patients_grouped['SliceNumber'] ]
    patients_without_hemorrhage = patients_grouped[patients_grouped['No_Hemorrhage'] == patients_grouped['SliceNumber'] ]

    hems = ['Intraventricular', 'Intraparenchymal', 'Subarachnoid','Epidural','Subdural']

    test_patients = np.full((5,), fill_value=-1)
    val_patients = np.full((5,), fill_value=-1)
    for i, hemorrhage in enumerate(hems):
        data_with_hemorrhage = np.array(patients_with_hemorrhage[patients_with_hemorrhage[hemorrhage] > 1].PatientNumber)

        taken = np.concatenate((test_patients, val_patients), axis=0)
        free_to_take = data_with_hemorrhage[np.isin(data_with_hemorrhage, taken, invert=True)]

        selected = np.random.choice(free_to_take, 2, replace=False)

        test = selected[0]
        val = selected[1]
        test_patients[i] = test
        val_patients[i] = val

    patients_numbers_without_hemorrhage = np.array(patients_without_hemorrhage.PatientNumber)
    test_negatives = np.random.choice(patients_numbers_without_hemorrhage, N_negatives_val_test, replace=False)
    patients_numbers_without_hemorrhage = patients_numbers_without_hemorrhage[np.isin(patients_numbers_without_hemorrhage, test_negatives, invert=True)]
    test_patients = np.concatenate((test_patients, test_negatives), axis=0)

    val_negatives = np.random.choice(patients_numbers_without_hemorrhage, N_negatives_val_test, replace=False)
    val_patients = np.concatenate((val_patients, val_negatives), axis=0)

    all_patients = patients_grouped.PatientNumber.to_numpy()
    taken = np.concatenate((test_patients, val_patients), axis=0)
    train_patients = all_patients[np.isin(all_patients, taken, invert=True)]

    print(test_patients, val_patients, train_patients)
    print(f"in total: {test_patients.shape[0] + val_patients.shape[0] + train_patients.shape[0]}")

    assert(np.intersect1d(val_patients, test_patients).size == 0)
    assert(np.intersect1d(val_patients, train_patients).size == 0)
    assert(np.intersect1d(test_patients, train_patients).size == 0)

    return val_patients, test_patients, train_patients

def main():
    val_patients, test_patients, train_patients = create_folds()

    base_directory = '../data/multiclass_mask'
    copy_files_in_folds(base_directory, val_patients, test_patients, train_patients)
    
    base_directory = '../data/ct_scans'
    copy_files_in_folds(base_directory, val_patients, test_patients, train_patients)

if __name__ == '__main__':
    main()
