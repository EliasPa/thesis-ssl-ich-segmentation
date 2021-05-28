import pandas as pd
import numpy as np
import argparse
from os import listdir
import seaborn as sns
import matplotlib.pyplot as plt

def init_arguments():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('-s', '--split-dir', type=str, required=True)
    args = args_parser.parse_args()
    return args

h_start_idx = 2
h_end_idx = 7

def analyze_set_fully(data_set_dir, patients):
    print(f'\n{data_set_dir}')
    patients_in_set = get_patients_in_set(data_set_dir, patients)

    indices = [0]
    indices.extend(range(h_start_idx, h_end_idx))

    grouped_patients = patients_in_set.iloc[:,indices].groupby(['PatientNumber']).sum(axis=0).reset_index()
    hemorrhages = patients_in_set.iloc[:, h_start_idx:h_end_idx]
    bin_hem_sum = hemorrhages.sum(axis=1).clip(0,1)
    #hem_sum = hemorrhages.sum(axis=1).clip(0,1)
    
    total_hemorrhagic_slices = bin_hem_sum.sum()
    print(f"Total slices with hemorrhage {total_hemorrhagic_slices}")
    #print(f"Empty slices {bin_hem_sum[bin_hem_sum == 0].shape[0]}")
    print(total_hemorrhagic_slices / (patients_in_set.shape[0]*1.0))

    print(hemorrhages.sum(axis=0))
    #print("grouped")
    print(grouped_patients.iloc[:,1:].clip(0,1))#.sum(axis=0))
    
    print("Intraventriculars")
    intras = grouped_patients[grouped_patients.Intraventricular != 0].PatientNumber.to_numpy()
    print(intras)

    print("Subarachnoid")
    subarachs = grouped_patients[grouped_patients.Subarachnoid != 0].PatientNumber.to_numpy()
    print(subarachs)

    print("Subdurals")
    subdurals = grouped_patients[grouped_patients.Subdural != 0].PatientNumber.to_numpy()
    print(subdurals)
    #print("patients with subdural {}")
    #print('\n')

    hem_sum = hemorrhages.sum(axis=0)
    sns.barplot(x=hem_sum, y=hem_sum.index)
    plt.title(f'{data_set_dir}')
    plt.show()

    return intras, subarachs, subdurals

def get_patients_in_set(data_set_dir, patients):
    files = listdir(data_set_dir)

    pids = []
    for file_name in files:
        pid = file_name.split('.')[0]
        if pid[0] == '0':
            pid = pid[1:]

        pid = int(pid)
        pids.append(pid)

    pids = np.array(pids)

    patients_in_set = patients[patients.PatientNumber.isin(pids)]

    return patients_in_set

def analyze_set(data_set_dir, patients):
    patients_in_set = get_patients_in_set(data_set_dir, patients)
    indices = [0]
    indices.extend(range(h_start_idx, h_end_idx))
    grouped_patients = patients_in_set.iloc[:,indices].groupby(['PatientNumber']).sum(axis=0).reset_index()
    intras = grouped_patients[grouped_patients.Intraventricular != 0].PatientNumber.to_numpy()
    subarachs = grouped_patients[grouped_patients.Subarachnoid != 0].PatientNumber.to_numpy()
    subdurals = grouped_patients[grouped_patients.Subdural != 0].PatientNumber.to_numpy()
    return intras, subarachs, subdurals


def main(args):
    patients = pd.read_csv('../data/hemorrhage_diagnosis_raw_ct.csv')

    #h_start_idx = 2
    #h_end_idx = 7
    #hemorrhages = patients_in_set.iloc[:, h_start_idx:h_end_idx]
    #print(hemorrhages.sum(axis=0))

    test_dir = f'{args.split_dir}/test/'
    train_dir = f'{args.split_dir}/train/'
    validation_dir = f'{args.split_dir}/validation/'

    analyze_set_fully(test_dir, patients)
    analyze_set_fully(train_dir, patients)
    analyze_set_fully(validation_dir, patients)


if __name__ == '__main__':
    args = init_arguments()
    main(args)
    pass
