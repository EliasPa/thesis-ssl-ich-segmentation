import torch
import numpy as np
from torch.utils.data import Subset
from os import listdir

# TODO: Now validation set includes samples from supervised and unsupervised set.
def random_split_equal_classes(data, sizes, n_classes):
    print(data)
    targets = np.array(data.targets)
    print(targets)
    classes = range(n_classes)
    subsets = []
    #indices = []
    for size in sizes:
        to_subset = np.array([])
        size_per_class = int(size / n_classes)
        for c in classes:
            samples_from_class = np.where(targets == c)[0]
            to_subset = np.concatenate((to_subset, np.random.choice(samples_from_class, size_per_class, False)))
            
        rand_perm = np.random.permutation(to_subset)
        indices = [int(i) for i in rand_perm]
        subset = Subset(data, indices)
        subsets.append(subset)
        #subsets.append(subset.indices)

    return tuple(subsets)

"""
Splits a dataset into supervised, unsupervised and validation sets. Uses id order to enable
deterministic behavior.
"""
def split_set(dataset, N_s, N_u, N_v):
    N = N_s + N_u + N_v
    assert(len(dataset) >= N)
    ids = sorted(dataset.ids)[:N]

    validation_ids = ids[-N_v:]
    supervised_ids = ids[:N_s]
    unsupervised_ids = ids[N_s:(N_u + N_s)]

    np_s = np.array(supervised_ids)
    np_u = np.array(unsupervised_ids)
    np_v = np.array(validation_ids)
    assert(np.intersect1d(np_s, np_u).size == 0)
    assert(np.intersect1d(np_v, np_u).size == 0)
    assert(np.intersect1d(np_v, np_s).size == 0)

    supervised_set = Subset(dataset, supervised_ids)
    unsupervised_set = Subset(dataset, unsupervised_ids)
    validation_set = Subset(dataset, validation_ids)

    return supervised_set, unsupervised_set, validation_set
