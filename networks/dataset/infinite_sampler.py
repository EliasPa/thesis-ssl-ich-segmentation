import itertools
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
import skimage.io as io
import os
import torchvision.transforms.functional as TF
import random as r

class InfiniteSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        def generate_infinite_iterables():
            while True: yield np.random.permutation(self.indices)

        return itertools.chain.from_iterable(generate_infinite_iterables())

    def __len__(self):
        return len(self.indices)

class SemiSupervisedSampler(Sampler):

    """
    Iterates over the unsupervised indices once, and many times over
    the supervised indices as needed (assuming there are less supervised samples
    than unsupervised samples).

    The intended use is to follow the setup in Mean-Teacher:
    https://arxiv.org/pdf/1703.01780.pdf

    """
    def __init__(self, s_indices, u_indices):
        self.s_indices = s_indices
        self.u_indices = u_indices

        if (len(s_indices) > len(u_indices)):
            print("Note: More supervised than unsupervised samples in semi supervised sampler...")

    def __iter__(self):
        def generate_infinite_iterables():
            while True: yield np.random.permutation(self.s_indices)

        zipped = zip(np.random.permutation(self.u_indices), itertools.chain.from_iterable(generate_infinite_iterables()))
        return iter([(u_i, s_i) for u_i, s_i in zipped])

    def __len__(self):
        return len(self.u_indices)


class TwoViewsDataset(Dataset):

    """
    Applies two different augmentation for each data point, providing two 'views'
    of the same data point.

    The itended use is to follow setup
    in FixMatch: https://arxiv.org/ftp/arxiv/papers/2001/2001.07685.pdf

    Args:
        - dataset_u         Unsupervised dataset, Map-like
        - dataset_s         Supervised dataset, Map-like
        - transform_u       Transformation for unsupervised examples
        - transform_s       Transformation for supervised examples
    """
    def __init__(self, dataset, transform_w, transform_s):
        self.dataset = dataset
        self.transform_w = transform_w
        self.transform_s = transform_s

    """
    Args:
        - index             Index of data point to fetch
    returns
        - ('weak' view, 'strong' view)      This depends on the actual augmentations fed to this class
                                            but FixMatch https://arxiv.org/ftp/arxiv/papers/2001/2001.07685.pdf
                                            uses this 'weak' / 'strong' naming scheme 
    """
    def __getitem__(self, index):
        print(index)
        image, *rest = self.dataset[index]
        return self.transform_w(image), self.transform_s(image), rest

    def __len__(self):
        return len(self.dataset)

class AugmentationWrapperDataset(Dataset):

    def __init__(self, dataset, transforms):

        self.dataset = dataset
        self.transforms = transforms
    
    def __getitem__(self, index):
        img, label = self.dataset[index]
        return self.transforms(img), label

    def __len__(self):
        return len(self.dataset)

class SemiSupervisedDataset(Dataset):

    """
    Combines two datasets.

    Args:
        - dataset_u         Unsupervised dataset, Map-like
        - dataset_s         Supervised dataset, Map-like
    """
    def __init__(self, dataset_u, dataset_s):
        self.dataset_u = dataset_u
        self.dataset_s = dataset_s

    def __getitem__(self, indices):
        print(indices)
        u_i, s_i = indices
        return self.dataset_u[u_i], self.dataset_s[s_i]
        
    def __len__(self):
        return len(self.dataset_u)

class MNISTLesionDatasetRAMHeavy(Dataset):

    """
    Simple dataset for MNIST with synthetic lesions.
    Requires data to be generated prior to use.

    Args:
        - image_path        path to images
        - mask_path         path to masks
    """
    def __init__(self, image_path, mask_path, file_extension='*.png'):
        image_path = os.path.join(image_path, file_extension)
        mask_path = os.path.join(mask_path, file_extension)
        self.images = io.imread_collection(image_path)
        self.masks = io.imread_collection(mask_path)

        print(len(self.images), len(self.masks))

    def __getitem__(self, index):
        return self.images[index], self.masks[index]
        
    def __len__(self):
        return len(self.images)

class MNISTLesionDatasetSlow(Dataset):

    """
    Simple dataset for MNIST with synthetic lesions.
    Requires data to be generated prior to use.

    Args:
        - image_path        path to images
        - mask_path         path to masks
    """
    def __init__(self, image_path, mask_path, n_classes, file_extension='*.png'):

        image_files = sorted(os.listdir(image_path))
        mask_files = sorted(os.listdir(mask_path))

        image_urls = []
        mask_urls = []

        for im in image_files:
            image_urls.append(os.path.join(image_path, im))
        
        for im in mask_files:
            mask_urls.append(os.path.join(mask_path, im))

        default_augmentations = {
                'random_rot': 15,
                'random_scale': (0.9, 1.1),
                'random_shear': 5,
                'random_translation': (5, 5),
                'flip': 0.5,
        }
        self.augmentations = default_augmentations

        self.image_urls = image_urls
        self.mask_urls = mask_urls

        self.ids = range(len(image_files))
        self.n_classes = n_classes

    def transform(self, img, flip, random_rot, random_scale, random_shear, random_translation, resample=TF.InterpolationMode.NEAREST, is_mask=False):
        # Random flip:
        if flip:
            img = TF.vflip(img)

        # Apply random affine
        img = TF.affine(img=img, angle=random_rot, scale=random_scale, shear=random_shear, translate=random_translation, interpolation=resample)
        if is_mask:
            img = torch.tensor(np.asarray(img, dtype=np.uint8))
        else:
            img = TF.to_tensor(img)
        return img

    def __getitem__(self, index):
        image_url = self.image_urls[index]
        mask_url = self.mask_urls[index]
        
        try:
            random_rot = r.uniform(-self.augmentations['random_rot'], self.augmentations['random_rot'])
            random_scale = r.uniform(self.augmentations['random_scale'][0], self.augmentations['random_scale'][1])
            random_shear = r.uniform(-self.augmentations['random_shear'], self.augmentations['random_shear'])
            random_translation_x = r.uniform(-self.augmentations['random_translation'][0], self.augmentations['random_translation'][1])
            random_translation_y = r.uniform(-self.augmentations['random_translation'][0], self.augmentations['random_translation'][1])
            random_translation = (random_translation_x, random_translation_y)
            flip = r.uniform(0, 1) < self.augmentations['flip']

            img = Image.open(image_url)
            img = self.transform(img=img, flip = flip, random_rot=random_rot, random_scale=random_scale, random_shear=random_shear, random_translation=random_translation, resample=TF.InterpolationMode.BILINEAR)

            mask = Image.open(mask_url)
            mask = self.transform(img=mask, is_mask=True, flip = flip, random_rot=random_rot, random_scale=random_scale, random_shear=random_shear, random_translation=random_translation)

            mask = mask.clip(0, self.n_classes - 1)
            mask = mask.long()

        except Exception as e:
            print(index, e)
            raise e

        return img, mask
        
    def __len__(self):
        return len(self.image_urls)
