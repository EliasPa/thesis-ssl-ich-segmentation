import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from .infinite_sampler import InfiniteSampler, AugmentationWrapperDataset
import numpy as np
import torch
import nibabel as nib
from os import listdir
from os.path import splitext
import random as r
import PIL
import torchvision.transforms.functional as TF
from glob import glob
from tqdm import tqdm
import os
import functools
from multiprocessing import Pool
import hashlib

class CtIchDataset(Dataset):

    def __init__(self, img_dir, resize, n_classes, experiment_data_path=None, N=-1, mask_dir=None, empty_mask_discard_probability=0.95, num_threads=1, augment=True, override_transforms=None):
        
        self.ids = []
        self.separator = '$'
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.window = (0,80)
        self.resize = resize
        self.experiment_data_path = experiment_data_path
        self.n_classes = n_classes
        self.augment = augment
        self.override_transforms = override_transforms

        default_augmentations = {
                'random_rot': 15,
                'random_scale': (0.9, 1.1),
                'random_shear': 5,
                'random_translation': (20, 20),
                'flip': 0.5,
        }

        if override_transforms is not None:
            self.augmentations = override_transforms
        else:
            self.augmentations = default_augmentations

        self.skipped_masks = None
        nof_included_zero_masks = 0

        # Create a skip array from masks that don't contain any lesions. Lesion are skipped with probability
        # determined by empty_mask_discard_probability
        if mask_dir is not None:
            self.skipped_masks = []
            for mask in tqdm(sorted(listdir(mask_dir))):
                if not mask.startswith('.'):
                    name = splitext(mask)[0]
                    img = nib.load(f'{mask_dir}/{mask}')
                    image_data = img.get_fdata()
                    depth = image_data.shape[2]
                    for i in range(depth):
                        slice_mask = image_data[:,:,i]
                        if (slice_mask.sum() == 0 and r.uniform(0,1) < empty_mask_discard_probability):
                            self.skipped_masks.append(name+self.separator+str(i))
                        elif slice_mask.sum() == 0:
                            nof_included_zero_masks += 1

        if experiment_data_path is None or N == 0:
            files = sorted(listdir(self.img_dir))

            # Only take as many CT scans as requested
            N_ = len(files)+1, N if N != -1 else len(files)+1
            files = files[:min(N_)]
            N_files = len(files)
        else:
            
            # Select the pre-determined partition matching with N
            partitions = listdir(experiment_data_path)
            partition = None
            for partition_name in partitions:
                partition_N = int(partition_name.split('_')[1].split('.')[0])
                if N == partition_N:
                    partition = partition_name
                    break

            assert(partition != None) # N must match some partition.
            print("Selected partition: ", partition)
            files = []
            with open(os.path.join(experiment_data_path, partition)) as f:
                unprocessed_files = f.readlines()
                for file_name in unprocessed_files:
                    files.append(file_name.strip('\n'))
            
            N_files = len(files)
            print(f'{N_files} included in dataset')

            # Assert for reproducibility experiments. Comment out if not needed.
            self.reproducibility_safeguard(files, N_files)

        if num_threads == 1:
            self.ids = self.process_images(files)
        else:
            # Prepare files into chunks to be processed in parallel
            chunk_size = N_files // num_threads + 1
            chunks = [files[i:i+chunk_size] for i in range(0, N_files, chunk_size)]

            # Initiate thread pool and accumulate IDs in parallel
            pool = Pool(num_threads)
            for new_ids in tqdm(pool.map(functools.partial(self.process_images), chunks)):
                self.ids.extend(new_ids)
        
        print(f"Total masks: {len(self.ids)}. Negative samples: {nof_included_zero_masks}")

    def reproducibility_safeguard(self, files, N_files):
        if self.experiment_data_path is not None:

            md5 = hashlib.md5()
            for file_name in files:
                md5.update(file_name.encode())
            
            hashes = {
                100: '4ef6daf5752a05aa70903ddbbfece097',
                500: '949dd566d9314eb3118ad3a427312c75',
                1000: '5b6c714d889b0c8d4f75f6cc01716ccd',
                2000: '1237df5fdc72262d2fa85c55eb08f2b5',
                4000: 'e76b89e8d9f2cdd75d5498c32f2abd77',
                5000: '5aaa25fb59ca376ade7cf6e031c0210b',
                8000: '7892ec69f8ad32846de58f7a96667622',
                10000: '1fbf2d361d051af42813ce7641ebaae9',
            }

            md5_hash = md5.hexdigest()
            print(md5_hash)
            assert(md5_hash == hashes[N_files])

    def process_images(self, files):
        ids = []
        for f, file_name in enumerate(tqdm(files)):
            if not file_name.startswith('.'):
                name = splitext(file_name)[0]
                img = nib.load(os.path.join(self.img_dir, file_name))
                image_data = img.get_fdata()
                depth = image_data.shape[2]
                for i in range(depth):
                    img_id = name+self.separator+str(i)
                    
                    if self.skipped_masks is not None:
                        if img_id not in self.skipped_masks:
                            ids.append(img_id)
                    else:
                        ids.append(img_id)
        return ids

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
        patient_id, slice_id = self.ids[index].split(self.separator)
        slice_id = int(slice_id)
        img_file_name = glob(self.img_dir + patient_id + '.*')

        size = int(512 * self.resize)

        if self.mask_dir is not None:
            mask_file_name = glob(self.mask_dir + patient_id + '.*')
        else:
            mask = torch.full(size=(size, size), fill_value=-1)

        img = nib.load(img_file_name[0]).get_fdata()[:, :, slice_id]
        img = img.clip(self.window[0], self.window[1])

        if self.mask_dir is not None:
            mask = np.asarray(nib.load(mask_file_name[0]).dataobj, dtype=np.uint8)[:, :, slice_id]
        
        img = img.astype(np.float32)

        if self.mask_dir is not None:
            mask = mask.astype(np.uint8)

        to_pil = transforms.ToPILImage()
    
        img = to_pil(img)

        if self.mask_dir is not None:
            mask = to_pil(mask)

        img = img.resize((size, size), PIL.Image.BILINEAR)

        if self.mask_dir is not None:
            mask = mask.resize((size, size), PIL.Image.NEAREST)

        if self.augment:
            random_rot = r.uniform(-self.augmentations['random_rot'], self.augmentations['random_rot'])
            random_scale = r.uniform(self.augmentations['random_scale'][0], self.augmentations['random_scale'][1])
            random_shear = r.uniform(-self.augmentations['random_shear'], self.augmentations['random_shear'])
            random_translation_x = r.uniform(-self.augmentations['random_translation'][0], self.augmentations['random_translation'][1])
            random_translation_y = r.uniform(-self.augmentations['random_translation'][0], self.augmentations['random_translation'][1])
            random_translation = (random_translation_x, random_translation_y)
            flip = r.uniform(0, 1) < self.augmentations['flip']

            img = self.transform(img=img, flip = flip, random_rot=random_rot, random_scale=random_scale, random_shear=random_shear, random_translation=random_translation, resample=TF.InterpolationMode.BILINEAR)
            
            if self.mask_dir is not None:
                mask = self.transform(img=mask, is_mask=True, flip = flip, random_rot=random_rot, random_scale=random_scale, random_shear=random_shear, random_translation=random_translation)
                mask = mask.long()
        else:
            img = TF.to_tensor(img)
            if self.mask_dir is not None:
                mask = torch.tensor(np.asarray(mask, dtype=np.uint8))
                mask = mask.long()

        img = img / self.window[-1]
        mask = mask.clip(0, self.n_classes - 1)

        return img, mask

    def __len__(self):
        return len(self.ids)

def create_all_dataloaders(N, batch_size_s, batch_size_u, sizes, pin_memory, transform_u, transform_s, transform_t, image_path, mask_path, u_image_path, train=True, path='./data/mnist', regular_mnist=False, n_classes=6, experiment_data_path='../data/experiment_data/'):
    assert(len(sizes) == 3)

    supervised_data = CtIchDataset(img_dir=image_path, mask_dir=mask_path, N=-1, n_classes=n_classes)
    supervised_size = int(sizes[0] * len(supervised_data))
    validation_size = int(sizes[1] * len(supervised_data))

    unsupervised_set = CtIchDataset(img_dir=u_image_path, N=N, n_classes=n_classes, experiment_data_path=experiment_data_path)
    
    supervised_set, rest = torch.utils.data.random_split(supervised_data, [supervised_size, len(supervised_data) - supervised_size])
    validation_set, test_set = torch.utils.data.random_split(rest, [validation_size, len(rest) - validation_size])

    print(f'Unsupervised: \n\t- train: {len(unsupervised_set)}\n\tSupervised:\n\t- train: {len(supervised_set)}\n\t- validation: {len(validation_set)}\n\t- test: {len(test_set)}\n')

    u_infinite_sampler = InfiniteSampler(list(range(len(unsupervised_set))))
    s_infinite_sampler = InfiniteSampler(list(range(len(supervised_set))))

    unsupervised_set = AugmentationWrapperDataset(unsupervised_set, transform_u)
    supervised_set = AugmentationWrapperDataset(supervised_set, transform_s)
    validation_set = AugmentationWrapperDataset(validation_set, transform_t)
    test_set = AugmentationWrapperDataset(test_set, transform_t)

    u_loader = DataLoader(dataset=unsupervised_set, batch_size=batch_size_u, pin_memory=pin_memory, sampler=u_infinite_sampler)
    s_loader = DataLoader(dataset=supervised_set, batch_size=batch_size_s, pin_memory=pin_memory, shuffle=True)
    s_loader_inf = DataLoader(dataset=supervised_set, batch_size=batch_size_s, pin_memory=pin_memory, sampler=s_infinite_sampler)
    v_loader = DataLoader(dataset=validation_set, batch_size=batch_size_s, pin_memory=pin_memory, shuffle=True)
    t_loader = DataLoader(dataset=test_set, batch_size=batch_size_s, pin_memory=pin_memory, shuffle=True)

    return u_loader, s_loader, s_loader_inf, v_loader, t_loader

def create_all_dataloaders_folded(N, num_threads, batch_size_s, batch_size_t, batch_size_u, sizes, pin_memory, transform_u, transform_s, transform_t, image_path, mask_path, u_image_path, fold, n_classes=6, resize=0.5, num_workers=1, experiment_data_path='../data/experiment_data/', base_augment_train=True, override_transforms=None):
    validation_image_path = os.path.join(image_path, f'fold_{fold}', 'validation/').replace("\\", "/")
    test_image_path = os.path.join(image_path, f'fold_{fold}', 'test/').replace("\\", "/")
    train_image_path = os.path.join(image_path, f'fold_{fold}', 'train/').replace("\\", "/")
    
    validation_mask_path = os.path.join(mask_path, f'fold_{fold}', 'validation/').replace("\\", "/")
    test_mask_path = os.path.join(mask_path, f'fold_{fold}', 'test/').replace("\\", "/")
    train_mask_path = os.path.join(mask_path, f'fold_{fold}', 'train/').replace("\\", "/")

    unsupervised_set = CtIchDataset(img_dir=u_image_path, N=N, resize=resize, n_classes=n_classes, experiment_data_path=experiment_data_path, override_transforms=override_transforms)
    
    validation_set = CtIchDataset(img_dir=validation_image_path, mask_dir=validation_mask_path, N=-1, empty_mask_discard_probability=0, num_threads=num_threads, resize=resize, n_classes=n_classes, augment=False)
    test_set = CtIchDataset(img_dir=test_image_path, mask_dir=test_mask_path, N=-1, empty_mask_discard_probability=0, num_threads=num_threads, resize=resize, n_classes=n_classes, augment=False)
    supervised_set = CtIchDataset(img_dir=train_image_path, mask_dir=train_mask_path, N=-1, num_threads=num_threads, resize=resize, n_classes=n_classes, augment=base_augment_train, override_transforms=override_transforms)

    print(f'Unsupervised: \n\t- train: {len(unsupervised_set)}\n\tSupervised:\n\t- train: {len(supervised_set)}\n\t- validation: {len(validation_set)}\n\t- test: {len(test_set)}\n')

    validation_ids = np.unique(np.array([ID.split('$')[0] for ID in validation_set.ids]))
    test_ids = np.unique(np.array([ID.split('$')[0] for ID in test_set.ids]))
    train_ids = np.unique(np.array([ID.split('$')[0] for ID in supervised_set.ids]))
    
    print(f'Validation IDs: \n\t{validation_ids}')
    print(f'Test IDs: \n\t{test_ids}')
    print(f'Train IDs: \n\t{train_ids}')

    u_infinite_sampler = InfiniteSampler(list(range(len(unsupervised_set))))
    s_infinite_sampler = InfiniteSampler(list(range(len(supervised_set))))

    unsupervised_set = AugmentationWrapperDataset(unsupervised_set, transform_u)
    supervised_set = AugmentationWrapperDataset(supervised_set, transform_s)
    validation_set = AugmentationWrapperDataset(validation_set, transform_t)
    test_set = AugmentationWrapperDataset(test_set, transform_t)

    u_loader = DataLoader(dataset=unsupervised_set, batch_size=batch_size_u, pin_memory=pin_memory, sampler=u_infinite_sampler, num_workers=num_workers)
    s_loader = DataLoader(dataset=supervised_set, batch_size=batch_size_s, pin_memory=pin_memory, shuffle=True, num_workers=num_workers)
    s_loader_inf = DataLoader(dataset=supervised_set, batch_size=batch_size_s, pin_memory=pin_memory, sampler=s_infinite_sampler, num_workers=num_workers)
    v_loader = DataLoader(dataset=validation_set, batch_size=batch_size_t, pin_memory=pin_memory, shuffle=True, num_workers=num_workers)
    t_loader = DataLoader(dataset=test_set, batch_size=batch_size_t, pin_memory=pin_memory, shuffle=True, num_workers=num_workers)

    return u_loader, s_loader, s_loader_inf, v_loader, t_loader
