"""
Author: Elias Panula
"""
import torchvision
import torchvision.transforms as transforms
from skimage import filters
from torch.utils.data import DataLoader
from .utils import split_set
from skimage import morphology
import random
import numpy as np
import torch
from .infinite_sampler import InfiniteSampler, SemiSupervisedDataset, SemiSupervisedSampler, TwoViewsDataset, AugmentationWrapperDataset, MNISTLesionDatasetRAMHeavy, MNISTLesionDatasetSlow

def lesion_collate(batch):
    
    c, w, h = 1, 32, 32
    n_classes = 5

    modified_batch = torch.zeros(len(batch), c, w, h)
    target = torch.zeros(len(batch), c, w, h)
    weak_labels = torch.zeros(len(batch), n_classes)
    weak_labels[:,0] = 1
    for i, example in enumerate(batch):
        
        lesion_class = random.randint(0, n_classes - 1)

        digit, _ = example
        digit = digit.squeeze(0)
        digit = np.array(digit) * 255

        if digit.shape[0] < 32:
            digit = np.pad(digit, pad_width=((2,2),(2,2)), mode='minimum')

        if lesion_class < 2:
            sobel = 1 - filters.sobel_h(digit)
        elif lesion_class >= 2:
            sobel = 1 - filters.sobel_v(digit)

        sobel = filters.gaussian(sobel)
        
        if lesion_class < 2:
            sobel = - filters.sobel_h(digit)
        elif lesion_class >= 2:
            sobel = - filters.sobel_v(digit)

        sobel = np.roll(sobel, 2)
        sobel = filters.gaussian(sobel)
        min_delta = np.abs(sobel.min())
        sobel = (sobel + min_delta)
        sobel = sobel / sobel.max()
        sobel = sobel * 255

        # mask
        sobel = (sobel - digit).clip(0,255) / 255.0
        sobel[sobel > 0.7] = 1
        sobel[sobel <= 0.7] = 0
        nzeros = np.nonzero(sobel)
        random_zero_index = random.randint(0, nzeros[0].shape[0] - 1)
        seed = (nzeros[0][random_zero_index], nzeros[1][random_zero_index])
        sobel = morphology.flood_fill(sobel, seed, 255)
        sobel[sobel < 255] = 0.0
        sobel = morphology.binary_closing(sobel)
        sobel = filters.gaussian(sobel, 1)
        sobel = sobel * 255

        grayscale = get_class_grayscale(lesion_class)
        
        if lesion_class != 0:
            final_input_image = torch.tensor((digit+(sobel*grayscale)).clip(0,255))
            modified_batch[i, :, :, :] = final_input_image.squeeze()
        else:
            modified_batch[i, :, :, :] = torch.tensor(digit.clip(0,255).squeeze())

        if lesion_class != 0:
            mask = (final_input_image - digit)
            mask[sobel <= 30] = 0
            mask[sobel > 30] = lesion_class
            target[i, :, :, :] = mask.clone().detach().squeeze()
            weak_labels[i, lesion_class] = 1

    return modified_batch.repeat(1, 3, 1, 1) / 256 , weak_labels, target.long()

def no_lesion_collate(batch):
    
    c, w, h = 1, 32, 32
    n_classes = 5

    modified_batch = torch.zeros(len(batch), c, w, h)
    target = torch.zeros(len(batch), c, w, h)
    weak_labels = torch.zeros(len(batch), n_classes)
    weak_labels[:,0] = 1
    affine = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=180, scale=(0.8, 1.2), shear=5),
        transforms.ToTensor()
    ])
    for i, example in enumerate(batch):
        
        digit, _ = example
        digit = digit.squeeze(0)

        if digit.shape[0] < 32:
            digit = np.pad(digit, pad_width=((2,2),(2,2)), mode='minimum')
        
        digit = torch.tensor(digit.clip(0,1).squeeze())
        modified_batch[i, :, :, :] = affine(digit)

    modified_batch = modified_batch.repeat(1, 3, 1, 1)

    return modified_batch, weak_labels, target.long()

def get_class_grayscale(lesion_class):
    if lesion_class == 0 or lesion_class == 2:
        return random.uniform(0.7, 0.8)
    else:
        return random.uniform(0.9, 1.1)

def consistency_collate(batch):
    c, w, h = 1, 32, 32
    modified_batch = torch.zeros(len(batch), c, w, h)
    target = torch.zeros(len(batch), c, w, h)
    
    for i, example in enumerate(batch):
        
        digit, _ = example
        digit = digit.squeeze(0)
        digit = np.pad(digit, pad_width=((2,2),(2,2)), mode='minimum')
        
        digit = torch.tensor(digit.clip(0,1).squeeze())
        modified_batch[i, :, :, :] = digit

    modified_batch = modified_batch.repeat(1, 3, 1, 1)

    return modified_batch, torch.tensor([digit for _, digit in batch]), target.long()

def create_all_dataloaders(batch_size_s, batch_size_u, sizes, pin_memory, transform_u, transform_s, transform_t, image_path, mask_path, train=True, path='./data/mnist', regular_mnist=False):
    assert(len(sizes) == 3)
    mnist = MNISTLesionDatasetSlow(image_path, mask_path)
    unsupervised_size = int(sizes[0] * len(mnist))
    supervised_size = int(sizes[1] * len(mnist))
    
    unsupervised_set, rest = torch.utils.data.random_split(mnist, [unsupervised_size, len(mnist) - unsupervised_size])
    supervised_set, test_set = torch.utils.data.random_split(rest, [supervised_size, len(rest) - supervised_size])

    print(f'Unsupervised: {len(unsupervised_set)}, Supervised: {len(supervised_set)}, Test: {len(test_set)}')

    u_infinite_sampler = InfiniteSampler(list(range(len(unsupervised_set))))
    s_infinite_sampler = InfiniteSampler(list(range(len(supervised_set))))

    unsupervised_set = AugmentationWrapperDataset(unsupervised_set, transform_u)
    supervised_set = AugmentationWrapperDataset(supervised_set, transform_s)
    test_set = AugmentationWrapperDataset(test_set, transform_t)

    if regular_mnist:
        u_loader = DataLoader(dataset=unsupervised_set, collate_fn=consistency_collate, batch_size=batch_size_u, pin_memory=pin_memory, sampler=u_infinite_sampler)
        s_loader = DataLoader(dataset=supervised_set, collate_fn=consistency_collate, batch_size=batch_size_s, pin_memory=pin_memory, shuffle=True)
        s_loader_inf = DataLoader(dataset=supervised_set, collate_fn=consistency_collate, batch_size=batch_size_s, pin_memory=pin_memory, sampler=s_infinite_sampler)
        t_loader = DataLoader(dataset=test_set, collate_fn=consistency_collate, batch_size=batch_size_s, pin_memory=pin_memory, shuffle=True)
    else:
        u_loader = DataLoader(dataset=unsupervised_set, batch_size=batch_size_u, pin_memory=pin_memory, sampler=u_infinite_sampler)
        s_loader = DataLoader(dataset=supervised_set, batch_size=batch_size_s, pin_memory=pin_memory, shuffle=True)
        s_loader_inf = DataLoader(dataset=supervised_set, batch_size=batch_size_s, pin_memory=pin_memory, sampler=s_infinite_sampler)
        t_loader = DataLoader(dataset=test_set, batch_size=batch_size_s, pin_memory=pin_memory, shuffle=True)

    return u_loader, s_loader, s_loader_inf, t_loader

def create_two_way_loaders(batch_size, sizes, transform_w, transform_s, pin_memory=True, train=True, path='./data/cifar10'):#, regular_mnist=False):
    assert(len(sizes) == 3)
    data = torchvision.datasets.EMNIST(path, split='mnist', train=train, download=True, transform=transforms.ToTensor())
    
    unsupervised_size = int(sizes[0] * len(data))
    supervised_size = int(sizes[1] * len(data))
    
    unsupervised_set, rest = torch.utils.data.random_split(data, [unsupervised_size, len(data) - unsupervised_size])
    supervised_set, validation_set = torch.utils.data.random_split(rest, [supervised_size, len(rest) - supervised_size])

    print(f'Unsupervised: {len(unsupervised_set)}, Supervised: {len(supervised_set)}, Test: {len(validation_set)}')

    two_views_dataset_unsupervised = TwoViewsDataset(unsupervised_set, transform_w=transform_w, transform_s=transform_s)
    two_views_dataset_supervised = TwoViewsDataset(supervised_set, transform_w=transform_w, transform_s=transform_s)

    combined = SemiSupervisedDataset(two_views_dataset_unsupervised, two_views_dataset_supervised)
    sampler = SemiSupervisedSampler(s_indices=list(range(len(two_views_dataset_supervised))), u_indices=list(range(len(two_views_dataset_unsupervised))))
    ss_train_loader = DataLoader(combined, sampler=sampler, batch_size=batch_size, pin_memory=pin_memory)

    s_train_loader = DataLoader(dataset=supervised_set, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
    validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)

    return ss_train_loader, s_train_loader, validation_loader

def create_dataloader_from_path(batch_size, pin_memory, image_path, mask_path, size=1):
    mnist = MNISTLesionDatasetSlow(image_path, mask_path, n_classes=5)
    size = int(size * len(mnist))
    mnist, _ = torch.utils.data.random_split(mnist, [size, len(mnist) - size])
    loader = DataLoader(dataset=mnist, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
    return loader

def create_dataloaders_from_path(
        n_classes,
        batch_size_s,
        batch_size_u,
        batch_size_t,
        transform_s,
        transform_u,
        transform_t,
        pin_memory,
        base_path,
        N_s=9000,
        N_u=50000,
        N_v=1000
    ):

    test_base_path = base_path + '/test/'
    train_base_path = base_path + '/train/'

    train_img_path = train_base_path + '/image/'
    train_mask_path = train_base_path + '/mask/'
   
    test_img_path = test_base_path + '/image/'
    test_mask_path = test_base_path + '/mask/'

    # Data sets
    train_set = MNISTLesionDatasetSlow(train_img_path, train_mask_path, n_classes)
    supervised_set, unsupervised_set, validation_set = split_set(train_set, N_s, N_u, N_v)
    test_set = MNISTLesionDatasetSlow(test_img_path, test_mask_path, n_classes)

    # Samplers
    u_infinite_sampler = InfiniteSampler(list(range(len(unsupervised_set))))
    s_infinite_sampler = InfiniteSampler(list(range(len(supervised_set))))

    # Data set augmentations (pixel-level)
    unsupervised_set = AugmentationWrapperDataset(unsupervised_set, transform_u)
    supervised_set = AugmentationWrapperDataset(supervised_set, transform_s)
    validation_set = AugmentationWrapperDataset(validation_set, transform_t)
    test_set = AugmentationWrapperDataset(test_set, transform_t)

    # Data loaders
    supervised_loader = DataLoader(dataset=supervised_set, batch_size=batch_size_s, pin_memory=pin_memory, shuffle=True)
    inf_supervised_loader = DataLoader(dataset=supervised_set, batch_size=batch_size_s, pin_memory=pin_memory, sampler=s_infinite_sampler)
    unsupervised_loader = DataLoader(dataset=unsupervised_set, batch_size=batch_size_u, pin_memory=pin_memory, sampler=u_infinite_sampler)
    validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size_t, pin_memory=pin_memory, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size_t, pin_memory=pin_memory, shuffle=False)

    return unsupervised_loader, supervised_loader, inf_supervised_loader, validation_loader, test_loader

def create_dataloader(batch_size, pin_memory, train=True, size=1, path='./data/mnist', regular_mnist=False):
    mnist = torchvision.datasets.EMNIST(path, split='mnist', train=train, download=True, transform=transforms.ToTensor())
    size = int(size * len(mnist))
    mnist, _ = torch.utils.data.random_split(mnist, [size, len(mnist) - size])

    if regular_mnist:
        loader = DataLoader(dataset=mnist, collate_fn=consistency_collate, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
    else:    
        loader = DataLoader(dataset=mnist, collate_fn=lesion_collate, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
    return loader

def create_dataloader_no_lesion(batch_size, pin_memory, train=True, size=1, path='./data/mnist'):
    mnist = torchvision.datasets.EMNIST(path, split='mnist', train=train, download=True, transform=transforms.ToTensor())
    size = int(size * len(mnist))
    _, mnist = torch.utils.data.random_split(mnist, [size, len(mnist) - size])
    loader = DataLoader(dataset=mnist, collate_fn=no_lesion_collate, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
    return loader

def labels_to_rgb(label_map, color_map):
    """
    label_map: (width, height) values in [0, N_classes - 1]
    color_map: (N_classes, n_channels) n_channels=3 for RGB
    """

    n_channels = color_map.shape[1]
    color_map = color_map.T

    if color_map.max() > 1:
        color_map = color_map / 255.0

    print(color_map)
    indexed = torch.index_select(input=color_map, dim=1, index=label_map.flatten())
    reshaped_colored_image = indexed.reshape(n_channels, label_map.shape[0],label_map.shape[1])

    return reshaped_colored_image

cmap = torch.tensor(
    [
        [0,0,0],
        [0,1,0],
        [0,0,1.0],
        [1,0,0],
        [1,1,1]
    ]
)
