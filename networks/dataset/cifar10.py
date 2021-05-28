import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .infinite_sampler import InfiniteSampler, SemiSupervisedDataset, SemiSupervisedSampler, TwoViewsDataset, AugmentationWrapperDataset
import torch
from .utils import random_split_equal_classes

def create_all_dataloaders(batch_size, sizes, pin_memory, train=True, path='./data/cifar10'):
    assert(len(sizes) == 3)
    data = torchvision.datasets.CIFAR10(path, train=train, download=True, transform=transforms.ToTensor())
    
    unsupervised_size = int(sizes[0] * len(data))
    supervised_size = int(sizes[1] * len(data))
    validation_size = int(sizes[2] * len(data))

    rest, supervised_set = random_split_equal_classes(data, [len(data) - supervised_size, supervised_size], 10)
    validation_set, unsupervised_set = torch.utils.data.random_split(rest, [validation_size, len(rest) - validation_size])

    print(f'Unsupervised: {len(unsupervised_set)}, Supervised: {len(supervised_set)}, Test: {len(validation_set)}')

    infinite_sampler = InfiniteSampler(list(range(len(supervised_set))))
    u_infinite_sampler = InfiniteSampler(list(range(len(unsupervised_set))))

    u_loader = DataLoader(dataset=unsupervised_set, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
    u_loader_inf = DataLoader(dataset=unsupervised_set, batch_size=batch_size, pin_memory=pin_memory, sampler=u_infinite_sampler)
    s_loader = DataLoader(dataset=supervised_set, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
    s_loader_inf = DataLoader(dataset=supervised_set, batch_size=batch_size, pin_memory=pin_memory, sampler=infinite_sampler)
    v_loader = DataLoader(dataset=validation_set, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)

    return u_loader, u_loader_inf, s_loader, s_loader_inf, v_loader

def create_dataloader(batch_size, pin_memory, train=True, size=1, path='./data/cifar10', transform=transforms.ToTensor()):
    data = torchvision.datasets.CIFAR10(path, train=train, download=True, transform=transform)
    size = int(size * len(data))
    data, _ = torch.utils.data.random_split(data, [size, len(data) - size])

    loader = DataLoader(dataset=data, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
    
    return loader

def create_separate_two_way_loaders(batch_size, u_batch_size, sizes, pin_memory, transform_w, transform_s, transform_v, num_workers=2, path='./data/cifar10', train=True):
    assert(len(sizes) == 3)
    mnist = torchvision.datasets.CIFAR10(path, train=train, download=True, transform=transforms.ToTensor())
    
    unsupervised_size = int(sizes[0] * len(mnist))
    supervised_size = int(sizes[1] * len(mnist))
    
    unsupervised_set, rest = torch.utils.data.random_split(mnist, [unsupervised_size, len(mnist) - unsupervised_size])
    supervised_set, test_set = torch.utils.data.random_split(rest, [supervised_size, len(rest) - supervised_size])

    unsupervised_set = TwoViewsDataset(unsupervised_set, transform_w, transform_s)
    supervised_set = AugmentationWrapperDataset(supervised_set, transform_w)
    test_set = AugmentationWrapperDataset(test_set, transform_w)

    print(f'Unsupervised: {len(unsupervised_set)}, Supervised: {len(supervised_set)}, Test: {len(test_set)}')

    u_infinite_sampler = InfiniteSampler(list(range(len(unsupervised_set))))
    s_infinite_sampler = InfiniteSampler(list(range(len(supervised_set))))

    u_loader = DataLoader(dataset=unsupervised_set, batch_size=u_batch_size, pin_memory=pin_memory, sampler=u_infinite_sampler)
    s_loader = DataLoader(dataset=supervised_set, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
    s_loader_inf = DataLoader(dataset=supervised_set, batch_size=batch_size, pin_memory=pin_memory, sampler=s_infinite_sampler)
    t_loader = DataLoader(dataset=test_set, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)

    return u_loader, s_loader, s_loader_inf, t_loader

def create_two_way_loaders(batch_size, batch_size_s, sizes, transform_w, transform_s, transform_v, pin_memory=True, train=True, path='./data/cifar10', num_workers=2):#, regular_mnist=False):
    assert(len(sizes) == 3)
    data = torchvision.datasets.CIFAR10(path, train=train, download=True, transform=transforms.ToTensor())

    unsupervised_size = int(sizes[0] * len(data))
    supervised_size = int(sizes[1] * len(data))
    validation_size = int(sizes[2] * len(data))

    rest, supervised_set = random_split_equal_classes(data, [len(data) - supervised_size, supervised_size], 10)
    validation_set, unsupervised_set = torch.utils.data.random_split(rest, [validation_size, len(rest) - validation_size])

    supervised_set = AugmentationWrapperDataset(supervised_set, transform_w)
    validation_set = AugmentationWrapperDataset(validation_set, transform_v)

    print(f'Semi-supervised: {len(unsupervised_set)}, Supervised: {len(supervised_set)}, Validation: {len(validation_set)}')

    two_views_dataset_unsupervised = TwoViewsDataset(unsupervised_set, transform_w=transform_w, transform_s=transform_s)

    combined = SemiSupervisedDataset(two_views_dataset_unsupervised, supervised_set)
    sampler = SemiSupervisedSampler(s_indices=list(range(len(supervised_set))), u_indices=list(range(len(two_views_dataset_unsupervised))))
    ss_train_loader = DataLoader(combined, sampler=sampler, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)

    s_train_loader = DataLoader(dataset=supervised_set, batch_size=batch_size_s, pin_memory=pin_memory, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size, pin_memory=pin_memory, shuffle=True, num_workers=num_workers)

    return ss_train_loader, s_train_loader, validation_loader
