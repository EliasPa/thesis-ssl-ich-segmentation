import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

class PILToTensor(object):

    def __init__(self):
        pass

    def __call__(self, x):
        x = torch.tensor(np.asarray(x, dtype=np.uint8)).long()
        return x

class ModifyLabels(object):

    def __init__(self):
        self.map = { # from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            7: 1,
            8: 2,
            9: 0,
            10: 0,
            11: 3,
            12: 4,
            13: 5,
            14: 0,
            15: 0,
            16: 0,
            17: 6,
            18: 0,
            19: 7,
            20: 8,
            21: 9,
            22: 10,
            23: 11,
            24: 12,
            25: 13,
            26: 14,
            27: 15,
            28: 16,
            29: 0,
            30: 0,
            31: 17,
            32: 18,
            33: 19,
            -1: 0,
        }

    def __call__(self, x):
        modified_target = torch.zeros_like(x)

        for key in self.map:
            new_label = self.map[key]
            modified_target[x == key] = new_label

        return modified_target

def create_all_dataloaders(batch_size_s, batch_size_v, batch_size_t, a_image_transforms=transforms.Compose([]), a_mask_transforms=transforms.Compose([]), path='../../data/cityscapes/', mode='fine', size=(256, 512)):

    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size, interpolation=transforms.functional.InterpolationMode.BILINEAR)
    ])

    mask_transforms = transforms.Compose([
        transforms.Resize(size,
        interpolation=transforms.functional.InterpolationMode.NEAREST),
        PILToTensor(),
        ModifyLabels()
    ])

    # Additional transforms requested by user...
    image_transforms.transforms.extend(a_image_transforms.transforms)
    mask_transforms.transforms.extend(a_mask_transforms.transforms)

    supervised_data = torchvision.datasets.Cityscapes(path, split='train', mode=mode, target_type='semantic', transform=image_transforms, target_transform=mask_transforms)
    validation_data = torchvision.datasets.Cityscapes(path, split='val', mode=mode, target_type='semantic', transform=image_transforms, target_transform=mask_transforms)
    test_data = torchvision.datasets.Cityscapes(path, split='test', mode=mode, target_type='semantic', transform=image_transforms, target_transform=mask_transforms)

    supervised_loader = DataLoader(dataset=supervised_data, batch_size=batch_size_s, pin_memory=True, shuffle=True)
    validation_loader = DataLoader(dataset=validation_data, batch_size=batch_size_v, pin_memory=True, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size_t, pin_memory=True, shuffle=True)

    return supervised_loader, validation_loader, test_loader
