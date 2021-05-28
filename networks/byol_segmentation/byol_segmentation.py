import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
import numpy as np
import random
import time
import PIL

class Transformation(nn.Module):

    def __init__(self, rot=30.0, translation=(4,4), scale=(0.8,1.2), flip_probability=0.5, shear_angle=10):
        super(Transformation, self).__init__()
        self.rot = rot
        self.translation = translation
        self.scale = scale
        self.flip_probability = flip_probability
        self.nof_transform_features = 6
        self.shear_angle = shear_angle

    def inverse_transformation(self, images, transforms):

        transformed = torch.zeros(images.shape)

        for i, img in enumerate(images):
            img = TF.to_pil_image(img.cpu())
            img_transform = transforms[i]

            flip =                  img_transform['flip']
            random_rot =            -img_transform['rotation']
            random_scale =          1 / img_transform['scale'] # x*A = a, A = a/x = a*(1/x)
            random_translation =    (-img_transform['translation_x'], -img_transform['translation_y'])
            random_shear =          -img_transform['shear']

            transformed[i, :, :, :] = self.transform(img, flip, random_rot, random_scale, random_shear, random_translation, PIL.Image.BICUBIC)

        print("max:", transformed.max())
        return transformed


    def transform(self, img, flip, random_rot, random_scale, random_shear, random_translation, resample=PIL.Image.NEAREST, is_mask=False):
        # Random flip:
        if flip:
            img = TF.hflip(img)
        
        # Apply random affine
        if is_mask:
            img = TF.affine(img=img, angle=random_rot, scale=random_scale, shear=random_shear, translate=random_translation, resample=resample, fill=-1)
        else:
            img = TF.affine(img=img, angle=random_rot, scale=random_scale, shear=random_shear, translate=random_translation, resample=resample)

        if is_mask and not torch.is_tensor(img):
            img = torch.tensor(np.asarray(img, dtype=np.uint8))
        elif not torch.is_tensor(img):
            img = TF.to_tensor(img)

        return img

    def helper_forward(self, x, prior_transforms, is_mask):
        start_time = time.time()
        device = x.device
        transformed = torch.zeros(x.shape).to(device)
        applied_transforms = []
        applied_transforms_tensor = torch.zeros((x.shape[0], self.nof_transform_features)).to(device)

        for i, img in enumerate(x):
            image_transforms = {}

            if prior_transforms is None:
                img = TF.to_pil_image(img.cpu())
                flip = random.random() > self.flip_probability
                random_rot = random.uniform(-self.rot, self.rot)
                random_scale = random.uniform(self.scale[0], self.scale[1])
                random_translation = random.uniform(-self.translation[0], self.translation[0]), random.uniform(-self.translation[1], self.translation[1])
                random_shear = random.uniform(-self.shear_angle, self.shear_angle)

                image_transforms['flip'] = 1 if flip else 0
                image_transforms['rotation'] =      random_rot # angle
                image_transforms['scale'] =         random_scale # multiplier
                image_transforms['shear'] =         random_shear # angle
                image_transforms['translation_x'] = random_translation[0] # x
                image_transforms['translation_y'] = random_translation[1] # y
                applied_transforms.append(image_transforms)

                applied_transforms_tensor[i, 0] = image_transforms['flip']
                applied_transforms_tensor[i, 1] = random_rot
                applied_transforms_tensor[i, 2] = random_scale
                applied_transforms_tensor[i, 3] = random_shear
                applied_transforms_tensor[i, 4] = random_translation[0]
                applied_transforms_tensor[i, 5] = random_translation[1]
            else:
                img = img.type(torch.uint8)
                img = TF.to_pil_image(img.cpu())
                prior = prior_transforms[i]
                flip =                  prior['flip']
                random_rot =            prior['rotation']
                random_scale =          prior['scale']
                random_translation =    (prior['translation_x'], prior['translation_y'])
                random_shear =          prior['shear']

            resample = PIL.Image.BILINEAR
            if is_mask:
                resample=PIL.Image.NEAREST
            
            img = self.transform(img=img, flip = flip, random_rot=random_rot, random_scale=random_scale, random_shear=random_shear, random_translation=random_translation, is_mask=is_mask, resample=resample)

            if prior_transforms is not None and not is_mask:
                img = img * 255

            transformed[i] = img
        
        applied_transforms = applied_transforms if prior_transforms is None else prior_transforms
        return transformed, applied_transforms, applied_transforms_tensor

    def forward(self, x, prior_transforms=None, is_mask=False, calculate_gradients=False):

        if not calculate_gradients:
            with torch.no_grad():
                return self.helper_forward(
                    x=x,
                    prior_transforms=prior_transforms,
                    is_mask=is_mask
                )
        else:
            return self.helper_forward(
                    x=x,
                    prior_transforms=prior_transforms,
                    is_mask=is_mask
                )

class SegmentationProjector(nn.Module):

    def __init__(self, n_input_channels, n_output_channels):
        super(SegmentationProjector, self).__init__()
        self.conv = nn.Conv2d(n_input_channels, n_output_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))
