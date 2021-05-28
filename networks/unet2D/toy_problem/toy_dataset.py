import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import random

# Dataset imports:
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF

class ShapeGenerator():

    """
    ShapeGenerator generates random images. Currently supports only one image per batch.
    Currently supported shapes: Square, Circle(ish), Cross sign
    """
    def __init__(self):

        circle = np.array([
            [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
            [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
            [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
            [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
            [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
            [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
            [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
            [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
            [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
            [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
            [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
            [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0]
        ])

        self.circle = torch.tensor(circle.repeat(3, axis=0).repeat(3,axis=1))

        cross = np.array([
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0]
        ])
        self.cross = torch.tensor(cross.repeat(3, axis=0).repeat(3,axis=1))

        self.circle_w = self.circle.shape[0]
        self.circle_h = self.circle.shape[1]

        self.cross_w = self.cross.shape[0]
        self.cross_h = self.cross.shape[1]

    """
    Generates an image of shape (N, 3, w, h). Currently only, N=1 is supported.

    return      image, labels
    """
    def generate(self, N, w=512, h=512):
        image = torch.ones((N, 3, w, h))
        segmentation_mask = torch.zeros((N, 4, w, h))

        # randomly select shapes for image
        bgs = np.ones(N)
        rects = np.random.choice([0,1], N)
        rects_idx = np.where(rects == 0)
        circles = np.random.choice([0,1], N)
        circles_idx = np.where(circles == 0)
        crosses = np.random.choice([0,1], N)
        crosses_idx = np.where(crosses == 0)

        # rectangle vertices
        points = np.array([np.random.choice(range(0,w // 4)), np.random.choice(range(0,w // 2))])

        # determine which coordinates should be outside of rectangle
        fill_x = np.concatenate([np.arange(0, points.min()), np.arange(points.max(), w)])
        fill_y = np.concatenate([np.arange(0, points.min()), np.arange(points.max(), h)])

        # fill coordinates outside of rectangle as 0
        image[:,:,fill_x, :] = 0
        image[:,:,:, fill_y] = 0
        image[rects_idx,:,:,:] = 0

        # determine and apply color to squares
        aux_tensor = torch.ones((3,3))
        rgb = aux_tensor.new_full((N,w,h,3), 1)
        rgb[:,:,:,:] = torch.tensor([0.1,0.5,0.6])
        rgb = rgb.permute(0,3,1,2)
        #segmentation_mask = image
        segmentation_mask[:,1,:,:] = image[:,0,:,:]
        image = image * rgb


        # Create a temp image for circles
        temp_image = torch.zeros((N, 3, w, h))
        r_i = random.randint(0,w-self.circle_w)
        r_j = random.randint(0,h-self.circle_h)

        ## randomize color for circle
        r = torch.rand(3).unsqueeze(0)
        helper = torch.cat(w*[r]).unsqueeze(0)
        circle_color = torch.cat(h*[helper]).unsqueeze(0).permute(0,3,1,2)

        ## fill circle to temporary image
        temp_image[:,:,r_i:r_i+self.circle_w, r_j:r_j+self.circle_h] = self.circle
        temp_image[circles_idx,:,:,:] = 0

        ## apply color for circle
        temp_image_binary = temp_image
        temp_image = temp_image * circle_color

        ## mask the circle on top of cross
        mask = temp_image > 0
        one_mask = torch.ones((N,3,w,h))
        #b = image

        segmentation_mask[:,2,:,:] = temp_image_binary[:,0,:,:]
        # combine rectange and circle
        image = (one_mask - mask*1)*image + mask*temp_image
        #segmentation_mask = (one_mask - mask*1)*segmentation_mask + mask*temp_image_binary*2

        # randomize color for cross
        r = torch.rand(3).unsqueeze(0)
        helper = torch.cat(w*[r]).unsqueeze(0)
        cross_color = torch.cat(h*[helper]).unsqueeze(0).permute(0,3,1,2)

        # create temp image for cross
        temp_image = torch.zeros((N, 3, w, h))
        r_i = random.randint(0,w-self.cross_w)
        r_j = random.randint(0,h-self.cross_h)

        # fill cross to temp image
        temp_image[:,:,r_i:r_i+self.cross_w, r_j:r_j+self.cross_h] = self.cross 
        temp_image[crosses_idx,:,:,:] = 0

        # apply color for cross
        temp_image_binary = temp_image
        temp_image = temp_image * cross_color

        # mask the cross on top of existing image
        mask = temp_image > 0
        one_mask = torch.ones((N,3,w,h))
        b = image
        
        segmentation_mask[:,3,:,:] = temp_image_binary[:,0,:,:]
        # finally, combine cross with rest of the image
        image = (one_mask - mask*1)*image + mask*temp_image
        #segmentation_mask = (one_mask - mask*1)*segmentation_mask + mask*temp_image_binary*3

        # background class segmentation:
        background_plane = torch.ones(1, w, h)
        empty = torch.zeros(1, w, h)
        stacked_image = image.sum(dim=1)
        background_plane = torch.where(stacked_image > 0, empty, background_plane)
        segmentation_mask[:,0,:,:] = background_plane

        return image, torch.tensor(np.stack([bgs, rects, circles, crosses]).T), segmentation_mask

class ToyDataSet(Dataset):

    """
    A simple toy dataset used for weak and fully supervised segmentation.
    - images        Image tensor (N, 3, image_w, image_h)
    - labels        Labels to tell which objects are found in the image (N, 3, image_w, image_h)

    Roadmap:
        [X] basic functionality
        [X] apply flip
        [ ] apply rotation, currently having issues with random rotation 
    """
    def __init__(self, images, labels, masks):
        self.images = images
        self.labels = labels
        self.masks = masks

    def transform(self, img, mask, max_rot=30, flip_p=0.5):
        img = TF.to_pil_image(img)
        mask = TF.to_pil_image(mask)

        # Random flip:
        if random.random() > flip_p:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # Apply rotation:
        #random_rot = random.uniform(-max_rot, max_rot)
        #filler = 0.0 if img.mode.startswith("F") else 0
        #img = TF.rotate(img, random_rot, fill=(0,))#[filler]*len(img.getbands())
        #mask = TF.rotate(mask, random_rot, fill=(0,))

        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)

        return img, mask

    def __getitem__(self, idx):

        #transform = transforms.Compose([
        #    transforms.ToPILImage(),
        #    transforms.RandomVerticalFlip(),
        #    transforms.RandomRotation(30),
        #    transforms.ToTensor()
        #])

        image = self.images[idx]
        mask = self.masks[idx]
        image, mask = self.transform(image.cpu(), mask.cpu())
        return (image, self.labels[idx], mask)

    def __len__(self):
        return self.images.shape[0]

def generate_dataset(N = 100, n_classes = 4, w = 128, h = 128):
    generator = ShapeGenerator()

    images = torch.zeros((N,3,w,h))
    labels = torch.zeros((N,n_classes))
    masks = torch.zeros((N,n_classes,w,h))

    for i in range(N):
        data = generator.generate(1, w, h)
        images[i, :, :, :] = data[0]
        labels[i,:] = data[1]
        masks[i, :, :, :] = data[2]

    return ToyDataSet(images, labels, masks)
