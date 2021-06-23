import networks.dataset.mnist as mnist_lesion
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import networks.unet2D.utils as utils
data_set_size = 1

def create_loader(train, data_set_size):
    torch.manual_seed(123)
    np.random.seed(123)
    mnist = torchvision.datasets.EMNIST('./data/mnist', split='mnist', train=train, download=True, transform=transforms.ToTensor())
    size = int(data_set_size * len(mnist))
    mnist, _ = torch.utils.data.random_split(mnist, [size, len(mnist) - size])

    loader = DataLoader(dataset=mnist, batch_size=20, pin_memory=True, shuffle=True)
    print(len(mnist))

    return loader


def save_examples(loader, train):


    color_map = torch.tensor([
                [0,0,0], # bg: black
                [0,1.0,0], # green, IVH
                [1.0,0,0], # red, IPH
                [0,0,1.0], # blue, SAH
                [0.5,0,0.5], # EDH, magenta?
        ])
    for i, batch in enumerate(tqdm(loader)):
        #image = (image.squeeze().permute(1,2,0) * 255).numpy().astype('uint8')
        #img = Image.fromarray(image, "RGB")
        #img.save(f'./examples/example_mnist_lesion_{i}.png')
        #batch = [(digit.squeeze(0), label)]
        old_digits, labels = batch
        labels = labels.squeeze(0)
        print(old_digits.shape, labels.shape)
        digits = torch.zeros((old_digits.shape[0],3,32,32))
        digits_with_lesion = torch.zeros((old_digits.shape[0],3,32,32))
        targets = torch.zeros((old_digits.shape[0],32,32)).long()

        for j, digit in enumerate(old_digits):
            #digit, _ = batch[0]
            print(j)
            print(labels)
            new_batch = [(digit.squeeze(0), labels[j])]
            digit_with_lesion, _, target = mnist_lesion.lesion_collate(new_batch)
            digit = digit.squeeze(0)
            digit = np.array(digit) * 255
            digit = np.pad(digit, pad_width=((2,2),(2,2)), mode='minimum') / 255.0
            digit = torch.tensor(digit).unsqueeze(0)
            print(digit.shape, digits.shape)
            digits[j] = digit
            digits_with_lesion[j] = digit_with_lesion
            print("target", target.shape)
            targets[j] = target.squeeze(0).squeeze(0)

    
        print("done")
        print(digits.shape)
        print(digits_with_lesion.shape)
        print(targets.dtype)
        target_rgb = utils.labels_to_rgb_batched(targets, color_map).permute(0,3,1,2)
        utils.save_all_grid(digits, target_rgb, digits_with_lesion, file_name='test.png')
        break

if __name__ == '__main__':
    train_loader = create_loader(True, data_set_size)
    save_examples(train_loader, True)

