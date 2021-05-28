import networks.dataset.mnist as mnist_lesion
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
data_set_size = 1

def create_loader(train, data_set_size):
    torch.manual_seed(123)
    np.random.seed(123)
    mnist = torchvision.datasets.EMNIST('./data/mnist', split='mnist', train=train, download=True, transform=transforms.ToTensor())
    size = int(data_set_size * len(mnist))
    mnist, _ = torch.utils.data.random_split(mnist, [size, len(mnist) - size])

    loader = DataLoader(dataset=mnist, collate_fn=mnist_lesion.lesion_collate, batch_size=1, pin_memory=True, shuffle=True)
    print(len(mnist))

    return loader


def save_images(loader, train):
    folder = 'train' if train else 'test'

    for i, (image, _, mask) in enumerate(tqdm(loader)):
        image = (image.squeeze().permute(1,2,0) * 255).numpy().astype('uint8')
        img = Image.fromarray(image, "RGB")
        img.save(f'./data/mnist/mnist_lesion/{folder}/image/{i}.png')

        msk = Image.fromarray((mask.squeeze()).numpy().astype('uint8'))
        msk.save(f'./data/mnist/mnist_lesion/{folder}/mask/{i}.png')

if __name__ == '__main__':
    train_loader = create_loader(True, data_set_size)
    save_images(train_loader, True)

    test_loader = create_loader(False, data_set_size)
    save_images(test_loader, False)
