import torch
import os
import torchvision

# TODO: Implement gaussian init recommended by U-net authors
def gaussian_init(model, mean=1, variance=1):
    pass

# include .pth at the end of model name. Subdirectories are not created.
def save_model(model, model_name, task_name):
    base_path = './output/tasks'
    task_path = f'{base_path}/{task_name}'

    try:
        os.makedirs(name=task_path, exist_ok=True)
    except FileExistsError:
        print("Directory creation failed. Aborting saving of model.")
        return

    path = f'{task_path}/{model_name}'
    torch.save(model.state_dict(), path)
    print(f'Saved model to {path}')


def save_grid(label_tensor, img_tensor, mask_tensor, file_name = f'progress_grid.jpg', tensor=False):
    
    grid_tensor = torch.zeros((3, 3, img_tensor.shape[1], img_tensor.shape[2]))

    grid_tensor[0,:,:,:] = img_tensor
    grid_tensor[1] = label_tensor
    grid_tensor[2] = mask_tensor

    grid = torchvision.utils.make_grid(grid_tensor)
    torchvision.utils.save_image(grid, file_name)


def save_batch_grid(images, file_name = f'progress_grid.jpg'):
    grid = torchvision.utils.make_grid(images)

    torchvision.utils.save_image(grid, file_name)


def save_all_grid(images, labels, ground_truth_masks, file_name = f'progress_grid.jpg'):

    b_size = images.shape[0]
    n_channels = images.shape[1]
    w = images.shape[2]
    h = images.shape[3]
    tensors = torch.stack((images, ground_truth_masks, labels), dim=1).reshape(b_size*3, n_channels, w, h)

    grid = torchvision.utils.make_grid(tensors, nrow=3)
    
    torchvision.utils.save_image(grid, file_name)

def save_overlay_grid(images, labels, ground_truth_masks, file_name = f'progress_grid.jpg'):

    nrow = 3
    b_size = images.shape[0]
    n_channels = images.shape[1]
    w = images.shape[2]
    h = images.shape[3]
    images_and_predicted = images * (labels == 0).all(1).unsqueeze(1) + labels
    images_and_gt = images * (ground_truth_masks == 0).all(1).unsqueeze(1) + ground_truth_masks
    tensors = torch.stack((images, images_and_predicted, images_and_gt), dim=1).reshape(b_size*nrow, n_channels, w, h)
    
    grid = torchvision.utils.make_grid(tensors, nrow=nrow)
    
    torchvision.utils.save_image(grid, file_name)


def labels_to_rgb_batched(label_map, color_map):
    """
    label_map: (width, height) values in [0, N_classes - 1]
    color_map: (N_classes, n_channels) n_channels=3 for RGB
    """

    n_channels = color_map.shape[1]
    color_map = color_map.T

    if color_map.max() > 1:
        color_map = color_map / 255.0

    indexed = torch.index_select(input=color_map, dim=1, index=label_map.flatten()).T
    reshaped_colored_image = indexed.reshape(label_map.shape[0], label_map.shape[1], label_map.shape[2], n_channels)

    return reshaped_colored_image

def labels_to_rgb(label_map, color_map):
    """
    label_map: (width, height) values in [0, N_classes - 1]
    color_map: (N_classes, n_channels) n_channels=3 for RGB
    """

    n_channels = color_map.shape[1]
    color_map = color_map.T

    if color_map.max() > 1:
        color_map = color_map / 255.0

    indexed = torch.index_select(input=color_map, dim=1, index=label_map.flatten())
    reshaped_colored_image = indexed.reshape(n_channels, label_map.shape[0],label_map.shape[1])

    return reshaped_colored_image
