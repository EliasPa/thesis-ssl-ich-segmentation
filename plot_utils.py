import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from scipy import signal

def plot_masked_nifti(image, mask):
    plot_masked(torch.tensor(image.get_fdata()), torch.tensor(mask.get_fdata()))

def plot_masked(image_data, mask_data):
    w = image_data.shape[0]
    h = image_data.shape[1]

    mask_data = mask_data.squeeze().cpu().numpy()
    image_data = image_data.squeeze().cpu().numpy()
    mask_img = Image.fromarray(mask_data)
    src_img = Image.fromarray(image_data)

    mask = np.array(mask_img)
    src = np.array(src_img)
    if (mask.sum() > 0):

        print(mask.shape)
        edges = get_mask_edges_conv(mask)
        non_zero = np.nonzero(edges)

        full_mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
        full_src = np.repeat(src[:,:,np.newaxis], 3, axis=2)
        full_edges = np.repeat(edges[:,:,np.newaxis], 3, axis=2)

        overlay_mask = full_mask*np.full((w,h,3), [0.5,0,0])
        overlay_edges = full_edges*np.full((w,h,3), [0,0,1])
        overlay_src = full_src+(overlay_mask*10)
        overlay_src[non_zero[0],non_zero[1], :] = (overlay_edges[non_zero[0],non_zero[1], :]*10)

        plt.imshow(overlay_src, origin='lower')
        plt.title('overlay_src')
        plt.show()


def plot_grid(image_data, mask_data):
    mask_data = mask_data.squeeze().cpu().numpy()
    image_data = image_data.squeeze().cpu().numpy()
    mask_img = Image.fromarray(mask_data)
    src_img = Image.fromarray(image_data)
    
    mask = np.array(mask_img)
    src = np.array(src_img)
    if (mask.sum() > 0):

        print(mask.shape)
        edges = get_mask_edges_conv(mask)
        non_zero = np.nonzero(edges)

        width = mask.shape[0]

        full_mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
        full_src = np.repeat(src[:,:,np.newaxis], 3, axis=2)
        full_edges = np.repeat(edges[:,:,np.newaxis], 3, axis=2)

        overlay_mask = full_mask*np.full((512,512,3), [0.5,0,0])
        overlay_edges = full_edges*np.full((512,512,3), [0,0,1])
        non_zero_idx = np.nonzero(overlay_edges)
        overlay_src = full_src+(overlay_mask*10)
        overlay_src[non_zero[0],non_zero[1], :] = (overlay_edges[non_zero[0],non_zero[1], :]*10)

        grid = np.zeros((width*2, width, 3))
        grid[0:overlay_src.shape[0],:,:] = overlay_src
        grid[width:width*2,:,:] = full_mask

        plt.imshow(grid, origin='bottom')
        plt.title('grid')
        plt.show()

        plt.imshow(overlay_src, origin='bottom')
        plt.title('overlay_src')
        plt.show()

def apply_filter(edge_filter, data, center_x, center_y):
    f_w = edge_filter.shape[0]
    f_h = edge_filter.shape[1]
    half_w = f_w // 2
    half_h = f_h // 2

    result = np.zeros(edge_filter.shape)

    for i in range(-half_w, half_w):
        for j in range(- half_h, + half_h):
            data_i = center_x + i
            data_j = center_y + j
            result[i,j] = data[data_i,data_j] * edge_filter[i,j]

    return result.sum()


def apply_fill_filter(size, data, center_x, center_y):
    f_w = size[0]
    f_h = size[1]
    half_w = f_w // 2
    half_h = f_h // 2

    result = np.zeros(size)

    for i in range(-half_w, half_w):
        for j in range(- half_h, + half_h):
            data_i = center_x + i
            data_j = center_y + j
            data_el = data[data_i,data_j]
            if data_el > 0:
                result[i,j] = 1

    return result.sum() > 0

def get_mask_edges_conv(mask):

    sobel_horizontal = [
        [1,2,1],
        [0,0,0],
        [-1,-2,-1]
    ]

    sobel_vertical = [
        [1,0,-1],
        [2,0,-2],
        [1,0,-1]
    ]

    res = signal.convolve2d(mask, sobel_horizontal, mode="same")
    res2 = signal.convolve2d(mask, sobel_vertical, mode="same")

    full = (res**2 + res2**2)**(1/2)
    full = full.clip(0,1)
    
    return full

def get_mask_edges(mask):
    filter_w = 3
    filter_h = 3

    edges = np.zeros(mask.shape)

    sobel_horizontal = [
        [1,2,1],
        [0,0,0],
        [-1,-2,-1]
    ]

    sobel_vertical = [
        [1,0,-1],
        [2,0,-2],
        [1,0,-1]
    ]

    sobel_horizontal = np.array(sobel_horizontal)
    sobel_vertical = np.array(sobel_vertical)

    for i in range(filter_w // 2, mask.shape[0] - filter_w // 2):
        for j in range(filter_h // 2, mask.shape[1] - filter_h // 2):
            sobel = (apply_filter(sobel_vertical, mask, i,j)**2 + apply_filter(sobel_horizontal, mask, i,j)**2)**(1/2)
            edges[i,j] = sobel

    filled_edges = np.zeros(edges.shape)
    for i in range(filter_w // 2, mask.shape[0] - filter_w // 2):
        for j in range(filter_h // 2, mask.shape[1] - filter_h // 2):
            can_fill = apply_fill_filter((3,3), edges, i, j)
            if can_fill:
                filled_edges[i,j] = 1

    return filled_edges

def labels_to_rgb(label_map, color_map):
    """
    label_map: (width, height) values in [0, N_classes - 1]
    color_map: (N_classes, n_channels) n_channels=3 for RGB
    """

    n_channels = color_map.shape[1]
    color_map = color_map.T

    print(color_map)

    if color_map.max() > 1:
        color_map = color_map / 255.0

    indexed = torch.index_select(input=color_map, dim=1, index=label_map.flatten())
    reshaped_colored_image = indexed.reshape(n_channels, label_map.shape[0],label_map.shape[1])

    return reshaped_colored_image
