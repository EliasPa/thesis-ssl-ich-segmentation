import torch
import math
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL

def get_size(tensor):
    return tensor.nelement() * tensor.element_size()

def CLE(f, f_hat, labels, N, n_classes, tau, down_sample_factor=2, background_index=0, ignore_bg=False):

    """
    Calculates contrastive clustering loss from: https://arxiv.org/pdf/2012.06985.pdf

    Args:
        f (Tensor): feature maps, shape (BxCxWxH)
        f_hat (Tensor): feature maps for perturbed images, shape (BxCxWxH)
        labels (Tensor): either the ground-truth segmentation mask or from pseudo-labeling (BxWxH)
        N (int): how many pixels to sample from input f and f_hat (GPU memory optimization)

    TODO:
        [ ] Finish implementation
    """
    device = f.device
    w = int(f.shape[3] // down_sample_factor)
    h = int(f.shape[2] // down_sample_factor)
    f = F.interpolate(f, size=(w,h), mode='bilinear')
    f_hat = F.interpolate(f_hat, size=(w,h), mode='bilinear')

    labels_ = torch.zeros((labels.shape[0], w,h)).to(device)
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    for i, l in enumerate(labels):
        as_img = to_pil(l.int())
        labels_[i] = to_tensor(as_img.resize((h,w), PIL.Image.NEAREST))

    labels = labels_.long()

    total_loss = torch.zeros((1,), requires_grad=True).to(f_hat.device)
    epsilon = 0.01

    N_sampled = 0

    for i, feature_map in enumerate(f):
        feature_map_hat = f_hat[i].permute(1,2,0).view(-1, n_classes)
        feature_map = feature_map.permute(1,2,0).view(-1, n_classes)

        label = labels[i]
        label = label.view(-1)
        not_in_bg_class = ~torch.eq(label, background_index)

        with torch.no_grad():
            if ignore_bg:
                N_to_select_from = label[not_in_bg_class].shape[0]
            else:
                N_to_select_from = N

        N = min(N_to_select_from, N)
        N = int(math.sqrt(N))**2
        N_sampled += N

        if N != 0:
          
            if ignore_bg:
                
                selected_feature_indices = not_in_bg_class.nonzero()
                selected_feature_indices = selected_feature_indices.squeeze(1)
                selected_feature_indices = selected_feature_indices[torch.randperm(selected_feature_indices.shape[0])]
                selected_feature_indices = selected_feature_indices[:N]
            else:
                probabilities = torch.ones((N_to_select_from,))
                selected_feature_indices = torch.multinomial(probabilities, N)

            selected_feature_indices, _ = selected_feature_indices.sort()
            feature_map = feature_map[selected_feature_indices]
            feature_map_hat = feature_map_hat[selected_feature_indices]
            label = label[selected_feature_indices]
            del selected_feature_indices

            label_reshape = int(math.sqrt(label.shape[0]))
            label_one_hot = torch.nn.functional.one_hot(label.reshape(label_reshape, label_reshape), n_classes).contiguous() * 1.0
            label_one_hot = label_one_hot.view(-1, n_classes)
            del label_reshape

            pairwise_dots = torch.tensordot(feature_map, feature_map_hat, dims=([1], [1])) / tau
            pairwise_equality_indicator = torch.tensordot(label_one_hot, label_one_hot, dims=([1], [1]))

            class_count_list = torch.zeros((n_classes,)).long().to(device)
            uniques, class_counts = torch.unique(label, return_counts=True)
            class_count_list[uniques] = class_counts # assuming uniques are class indices...

            softmax = F.log_softmax(pairwise_dots, dim=1)
            
            class_sums = torch.index_select(input=class_count_list, dim=0, index=label).T
            class_sums = class_sums + epsilon
            class_normalized_softmax = (softmax / class_sums) * pairwise_equality_indicator
            total_loss = total_loss + (class_normalized_softmax.sum() / N)

    cle_data = {
        'loss': -total_loss / f_hat.shape[0],
        'N': N_sampled
    }

    return cle_data



def batch_wise_CLE(f, f_hat, labels, N, n_classes, tau, down_sample_factor=2, background_index=0):

    """
    Calculates contrastive clustering loss from: https://arxiv.org/pdf/2012.06985.pdf

    Args:
        f (Tensor): feature maps, shape (BxCxWxH)
        f_hat (Tensor): feature maps for perturbed images, shape (BxCxWxH)
        labels (Tensor): either the ground-truth segmentation mask or from pseudo-labeling (BxWxH)
        N (int): how many pixels to sample from input f and f_hat (GPU memory optimization)

    TODO:
        [ ] Finish implementation
    """
    device = f.device
    b_size = f.shape[0]
    w = f.shape[2] // down_sample_factor
    h = f.shape[3] // down_sample_factor
    f = F.interpolate(f, size=(w,h), mode='bilinear')
    f_hat = F.interpolate(f_hat, size=(w,h), mode='bilinear')

    labels_ = torch.zeros((labels.shape[0], w,h)).to(device)
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    for i, l in enumerate(labels):
        as_img = to_pil(l.int())
        labels_[i] = to_tensor(as_img.resize((w,h), PIL.Image.NEAREST))

    labels = labels_.long()
    epsilon = 0.01

    feature_map_hat = f_hat.permute(0,2,3,1).contiguous().view(-1, n_classes)
    feature_map = f.permute(0,2,3,1).contiguous().view(-1, n_classes)
    label = labels.view(-1)

    not_in_bg_class = ~torch.eq(label, background_index)
    
    N_to_select_from = label[not_in_bg_class].shape[0]
    N = min(N_to_select_from, N)
    N = int(math.sqrt(N))**2

    if N == 0:
        cle_data = {
            'loss': torch.zeros((1,), requires_grad=True).to(f_hat.device),
            'N': N
        }
        return cle_data
    
    probabilities = torch.zeros_like(label).float()
    probabilities[not_in_bg_class] = 1.0
    selected_feature_indices = torch.multinomial(probabilities, N)
    selected_feature_indices, _ = selected_feature_indices.sort()
    feature_map = feature_map[selected_feature_indices]
    feature_map_hat = feature_map_hat[selected_feature_indices]
    label = label[selected_feature_indices]
    del selected_feature_indices

    label_reshape = int(math.sqrt(label.shape[0]))
    label_one_hot = torch.nn.functional.one_hot(label.reshape(label_reshape, label_reshape), n_classes).contiguous() * 1.0
    label_one_hot = label_one_hot.view(-1, n_classes)

    pairwise_dots = torch.tensordot(feature_map, feature_map_hat, dims=([1], [1])) / tau
    pairwise_equality_indicator = torch.tensordot(label_one_hot, label_one_hot, dims=([1], [1]))


    class_count_list = torch.zeros((n_classes,)).long().to(device)
    uniques, class_counts = torch.unique(label, return_counts=True)
    class_count_list[uniques] = class_counts # assuming uniques are class indices...

    softmax = F.log_softmax(pairwise_dots, dim=1)
    
    class_sums = torch.index_select(input=class_count_list, dim=0, index=label).T
    class_sums = class_sums + epsilon
    class_normalized_softmax = (softmax / class_sums) * pairwise_equality_indicator
    total_loss = class_normalized_softmax.sum() / N

    cle_data = {
        'loss': -total_loss / b_size,
        'N': N
    }

    return cle_data
