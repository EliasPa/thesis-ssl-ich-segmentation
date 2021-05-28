import torch

"""
Hint: target_shape can be e.g. (batch_size, n_classes, w, h) 
"""
def one_hot_encode(tensor, n_classes, target_shape):
    one_hot_target = torch.zeros(target_shape).to(tensor.device)
    for c in range(n_classes):
        full = torch.full((one_hot_target.shape[2], one_hot_target.shape[3]), c).to(tensor.device)
        indices = tensor == full
        one_hot_target[:, c, :, :] = indices * 1.0

    return one_hot_target

def dice_loss(prediction, target):
    dice, _, _, _ = destructed_dice_loss(prediction, target)
    return dice

def destructed_dice_loss(prediction, target):
    n_classes = prediction.shape[1]
    one_hot_target = one_hot_encode(target, n_classes, prediction.shape)
    spread_prediction = prediction.view(-1)
    target_prediction = one_hot_target.view(-1)
    intersection = torch.dot(spread_prediction, target_prediction)
    bottom = torch.dot(spread_prediction, spread_prediction) + torch.dot(target_prediction, target_prediction)
    epsilon = 0.0001

    dice_coefficient = (2 * intersection) / (bottom + epsilon)

    return 1 - (dice_coefficient + torch.tensor(0)), 2*intersection, bottom, epsilon

def multiclass_dice_loss(prediction, target, reduction='mean', ignore_index=None):
    dice, _, _, _ = destructed_multiclass_dice_loss(prediction, target, reduction, ignore_index)
    return dice

def destructed_multiclass_dice_loss(prediction, target, reduction='mean', ignore_index=None):
    n_classes = prediction.shape[1]
    one_hot_target = one_hot_encode(target, n_classes, prediction.shape)

    if ignore_index is not None:
        #with torch.no_grad():
        ignore_mask = torch.logical_not(torch.eq(target, ignore_index)).long()
        pred_ignore_mask = ignore_mask.unsqueeze(1).repeat(1,n_classes,1,1)
        #print(pred_ignore_mask)
        #print(pred_ignore_mask.shape)
        prediction = prediction * pred_ignore_mask
        target = target * ignore_mask
        #print(target)
        #print(prediction)

    spread_prediction = prediction.view(prediction.shape[0], prediction.shape[1], -1)
    target_prediction = one_hot_target.view(one_hot_target.shape[0], one_hot_target.shape[1],-1)

    # Thanks to the discussion in: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
    intersection = (spread_prediction*target_prediction).sum(0).sum(1)
    bottom = (spread_prediction * spread_prediction).sum(0).sum(1) + (target_prediction * target_prediction).sum(0).sum(1)
    epsilon = 0.0001

    dice_coefficient = (2 * intersection) / (bottom + epsilon)
    if reduction == 'mean':
        dice_coefficient = dice_coefficient.mean()
    elif reduction == 'sum':
        dice_coefficient = dice_coefficient.sum()
    elif reduction == 'no-bg':
        dice_coefficient = dice_coefficient[1:].mean()
    elif reduction == 'no-bg-sum':
        dice_coefficient = dice_coefficient[1:].sum()
    elif reduction == 'none':
        pass
    
    return 1 - (dice_coefficient + torch.tensor(0)), 2*intersection, bottom, epsilon
    
def destructed_intersection_over_union_loss(prediction, target):
    n_classes = prediction.shape[1]
    one_hot_target = one_hot_encode(target, n_classes, prediction.shape)
    spread_prediction = prediction.view(-1)
    target_prediction = one_hot_target.view(-1)
    intersection = torch.dot(spread_prediction, target_prediction)
    union = torch.dot(spread_prediction, spread_prediction) + torch.dot(target_prediction, target_prediction) - intersection
    epsilon = 0.0001

    iou = (intersection) / (union + epsilon)

    return 1 - (iou + torch.tensor(0)), intersection, union, epsilon

def intersection_over_union_loss(prediction, target):
    iou, _, _, _ = destructed_intersection_over_union_loss(prediction, target)
    return iou
