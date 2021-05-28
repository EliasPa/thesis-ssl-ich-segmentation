import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Subset
from unet2D.loss import intersection_over_union_loss, destructed_intersection_over_union_loss, multiclass_dice_loss, destructed_dice_loss
from tqdm import tqdm
from PIL import ImageDraw
import PIL
from unet2D.utils import labels_to_rgb_batched
import unet2D.utils as unet_utils
import torch.nn.functional as F
from mean_teacher.projection_head import ProjectionHead
from metrics import test_metrics, meters
import cutmix_segmentation.utils as cutmix_utils
from enum import Enum
import inspect
from sklearn.metrics import confusion_matrix

class EMA(nn.Module):

    def __init__(self, mean=None, a=0.5, name="default", defer=True):
        super(EMA, self).__init__()

        self.a = a
        self.name = name
        if defer:
            self.init(mean)

    def init(self, mean):
        self.mean = mean
        
    def forward(self, x):
        self.mean = self.a*x + (1-self.a)*self.mean
        return self.mean


class EMAModel(nn.Module):

    def __init__(self, model, alpha=0.09):
        super(EMAModel, self).__init__()
        self.model = model
        self.alpha = alpha

        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.model(x)

    def calc_ema(self, previous_mean, incoming_value, alpha=0.99):
        return alpha * incoming_value + (1 - alpha) * previous_mean

    def update(self, online):
        for ema_p, online_p in zip(self.parameters(), online.parameters()):
            ema_p.data = self.calc_ema(ema_p.data, online_p.data, alpha=self.alpha)

def initialize_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def accuracy(pred, labels):
    dim = 1 if len(pred.shape) > 1 else 0
    values = torch.argmax(pred, dim=dim)
    tp = (labels == values).sum() * 1.0
    tp_fn = labels.shape[0] * 1.0
    return tp / tp_fn

def get_n_true_positives(pred, labels):
    values = torch.argmax(pred, dim=-1)
    tp = (labels == values).sum().item() * 1.0
    return tp

# TODO: Not working as expected
def class_wise_accuracy(pred, labels, n_classes=10):
    dim = 1 if len(pred.shape) > 1 else 0
    values = torch.argmax(pred, dim=dim)
    prediction_hot = torch.nn.functional.one_hot(values, n_classes)
    prediction_hot[prediction_hot == 0] = -1
    label_hot = torch.nn.functional.one_hot(labels, 10)
    t = ((prediction_hot == label_hot) * 1.0).sum(dim=0)
    n_predictions = label_hot.sum(0)*1.0
    acc = t / n_predictions
    acc[acc != acc] = 0
    return acc, n_predictions

def validate_classifier(net, validation_loader, criterion, epoch, device):
    net.eval()
    with torch.no_grad():
        validation_loss = 0
        m_val_acc_sum = 0
        val_size = 0
        for i, (images, labels) in enumerate(validation_loader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            loss = criterion(output, labels)
            validation_loss += loss.item()
            pred = torch.softmax(output, dim=-1)
            m_val_acc_sum += get_n_true_positives(pred, labels)
            val_size += labels.shape[0]

    return validation_loss, m_val_acc_sum / (val_size * 1.0)

def test_accuracies(net, test_loader, device):    
    net.eval()
    m_acc_sum = 0
    n_classes = 10
    c_m_acc_sum = torch.zeros((n_classes,)).to(device)
    classes_seen = torch.zeros((n_classes,)).to(device)
    N_test = 0.0

    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        if images.shape[1] == 1:
            images = images.repeat(1,3,1,1)
        labels = labels.to(device)

        if len(labels.shape) > 1:
            labels = torch.argmax(labels[:,1:labels.shape[1]], dim=1)
        
        pred = net(images)
        
        acc = get_n_true_positives(pred, labels)

        c_acc, n_predictions = class_wise_accuracy(pred, labels)
        c_semi_acc = c_semi_acc.cpu()
        c_m_acc_sum = c_acc + c_m_acc_sum
        classes_seen = classes_seen + n_predictions

        m_acc_sum = acc + m_acc_sum
        N_test += labels.shape[0]

    acc = m_acc_sum / N_test
    class_wise_acc = c_m_acc_sum / classes_seen
    print(f'Overall Accuracy:\n {m_acc_sum / N_test}')
    print(f'Classwise Accuracy:\n {c_m_acc_sum / classes_seen}')
    return acc, class_wise_acc

def random_split_equal_classes(data, sizes, n_classes):
    print(data)
    targets = np.array(data.targets)
    classes = range(n_classes)
    subsets = []
    for size in sizes:
        to_subset = np.array([])
        size_per_class = int(size / n_classes)
        for c in classes:
            samples_from_class = np.where(targets == c)[0]
            to_subset = np.concatenate((to_subset, np.random.choice(samples_from_class, size_per_class, False)))
            
        subsets.append(Subset(data, np.random.permutation(to_subset)))

    return tuple(subsets)

def filter_decay_params(net, decay):
    with_decay = []
    no_decay = []
    for n, p in net.named_parameters():
        if n == 'bias' or n == 'bn':
            no_decay.append(p)
        else:
            with_decay.append(p)

    return [{'params': with_decay, 'weight_decay': decay }, {'params': no_decay, 'weight_decay': 0.0 }]

def extract_to_writer(writer, metrics_dict, prefix, write_index):
    for key in metrics_dict.keys():
        if not hasattr(metrics_dict[key], "__len__"):
            writer.add_scalar(f'{prefix}/{key}', metrics_dict[key], write_index)

def calc_alpha(start_alpha, target_alpha, training_step, training_steps):
    x = training_step / training_steps
    sigmoid_fac = np.e**(-5*(1-x)**2)
    delta = target_alpha - start_alpha

    return start_alpha + sigmoid_fac * delta

def calc_ema(previous_mean, incoming_value, alpha=0.99):
    return alpha * incoming_value + (1 - alpha) * previous_mean

def ema(net, ema_net, alpha):
    for teacher_p, student_p in zip(ema_net.parameters(), net.parameters()):
        teacher_p.data = calc_ema(teacher_p.data, student_p.data, alpha=alpha)

def get_inverse_frequencies(loader, device, n_classes):
    with torch.no_grad():
        counts = torch.zeros((n_classes,)).long().to(device)
        for (_, mask) in tqdm(loader):
            flat_mask = mask.view(-1).to(device)
            classes_present, class_counts = torch.unique(flat_mask, return_counts=True)
            counts[classes_present] = class_counts
        total = counts.sum().float()
        frequencies = counts.float() / total
        inverse_frequencies = (1 - frequencies).tolist()
        print(f'Inverse frequencies: {inverse_frequencies}')
        return inverse_frequencies

def get_color_map(n_classes):
    if n_classes == 6:    
        color_map = torch.tensor([
                [0,0,0], # bg: black
                [0,1.0,0], # green, IVH
                [1.0,0,0], # red, IPH
                [0,0,1.0], # blue, SAH
                [0.5,0,0.5], # EDH, magenta?
                [0,0.5,0.9] # SDH, cyan?
        ])
    elif n_classes == 2:
        color_map = torch.tensor([
                [0,0,0], # bg: black
                [1.0,0.2,0.2], # any blood: white
        ])
    else:
        color_map = torch.rand((n_classes, 3)) # random colors

    return color_map

def get_hemorrhage_color_map(device):
    color_map = torch.tensor([
            [0,0,0], # bg: black
            [0,1.0,0], # green, IVH
            [1.0,0,0], # red, IPH
            [0,0,1.0], # blue, SAH
            [0.5,0,0.5], # EDH, magenta
            [0,0.5,0.9] # SDH, cyan
    ]).to(device)

    return color_map


def save(net, optimizer, last_epoch, test_metrics, lr, save_base_path, hps, date, mode='semi'):
    d = {
        'net': net.state_dict(),
        'optimizer': optimizer,
        'last_epoch': last_epoch,
        'test_metrics': test_metrics,
        'lr': lr,
        'hps': hps,
        'date': date
    }

    torch.save(d, f'{save_base_path}{mode}.pth')

def save_best_ema(save_args, save_ema_args, best_path='_checkpoint_best', best_ema_path='_checkpoint_ema_best'):
    save_best(save_args, best_path)
    save_best(save_ema_args, best_ema_path)

def save_best(save_args, best_path='_checkpoint_best'):

    latest_path=f'{best_path}_LATEST'
    new_metrics = save_args['test_metrics']
    best_save_path = f"{save_args['save_base_path']}{best_path}.pth"
    assert(save_args['date'] is not None)
    try:
        old_model_data = load(best_save_path)
        old_metrics = old_model_data['test_metrics']
        old_date = old_model_data['date']

        assert(save_args['date'] == old_date)

        print(save_args['date'], old_date)
        if new_metrics['any_blood_dice'] > old_metrics['any_blood_dice']:
            save_args['mode'] = best_path
            save(**save_args)
            print(f"Overwrote new best model to {best_save_path} with {new_metrics['any_blood_dice']}")
        else:
            print(f"Ignoring new best candidate {new_metrics['any_blood_dice']}")
    except Exception as e:
        print(e)
        save_args['mode'] = best_path
        save(**save_args)
        print(f"Saved new best model to {best_save_path} with {new_metrics['any_blood_dice']}")
    
def load(path):
    state_dict = torch.load(path)
    return state_dict

def test(
    net,
    test_loader,
    color_map,
    n_classes,
    n_channels=1,
    visualize_every=20,
    visualization_path='../output/tasks/MNIST_MT_SEG/',
    thresholding_enabled=True,
    full_range_of_thresholds=False):

    net.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    device = color_map.device
    no_bg_mean_dice_accumulator = 0

    n_classes_bg_v_any = 2 # Two classes, bg vs. all blood
    thresholds = np.linspace(0,1,20) if full_range_of_thresholds else [1.0 / n_classes_bg_v_any]
    slice_level_cms = np.zeros((len(thresholds), n_classes_bg_v_any, n_classes_bg_v_any))
    pixel_level_cms = np.zeros((len(thresholds), n_classes_bg_v_any, n_classes_bg_v_any))

    mean_dice_meter = meters.LegacySegmentationMeter(n_classes=n_classes, scoring_function=destructed_dice_loss)
    mean_iou_meter = meters.LegacySegmentationMeter(n_classes=n_classes, scoring_function=destructed_intersection_over_union_loss)
    any_blood_dice_meter = meters.DiceMeter(n_classes=n_classes_bg_v_any, reduction_mode=meters.SegmentationMetricReductionMode.EMPTY_SET_IGNORE_BINARY_NO_BG)
    any_blood_iou_meter = meters.IoUMeter(n_classes=n_classes_bg_v_any, reduction_mode=meters.SegmentationMetricReductionMode.EMPTY_SET_IGNORE_BINARY_NO_BG)
    no_bg_ignored_sets_dice_meter = meters.DiceMeter(n_classes=n_classes, reduction_mode=meters.SegmentationMetricReductionMode.EMPTY_SET_IGNORE_MULTICLASS_NO_BG)
    no_bg_ignored_sets_iou_meter = meters.IoUMeter(n_classes=n_classes, reduction_mode=meters.SegmentationMetricReductionMode.EMPTY_SET_IGNORE_MULTICLASS_NO_BG)

    for i, (images, masks) in enumerate(tqdm(test_loader)):
        images = images.to(device)
        if images.shape[1] == 1:
            images = images.repeat(1,n_channels,1,1)
        masks = masks.to(device).squeeze(1)
        masks = masks.clip(0, n_classes - 1)
        
        output = net(images)

        pred = F.softmax(output, dim=1)
        total_loss += criterion(pred, masks).item()

        mean_dice_meter.update(pred, masks)
        mean_iou_meter.update(pred, masks)

        class_index_pred = torch.argmax(pred, dim=1)
        
        no_bg_ignored_sets_dice_meter.update(class_index_pred, masks)
        no_bg_ignored_sets_iou_meter.update(class_index_pred, masks)

        any_blood_dice_meter.update(class_index_pred.clip(0,1), masks.clip(0,1))
        any_blood_iou_meter.update(class_index_pred.clip(0,1), masks.clip(0,1))

        # No-background Dice score:
        no_bg_mean_dice_accumulator += (1 - multiclass_dice_loss(pred, masks, reduction='no-bg')).item()

        class_map = pred[:,1:,:,:].squeeze(1)
        
        for t, threshold in enumerate(thresholds):

            threshold_mask = torch.ge(class_map, threshold).long()

            """
            To calculate ROC stats, we transform this into a binary problem: bg vs. any blood
            """
            thresholded_max_labels = threshold_mask
            pixel_level_cms[t] += cutmix_utils.get_confusion_matrix(thresholded_max_labels, masks.clip(0,1),  n_classes=n_classes)

            thresholded_slice_prediction = thresholded_max_labels.sum(dim=2).sum(dim=1).clip(0,1).cpu().numpy()
            thresholded_slice_label = masks.sum(dim=2).sum(dim=1).clip(0,1).cpu().numpy()
            slice_level_cms[t] += confusion_matrix(thresholded_slice_label, thresholded_slice_prediction, labels=[0,1])

        if i % visualize_every == 0:
            pred_as_image = labels_to_rgb_batched(torch.argmax(output, dim=1), color_map).permute(0,3,1,2)
            gt_as_image = labels_to_rgb_batched(masks, color_map).permute(0,3,1,2)
            if images.shape[1] == 1:
                images_as_image = images.repeat(1,3,1,1)
            else:
                images_as_image = images

            for idx, img in enumerate(pred_as_image):
                dice = 1 - multiclass_dice_loss(pred[idx].unsqueeze(0), masks[idx].unsqueeze(0), reduction='none')
                iou = 1 - intersection_over_union_loss(pred[idx].unsqueeze(0), masks[idx].unsqueeze(0))
                img = np.asarray(img.permute(1,2,0).cpu()*255, dtype=np.uint8)

                text = "Dice: \n{}\nIoU: {:.3f}".format(
                    f"[{ ''.join(['{:.1f}'.format(d.item()*100) + (', ' if d_i < dice.shape[0] else '') for d_i, d in enumerate(dice)]) }]",
                    iou*100
                )
                img = PIL.Image.fromarray(img)
                ImageDraw.Draw(img).text((1,1), text, (0,255,255))
                pred_as_image[idx] = torch.tensor(np.asarray(img)).permute(2,0,1).to(device) / 255.0

            unet_utils.save_overlay_grid(images_as_image, pred_as_image, gt_as_image, f'{visualization_path}_test_output_{i}.png')

    no_bg_mean_dice = no_bg_mean_dice_accumulator / len(test_loader)
    
    no_bg_ignored_sets_dice = no_bg_ignored_sets_dice_meter.get_final_metric()
    no_bg_ignored_sets_iou = no_bg_ignored_sets_iou_meter.get_final_metric()
    
    mean_dice = mean_dice_meter.get_final_metric()
    mean_iou = mean_iou_meter.get_final_metric()
    
    any_blood_dice = any_blood_dice_meter.get_final_metric()
    any_blood_iou = any_blood_iou_meter.get_final_metric()

    slice_level_stats = test_metrics.get_statistics_from_cms(slice_level_cms)
    pixel_level_stats = test_metrics.get_statistics_from_cms(pixel_level_cms)
    
    metrics = {
        'total_loss': total_loss,
        'mean_dice': mean_dice,
        'mean_iou': mean_iou,
        'no_bg_mean_dice': no_bg_mean_dice,
        'no_bg_ignored_sets_dice': no_bg_ignored_sets_dice,
        'no_bg_ignored_sets_iou': no_bg_ignored_sets_iou,
        'any_blood_dice': any_blood_dice,
        'any_blood_iou': any_blood_iou,
    }

    for key in slice_level_stats.keys():
        metrics[f'slice_{key}'] = slice_level_stats[key]
        metrics[f'pixel_{key}'] = pixel_level_stats[key]

    return metrics


def test_ensemble(nets, test_loader, color_map, n_classes, n_channels=1, visualize_every=5, visualization_path='../output/tasks/MNIST_MT_SEG/'):

    with torch.no_grad():
        iou = 0.0
        dice_top = 0
        dice_bottom = 0
        dice_epsilon = 0
        no_bg_mean_dice_accumulator = 0

        iou_top = 0
        iou_bottom = 0
        iou_epsilon = 0

        criterion = nn.CrossEntropyLoss()
        total_loss = 0

        device = color_map.device

        class_index_dice_top = 0
        class_index_dice_bottom = 0
        ignored_sets_dice_top = 0
        ignored_sets_dice_bottom = 0
        no_bg_ignored_sets_dice_top = 0
        no_bg_ignored_sets_dice_bottom = 0

        thresholds = np.linspace(0,1,len(nets)*2)
        n_classes_bg_v_any = 2 # Two classes, bg vs. all blood
        precisions = np.zeros((len(thresholds), n_classes_bg_v_any))
        recalls = np.zeros((len(thresholds), n_classes_bg_v_any))
        sensitivities = np.zeros((len(thresholds), n_classes_bg_v_any))
        specificities = np.zeros((len(thresholds), n_classes_bg_v_any))
        tps = np.zeros((len(thresholds), n_classes_bg_v_any))
        fps = np.zeros((len(thresholds), n_classes_bg_v_any))
        tns = np.zeros((len(thresholds), n_classes_bg_v_any))
        fns = np.zeros((len(thresholds), n_classes_bg_v_any))
        epsilon = 0.000001

        any_blood_dice_top = 0
        any_blood_dice_bottom = 0

        # TODO: Batch size averaging in multiclass dice loss?
        for i, (images, masks) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            images = images.repeat(1,n_channels,1,1)
            masks = masks.to(device)
            ensemble_prediction = torch.zeros_like(masks)
            masks = masks.squeeze(1)
            
            for n, net in enumerate(nets):
                net.eval()
                output = net(images)

                pred = F.softmax(output, dim=1)
                total_loss += criterion(pred, masks).item()

                _, dt, db, de = destructed_dice_loss(pred, masks)
                dice_bottom += db.item()
                dice_top += dt.item()
                dice_epsilon = de

                num_classes = output.shape[1]
                class_index_pred = torch.argmax(pred, dim=1)
                binary_prediction = torch.clip(class_index_pred, min=0, max=1)
                ensemble_prediction = ensemble_prediction + binary_prediction

                _, cdt, cdb = test_metrics.dice(class_index_pred, masks, num_classes, segmentation_mode=test_metrics.SegmentationMetricMode.DEFAULT)
                class_index_dice_top += cdt.sum()
                class_index_dice_bottom += cdb.sum()

                # The return values might have np.nan values, since empty sets are ignored:
                _, cdt, cdb = test_metrics.dice(class_index_pred, masks, num_classes, segmentation_mode=test_metrics.SegmentationMetricMode.EMPTY_SET_IGNORE)
                ignored_sets_dice_top += cdt[~np.isnan(cdt)].sum()
                ignored_sets_dice_bottom += cdb[~np.isnan(cdb)].sum()
                no_bg_ignored_sets_dice_top += cdt[1:][~np.isnan(cdt[1:])].sum() # don't include first class (bg)
                no_bg_ignored_sets_dice_bottom += cdb[1:][~np.isnan(cdb[1:])].sum() # don't include first class (bg)

                 # The return values might have np.nan values, since empty sets are ignored:
                _, cdt, cdb = test_metrics.dice(class_index_pred.clip(0,1), masks.clip(0,1), num_classes=2, segmentation_mode=test_metrics.SegmentationMetricMode.EMPTY_SET_IGNORE)
                any_blood_dice_top += cdt[1] if not np.isnan(cdt[1:])[0] else 0#[~np.isnan(cdt[1])].sum() # don't include first class (bg)
                any_blood_dice_bottom += cdb[1] if not np.isnan(cdb[1:])[0] else 0#[~np.isnan(cdb[1])].sum() # don't include first class (bg)

                # No-background Dice score:
                no_bg_mean_dice_accumulator += (1 - multiclass_dice_loss(pred, masks, reduction='no-bg')).item()

                _, it, ib, ie = destructed_intersection_over_union_loss(pred, masks)
                iou_bottom += ib.item()
                iou_top += it.item()
                iou_epsilon = ie

            ensemble_prediction = ensemble_prediction.float() / len(nets)

            for t, threshold in enumerate(thresholds):
                threshold_mask = torch.ge(ensemble_prediction, threshold).long() # OVER threshold

                """
                To calculate ROC stats, we transform this into a binary problem: bg vs. any blood
                """
                clipped_masks = masks.clip(0,1)

                cm = cutmix_utils.get_confusion_matrix(clipped_masks, threshold_mask,  n_classes=n_classes_bg_v_any)

                stats = test_metrics.get_statistics_from_cm(cm, n_classes=n_classes_bg_v_any)

                tps[t] += stats['base_stats'][0]
                tns[t] += stats['base_stats'][1]
                fps[t] += stats['base_stats'][2]
                fns[t] += stats['base_stats'][3]
                
                precisions[t] += stats['precisions']
                recalls[t] += stats['recalls']
                sensitivities[t] += stats['sensitivities']
                specificities[t] += stats['specificities']

            if i % visualize_every == 0:
                pred_as_image = labels_to_rgb_batched(torch.argmax(output, dim=1), color_map).permute(0,3,1,2)
                gt_as_image = labels_to_rgb_batched(masks, color_map).permute(0,3,1,2)

                if images.shape[1] == 1:
                    images_as_image = images.repeat(1,3,1,1)
                else:
                    images_as_image = images

                for idx, img in enumerate(pred_as_image):
                    dice = 1 - multiclass_dice_loss(pred[idx].unsqueeze(0), masks[idx].unsqueeze(0), reduction='none')
                    iou = 1 - intersection_over_union_loss(pred[idx].unsqueeze(0), masks[idx].unsqueeze(0))
                    img = np.asarray(img.permute(1,2,0).cpu()*255, dtype=np.uint8)

                    text = "Dice: \n{}\nIoU: {:.3f}".format(
                        f"[{ ''.join(['{:.1f}'.format(d.item()*100) + (', ' if d_i < dice.shape[0] else '') for d_i, d in enumerate(dice)]) }]",
                        iou*100
                    )
                    img = PIL.Image.fromarray(img)
                    ImageDraw.Draw(img).text((1,1), text, (0,255,255))
                    pred_as_image[idx] = torch.tensor(np.asarray(img)).permute(2,0,1).to(device) / 255.0

                unet_utils.save_overlay_grid(images_as_image, pred_as_image, gt_as_image, f'{visualization_path}_test_output_{i}.png')

        mean_dice =(dice_top / (dice_bottom + dice_epsilon))
        mean_iou = (iou_top / (iou_bottom + iou_epsilon))
        no_bg_mean_dice = no_bg_mean_dice_accumulator / len(test_loader)
        class_index_dice = class_index_dice_top / (class_index_dice_bottom + dice_epsilon)
        ignored_sets_dice = ignored_sets_dice_top / (ignored_sets_dice_bottom + dice_epsilon)
        no_bg_ignored_sets_dice = no_bg_ignored_sets_dice_top / (no_bg_ignored_sets_dice_bottom + dice_epsilon)
        any_blood_dice = any_blood_dice_top / (any_blood_dice_bottom + epsilon)

        metrics = {
            'total_loss': total_loss,
            'mean_dice': mean_dice,
            'mean_iou': mean_iou,
            'no_bg_mean_dice': no_bg_mean_dice,
            'class_index_dice': class_index_dice,
            'ignored_sets_dice': ignored_sets_dice,
            'no_bg_ignored_sets_dice': no_bg_ignored_sets_dice,
            'precisions': precisions / len(test_loader),
            'recalls': recalls / len(test_loader),
            'sensitivities': sensitivities / len(test_loader),
            'specificities': specificities / len(test_loader),
            'fpr': 1 - tns[:,1] / (tns[:,1] + fps[:,1] + epsilon),
            'tpr': tps[:,1] / (tps[:,1] + fns[:,1] + epsilon),
            'any_blood_dice': any_blood_dice
        }

        return metrics

class SegmentationNetWithProjectionHead(nn.Module):

    def __init__(self, net, n_classes, n_projection_input_channels=2048, n_hidden_channels=256):
        super(SegmentationNetWithProjectionHead, self).__init__()
        self.net = net
        self.projection_head = ProjectionHead(n_classes=n_classes, n_input_channels=n_projection_input_channels, n_hidden_dims=n_hidden_channels)

    """
    Segmentation without projection.
    """
    def segment(self, x):
        return self.net(x)

    """
    Forward pass consisting of a segmentation and a projection.
    This method should be used for pretraining only.
    """
    def forward(self, x, classify=True):
        if 'classify' in inspect.getfullargspec(self.net.forward)[0]:
            segmentation = self.net(x, classify=classify)
        else:
            segmentation = self.net(x)

        projection = self.projection_head(segmentation)
        return projection

class ExperimentType(Enum):
    CT = 'ct'
    CITYSCAPES = 'cityscapes'
    MNIST = 'mnist'

    def __str__(self):
        return str(self.value)

class ModelType(Enum):
    UNET = 'unet'
    DEEPLABV3_RESNET50 = 'deeplabv3plus_resnet50'

    def __str__(self):
        return str(self.value)

class CLEMode(Enum):
    REAL_LABELS='real_labels'
    PSEUDO_LABELS='pseudo_labels'

    def __str__(self):
        return str(self.value)

class FixMatchType(Enum):
    CUTMIX='cutmix'
    REGULAR='regular'

    def __str__(self):
        return str(self.value)

class OptimizerType(Enum):
    SGD='sgd'
    ADAM='adam'

    def __str__(self):
        return str(self.value)

class SchedulerType(Enum):
    COSINE='cosine'
    COSINE_WARM_RESTARTS='cosine_warm_restarts'

    def __str__(self):
        return str(self.value)

class AugmentationType(Enum):
    NONE='none'
    MILD='mild'
    FULL='full'
    EXTREME='extreme'

    def __str__(self):
        return str(self.value)
