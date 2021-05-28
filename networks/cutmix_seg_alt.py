import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
from dataset.mnist import create_dataloaders_from_path
from dataset.ct import create_all_dataloaders_folded
from unet2D import utils
import math
import argparse
from unet2D.unet import Unet2D
import fix_match.utils as fm_utils
import torch.nn.functional as F
import fix_match.rand_aug as rand_aug
from unet2D.loss import intersection_over_union_loss, multiclass_dice_loss
from tqdm import tqdm
import utils as gen_utils
from focal_loss_alt.focal_loss import FocalLoss, focal_loss_function
from unet2D.loss import intersection_over_union_loss
from tqdm import tqdm
import mean_teacher.utils as mt_utils
import random
from torch.utils.tensorboard import SummaryWriter
from unet2D.utils import labels_to_rgb_batched
import time
from datetime import datetime
from utils import ExperimentType, OptimizerType, AugmentationType
from augmentation.additive_gaussian_noise import AdditiveGaussianNoiseTransform
import arguments

def initialise_arguments():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('-m', '--model', type=str, help="Base path to model")
    args_parser.add_argument('-n', '--experiment-name', type=str, default='mt_cut_mix', help="Name of experiment. E.g. baseline")
    args_parser.add_argument('-r', '--relu-type', type=str, help="ReLU type used when training: normal or leaky")
    args_parser.add_argument('-lt', '--loss-type', type=str, default='multiclass_dice_sum', help="Type of loss function used for training")
    args_parser.add_argument('--loss-reduction', type=str, default='no-bg', help="Loss function reduction")
    args_parser.add_argument('-c', '--n-classes', type=int, default=6, help="Number of classes to predict.")
    args_parser.add_argument('-ch', '--n-channels', type=int, default=1, help="Number of channels in input image.")
    args_parser.add_argument('-et', '--experiment-type', type=ExperimentType, default='ct', help="ct or cityscapes", choices=list(ExperimentType))
    args_parser.add_argument('--optimizer', type=OptimizerType, default='adam', help="sgd or adam", choices=list(OptimizerType))
    args_parser.add_argument('-el', '--experiment-log-prefix', type=str, default='', help="prefix added to logs")
    args_parser.add_argument('--tau', '--confidence-threshold', type=float, default=0.97, help="Confidence threshold [0,1]")
    args_parser.add_argument('-ct', '--consistency-loss-type', default='mse-confidence', type=str, help="Consistency loss function type")
    args_parser.add_argument('-dr', '--consistency-dice-reduction', default='no-bg', type=str, help="Dice reduction type for consistency loss. Required if consistency loss type is Dice loss.")
    args_parser.add_argument('-u', '--unsupervised-training-path', type=str, default='E:\\kaggle\\rsna-intracranial-hemorrhage-detection\\nifti_down_sampled\\', help="Path to unsupervised training data")
    args_parser.add_argument('--cut-mix-lower-bound', type=int, default=40, help="CutMix lower bound e.g. (40%)")
    args_parser.add_argument('--cut-mix-upper-bound', type=int, default=50, help="CutMix upper bound e.g. (50%)")
    args_parser.add_argument('-e', '--epochs', type=int, default=100, help="Number of epochs.")
    args_parser.add_argument('-f', '--fold', type=int, default=0, help="Fold index")
    args_parser.add_argument('--lambd', type=float, default=1.0, help="Consistency loss weight.")
    args_parser.add_argument('--ema-alpha', type=float, default=0.99, help="EMA alpha.")
    args_parser.add_argument('--N-ramp-up-consistency', type=int, default=5, help="Ramp up epochs for consistency loss weight.")
    args_parser.add_argument('--lr', type=float, default=3e-5, help="Learning rate.")
    args_parser.add_argument('--seed', type=int, default=123, help="Seed.")
    args_parser.add_argument('--batch-size-s', type=int, default=2, help="Supervised batch size.")
    args_parser.add_argument('--batch-size-u', type=int, default=4, help="Unsupervised batch size.")
    args_parser.add_argument('--batch-size-t', type=int, default=2, help="Test/validation batch size.")
    args_parser.add_argument('-nu', '--N-unsupervised', type=int, default=1000, help="Number of epochs.")
    args_parser.add_argument('-ns', '--N-supervised-mnist', type=int, default=10, help="Number of supervised samples in MNIST case.")
    args_parser.add_argument('--N-iter', type=int, default=512, help="N iterations in epoch")
    args_parser.add_argument('--consistency-focal-gamma', type=float, default=0.5, help="Gamma of focal loss.")
    args_parser.add_argument('--focal-gamma', type=float, default=2, help="Gamma of focal loss.")
    args_parser.add_argument('--per-pixel-confidence', type=arguments.bool_arg, default=True, help="If True, confidence masks are per pixel. If False, mean reduction is used.")
    args_parser.add_argument('--shuffle', type=arguments.bool_arg, default=False, help="Should shuffle model inputs prior to forward pass")
    args_parser.add_argument('--std', type=float, default=0.01, help="Gaussian noise standard deviation")
    args_parser.add_argument('--use-dropout', type=arguments.bool_arg, default=False, help="Should use dropout in model")
    args_parser.add_argument('--skip-checkpoint', type=arguments.bool_arg, default=False, help="Should skip saving checkpoints")
    args_parser.add_argument('-z', '--zero-gradients-every', type=int, default=1, help="How often to zero gradients")
    args_parser.add_argument('-at', '--augmentation-type', type=AugmentationType, default='full', help="Augmentation type", choices=list(AugmentationType))
    args_parser.add_argument('--legacy-unet', type=arguments.bool_arg, default=True, help="Should use legacy U-net, or latest version")
    hps = args_parser.parse_args()
    return hps

num_threads = 1
num_data_loader_workers = 1
device = torch.device('cuda:0')
date = None
weight_decay=4e-5
momentum=0.9

sizes = [
    0.8,
    0.1,
    0.1
]

def main():
    hps = initialise_arguments()
    torch.cuda.empty_cache()
    run_experiment(hps)

def run_experiment(hps):
    print(f"Using fold {hps.fold}...")
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)
    random.seed(hps.seed)
    torch.backends.cudnn.benchmark=False

    start_time = time.time()

    
    transform_u = transforms.Compose([])

    if hps.std > 0:
        transform_s = transforms.Compose([
            AdditiveGaussianNoiseTransform(mean=0.0, std=hps.std)
        ])
    else:
        transform_s = transforms.Compose([])

    transform_t = transforms.Compose([])

    if hps.experiment_type == ExperimentType.CT:

        if hps.augmentation_type == AugmentationType.NONE:
            override_augmentations = None
            base_augment_train = False
        elif hps.augmentation_type == AugmentationType.MILD:
            override_augmentations = {
                'random_rot': 3,
                'random_scale': (0.98, 1.02),
                'random_shear': 2,
                'random_translation': (5, 5),
                'flip': 0.5,
            }
            base_augment_train = True
        elif hps.augmentation_type == AugmentationType.FULL:
            override_augmentations = None
            base_augment_train = True
        else:
            raise Exception(f'Augmentation type {hps.augmentation_type} not implemented.')
        
        loaders = create_all_dataloaders_folded(
            N=hps.N_unsupervised,
            num_threads=num_threads,
            num_workers=num_data_loader_workers,
            fold=hps.fold,
            batch_size_s=hps.batch_size_s,
            batch_size_u=hps.batch_size_u,
            batch_size_t=hps.batch_size_t,
            sizes=sizes,
            pin_memory=True,
            resize=0.5,
            transform_u=transform_u,
            transform_s=transform_s,
            transform_t=transform_t,
            n_classes=hps.n_classes,
            image_path='../data/ct_scans/',
            u_image_path=hps.unsupervised_training_path,
            mask_path='../data/multiclass_mask/',
            override_transforms=override_augmentations,
            base_augment_train=base_augment_train
        )

        unsupervised_loader_inf, supervised_loader, supervised_loader_inf, validation_loader, test_loader = loaders
    elif hps.experiment_type == ExperimentType.MNIST:
        loaders = create_dataloaders_from_path(
            n_classes=hps.n_classes,
            batch_size_s=hps.batch_size_s,
            batch_size_u=hps.batch_size_u,
            batch_size_t=hps.batch_size_t,
            transform_u=transform_u,
            transform_s=transform_s,
            transform_t=transform_t,
            pin_memory=True,
            base_path='../data/mnist/mnist_lesion/',
            N_s=hps.N_supervised_mnist,
            N_u=hps.N_unsupervised,
            N_v=1000
        )
        unsupervised_loader_inf, supervised_loader, supervised_loader_inf, validation_loader, test_loader = loaders
    else:
        raise Exception(f'Experiment type {hps.experiment_type} not implemented.')

    print(f'Data loading took: {time.time() - start_time}s using {num_threads} threads.')

    color_map = gen_utils.get_color_map(hps.n_classes).to(device)

    global save_base_path
    save_base_path = f'../experiment_runs/{hps.experiment_name}/models/{hps.fold}_{hps.experiment_log_prefix}_{hps.seed}'

    semi_net = Unet2D(n_input_channels=hps.n_channels, n_classes=hps.n_classes, use_dropout=hps.use_dropout, legacy=hps.legacy_unet).to(device)
    gen_utils.initialize_model(semi_net)

    ema_model_internal = Unet2D(n_input_channels=hps.n_channels, n_classes=hps.n_classes, use_dropout=hps.use_dropout, legacy=hps.legacy_unet).to(device)
    ema_net = mt_utils.EMAWeightOptimizer(target_net=ema_model_internal, source_net=semi_net)

    if hps.optimizer == OptimizerType.SGD:
        optimizer = torch.optim.SGD(semi_net.parameters(), lr=hps.lr, weight_decay=weight_decay, momentum=momentum)
    elif hps.optimizer == OptimizerType.ADAM:
        optimizer = torch.optim.Adam(semi_net.parameters(), lr=hps.lr)
    else:
        raise Exception(f'Did not recognize optimizer type {hps.optimizer}')
    
    if 'focal' in (hps.loss_type, hps.consistency_loss_type):
        global focal_alphas
        focal_alphas = gen_utils.get_inverse_frequencies(supervised_loader, device, hps.n_classes)

    if hps.loss_type == 'focal':
        criterion = FocalLoss(gamma=hps.focal_gamma, alpha=focal_alphas, size_average=True)
    elif hps.loss_type == 'dice':
        criterion = multiclass_dice_loss
    elif hps.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception(f'Loss type {hps.loss_type} not defined')
    
    scheduler = fm_utils.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=hps.epochs)
    last_epoch = 0

    if hps.model:
        print("Attempting to load previous network from checkpoint...")
        try:
            state_dict = gen_utils.load(f'{hps.model}semi.pth')
            net = state_dict['net']
            optimizer = state_dict['optimizer']
            last_epoch = state_dict['last_epoch']
            semi_net.load_state_dict(net)

            ema_state_dict = gen_utils.load(f'{hps.model}ema.pth')
            net = ema_state_dict['net']
            ema_model_internal.load_state_dict(net)
            ema_net = mt_utils.EMAWeightOptimizer(target_net=ema_model_internal, source_net=semi_net, initialize_to_source=False)
            print("Success.")
        except Exception as e:
            print("Failed to load previous net")
            print(e)
    
    # Step the scheduler until it reaches where training was last stopped:
    print(f"Stepping scheduler {last_epoch} times")
    for _ in tqdm(range(last_epoch)):
        scheduler.step()

    print("Starting training...")
    global date
    date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    writer = SummaryWriter(log_dir=f'../experiment_runs/{hps.experiment_name}/{hps.experiment_log_prefix}_{hps.seed}_fold_{hps.fold}/{date}')
    hps.N_iter = max(int(len(supervised_loader)), 10)
    last_epoch = train(
        net=semi_net,
        ema_net=ema_net,
        unsupervised_loader=unsupervised_loader_inf,
        supervised_loader=supervised_loader_inf,
        validation_loader=validation_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        writer=writer,
        hps=hps,
        color_map=color_map,
        last_epoch=last_epoch
    )

    print("Training done. Starting testing...")
    test_metrics = gen_utils.test(
        net=semi_net,
        test_loader=test_loader,
        color_map=color_map,
        visualize_every=1,
        n_classes=hps.n_classes,
        visualization_path=f'../output/tasks/{hps.experiment_name}/{hps.fold}_full')

    gen_utils.extract_to_writer(writer, test_metrics, prefix='supervised/test', write_index=hps.fold)
    
    print("Testing done. Saving model...")
    gen_utils.save(semi_net, optimizer, last_epoch, test_metrics, scheduler.get_last_lr()[0], save_base_path, hps=hps, mode='_full_semi', date=date)
    
    if hasattr(ema_net, 'target_net'):
        gen_utils.save(ema_net.target_net, optimizer, last_epoch, test_metrics, scheduler.get_last_lr()[0], save_base_path, hps=hps, mode='_full_ema', date=date)
    else:
        gen_utils.save(ema_net, optimizer, last_epoch, test_metrics, scheduler.get_last_lr()[0], save_base_path, hps=hps, mode='_full_ema', date=date)

    print(f'Fold {hps.fold} finished. Metrics: {test_metrics}')


def train(net, ema_net, unsupervised_loader, supervised_loader, validation_loader, optimizer, criterion, scheduler, writer, hps, color_map, reduction=None, last_epoch=0):

    mode = "RGB" if hps.n_channels == 3 else None
    to_pil = transforms.ToPILImage(mode=mode)
    to_tensor = transforms.ToTensor()

    unsupervised_iter = iter(unsupervised_loader)
    supervised_iter = iter(supervised_loader)

    optimizer.zero_grad()
    for epoch in range(last_epoch, hps.epochs):
        total_loss = 0

        data_start = time.time()
        p_bar = tqdm(range(hps.N_iter))

        if hasattr(ema_net, 'target_net'):
            ema_net.target_net.train()
        else:
            ema_net.train()

        accumulated_loss = 0

        net.train()
        for i in p_bar:

            images, _ = unsupervised_iter.next()
            s_images, labels = supervised_iter.next()
            images = images.to(device)
            s_images = s_images.to(device)
            labels = labels.to(device)

            labels = labels.squeeze(1)
            if images.shape[1] == 1:
                images = images.repeat(1,hps.n_channels,1,1)
            
            # Network:
            #
            # upper -------> teacher --> upper_seg -----------------------------------------
            #         |                                                                    |
            #         \/                                                                  \/
            # mask ->mixed-> student --> mixed_seg --> loss_c <-- pseudo_seg <--- mixed_seg_post
            #         /\                                                                  /\
            #         |                                                                    |
            # lower -------> teacher --> lower_seg ----------------------------------------- 

            
            # Supervised loss:
            supervised_output = net(s_images)
            if hps.loss_type == 'dice':
                supervised_output = F.softmax(supervised_output, dim=1)
                loss_s = criterion(supervised_output, labels, reduction=hps.loss_reduction)
            else:
                loss_s = criterion(supervised_output, labels)
            
            loss_s = loss_s / hps.zero_gradients_every
            loss_s.backward()

            # Prepare CutMix images and masks
            with torch.no_grad():
                upper = images[0:images.shape[0] // 2]
                lower = images[(images.shape[0] // 2):images.shape[0]]

                upper_erased = torch.empty_like(upper)
                masks = torch.empty_like(upper)
                for j, im in enumerate(upper):
                    im_np = im
                    erased, mask = rand_aug.CutoutR(to_pil(im_np), random.randrange(hps.cut_mix_lower_bound, hps.cut_mix_upper_bound) / 100.0)
                    mask_as_tensor = to_tensor(mask)
                    upper_erased[j], masks[j] = to_tensor(erased), mask_as_tensor

                mixed = (lower * masks) + upper_erased
                masks = masks[:,0,:,:].unsqueeze(1)

                torchvision.utils.save_image(mixed, f'../output/tasks/{hps.experiment_name}/mixed.png')
                torchvision.utils.save_image(upper, f'../output/tasks/{hps.experiment_name}/upper.png')
                torchvision.utils.save_image(upper_erased, f'../output/tasks/{hps.experiment_name}/upper_erased.png')

                combined = mixed

            # shuffle inputs
            if hps.shuffle:
                perm = torch.randperm(combined.shape[0])
                combined = combined[perm]
        
            summary_idx = i + (hps.N_iter * epoch)

            data_time = time.time() - data_start

            student_output = net(combined) # forward pass

            if hps.shuffle:
                student_output = student_output[torch.argsort(perm)] # deshuffle inputs

            with torch.no_grad():
                if hasattr(ema_net, 'ema'):
                    upper_seg = ema_net.ema(upper).detach()
                    lower_seg = ema_net.ema(lower).detach()
                elif hasattr(ema_net, 'target_net'):
                    upper_seg = ema_net.target_net(upper).detach()
                    lower_seg = ema_net.target_net(lower).detach()
                else:
                    upper_seg = ema_net(upper).detach()
                    lower_seg = ema_net(lower).detach()
            
            mixed_seg = student_output
            mixed_seg_post = (lower_seg * masks) + upper_seg * (1 - masks)

            # pseudo labels per pixel.
            mixed_seg_post = mixed_seg_post.detach()
            mixed_seg_post = F.softmax(mixed_seg_post, dim=1)
            pseudo_values, pseudo_seg = torch.max(mixed_seg_post, dim=1)
            pseudo_mask = torch.ge(pseudo_values, hps.tau)

            if not hps.per_pixel_confidence:
                pseudo_mask = pseudo_mask.float()
                pseudo_mask = pseudo_mask.mean() # As in the original paper's confidence masking https://ueaeprints.uea.ac.uk/id/eprint/78164/1/1906.01916.pdf
                writer.add_scalar(f'semi/train/mean_confidence_mask', pseudo_mask.item(), summary_idx)

            # Loss:
            if hps.N_ramp_up_consistency != -1:
                current_lambd = gen_utils.calc_alpha(start_alpha=0, target_alpha=hps.lambd, training_step=np.min([i + (hps.N_iter*epoch), hps.N_iter*hps.N_ramp_up_consistency]), training_steps=hps.N_ramp_up_consistency*hps.N_iter)
            else:
                current_lambd = hps.lambd
            
            if hps.consistency_loss_type == 'cross_entropy':
                loss_c = F.cross_entropy(mixed_seg, pseudo_seg, reduction='none')
                loss_c = loss_c * pseudo_mask
                loss_c = loss_c.mean()
            elif hps.consistency_loss_type == 'focal':
                loss_c = focal_loss_function(
                    input=mixed_seg,
                    target=pseudo_seg,
                    reduction='none',
                    xe_reduction='none',
                    alpha=focal_alphas,
                    gamma=hps.consistency_focal_gamma
                )
                loss_c = loss_c * pseudo_mask
                loss_c = loss_c.mean()
            elif hps.consistency_loss_type == 'mse':
                loss_c = F.mse_loss(F.softmax(mixed_seg, dim=1), mixed_seg_post, reduction='mean')
            elif hps.consistency_loss_type == 'mse-confidence':
                loss_c = F.mse_loss(F.softmax(mixed_seg, dim=1), mixed_seg_post, reduction='none')
                loss_c = loss_c.sum(dim=1) * pseudo_mask
                loss_c = loss_c.mean()
            elif hps.consistency_loss_type == 'mse-manual-confidence':
                difference = F.softmax(mixed_seg, dim=1) - mixed_seg_post
                loss_c = (difference * difference) # Mask as in the original implementation
                loss_c = loss_c.sum(dim=1) * pseudo_mask
                loss_c = loss_c.mean()
            elif hps.consistency_loss_type == 'dice':
                pseudo_seg = pseudo_seg * pseudo_mask
                loss_c = multiclass_dice_loss(F.softmax(mixed_seg, dim=1), pseudo_seg, reduction=hps.consistency_dice_reduction)
            elif hps.consistency_loss_type == 'dice-mse':
                loss_mse = F.mse_loss(F.softmax(mixed_seg, dim=1), mixed_seg_post, reduction='mean')
                loss_dice = multiclass_dice_loss(F.softmax(mixed_seg, dim=1), pseudo_seg, reduction=hps.consistency_dice_reduction)
                loss_c = loss_dice + loss_mse
            elif hps.consistency_loss_type == 'cross_dice':
                loss_xe = F.cross_entropy(mixed_seg, pseudo_seg, reduction='none')
                loss_xe = loss_xe * pseudo_mask
                loss_xe = loss_xe.mean()
                
                loss_dice = multiclass_dice_loss(F.softmax(mixed_seg, dim=1), pseudo_seg, reduction=hps.consistency_dice_reduction)
                loss_c = loss_xe + loss_dice
            else:
                raise Exception(f'Consistency loss {hps.consistency_loss_type} not implemented.')

            loss_c = loss_c / hps.zero_gradients_every
            loss_c = current_lambd * loss_c
            loss_c.backward()

            loss = loss_s + loss_c

            # Step optimizer:
            accumulated_loss += loss.item()
            if (i + 1) % hps.zero_gradients_every == 0:
                optimizer.step()
                optimizer.zero_grad()
                writer.add_scalar(f'semi/train/accumulated_loss', accumulated_loss, summary_idx)
                accumulated_loss = 0

            total_loss += loss.item()

            if hasattr(ema_net, 'target_net'):
                ema_net.step(hps.ema_alpha)
            else:
                gen_utils.ema(net, ema_net, hps.ema_alpha)

            with torch.no_grad():
                if i % 10 == 0:
                    p_bar.set_description('Loss={:.2f}. Loss_s={:.2f}. Loss_c={:.2f} Data time={:.2f}s. Full iteration time={:.2f}s'.format(
                        loss.item(),
                        loss_s.item(),
                        loss_c.item(),
                        data_time,
                        time.time() - data_start))

                    writer.add_scalar(f'semi/train/loss', loss.item(), summary_idx)
                    writer.add_scalar(f'semi/train/loss_s', loss_s.item(), summary_idx)
                    writer.add_scalar(f'semi/train/loss_c', loss_c.item(), summary_idx)
                    
                    writer.add_scalar(f'semi/train/lambda', current_lambd, summary_idx)

                    # if not already applied:
                    if hps.loss_type != 'dice':
                        supervised_output = F.softmax(supervised_output, dim=1)
                    
                    dice = (1 - multiclass_dice_loss(supervised_output, labels))
                    iou = (1 - intersection_over_union_loss(supervised_output, labels))

                    writer.add_scalar(f'semi/train/dice_score', dice.item(), summary_idx)
                    writer.add_scalar(f'semi/train/iou_score', iou.item(), summary_idx)

                    pseudo_seg_as_image = labels_to_rgb_batched(pseudo_seg, color_map).permute(0,3,1,2)
                    mixed_seg = torch.argmax(mixed_seg, dim=1)
                    mixed_seg_as_image = labels_to_rgb_batched(mixed_seg, color_map).permute(0,3,1,2)
                    supervised_seg_as_image = labels_to_rgb_batched(torch.argmax(supervised_output, dim=1), color_map).permute(0,3,1,2)
                    supervised_label_as_image = labels_to_rgb_batched(labels, color_map).permute(0,3,1,2)
                   
                    if mixed.shape[1] == 1:
                        mixed_as_image = mixed.repeat(1,3,1,1)
                        s_images_as_image = s_images.repeat(1,3,1,1)
                    else:
                        mixed_as_image = mixed
                        s_images_as_image = s_images

                    utils.save_overlay_grid(mixed_as_image, mixed_seg_as_image, pseudo_seg_as_image, f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}_{hps.fold}_output.png')
                    utils.save_overlay_grid(s_images_as_image, supervised_seg_as_image, supervised_label_as_image, f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}_{hps.fold}_supervised_output.png')
                    
        scheduler.step()
        writer.add_scalar(f'semi/train/lr', scheduler.get_last_lr()[0], epoch)
        
        validation_metrics = gen_utils.test(net, validation_loader, color_map, n_classes=hps.n_classes, visualization_path=f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}_{hps.fold}_semi_train', thresholding_enabled=False)
        gen_utils.extract_to_writer(writer, validation_metrics, prefix='semi/validation', write_index=epoch)
        
        if hasattr(ema_net, 'target_net'):
            ema_validation_metrics = gen_utils.test(ema_net.target_net, validation_loader, color_map, n_classes=hps.n_classes, visualization_path=f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}_{hps.fold}_ema_train', thresholding_enabled=False)
        else:
            ema_validation_metrics = gen_utils.test(ema_net, validation_loader, color_map, n_classes=hps.n_classes, visualization_path=f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}_{hps.fold}_ema_train', thresholding_enabled=False)
        
        gen_utils.extract_to_writer(writer, ema_validation_metrics, prefix='semi/validation/ema', write_index=epoch)

        print(f'Total loss for epoch {epoch} was {total_loss}')
        writer.add_scalar(f'semi/train/total_loss', total_loss, epoch)

        last_epoch = epoch

        save_args = {
            'net': net,
            'optimizer': optimizer,
            'last_epoch': last_epoch,
            'test_metrics': validation_metrics,
            'lr': scheduler.get_last_lr()[0],
            'save_base_path': save_base_path,
            'hps': hps,
            'mode': f'_checkpoint_{epoch}',
            'date': date
        }

        save_ema_args = save_args.copy()
        save_ema_args['net'] = ema_net
        save_ema_args['test_metrics'] = ema_validation_metrics
        save_ema_args['mode'] = f'_checkpoint_ema_{epoch}'

        if hasattr(ema_net, 'target_net'):
            save_ema_args['net'] = ema_net.target_net
        if epoch % 50 == 0 and not hps.skip_checkpoint:
            gen_utils.save(**save_args)
            gen_utils.save(**save_ema_args)

        gen_utils.save_best_ema(save_args, save_ema_args)
        print('----------------------------------------------')
        
    writer.close()
    print(f'Training done ({hps.epochs} epochs).')
    print('===============================================')
    return last_epoch

if __name__ == '__main__':
    main()
