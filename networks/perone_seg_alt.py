
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from dataset.mnist import create_dataloaders_from_path
from dataset.ct import create_all_dataloaders_folded
import unet2D.utils as unet_utils
from unet2D.unet import Unet2D
import fix_match.utils as fm_utils
import torch.nn.functional as F
from unet2D.loss import intersection_over_union_loss, multiclass_dice_loss
from tqdm import tqdm
import utils as gen_utils
from focal_loss_alt.focal_loss import FocalLoss, focal_loss_function
from byol_segmentation.byol_segmentation import Transformation
import random
from utils import test
import mean_teacher.utils as mt_utils
import random
from torch.utils.tensorboard import SummaryWriter
from unet2D.utils import labels_to_rgb_batched
import time
from augmentation.additive_gaussian_noise import AdditiveGaussianNoise, AdditiveGaussianNoiseTransform
import argparse
from datetime import datetime
import arguments
from utils import ExperimentType, OptimizerType, AugmentationType

def initialise_arguments():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('-m', '--model', type=str, help="Base path to model")
    args_parser.add_argument('-r', '--relu-type', type=str, help="ReLU type used when training: normal or leaky")
    args_parser.add_argument('-n', '--experiment-name', type=str, default='mt_seg', help="Name of experiment. E.g. baseline")
    args_parser.add_argument('-lt', '--loss-type', type=str, help="Loss function type")
    args_parser.add_argument('--loss-reduction', type=str, default='no-bg', help="Loss function reduction")
    args_parser.add_argument('-ct', '--consistency-loss-type', default='cross_dice', type=str, help="Consistency loss function type")
    args_parser.add_argument('-dr', '--consistency-dice-reduction', default='mean', type=str, help="Dice reduction type for consistency loss. Required if consistency loss type is Dice loss.")
    args_parser.add_argument('-el', '--experiment-log-prefix', type=str, default='', help="prefix added to logs")
    args_parser.add_argument('-e', '--epochs', type=int, default=150, help="Number of epochs.")
    args_parser.add_argument('-nu', '--N-unsupervised', type=int, default=1000, help="Number of unsupervised samples.")
    args_parser.add_argument('-ns', '--N-supervised-mnist', type=int, default=10, help="Number of supervised samples in MNIST case.")
    args_parser.add_argument('-f', '--fold', type=int, default=0, help="Fold index.")
    args_parser.add_argument('-c', '--n-classes', type=int, default=6, help="Number of epochs.")
    args_parser.add_argument('-ch', '--n-channels', type=int, default=1, help="Number of channels in input image.")
    args_parser.add_argument('--N-ramp-up', type=int, default=10, help="Ramp up epochs for EMA alpha.")
    args_parser.add_argument('--N-ramp-up-consistency', type=int, default=80, help="Ramp up epochs for consistency loss weight.")
    args_parser.add_argument('--lambd', type=float, default=2.9, help="Consistency loss weight.")
    args_parser.add_argument('--seed', type=int, default=123, help="Seed.")
    args_parser.add_argument('--ema-alpha', type=float, default=0.999, help="EMA alpha.")
    args_parser.add_argument('--lr', type=float, default=3e-3, help="Learning rate.")
    args_parser.add_argument('--start-alpha', type=float, default=0.99, help="EMA start alpha (before ramp up).")
    args_parser.add_argument('-u', '--unsupervised-training-path', type=str, default='E:\\kaggle\\rsna-intracranial-hemorrhage-detection\\nifti_down_sampled\\', help="Path to unsupervised training data")
    args_parser.add_argument('--batch-size-s', type=int, default=4, help="Supervised batch size.")
    args_parser.add_argument('--batch-size-u', type=int, default=2, help="Unsupervised batch size.")
    args_parser.add_argument('--batch-size-t', type=int, default=1, help="Test/validation batch size.")
    args_parser.add_argument('--focal-gamma', type=float, default=2, help="Gamma of focal loss.")
    args_parser.add_argument('--consistency-focal-gamma', type=float, default=0.5, help="Gamma of focal loss.")
    args_parser.add_argument('--std', type=float, default=0.1, help="Gaussian noise standard deviation")
    args_parser.add_argument('--experiment-type', type=ExperimentType, default='ct', help="ct or mnist", choices=list(ExperimentType))
    args_parser.add_argument('--N-iter', type=int, default=512, help="N iterations in epoch")
    args_parser.add_argument('--use-dropout', type=arguments.bool_arg, default=False, help="Should use dropout in model")
    args_parser.add_argument('-a', '--base-augment-train', type=arguments.bool_arg, default=False, help="Should use augment train data automatically in data loader")
    args_parser.add_argument('--shuffle', type=arguments.bool_arg, default=False, help="Should shuffle model inputs prior to forward pass")
    args_parser.add_argument('--skip-checkpoint', type=arguments.bool_arg, default=False, help="Should skip saving checkpoints")
    args_parser.add_argument('--optimizer', type=OptimizerType, default='adam', help="sgd or adam", choices=list(OptimizerType))
    args_parser.add_argument('--confidence-threshold', type=float, default=0, help="Confidence threshold [0,1]")
    args_parser.add_argument('--legacy-unet', type=arguments.bool_arg, default=True, help="Should use legacy U-net, or latest version")
    args_parser.add_argument('--always-include-consistency', type=arguments.bool_arg, default=True, help="Even if consistency weight is 0.0, should still calculate the unsupervised path.")
    args_parser.add_argument('-z', '--zero-gradients-every', type=int, default=1, help="How often to zero gradients")
    args_parser.add_argument('--consistency-augmentation-type', type=AugmentationType, default='mild', help="Consistency augmentation type", choices=list(AugmentationType))
    hps = args_parser.parse_args()
    return hps

num_threads = 1
num_data_loader_workers = 1
date = None

sizes = [
    0.8,
    0.1,
    0.1
]

def train(
        net,
        ema_net,
        transformation_network,
        unsupervised_loader,
        supervised_loader,
        validation_loader,
        optimizer,
        criterion,
        scheduler,
        writer,
        hps,
        last_epoch,
        color_map,
        device
    ):

    unsupervised_iter = iter(unsupervised_loader)
    supervised_iter = iter(supervised_loader)

    gaussian_noise_module = AdditiveGaussianNoise(mean=0, std=hps.std)

    for epoch in range(last_epoch, hps.epochs):
        total_loss = 0

        data_start = time.time()
        p_bar = tqdm(range(hps.N_iter))

        net.train()
        accumulated_loss = 0
        for i in p_bar:

            images, _ = unsupervised_iter.next()
            s_images, labels = supervised_iter.next()
            images = images.to(device)
            s_images = s_images.to(device)
            labels = labels.to(device)

            labels = labels.squeeze(1)

            if images.shape[1] == 1:
                images = images.repeat(1, hps.n_channels, 1, 1)
                s_images = s_images.repeat(1, hps.n_channels, 1, 1)
            
            # Supervised loss:
            supervised_output = net(s_images)
            if hps.loss_type == 'dice':
                supervised_output = F.softmax(supervised_output, dim=1)
                loss_s = criterion(supervised_output, labels, reduction=hps.loss_reduction)
            else:
                loss_s = criterion(supervised_output, labels)

            loss_s = loss_s / hps.zero_gradients_every
            loss_s.backward()

            summary_idx = i + (hps.N_iter * epoch)
            if hps.lambd > 0 or hps.always_include_consistency:

                student_images, transformations, _ = transformation_network(images.clone())
                teacher_images = gaussian_noise_module(images.clone())

                combined = student_images
                with torch.no_grad():
                    combined = gaussian_noise_module(combined)

                # shuffle inputs
                if hps.shuffle:
                    perm = torch.randperm(combined.shape[0])
                    combined = combined[perm]
            
                data_time = time.time() - data_start
            
                student_output = net(combined) # forward pass

                if hps.shuffle:
                    student_output = student_output[torch.argsort(perm)] # deshuffle inputs

                with torch.no_grad():
                    if hasattr(ema_net, 'ema'):
                        teacher_output = ema_net.ema(teacher_images).detach()
                    elif hasattr(ema_net, 'target_net'):
                        teacher_output = ema_net.target_net(teacher_images).detach()
                    else:
                        teacher_output = ema_net(teacher_images).detach()
                
                # Align segmentation maps
                with torch.no_grad():
                    teacher_output = teacher_output.detach()
                    teacher_max_values, teacher_output = torch.max(teacher_output, dim=1)
                    teacher_output_pre = teacher_output.clone()
                    teacher_output, _, _ = transformation_network(teacher_output, transformations, is_mask=True)
                    teacher_output = teacher_output.long()
                    teacher_max_values, _, _ = transformation_network(teacher_max_values, transformations, is_mask=False) # Transform confidence map

                if i % (hps.N_iter // 2) == 0:
                    teacher_output_pre_as_image = labels_to_rgb_batched(teacher_output_pre, color_map).permute(0,3,1,2)
                    teacher_output_as_image = labels_to_rgb_batched(teacher_output, color_map).permute(0,3,1,2)

                    if images.shape[1] == 1:
                        student_images = student_images.repeat(1,3,1,1)

                    unet_utils.save_all_grid(student_images, teacher_output_as_image, teacher_output_pre_as_image, f'../output/tasks/{hps.experiment_name}/transformed_teacher_output.png')

                    del teacher_output_pre_as_image
                    del teacher_output_as_image

                if hps.confidence_threshold > 0:
                    pseudo_mask = torch.ge(teacher_max_values, hps.confidence_threshold)
                    teacher_output[pseudo_mask] = -1

                # Consistency loss
                if hps.N_ramp_up_consistency != -1:
                    current_lambd = (np.min([i + (hps.N_iter*epoch), hps.N_ramp_up_consistency*hps.N_iter])/(hps.N_ramp_up_consistency*hps.N_iter))*hps.lambd
                else:
                    current_lambd = hps.lambd

                if hps.consistency_loss_type == 'cross_entropy':
                    loss_c = F.cross_entropy(student_output, teacher_output, reduction='mean', ignore_index=-1)
                elif hps.consistency_loss_type == 'focal':
                    loss_c = focal_loss_function(
                        input=student_output,
                        target=teacher_output,
                        reduction=hps.consistency_dice_reduction,
                        xe_reduction=hps.consistency_dice_reduction,
                        alpha=focal_alphas,
                        gamma=hps.consistency_focal_gamma
                    )
                elif hps.consistency_loss_type == 'dice':
                    loss_c = multiclass_dice_loss(student_output, teacher_output, reduction=hps.consistency_dice_reduction, ignore_index=-1)
                elif hps.consistency_loss_type == 'cross_dice':
                    loss_xe = F.cross_entropy(student_output, teacher_output, reduction='mean', ignore_index=-1)
                    loss_dice = multiclass_dice_loss(student_output, teacher_output, reduction=hps.consistency_dice_reduction, ignore_index=-1)
                    loss_c = loss_xe + loss_dice
                else:
                    raise Exception(f'Consistency type not defined: {hps.consistency_loss_type}')

                loss_c = current_lambd * loss_c
                loss_c = loss_c / hps.zero_gradients_every
                loss_c.backward()
            else:
                loss_c = torch.zeros_like(loss_s)
                data_time = time.time() - data_start
                current_lambd = 0

            with torch.no_grad():
                loss = loss_s + loss_c

            # Calculate gradient and update model:
            accumulated_loss += loss.item()
            if (i + 1) % hps.zero_gradients_every == 0:
                optimizer.step()
                optimizer.zero_grad()
                writer.add_scalar('semi/train/accumulated_loss', accumulated_loss, summary_idx)
                accumulated_loss = 0
            total_loss += loss.item()

            # EMA update:
            if hps.N_ramp_up != -1:
                alpha = gen_utils.calc_alpha(start_alpha=hps.start_alpha, target_alpha=hps.ema_alpha, training_step=np.min([i + (hps.N_iter*epoch), hps.N_iter*hps.N_ramp_up]), training_steps=hps.N_iter*hps.N_ramp_up)
            else:
                alpha = hps.ema_alpha

            if hasattr(ema_net, 'target_net'):
                ema_net.step(alpha)
            else:
                gen_utils.ema(net, ema_net, alpha)

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
                    writer.add_scalar(f'semi/train/lambd', current_lambd, summary_idx)
                    writer.add_scalar(f'semi/train/alpha', alpha, summary_idx)

                    dice = (1 - multiclass_dice_loss(supervised_output, labels))
                    iou = (1 - intersection_over_union_loss(supervised_output, labels))

                    writer.add_scalar(f'semi/train/dice_score', dice.item(), summary_idx)
                    writer.add_scalar(f'semi/train/iou_score', iou.item(), summary_idx)

                    supervised_seg_as_image = labels_to_rgb_batched(torch.argmax(supervised_output, dim=1), color_map).permute(0,3,1,2)
                    supervised_label_as_image = labels_to_rgb_batched(labels, color_map).permute(0,3,1,2)
                    
                    if s_images.shape[1] == 1:
                        s_images_as_image = s_images.repeat(1,3,1,1)
                    else:
                        s_images_as_image = s_images

                    unet_utils.save_overlay_grid(s_images_as_image, supervised_seg_as_image, supervised_label_as_image, f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}supervised_output.png')

                    if hps.lambd > 0 or hps.always_include_consistency:
                        teacher_output = teacher_output.clip(0, hps.n_classes - 1)
                        student_output = student_output.clip(0, hps.n_classes - 1)
                        
                        teacher_output_pre_as_image = labels_to_rgb_batched(teacher_output_pre, color_map).permute(0,3,1,2)
                        teacher_output_as_image = labels_to_rgb_batched(teacher_output, color_map).permute(0,3,1,2)
                        student_output_as_image = labels_to_rgb_batched(torch.argmax(student_output, dim=1), color_map).permute(0,3,1,2)

                        if student_images.shape[1] == 1:
                            student_images = student_images.repeat(1,3,1,1)

                        unet_utils.save_overlay_grid(student_images, student_output_as_image, teacher_output_as_image, f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}unsupervised_output.png')
                    
        scheduler.step()
        writer.add_scalar(f'semi/train/lr', scheduler.get_last_lr()[0], epoch)

        # Validate:
        validation_metrics = test(net, validation_loader, n_classes=hps.n_classes, color_map=color_map, visualization_path=f'../output/tasks/{hps.experiment_name}/semi_train', thresholding_enabled=False)
        gen_utils.extract_to_writer(writer, validation_metrics, 'semi/validation/', epoch)

        if hasattr(ema_net, 'target_net'):
            ema_validation_metrics = test(ema_net.target_net, validation_loader, n_classes=hps.n_classes, color_map=color_map, visualization_path=f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}ema_train_', thresholding_enabled=False)
        else:
            ema_validation_metrics = test(ema_net, validation_loader, n_classes=hps.n_classes, color_map=color_map, visualization_path=f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}ema_train_', thresholding_enabled=False)

        gen_utils.extract_to_writer(writer, ema_validation_metrics, prefix='semi/validation/ema', write_index=epoch)

        last_epoch = epoch
        
        # Save checkpoints:
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
            
        if epoch % 30 == 0 and not hps.skip_checkpoint:
            gen_utils.save(**save_args)
            gen_utils.save(**save_ema_args)

        gen_utils.save_best_ema(save_args, save_ema_args)

        print(f'Total loss for epoch {epoch} was {total_loss}')
        print('----------------------------------------------')
    writer.close()
    print(f'Training done ({hps.epochs} epochs).')
    print('===============================================')
    return last_epoch

def main():
    hps = initialise_arguments()
    torch.cuda.empty_cache()
    run_experiment(hps)

def run_experiment(hps):
    print(f"Using fold {hps.fold}...")
    print(f'Hyperparameters: {hps}')
    device = torch.device('cuda:0')
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)
    random.seed(hps.seed)
    torch.backends.cudnn.benchmark=False

    transform_u = transforms.Compose([])

    transform_s = transforms.Compose([
        AdditiveGaussianNoiseTransform(mean=0.0, std=hps.std)
    ])

    transform_t = transforms.Compose([])

    resize = 0.5
    start_time = time.time()

    if hps.experiment_type == ExperimentType.CT:
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
            resize=resize,
            transform_u=transform_u,
            transform_s=transform_s,
            transform_t=transform_t,
            n_classes=hps.n_classes,
            image_path='../data/ct_scans/',
            u_image_path=hps.unsupervised_training_path,
            mask_path='../data/multiclass_mask/',
            base_augment_train=hps.base_augment_train)

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

    global save_base_path
    save_base_path = f'../experiment_runs/{hps.experiment_name}/models/{hps.fold}_{hps.experiment_log_prefix}_{hps.seed}'


    color_map = gen_utils.get_color_map(hps.n_classes).to(device)

    semi_net = Unet2D(n_input_channels=hps.n_channels, n_classes=hps.n_classes, use_dropout=hps.use_dropout, legacy=hps.legacy_unet).to(device)
    gen_utils.initialize_model(semi_net)

    ema_model_internal = Unet2D(n_input_channels=hps.n_channels, n_classes=hps.n_classes, use_dropout=hps.use_dropout, legacy=hps.legacy_unet).to(device)
    ema_net = mt_utils.EMAWeightOptimizer(target_net=ema_model_internal, source_net=semi_net)

    if hps.experiment_type == ExperimentType.MNIST:
        translation = (2,2) 
        transform_dict = {
            'rot': 3,
            'translation': translation,
            'scale': (0.98,1.02),
            'flip_probability': 0.5,
            'shear_angle': 2,
        }
    elif hps.experiment_type == ExperimentType.CT:

        if hps.consistency_augmentation_type == AugmentationType.MILD:
            translation = (5,5) 
            transform_dict = {
                'rot': 3,
                'translation': translation,
                'scale': (0.98,1.02),
                'flip_probability': 0.5,
                'shear_angle': 2,
            }
        elif hps.consistency_augmentation_type == AugmentationType.FULL:
            width = 512 * resize
            translation = (width*0.1, width*0.1)
            transform_dict = {
                'rot': 15,
                'translation': translation,
                'scale': (0.9,1.1),
                'flip_probability': 0.5,
                'shear_angle': 5,
            }
        elif hps.consistency_augmentation_type == AugmentationType.EXTREME:
            width = 512 * resize
            translation = (width*0.1, width*0.1)
            transform_dict = {
                'rot': 30,
                'translation': translation,
                'scale': (0.8,1.2),
                'flip_probability': 0.5,
                'shear_angle': 10,
            }
        else:
            raise Exception(f'Augmentation type {hps.consistency_augmentation_type} not implemented.') 
    else:
        raise Exception(f'Experiment type {hps.experiment_type} not implemented.') 

    transformation_network = Transformation(**transform_dict)

    if 'focal' in (hps.loss_type, hps.consistency_loss_type):
        global focal_alphas
        focal_alphas = gen_utils.get_inverse_frequencies(supervised_loader, device, hps.n_classes)
    
    if hps.loss_type == 'dice':
        criterion = multiclass_dice_loss
    elif hps.loss_type == 'focal':
        criterion = FocalLoss(gamma=hps.focal_gamma, alpha=focal_alphas, size_average=True)
    elif hps.loss_type == 'cross-entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception(f'Loss not defined: {hps.loss_type}')

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
    
    if hps.optimizer == OptimizerType.SGD:
        weight_decay=4e-5
        momentum=0.9
        optimizer = torch.optim.SGD(semi_net.parameters(), lr=hps.lr, weight_decay=weight_decay, momentum=momentum)
    elif hps.optimizer == OptimizerType.ADAM:
        weight_decay=0.0006
        filtered_parameters = gen_utils.filter_decay_params(semi_net, weight_decay)
        optimizer = torch.optim.Adam(filtered_parameters, lr=hps.lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    else:
        raise Exception(f'Optimizer {hps.optimizer} not supported.')

    scheduler = fm_utils.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=hps.epochs)

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
        transformation_network=transformation_network,
        unsupervised_loader=unsupervised_loader_inf,
        supervised_loader=supervised_loader_inf,
        validation_loader=validation_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        writer=writer,
        last_epoch=last_epoch,
        hps=hps,
        color_map=color_map,
        device=device
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

    print("Training done. Starting testing...")

    ema_net = ema_net.target_net if hasattr(ema_net, 'target_net') else ema_net
    ema_test_metrics = gen_utils.test(
        net=ema_net,
        test_loader=test_loader,
        color_map=color_map,
        visualize_every=1,
        n_classes=hps.n_classes,
        visualization_path=f'../output/tasks/{hps.experiment_name}/{hps.fold}_full')

    gen_utils.extract_to_writer(writer, ema_test_metrics, prefix='supervised/test/ema', write_index=hps.fold)
    
    print("Testing done. Saving model...")
    gen_utils.save(semi_net, optimizer, last_epoch, test_metrics, scheduler.get_last_lr()[0], save_base_path, hps=hps, mode='_full_semi', date=date)
    gen_utils.save(ema_net, optimizer, last_epoch, test_metrics, scheduler.get_last_lr()[0], save_base_path, hps=hps, mode='_full_ema', date=date)
    
    print(f'Fold {hps.fold} finished. Metrics: {test_metrics}')

if __name__ == '__main__':
    main()
