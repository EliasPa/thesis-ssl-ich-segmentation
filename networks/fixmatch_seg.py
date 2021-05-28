import torch
import torch.nn as nn
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
from utils import ExperimentType, FixMatchType, OptimizerType, AugmentationType
import fixmatch_reg_train
from byol_segmentation.byol_segmentation import Transformation
from augmentation.additive_gaussian_noise import AdditiveGaussianNoiseTransform
import arguments

def initialise_arguments():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('-m', '--model', type=str, help="Base path to model")
    args_parser.add_argument('-n', '--experiment-name', type=str, default='fix_match_seg', help="Name of experiment. E.g. baseline")
    args_parser.add_argument('-r', '--relu-type', type=str, help="ReLU type used when training: normal or leaky")
    args_parser.add_argument('-lt', '--loss-type', type=str, default='multiclass_dice_sum', help="Type of loss function used for training")
    args_parser.add_argument('--loss-reduction', type=str, default='no-bg', help="Loss function reduction")
    args_parser.add_argument('-c', '--n-classes', type=int, default=6, help="Number of classes to predict.")
    args_parser.add_argument('-ch', '--n-channels', type=int, default=1, help="Number of channels in input image.")
    args_parser.add_argument('-et', '--experiment-type', type=ExperimentType, default='ct', help="ct or cityscapes", choices=list(ExperimentType))
    args_parser.add_argument('-fm', '--fix-match-type', type=FixMatchType, default='cutmix', help="cutmix or regular", choices=list(FixMatchType))
    args_parser.add_argument('-el', '--experiment-log-prefix', type=str, default='', help="prefix added to logs")
    args_parser.add_argument('--tau', type=float, default=0.95, help="Confidence threshold [0,1]")
    args_parser.add_argument('-ct', '--consistency-loss-type', default='mse-confidence', type=str, help="Consistency loss function type")
    args_parser.add_argument('-dr', '--consistency-dice-reduction', default='no-bg', type=str, help="Dice reduction type for consistency loss. Required if consistency loss type is Dice loss.")
    args_parser.add_argument('-u', '--unsupervised-training-path', type=str, default='E:\\kaggle\\rsna-intracranial-hemorrhage-detection\\nifti_down_sampled\\', help="Path to unsupervised training data")
    args_parser.add_argument('--N-ramp-up-consistency', type=int, default=-1, help="Ramp up epochs for consistency loss weight.")
    args_parser.add_argument('--cut-mix-lower-bound', type=int, default=40, help="CutMix lower bound e.g. (40%)")
    args_parser.add_argument('--cut-mix-upper-bound', type=int, default=50, help="CutMix upper bound e.g. (50%)")
    args_parser.add_argument('-e', '--epochs', type=int, default=100, help="Number of epochs.")
    args_parser.add_argument('--seed', type=int, default=123, help="Seed.")
    args_parser.add_argument('-f', '--fold', type=int, default=0, help="Fold index")
    args_parser.add_argument('--lambd', type=float, default=1.0, help="Consistency loss weight.")
    args_parser.add_argument('--ema-alpha', type=float, default=0.99, help="EMA alpha.")
    args_parser.add_argument('--lr', type=float, default=3e-5, help="Learning rate.")
    args_parser.add_argument('--batch-size-s', type=int, default=2, help="Supervised batch size.")
    args_parser.add_argument('--batch-size-u', type=int, default=4, help="Unsupervised batch size.")
    args_parser.add_argument('--batch-size-t', type=int, default=2, help="Test/validation batch size.")
    args_parser.add_argument('-nu', '--N-unsupervised', type=int, default=1000, help="Number of epochs.")
    args_parser.add_argument('-ns', '--N-supervised-mnist', type=int, default=10, help="Number of supervised samples in MNIST case.")
    args_parser.add_argument('--N-iter', type=int, default=512, help="N iterations in epoch")
    args_parser.add_argument('--focal-gamma', type=float, default=2, help="Gamma of focal loss.")
    args_parser.add_argument('--consistency-focal-gamma', type=float, default=0.5, help="Gamma of focal loss.")
    args_parser.add_argument('--ra-n', type=int, default=2, help="N of RandAugment.")
    args_parser.add_argument('--ra-m-max', type=int, default=30, help="Max M for RandAugment.")
    args_parser.add_argument('--std', type=float, default=0.01, help="Gaussian noise standard deviation")
    args_parser.add_argument('--use-dropout', type=arguments.bool_arg, default=False, help="Should use dropout in model")
    args_parser.add_argument('--shuffle', type=arguments.bool_arg, default=False, help="Should shuffle model inputs prior to forward pass")
    args_parser.add_argument('--optimizer', type=OptimizerType, default='adam', help="sgd or adam", choices=list(OptimizerType))
    args_parser.add_argument('-at', '--augmentation-type', type=AugmentationType, default='mild', help="Augmentation type", choices=list(AugmentationType))
    args_parser.add_argument('--skip-checkpoint', type=arguments.bool_arg, default=False, help="Should skip saving checkpoints")
    args_parser.add_argument('--legacy-unet', type=arguments.bool_arg, default=True, help="Should use legacy U-net, or latest version")
    args_parser.add_argument('-z', '--zero-gradients-every', type=int, default=1, help="How often to zero gradients")   
    args_parser.add_argument('--consistency-augmentation-type', type=AugmentationType, default='mild', help="Consistency augmentation type", choices=list(AugmentationType))
    hps = args_parser.parse_args()
    return hps

num_threads = 1
num_data_loader_workers = 1
device = torch.device('cuda:0')
date = None

focal_alphas = None
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
    print(hps)
    print(f"Using fold {hps.fold}...")
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)
    random.seed(hps.seed)
    torch.backends.cudnn.benchmark=False

    start_time = time.time()
    
    transform_u = transforms.Compose([])

    transform_s = transforms.Compose([
        AdditiveGaussianNoiseTransform(mean=0.0, std=hps.std)
    ])

    transform_t = transforms.Compose([])
    resize = 0.5
    if hps.experiment_type == ExperimentType.CT:

        if hps.augmentation_type == AugmentationType.NONE:
            override_augmentations = None
            base_augment_train = False
        elif hps.augmentation_type == AugmentationType.MILD:
            width = 512
            width = width * 0.5
            
            weak_tr = width * 0.1
            override_augmentations = {
                'random_rot': 0,
                'random_translation': (weak_tr, weak_tr),
                'random_scale': (1,1),
                'flip': 0.5,
                'random_shear': 0,
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
            resize=resize,
            transform_u=transform_u,
            transform_s=transform_s,
            transform_t=transform_t,
            n_classes=hps.n_classes,
            image_path='../data/ct_scans/',
            u_image_path=hps.unsupervised_training_path,
            mask_path='../data/multiclass_mask/',
            base_augment_train=base_augment_train,
            override_transforms=override_augmentations
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

    if hps.fix_match_type == FixMatchType.CUTMIX:
        raise Exception('Not implemented')
    elif hps.fix_match_type == FixMatchType.REGULAR:

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
        hps.N_iter = max(int(len(supervised_loader)), 10)
        last_epoch = fixmatch_reg_train.train(
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
            last_epoch=last_epoch,
            save_base_path=save_base_path,
            focal_alphas=focal_alphas,
            date=date,
            device=device,
            transformation_network=transformation_network
        )
    else:
        raise Exception(f'FM type {hps.fix_match_type} is not recognized.')

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

if __name__ == '__main__':
    main()
