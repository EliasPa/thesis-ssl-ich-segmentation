
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from dataset.mnist import create_dataloaders_from_path
from dataset.ct import create_all_dataloaders_folded
from unet2D import utils
from unet2D.unet import Unet2D
import fix_match.utils as fm_utils
import torch.nn.functional as F
from unet2D.loss import intersection_over_union_loss, multiclass_dice_loss
from tqdm import tqdm
import utils as gen_utils
from focal_loss_alt.focal_loss import FocalLoss
import random
from datetime import datetime
import random
from torch.utils.tensorboard import SummaryWriter
from unet2D.utils import labels_to_rgb_batched
import time
import argparse
from utils import ModelType
import arguments
from utils import ExperimentType, SchedulerType, AugmentationType, OptimizerType

def initialise_arguments():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--randomize-seed', type=bool, default=False, help="If true, will choose a new seed randomly at each fold")
    args_parser.add_argument('-r', '--relu-type', type=str, help="ReLU type used when training: normal or leaky")
    args_parser.add_argument('-m', '--model-checkpoint', type=str, help="Path to model checkpoint. If provided, continues training.")
    args_parser.add_argument('-n', '--experiment-name', type=str, default='baseline', help="Name of experiment. E.g. baseline")
    args_parser.add_argument('-el', '--experiment-log-prefix', type=str, default='', help="prefix added to logs")
    args_parser.add_argument('-e', '--epochs', type=int, default=100, help="Number of epochs.")
    args_parser.add_argument('-s', '--start-fold', type=int, default=0, help="Start fold.")
    args_parser.add_argument('-f', '--end-fold', type=int, default=5, help="End fold.")
    args_parser.add_argument('--lr', type=float, default=3e-2, help="Learning rate.")
    args_parser.add_argument('-c', '--n-classes', type=int, default=6, help="Number of epochs.")
    args_parser.add_argument('-ch', '--n-channels', type=int, default=1, help="Number of channels in input image.")
    args_parser.add_argument('--batch-size-s', type=int, default=8, help="Supervised batch size.")
    args_parser.add_argument('--batch-size-u', type=int, default=1, help="Unsupervised batch size.")
    args_parser.add_argument('--batch-size-t', type=int, default=1, help="Test/validation batch size.")
    args_parser.add_argument('-u', '--unsupervised_training_path', type=str, default='E:\\kaggle\\rsna-intracranial-hemorrhage-detection\\nifti_down_sampled\\', help="Path to unsupervised training data")
    args_parser.add_argument('--optimizer', type=OptimizerType, default='adam', help="sgd or adam", choices=list(OptimizerType))
    args_parser.add_argument('-ns', '--N-supervised-mnist', type=int, default=10, help="Number of supervised samples in MNIST case.")
    args_parser.add_argument('--use-checkpoint', type=bool, default=False, help="If true, training will continue from model saved as {save_base_path}full.pth")
    args_parser.add_argument('-lt', '--loss-type', type=str, help="Loss function type")
    args_parser.add_argument('--focal-gamma', type=float, default=2, help="Gamma of focal loss.")
    args_parser.add_argument('--loss-reduction', type=str, default='no-bg', help="Loss function reduction")
    args_parser.add_argument('--seed', type=int, default=123, help="Seed.")
    args_parser.add_argument('--model-type', type=ModelType, default='unet', help="model type, e.g. unet", choices=list(ModelType))
    args_parser.add_argument('--output-stride', type=int, default=8, help="output stride for deeplabv3+")
    args_parser.add_argument('--use-dropout', type=arguments.bool_arg, default=False, help="Should use dropout in model")
    args_parser.add_argument('--legacy-unet', type=arguments.bool_arg, default=True, help="Should use legacy U-net, or latest version")
    args_parser.add_argument('-et', '--experiment-type', type=ExperimentType, default='ct', help="ct or cityscapes", choices=list(ExperimentType))
    args_parser.add_argument('-st', '--scheduler-type', type=SchedulerType, default='cosine', help="Scheduler type", choices=list(SchedulerType))
    args_parser.add_argument('-at', '--augmentation-type', type=AugmentationType, default='full', help="Augmentation type", choices=list(AugmentationType))
    args_parser.add_argument('-z', '--zero-gradients-every', type=int, default=1, help="How often to zero gradients")
    args_parser.add_argument('--skip-checkpoint', type=arguments.bool_arg, default=True, help="Should skip saving checkpoints")
    args_parser.add_argument('--warm-restart-every', type=int, default=25, help="How often to warm restard scheduler")
    hps = args_parser.parse_args()
    return hps

N_unsupervised = 0
weight_decay=4e-5
momentum=0.9
device = torch.device('cuda:0')

num_threads = 1
batch_size_supervised = 4
save_base_path = None
date = None

sizes = []

transform_u = transforms.Compose([])

transform_s = transforms.Compose([])

transform_t = transforms.Compose([])

def save(net, optimizer, last_epoch, test_metrics, lr, hps, date, mode='semi'):
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
    
def load(path):
    state_dict = torch.load(path)
    return state_dict

def train_sup(net, loader, validation_loader, fold, writer, optimizer, scheduler, criterion, hps, color_map, last_epoch):
   
    net.train()
    for epoch in range(last_epoch, hps.epochs):
        total_loss = 0
        net.train()
        p_bar = tqdm(loader)
        accumulated_loss = 0
        for i, (images, masks) in enumerate(p_bar):
            images = images.to(device)

            if images.shape[1] == 1:
                images = images.repeat(1, hps.n_channels, 1, 1)

            masks = masks.to(device)
            masks = masks.squeeze(1)

            summary_idx = i + len(loader) * epoch

            output = net(images)
            
            if hps.loss_type == 'dice':
                output = F.softmax(output, dim=1)
                loss = criterion(output, masks, reduction=hps.loss_reduction)
            else:
                loss = criterion(output, masks)

            loss = loss / hps.zero_gradients_every
            loss.backward()
            accumulated_loss += loss.item()
            if (i+1) % hps.zero_gradients_every == 0:
                optimizer.step()
                optimizer.zero_grad()
                writer.add_scalar('supervised/train/accumulated_loss', accumulated_loss, summary_idx)
                accumulated_loss = 0
            
            total_loss += loss.item()

            with torch.no_grad():
                if i % 5 == 0:
                    
                    p_bar.set_description('Epoch {}. Iteration {}. Loss={:.2f}'.format(
                        epoch,
                        i,
                        loss.item()
                    ))
                    
                    pred_as_image = labels_to_rgb_batched(torch.argmax(output, dim=1), color_map).permute(0,3,1,2)
                    gt_as_image = labels_to_rgb_batched(masks, color_map).permute(0,3,1,2)

                    if hps.loss_type != 'dice':
                        output = F.softmax(output, dim=1)
                    
                    dice = (1 - multiclass_dice_loss(output, masks, reduction="sum").item())
                    iou = (1 - intersection_over_union_loss(output, masks).item())

                    writer.add_scalar(f'supervised/train/dice_score', dice, summary_idx)
                    writer.add_scalar(f'supervised/train/iou_score', iou, summary_idx)
                    writer.add_scalar(f'supervised/train/loss', loss.item(), summary_idx)

                    if images.shape[1] == 1:
                        images_as_image = images.repeat(1,3,1,1)
                    else:
                        images_as_image = images
                    utils.save_overlay_grid(images_as_image, pred_as_image, gt_as_image, f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}_{fold}_fully_supervised_training_output.png')
        
        scheduler.step()
        writer.add_scalar(f'supervised/train/lr', scheduler.get_last_lr()[0], epoch)

        validation_metrics = gen_utils.test(
            net=net,
            test_loader=validation_loader,
            color_map=color_map,
            n_classes=hps.n_classes,
            n_channels=hps.n_channels,
            visualization_path=f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}_{fold}_semi_train',
            thresholding_enabled=False
            )
        gen_utils.extract_to_writer(writer, validation_metrics, prefix='supervised/validation', write_index=epoch)

        writer.add_scalar(f'supervised/train/total_loss', total_loss, epoch)
        last_epoch = epoch
        
        save_args = {
            'net': net,
            'optimizer': optimizer,
            'last_epoch': last_epoch,
            'test_metrics': validation_metrics,
            'lr': scheduler.get_last_lr()[0],
            'hps': hps,
            'date': date,
            'save_base_path': save_base_path,
            'mode': f'_checkpoint_{epoch}'
        }

        if epoch % 50 == 0 and not hps.skip_checkpoint:
            gen_utils.save(**save_args)

        best_path = f'_checkpoint_best'
        gen_utils.save_best(save_args, best_path)

    writer.close()
    return last_epoch

def main():
    hps = initialise_arguments()
    torch.cuda.empty_cache()
    run_experiment(hps)

def run_experiment(hps):
    device = torch.device('cuda:0')

    print(f'Hyperparameters: {hps}')
    print(f'Training about to start with {hps.n_classes} classes...')

    for fold in range(hps.start_fold, hps.end_fold):

        print(f"Starting fold {fold}...")
    
        if hps.randomize_seed:
            global seed
            seed = random.randint(0, 2147483647)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        else:
            seed = hps.seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        start_time = time.time()

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
                N=N_unsupervised,
                n_classes=hps.n_classes,
                num_threads=num_threads,
                fold=fold,
                batch_size_s=hps.batch_size_s,
                batch_size_u=hps.batch_size_u,
                batch_size_t=hps.batch_size_t,
                sizes=sizes,
                pin_memory=True,
                resize=0.5,
                transform_u=transform_u,
                transform_s=transform_s,
                transform_t=transform_t,
                image_path='../data/ct_scans/',
                u_image_path=hps.unsupervised_training_path,
                mask_path='../data/multiclass_mask/',
                override_transforms=override_augmentations,
                base_augment_train=base_augment_train
            )

            _, supervised_loader, _, validation_loader, test_loader = loaders
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
                N_u=0,
                N_v=1000
            )
            _, supervised_loader, _, validation_loader, test_loader = loaders
        else:
            raise Exception(f'Experiment type {hps.experiment_type} not implemented.')

        print(f'Data loading took: {time.time() - start_time}s using {num_threads} threads.')
        
        color_map = gen_utils.get_color_map(hps.n_classes).to(device)
        print(f'Color map: {color_map}')
        
        global save_base_path
        save_base_path = f'../experiment_runs/{hps.experiment_name}/models/{fold}_{hps.experiment_log_prefix}_{seed}'

        if hps.model_type == ModelType.UNET:
            full_net = Unet2D(n_input_channels=hps.n_channels, n_classes=hps.n_classes, relu_type=hps.relu_type, use_dropout=hps.use_dropout, legacy=hps.legacy_unet).to(device)
            gen_utils.initialize_model(full_net)
        elif hps.model_type == ModelType.DEEPLABV3_RESNET50:
            raise Exception('Not implemented')

        if hps.loss_type == 'dice':
            criterion = multiclass_dice_loss
        elif hps.loss_type == 'focal':
            alphas = gen_utils.get_inverse_frequencies(supervised_loader, device, hps.n_classes)
            criterion = FocalLoss(gamma=hps.focal_gamma, alpha=alphas, size_average=True)
        elif hps.loss_type == 'cross-entropy':
            criterion = nn.CrossEntropyLoss()
        else:
            raise Exception(f'Loss not defined: {hps.loss_type}')
        last_epoch = 0

        optimizer = None
        if hps.use_checkpoint or hps.model_checkpoint is not None:
            print("Attempting to load previous network from checkpoint...")
            try:
                load_path = hps.model_checkpoint if hps.model_checkpoint is not None else f'{save_base_path}full.pth'
                state_dict = load(load_path)
                net = state_dict['net']
                optimizer = state_dict['optimizer']
                last_epoch = state_dict['last_epoch']
                test_metrics = state_dict['test_metrics']
                full_net.load_state_dict(net)
                print("Success.")
            except Exception as e:
                print("Failed to load previous net")
                print(e)

        if optimizer is None:
            if hps.optimizer == OptimizerType.SGD:
                optimizer = torch.optim.SGD(full_net.parameters(), lr=hps.lr, weight_decay=weight_decay, momentum=momentum)
            elif hps.optimizer == OptimizerType.ADAM:
                optimizer = torch.optim.Adam(full_net.parameters(), lr=hps.lr)
            else:
                raise Exception(f'Did not recognize optimizer type {hps.optimizer}')

        if hps.scheduler_type == SchedulerType.COSINE:
            scheduler = fm_utils.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=hps.epochs)
        elif hps.scheduler_type == SchedulerType.COSINE_WARM_RESTARTS:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=hps.warm_restart_every)
        else:
            raise Exception(f'Scheduler {hps.scheduler} not found.')

        # Step the scheduler until it reaches where training was last stopped:
        print(f"Stepping scheduler {last_epoch} times")
        for _ in tqdm(range(last_epoch)):
            scheduler.step()

        print("Starting training...")
        global date
        date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        writer = SummaryWriter(log_dir=f'../experiment_runs/{hps.experiment_name}/{hps.experiment_log_prefix}_{seed}_fold_{fold}/{date}')
        last_epoch = train_sup(
            net=full_net,
            loader=supervised_loader,
            validation_loader=validation_loader,
            fold=fold,
            writer=writer,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            last_epoch=last_epoch,
            hps=hps,
            color_map=color_map
            )

        print("Training done. Starting testing...")
        test_metrics = gen_utils.test(
            net=full_net,
            test_loader=test_loader,
            color_map=color_map,
            visualize_every=1,
            n_channels=hps.n_channels,
            n_classes=hps.n_classes,
            visualization_path=f'../output/tasks/{hps.experiment_name}/{fold}_full')

        gen_utils.extract_to_writer(writer, test_metrics, prefix='supervised/test', write_index=fold)
        
        print("Testing done. Saving model...")
        save(full_net, optimizer, last_epoch, test_metrics, scheduler.get_last_lr()[0], hps=hps, mode=f'{hps.experiment_log_prefix}_full', date=date)

        print(f'Fold {fold} finished. Metrics: {test_metrics}')

if __name__ == '__main__':
    main()
