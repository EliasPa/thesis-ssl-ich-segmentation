
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
import random
from datetime import datetime
import random
from torch.utils.tensorboard import SummaryWriter
from unet2D.utils import labels_to_rgb_batched
import time
import argparse
from cle.cle import CLE
import dataset.cityscapes as cityscapes
from byol_segmentation.byol_segmentation import Transformation
from utils import ExperimentType
from utils import ModelType
from utils import CLEMode
import arguments

def initialise_arguments():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--randomize-seed', type=arguments.bool_arg, default=False, help="If true, will choose a new seed randomly at each fold")
    args_parser.add_argument('-r', '--relu-type', type=str, help="ReLU type used when training: normal or leaky")
    args_parser.add_argument('-n', '--experiment-name', type=str, default='cle_pre', help="Name of experiment.")
    args_parser.add_argument('-m', '--model', type=str, help="Path to model checkpoint. If provided, continues training.")
    args_parser.add_argument('-e', '--epochs', type=int, default=100, help="Number of epochs.")
    args_parser.add_argument('-nc', '--nof-contrastive-pixels', type=int, default=2025, help="Number of pixels to sample for CLE loss.")
    args_parser.add_argument('-d', '--cle-down-sample-factor', type=float, default=4, help="Images and masks are downsampled by this factor prior to loss. E.g. 2 for half-resolution. ")
    args_parser.add_argument('-t', '--tau', type=float, default=0.07, help="Temperature parameter of CLE loss")
    args_parser.add_argument('-s', '--start-fold', type=int, default=0, help="Start fold.")
    args_parser.add_argument('-f', '--end-fold', type=int, default=5, help="End fold.")
    args_parser.add_argument('-c', '--n-classes', type=int, default=6, help="Number of classes to predict.")
    args_parser.add_argument('-nu', '--N-unsupervised', type=int, default=0, help="Number of unsupervised samples.")
    args_parser.add_argument('-ns', '--N-supervised-mnist', type=int, default=10, help="Number of supervised samples in MNIST case.")
    args_parser.add_argument('-ch', '--n-channels', type=int, default=1, help="Number of channels in input image.")
    args_parser.add_argument('-et', '--experiment-type', type=ExperimentType, default='ct', help="ct or cityscapes", choices=list(ExperimentType))
    args_parser.add_argument('-el', '--experiment-log-prefix', type=str, default='', help="prefix added to logs")
    args_parser.add_argument('-l', '--n-hidden-channels-in-projection-head', required=True, type=int, help="Hidden dimension in projection head of CLE")
    args_parser.add_argument('-i', '--ignore-background', required=False, default=False, type=arguments.bool_arg, help="If True, ignores background and samples only positive pixels.")
    args_parser.add_argument('-u', '--unsupervised-training-path', type=str, default='E:\\kaggle\\rsna-intracranial-hemorrhage-detection\\nifti_down_sampled\\', help="Path to unsupervised training data")        
    args_parser.add_argument('--lr', type=float, default=0.1, help="Learning rate.")
    args_parser.add_argument('--batch-size-s', type=int, default=4, help="Supervised batch size.")
    args_parser.add_argument('--batch-size-u', type=int, default=2, help="Unsupervised batch size.")
    args_parser.add_argument('--batch-size-t', type=int, default=1, help="Test/validation batch size.")
    args_parser.add_argument('--model-type', type=ModelType, default='unet', help="model type, e.g. unet", choices=list(ModelType))
    args_parser.add_argument('--output-stride', type=int, default=4, help="output stride for deeplabv3+")
    args_parser.add_argument('--cle-mode', type=CLEMode, default='real_labels', help="Training mode for CLE", choices=list(CLEMode))
    args_parser.add_argument('--N-iter', type=int, default=512, help="N iterations in epoch")
    args_parser.add_argument('--confidence-threshold', type=float, default=0.9, help="Pseudo labeling confidence threshold")
    args_parser.add_argument('--perturb-probability', type=float, default=0.8, help="Probability with which to perturb input images")
    args_parser.add_argument('--pre-trained-backbone', type=arguments.bool_arg, default=False, help="Deeplabv3+, pretrained backbone")
    args_parser.add_argument('--visualize', type=arguments.bool_arg, default=True, help="Visualize results.")
    hps = args_parser.parse_args()
    return hps

momentum=0.9
weight_decay = 4e-5
use_checkpoint = False
date = None

seed = 123
num_threads = 1
save_base_path = None

# Set seeds:
torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

sizes = []

transform_u = transforms.Compose([])

transform_s = transforms.Compose([])

transform_t = transforms.Compose([])

class AugmentationModule(nn.Module):

    def __init__(self, device, mean=0, std=0.1, is_geometric=True, rot=4.5, translation=(0,0), scale=(0.9,1.1), flip_probability=0.5, shear_angle=5):
        super(AugmentationModule, self).__init__()

        self.non_geometric_augmentations = transforms.Compose([
            ])
        
        self.regular_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor()
            ])

        self.geometric_augmentations = Transformation(
                    rot=rot,
                    translation=translation,
                    scale=scale,
                    flip_probability=flip_probability,
                    shear_angle=shear_angle
                )        

        self.is_geometric = is_geometric

    def geometric_transform(self, x, applied_transforms):
        return self.geometric_augmentations(x, applied_transforms)

    def forward(self, x):
        # with torch.no_grad():
        applied_transforms = None
        if self.is_geometric:
            x, applied_transforms, _ = self.geometric_augmentations(x)
        x = self.non_geometric_augmentations(x)
        for i, x_ in enumerate(x):
            x[i] = self.regular_transforms(x_)
        return x, applied_transforms

def prepare_images(images, hps):
    if images.shape[1] == 1:
        images = images.repeat(1,hps.n_channels,1,1)

    return images

def train_sup(net, loader, validation_loader, unsupervised_loader, augmentation_module, optimizer, scheduler, writer, color_map, fold, last_epoch, hps, device):
    net.train()
    supervised_iter = iter(loader)
    unsupervised_iter = iter(unsupervised_loader) if unsupervised_loader is not None else None
    for epoch in range(last_epoch, hps.epochs):
        total_loss = 0
        net.train()
        p_bar = tqdm(range(hps.N_iter))
        for i in p_bar:
            optimizer.zero_grad()
            if hps.cle_mode == CLEMode.REAL_LABELS:
                images, masks = next(supervised_iter)
                
                # Prepare images
                images = prepare_images(images, hps).to(device)

                # Prepare masks
                masks = masks.to(device)
                masks = masks.squeeze(1)
            elif hps.cle_mode == CLEMode.PSEUDO_LABELS:
                images, _ = next(unsupervised_iter)

                # Prepare images
                images = prepare_images(images, hps).to(device)

                # Prepare masks
                with torch.no_grad():
                    segmentation = net.segment(images)
                    mask_probabilities = F.softmax(segmentation, dim=1)
                    mask_max_values, mask_max_indices = torch.max(mask_probabilities, dim=1)
                    confidence_mask = torch.ge(mask_max_values, hps.confidence_threshold)
                    masks = mask_max_indices * confidence_mask
                    del mask_max_indices
                    del confidence_mask
                    del mask_max_values
                    del mask_probabilities
                    del segmentation
            else:
                raise Exception(f'CLEMode {hps.cle_mode} not supported.')

            # Generate I_hat (augmented images)
            images_hat = images.clone()
            geometric_transformations = None

            with torch.no_grad():
                if random.random() < hps.perturb_probability:
                    images_hat, geometric_transformations = augmentation_module(images_hat)

            # Concatenate original and augmented images to perform one forward pass on the segmentation net
            combined = torch.cat([images, images_hat])

            if hps.model_type == ModelType.DEEPLABV3_RESNET50:
                raise Exception('Not implemented')
            else:
                output = net(combined)

            # Extract F and F_hat (projected feature maps, for original and perturbed images, respectively)
            f, f_hat = output.chunk(2)
            if geometric_transformations is not None:
                f_hat, _, _ = augmentation_module.geometric_transform(f_hat, geometric_transformations)

            cle_data = CLE(
                f=f,
                f_hat=f_hat,
                labels=masks,
                N=hps.nof_contrastive_pixels,
                n_classes=hps.n_classes,
                tau=hps.tau,
                down_sample_factor=hps.cle_down_sample_factor,
                ignore_bg=hps.ignore_background
            )

            loss = cle_data['loss']
            N_pixels = cle_data['N']

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            summary_idx = i + hps.N_iter * epoch
            writer.add_scalar(f'supervised/train/N_pixels', N_pixels, summary_idx)
            writer.add_scalar(f'supervised/train/loss', loss.item(), summary_idx)

            p_bar.set_description('Loss={:.2f}. Epoch {}'.format(loss.item(), epoch))

            if i % 5 == 0 and hps.visualize:
                with torch.no_grad():
                    
                    f = net.segment(images).to(device)

                    f = F.interpolate(f, size=(masks.shape[2], masks.shape[1]), mode='bilinear')
                    f_hat = F.interpolate(f_hat, size=(masks.shape[2], masks.shape[1]), mode='bilinear')

                    pred_as_image = labels_to_rgb_batched(torch.argmax(f, dim=1), color_map).permute(0,3,1,2)
                    pred_hat_as_image = labels_to_rgb_batched(torch.argmax(f_hat, dim=1), color_map).permute(0,3,1,2)
                    gt_as_image = labels_to_rgb_batched(masks, color_map).permute(0,3,1,2)
                        
                    output = F.softmax(output, dim=1)
                    dice = (1 - multiclass_dice_loss(f, masks, reduction="sum").item())
                    iou = (1 - intersection_over_union_loss(f, masks).item())

                    writer.add_scalar(f'supervised/train/dice_score', dice, summary_idx)
                    writer.add_scalar(f'supervised/train/iou_score', iou, summary_idx)

                    if images.shape[1] == 1:
                        images_as_image = images.repeat(1,3,1,1)
                        images_hat_as_image = images_hat.repeat(1,3,1,1)
                    else:
                        images_as_image = images
                        images_hat_as_image = images_hat
                        
                    utils.save_overlay_grid(images_as_image, pred_as_image, gt_as_image, f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}contrastive_pretraining_output.png')
                    utils.save_overlay_grid(images_as_image, pred_hat_as_image, gt_as_image, f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}contrastive_pretraining_output_hat.png')
                    utils.save_all_grid(images_as_image, images_hat_as_image, images_hat_as_image, f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}original and augmented.png')

        scheduler.step()
        writer.add_scalar(f'supervised/train/lr', scheduler.get_last_lr()[0], epoch)

        metrics = {}

        if epoch % 10 == 0:
            save_args = {
                'net': net,
                'optimizer': optimizer,
                'last_epoch': epoch,
                'test_metrics': metrics,
                'lr': scheduler.get_last_lr()[0],
                'save_base_path': save_base_path,
                'hps': hps,
                'mode': f'checkpoint_{epoch}',
                'date': date
            }
            gen_utils.save(**save_args)

        print(f'Total loss for epoch {epoch} was {total_loss}')
        print('===============================================')
        last_epoch = epoch
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
            torch.manual_seed(seed) # 123
            np.random.seed(seed)
            random.seed(seed)

        start_time = time.time()

        if hps.experiment_type == ExperimentType.CT:
            print('Loading CT scans...')
            loaders = create_all_dataloaders_folded(
                N=hps.N_unsupervised,
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
                n_classes=hps.n_classes,
                image_path='../data/ct_scans/',
                u_image_path=hps.unsupervised_training_path,
                mask_path='../data/multiclass_mask/')

            unsupervised_loader, supervised_loader, supervised_loader_inf, validation_loader, _ = loaders

            if hps.cle_mode == CLEMode.REAL_LABELS:
                unsupervised_loader = None
        elif hps.experiment_type == ExperimentType.CITYSCAPES:
            print('Loading cityscapes...')
            loaders = cityscapes.create_all_dataloaders(
                batch_size_s=1,
                batch_size_v=1,
                batch_size_t=1,
                path='../../data/cityscapes/'
            )

            supervised_loader, validation_loader, _ = loaders
            unsupervised_loader = None
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

            unsupervised_loader, supervised_loader, supervised_loader_inf, validation_loader, _ = loaders

            if hps.cle_mode == CLEMode.REAL_LABELS:
                unsupervised_loader = None
        else:
            raise Exception(f'Unknown experiment type {hps.experiment_type}')
            
        if hps.cle_mode == CLEMode.PSEUDO_LABELS:
            assert(unsupervised_loader is not None and len(unsupervised_loader) != 0)
            assert(hps.model is not None)
        
        color_map = gen_utils.get_color_map(hps.n_classes).to(device)
        print(f'Color map: {color_map}')

        global save_base_path
        save_base_path = f'../experiment_runs/{hps.experiment_name}/models/{fold}_{hps.experiment_log_prefix}_'

        if hps.model_type == ModelType.UNET:
            net = Unet2D(n_input_channels=hps.n_channels, n_classes=hps.n_classes, relu_type=hps.relu_type).to(device)
        elif hps.model_type == ModelType.DEEPLABV3_RESNET50:
            raise Exception('Not implemented')
        
        full_net = gen_utils.SegmentationNetWithProjectionHead(net=net, n_classes=hps.n_classes, n_hidden_channels=hps.n_hidden_channels_in_projection_head).to(device)
        augmentation_module = AugmentationModule(device=device, is_geometric=False).to(device)
        

        print(f'Data loading took: {time.time() - start_time}s using {num_threads} threads.')

        last_epoch = 0

        if use_checkpoint or hps.model is not None:
            print("Attempting to load previous network from checkpoint...")
            state_dict = gen_utils.load(hps.model)
            if 'projection_head.net.0.weight' in state_dict['net']:
                full_net.load_state_dict(state_dict['net'])
                print('Loading SegmentationNetWithProjectionHead OK')
            else:
                print('Loading up fine-tuned net...')
                if hps.model_type == ModelType.UNET:
                    net = Unet2D(n_input_channels=hps.n_channels, n_classes=hps.n_classes, relu_type=hps.relu_type).to(device)
                elif hps.model_type == ModelType.DEEPLABV3_RESNET50:
                    raise Exception('Not implemented')
        
                net.load_state_dict(state_dict['net'])
                full_net = gen_utils.SegmentationNetWithProjectionHead(net=net, n_classes=hps.n_classes, n_hidden_channels=hps.n_hidden_channels_in_projection_head).to(device)
                print('Loading from fine-tuned U-net OK. Using a fresh Projection head.')

        optimizer = torch.optim.SGD(full_net.parameters(), lr=hps.lr, weight_decay=weight_decay, momentum=momentum)
        scheduler = fm_utils.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=hps.epochs)

        # Step the scheduler until it reaches where training was last stopped:
        print(f"Stepping scheduler {last_epoch} times")
        for _ in tqdm(range(last_epoch)):
            scheduler.step()

        print("Starting training...")
        global date
        date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        writer = SummaryWriter(log_dir=f'../experiment_runs/{hps.experiment_name}/{hps.experiment_log_prefix}_fold_{fold}/{date}')
        hps.N_iter = max(int(len(supervised_loader)), 10)
        last_epoch = train_sup(
            net=full_net,
            augmentation_module=augmentation_module,
            loader=supervised_loader_inf,
            validation_loader=validation_loader,
            unsupervised_loader=unsupervised_loader,
            fold=fold,
            writer=writer,
            optimizer=optimizer,
            scheduler=scheduler,
            last_epoch=last_epoch,
            hps=hps,
            color_map=color_map,
            device=device
            )

        test_metrics = {}
        gen_utils.extract_to_writer(writer, test_metrics, prefix='supervised/test', write_index=fold)
        
        gen_utils.save(full_net, optimizer, last_epoch, test_metrics, scheduler.get_last_lr()[0], save_base_path=save_base_path, hps=hps, mode=f'full_{seed}', date=date)

        print(f'Fold {fold} finished. Metrics: {test_metrics}')

if __name__ == '__main__':
    main()
