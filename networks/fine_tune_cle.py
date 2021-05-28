
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
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
from focal_loss.focalloss import FocalLoss
from utils import ExperimentType
import dataset.cityscapes as cityscapes
from utils import ModelType
import arguments

def initialise_arguments():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('-m', '--model', type=str, help="Path to model")
    args_parser.add_argument('-n', '--experiment-name', type=str, default='cle', help="Name of experiment. E.g. baseline")
    args_parser.add_argument('-r', '--relu-type', type=str, help="ReLU type used when training: normal or leaky")
    args_parser.add_argument('-lt', '--loss-type', type=str, default='focal', help="Type of loss function used for training")
    args_parser.add_argument('--loss-reduction', type=str, default='no-bg', help="Loss function reduction")
    args_parser.add_argument('-c', '--n-classes', type=int, default=6, help="Number of classes to predict.")
    args_parser.add_argument('-ch', '--n-channels', type=int, default=1, help="Number of channels in input image.")
    args_parser.add_argument('-et', '--experiment-type', type=ExperimentType, default='ct', help="ct or cityscapes", choices=list(ExperimentType))
    args_parser.add_argument('-el', '--experiment-log-prefix', type=str, default='', help="prefix added to logs")
    args_parser.add_argument('-l', '--n-hidden-channels-in-projection-head', required=True, type=int, help="Hidden dimension in projection head of CLE")
    args_parser.add_argument('-e', '--epochs', type=int, default=100, help="Number of epochs.")
    args_parser.add_argument('--lr', type=float, default=3e-3, help="Learning rate.")
    args_parser.add_argument('-u', '--unsupervised-training-path', type=str, default='E:\\kaggle\\rsna-intracranial-hemorrhage-detection\\nifti_down_sampled\\', help="Path to unsupervised training data")
    args_parser.add_argument('--focal-gamma', type=float, default=2, help="Gamma of focal loss.")
    args_parser.add_argument('--batch-size-s', type=int, default=4, help="Supervised batch size.")
    args_parser.add_argument('--batch-size-u', type=int, default=1, help="Unsupervised batch size.")
    args_parser.add_argument('--batch-size-t', type=int, default=2, help="Test/validation batch size.")
    args_parser.add_argument('-nu', '--N-unsupervised', type=int, default=0, help="Number of unsupervised samples.")
    args_parser.add_argument('-f', '--fold', type=int, default=0, help="Fold index.")
    args_parser.add_argument('--model-type', type=ModelType, default='unet', help="model type, e.g. unet", choices=list(ModelType))
    args_parser.add_argument('--pre-trained-backbone', type=arguments.bool_arg, default=False, help="Deeplabv3+, pretrained backbone")
    args_parser.add_argument('--output-stride', type=int, default=8, help="output stride for deeplabv3+")
    hps = args_parser.parse_args()
    return hps

weight_decay=4e-5
momentum=0.9
use_checkpoint = False
device = torch.device('cuda:0')

seed = 123
num_threads = 1
batch_size_supervised = 4
save_base_path = None
date = None

# Set seeds:
torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

sizes = []

transform_u = transforms.Compose([])

transform_s = transforms.Compose([
    #transforms.ToPILImage(),
    #transforms.ColorJitter(0.1, 0.1, 0.1),
    #transforms.ToTensor()#,
    #AdditiveGaussianNoiseTransform(mean=0.01, std=0.01) TODO: Why?
])

transform_t = transforms.Compose([])

def train(net, loader, validation_loader, writer , optimizer, scheduler, criterion, hps, color_map, last_epoch=0):
   
    net.train()
    for epoch in range(last_epoch, hps.epochs):
        total_loss = 0
        net.train()
        p_bar = tqdm(loader)
        for i, (images, masks) in enumerate(p_bar):
            images = images.to(device)
            if images.shape[1] == 1:
                images = images.repeat(1,hps.n_channels,1,1)

            masks = masks.to(device)
            masks = masks.squeeze(1)

            b_size = images.shape[0]
            if b_size == 1:
                images = images.repeat(2,1,1,1)
                masks = masks.repeat(2,1,1)

            output = net(images)
            optimizer.zero_grad()
            
            if hps.loss_type == 'dice':
                output = F.softmax(output, dim=1)
                loss = criterion(output, masks, reduction=hps.loss_reduction)
            else:
                loss = criterion(output, masks)
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                if i % 5 == 0:
                    summary_idx = i + len(loader) * epoch
                    p_bar.set_description('Epoch {}. Iteration {}. Loss={:.2f}'.format(
                        epoch,
                        i,
                        loss.item()
                    ))
                    
                    pred_as_image = labels_to_rgb_batched(torch.argmax(output, dim=1), color_map).permute(0,3,1,2)
                    gt_as_image = labels_to_rgb_batched(masks, color_map).permute(0,3,1,2)
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
                    #print(f'supervised/train/loss: {loss.item()} batch {i} / {len(loader)}')            
                    utils.save_overlay_grid(images_as_image, pred_as_image, gt_as_image, f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}_{hps.fold}_fully_supervised_training_output.png')
        
        scheduler.step()
        writer.add_scalar(f'supervised/train/lr', scheduler.get_last_lr()[0], epoch)

        validation_metrics = gen_utils.test(net, validation_loader, color_map, n_classes=hps.n_classes, n_channels=hps.n_channels, visualization_path=f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}_{hps.fold}_{hps.loss_type}_semi', thresholding_enabled=False)
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

        if epoch % 10 == 0:
            gen_utils.save(**save_args)

        gen_utils.save_best(save_args)
       
    writer.close()
    return last_epoch

def main(hps):
    torch.cuda.empty_cache()
    run_experiment(hps)

def run_experiment(hps):
    print(f"Starting fold {hps.fold}...")
    print(f"Hyperparameters: {hps}")
    device = torch.device('cuda:0')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    start_time = time.time()

    if hps.experiment_type == ExperimentType.CT:
        loaders = create_all_dataloaders_folded(
            N=hps.N_unsupervised,
            num_threads=num_threads,
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
            mask_path='../data/multiclass_mask/')
        _, supervised_loader, _, validation_loader, test_loader = loaders
    elif hps.experiment_type == ExperimentType.CITYSCAPES:
        print('Loading cityscapes...')
        loaders = cityscapes.create_all_dataloaders(
            batch_size_s=1,
            batch_size_v=1,
            batch_size_t=1,
            path='../../data/cityscapes/'
        )

        supervised_loader, validation_loader, test_loader = loaders
    else:
        raise Exception(f'Unknown experiment type {hps.experiment_type}')

    color_map = gen_utils.get_color_map(hps.n_classes).to(device)
    print(f'Data loading took: {time.time() - start_time}s using {num_threads} threads.')

    global save_base_path
    save_base_path = f'../experiment_runs/{hps.experiment_name}/models/{hps.fold}_{hps.loss_type}_{hps.experiment_log_prefix}'

    if hps.loss_type == 'focal':
        alphas = gen_utils.get_inverse_frequencies(supervised_loader, device, hps.n_classes)
        criterion = FocalLoss(gamma=hps.focal_gamma, alpha=alphas, size_average=True)
    elif hps.loss_type == 'dice':
        criterion = multiclass_dice_loss
    elif hps.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception(f'Loss type {hps.loss_type} not defined')
    
    last_epoch = 0

    if hps.model_type == ModelType.UNET:
        net = Unet2D(n_input_channels=hps.n_channels, n_classes=hps.n_classes, relu_type=hps.relu_type).to(device)
    elif hps.model_type == ModelType.DEEPLABV3_RESNET50:
        raise Exception('Not implemented')
        
    print("Attempting to load previous network from checkpoint...")
    try:
        state_dict = gen_utils.load(hps.model)

        if 'projection_head.net.0.weight' in state_dict['net']:
            print("###################################### found key projection_head.net.0.weight in state dict ##########################################")
            full_net = gen_utils.SegmentationNetWithProjectionHead(net=net, n_classes=hps.n_classes, n_hidden_channels=hps.n_hidden_channels_in_projection_head).to(device)
            full_net.load_state_dict(state_dict['net'])
            full_net = full_net.net
            print('SegmentationNetWithProjectionHead creation OK')
        else:
            net.load_state_dict(state_dict['net'])
            full_net = full_net.net
            print('Unet creation OK')

        optimizer = torch.optim.SGD(full_net.parameters(), lr=hps.lr, weight_decay=weight_decay, momentum=momentum)
        scheduler = fm_utils.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=hps.epochs)
        print("Net loaded successfully")
            
    except:
        print("Failed to load full dict")
        raise Exception('A model should be specified.')

    print("Starting training...")
    global date
    date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    writer = SummaryWriter(log_dir=f'../experiment_runs/{hps.experiment_name}/{hps.experiment_log_prefix}_fold_{hps.fold}/{date}')
    last_epoch = train(
        net=full_net,
        loader=supervised_loader,
        validation_loader=validation_loader,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        last_epoch=last_epoch,
        color_map=color_map,
        hps=hps)

    print("Training done. Starting testing...")
    test_metrics = gen_utils.test(
        net=full_net,
        test_loader=test_loader,
        color_map=color_map,
        visualize_every=1,
        n_classes=hps.n_classes,
        n_channels=hps.n_channels,
        visualization_path=f'../output/tasks/{hps.experiment_name}/{hps.fold}_full')

    gen_utils.extract_to_writer(writer, test_metrics, prefix='supervised/test', write_index=hps.fold)
    
    print("Testing done. Saving model...")
    gen_utils.save(full_net, optimizer, last_epoch, test_metrics, scheduler.get_last_lr()[0], mode='_full', save_base_path=save_base_path, hps=hps, date=date)

    print(f'Fold {hps.fold} finished. Metrics: {test_metrics}')

if __name__ == '__main__':
    hps = initialise_arguments()
    main(hps)
