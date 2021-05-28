import torch
import torch.nn
import torchvision
import numpy as np
import torch.nn.functional as F
import fix_match.rand_aug as rand_aug
from unet2D.loss import intersection_over_union_loss, multiclass_dice_loss
from tqdm import tqdm
import utils as gen_utils
from focal_loss_alt.focal_loss import focal_loss_function
from utils import test
from unet2D.loss import intersection_over_union_loss
from tqdm import tqdm
from unet2D.utils import labels_to_rgb_batched
import time
import unet2D.utils as unet_utils

def train(net, ema_net, transformation_network, unsupervised_loader, supervised_loader, validation_loader, optimizer, criterion, scheduler, writer, hps, last_epoch, color_map, device, focal_alphas, date, save_base_path):

    unsupervised_iter = iter(unsupervised_loader)
    supervised_iter = iter(supervised_loader)

    mode = "RGB" if hps.n_channels == 3 else None
    to_pil = torchvision.transforms.ToPILImage(mode=mode)
    to_tensor = torchvision.transforms.ToTensor()

    for epoch in range(last_epoch, hps.epochs):
        total_loss = 0

        data_start = time.time()
        p_bar = tqdm(range(hps.N_iter))

        accumulated_loss = 0
        net.train()
        for i in p_bar:

            if hps.ra_n == -1:
                strong_pixel_augment = rand_aug.RandAugment(n=hps.ra_n, m_max=hps.ra_m_max, aug_list=rand_aug.strong_augment_list_fm_seg)
            else:
                strong_pixel_augment = None

            images, _ = unsupervised_iter.next()
            s_images, labels = supervised_iter.next()
            images = images.to(device)
            s_images = s_images.to(device)
            labels = labels.to(device)

            labels = labels.squeeze(1)

            if images.shape[1] == 1:
                images = images.repeat(1,hps.n_channels,1,1)
                s_images = s_images.repeat(1,hps.n_channels,1,1)

            # Architecture
            # (images, _), (s_images, labels) -> a(images), a(s_images, labels) -> net() supervised_output, weak_output -> pseudo_labels
            #                                                   |                                                               |
            #                                                   |                                                               |
            #                                                   \/                                                             \/
            #                                               A(images) -----------> net(): strong_output --------------- A(pseudo_labels)
            #                                                                                                   | 
            #                                                                                                  \/
            #                                                   loss = loss_c(strong_output, pseudo_labels) + loss_s(so, labels)
        
            weakly_augmented_images = images.clone()
            strongly_transformed_images = images.clone()
            strongly_transformed_images, strong_transformations, _ = transformation_network(strongly_transformed_images) # strong geometric augmentations

            strongly_augmented_images = torch.zeros(strongly_transformed_images.shape).to(device)

            if strong_pixel_augment is not None:
                for j, im in enumerate(strongly_transformed_images):
                    strongly_augmented_images[j] = to_tensor(strong_pixel_augment(to_pil(im.cpu()))) # strong pixel augmentations, no need to invert
            else:
                strongly_augmented_images = strongly_transformed_images
                
            combined = torch.cat((s_images, weakly_augmented_images, strongly_augmented_images))

            # shuffle inputs
            if hps.shuffle:
                perm = torch.randperm(combined.shape[0])
                combined = combined[perm]
        
            summary_idx = i + (hps.N_iter * epoch)

            data_time = time.time() - data_start

            output = net(combined)

            if hps.shuffle:
                output = output[torch.argsort(perm)] # deshuffle inputs

            supervised_length = s_images.shape[0]
            supervised_output = output[:supervised_length] # Extract supervised outputs
            weak_output, strong_output = output[supervised_length:].chunk(2) # Extract all unsupervised outputs
            
            # Align segmentation maps
            weak_output_values = weak_output.detach().clone()
            weak_max_values, weak_output = torch.max(weak_output, dim=1)
            weak_output_pre = weak_output.detach().clone()
            weak_output, _, _ = transformation_network(weak_output, strong_transformations, is_mask=True, calculate_gradients=True) # Apply geometric T's to weak output
            # Pseudo labeling
            pseudo_mask = torch.ge(weak_max_values, hps.tau).long()
            inverted_mask = torch.logical_not(pseudo_mask)
            ignored_values = torch.full_like(pseudo_mask, fill_value=-1)

            weak_output = inverted_mask * ignored_values + pseudo_mask * weak_output
            weak_output = weak_output.long()

            with torch.no_grad():
                if i % (hps.N_iter // 2) == 0:
                    weak_output_pre_as_image = labels_to_rgb_batched(weak_output_pre.clip(0, hps.n_classes), color_map).permute(0,3,1,2)
                    weak_output_as_image = labels_to_rgb_batched(weak_output.clip(0, hps.n_classes), color_map).permute(0,3,1,2)

                    if s_images.shape[1] == 1:
                        s_images_as_image = s_images.repeat(1,3,1,1)
                        weakly_augmented_images_as_image = weakly_augmented_images.repeat(1,3,1,1)
                    else:
                        s_images_as_image = s_images
                        weakly_augmented_images_as_image = weakly_augmented_images

                    unet_utils.save_all_grid(weakly_augmented_images_as_image, weak_output_as_image, weak_output_pre_as_image, f'../output/tasks/{hps.experiment_name}/transformed_teacher_output.png')
                    torchvision.utils.save_image(weak_output_values[0,1,:,:], f'../output/tasks/{hps.experiment_name}/unsupervised_confidence.png')
                    del weak_output_pre_as_image
                    del weak_output_as_image

            # Supervised loss:
            if hps.loss_type == 'dice':
                supervised_output = F.softmax(supervised_output, dim=1)
                loss_s = criterion(supervised_output, labels, reduction=hps.loss_reduction)
            else:
                loss_s = criterion(supervised_output, labels)

            # Consistency loss
            if hps.N_ramp_up_consistency != -1:
                current_lambd = (np.min([i + (hps.N_iter*epoch), hps.N_ramp_up_consistency*hps.N_iter])/(hps.N_ramp_up_consistency*hps.N_iter))*hps.lambd
            else:
                current_lambd = hps.lambd
            
            if hps.consistency_loss_type == 'cross_entropy':
                loss_c = F.cross_entropy(strong_output, weak_output, reduction='mean', ignore_index=-1)
            elif hps.consistency_loss_type == 'focal':
                loss_c = focal_loss_function(
                    input=strong_output,
                    target=weak_output,
                    reduction='none',
                    xe_reduction='none',
                    alpha=focal_alphas,
                    gamma=hps.consistency_focal_gamma
                )

                loss_c = loss_c * pseudo_mask
                loss_c = loss_c.mean()
            elif hps.consistency_loss_type == 'dice':
                loss_c = multiclass_dice_loss(strong_output, weak_output, reduction=hps.consistency_dice_reduction, ignore_index=-1)
            elif hps.consistency_loss_type == 'cross_dice':
                loss_xe = F.cross_entropy(strong_output, weak_output, reduction='mean', ignore_index=-1)
                loss_dice = multiclass_dice_loss(strong_output, weak_output, reduction=hps.consistency_dice_reduction, ignore_index=-1)
                loss_c = loss_xe + loss_dice
            else:
                raise Exception(f'Consistency type not defined: {hps.consistency_loss_type}')

            loss_c = current_lambd * loss_c
            loss = loss_s + loss_c

            # Calculate gradient and update model:
            loss.backward()
            total_loss += loss.item()
            accumulated_loss += loss.item()

            if (i + 1) % hps.zero_gradients_every == 0:
                optimizer.step()
                writer.add_scalar(f'semi/train/accumulated_loss', accumulated_loss, summary_idx)
                accumulated_loss = 0
                optimizer.zero_grad()

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

                    dice = (1 - multiclass_dice_loss(supervised_output, labels))
                    iou = (1 - intersection_over_union_loss(supervised_output, labels))

                    writer.add_scalar(f'semi/train/dice_score', dice.item(), summary_idx)
                    writer.add_scalar(f'semi/train/iou_score', iou.item(), summary_idx)

                    supervised_seg_as_image = labels_to_rgb_batched(torch.argmax(supervised_output, dim=1), color_map).permute(0,3,1,2)
                    supervised_label_as_image = labels_to_rgb_batched(labels, color_map).permute(0,3,1,2)

                    weak_output_as_image = labels_to_rgb_batched(weak_output.clip(0, hps.n_classes), color_map).permute(0,3,1,2)
                    strong_output_as_image = labels_to_rgb_batched(torch.argmax(strong_output, dim=1), color_map).permute(0,3,1,2)

                    if s_images.shape[1] == 1:
                        s_images_as_image = s_images.repeat(1,3,1,1)
                        strongly_augmented_images_as_image = strongly_augmented_images.repeat(1,3,1,1)
                    else:
                        s_images_as_image = s_images
                        strongly_augmented_images_as_image = strongly_augmented_images

                    unet_utils.save_overlay_grid(strongly_augmented_images_as_image, strong_output_as_image, weak_output_as_image, f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}unsupervised_output.png')
                    unet_utils.save_overlay_grid(s_images_as_image, supervised_seg_as_image, supervised_label_as_image, f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}supervised_output.png')
                    
        scheduler.step()
        writer.add_scalar(f'semi/train/lr', scheduler.get_last_lr()[0], epoch)
        validation_metrics = test(net, validation_loader, n_classes=hps.n_classes, color_map=color_map, visualization_path=f'../output/tasks/{hps.experiment_name}/semi_train', thresholding_enabled=False)
        gen_utils.extract_to_writer(writer, validation_metrics, 'semi/validation/', epoch)

        if hasattr(ema_net, 'target_net'):
            ema_validation_metrics = test(ema_net.target_net, validation_loader, n_classes=hps.n_classes, color_map=color_map, visualization_path=f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}ema_train_', thresholding_enabled=False)
        else:
            ema_validation_metrics = test(ema_net, validation_loader, n_classes=hps.n_classes, color_map=color_map, visualization_path=f'../output/tasks/{hps.experiment_name}/{hps.experiment_log_prefix}ema_train_', thresholding_enabled=False)

        gen_utils.extract_to_writer(writer, ema_validation_metrics, prefix='semi/validation/ema', write_index=epoch)

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
