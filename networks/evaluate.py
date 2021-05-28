
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from mean_teacher.mean_teacher import MTResNet
import numpy as np
from dataset.mnist import create_all_dataloaders, create_dataloader, create_dataloaders_from_path
from dataset.ct import create_all_dataloaders, create_all_dataloaders_folded
from unet2D import utils
import math
from unet2D.unet import Unet2D
import fix_match.utils as fm_utils
import torch.nn.functional as F
import fix_match.rand_aug as rand_aug
from unet2D.loss import intersection_over_union_loss, multiclass_dice_loss
from tqdm import tqdm
import utils as gen_utils
from focal_loss.focalloss import FocalLoss
import random
from utils import test
from datetime import datetime
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter
from unet2D.loss import one_hot_encode
from unet2D.utils import labels_to_rgb_batched
import matplotlib.pyplot as plt
import time
from augmentation.additive_gaussian_noise import AdditiveGaussianNoiseTransform
import matplotlib.pyplot as plt
import argparse
import os
import matplotlib.pyplot as plt
import glob
from utils import ModelType, FixMatchType
import json
import arguments
from metrics.utils import StatisticEvaluator
from utils import ExperimentType

def initialise_arguments():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('-m', '--model', type=str, help="Path to model")
    args_parser.add_argument('--single-mode', default=True, type=arguments.bool_arg, help="If True, will evaluate --model. If False, will treat --model as a regexp for many models and evaluates all of them.")
    args_parser.add_argument('-n', '--experiment-name', type=str, help="Name of experiment. E.g. baseline")
    args_parser.add_argument('-et', '--experiment-type', type=ExperimentType, default='ct', help="ct or cityscapes", choices=list(ExperimentType))
    args_parser.add_argument('-r', '--relu-type', type=str, help="ReLU type used when training: normal or leaky")
    args_parser.add_argument('-e', '--ensemble', type=arguments.bool_arg, help="Should evaluate with ensemble prediction or not? This allows for calculation of AUC.")
    args_parser.add_argument('-v', '--visualize', type=arguments.bool_arg, help="Should visualize result.")
    args_parser.add_argument('-s', '--start-fold', type=int, default=0, help="If ensembling, specified range of folded models together with --end-fold")
    args_parser.add_argument('-f', '--end-fold', type=int, default=5, help="If ensembling, specified range of folded models together with --start-fold")
    args_parser.add_argument('-el', '--experiment-log-prefix', type=str, default='', help="prefix added to logs")
    args_parser.add_argument('-c', '--n-classes', type=int, default=6, help="Number of classes to predict.")
    args_parser.add_argument('-ch', '--n-channels', type=int, default=1, help="Number of channels in input image.")
    args_parser.add_argument('--model-type', type=ModelType, default='unet', help="model type, e.g. unet", choices=list(ModelType))
    args_parser.add_argument('--legacy-unet', type=arguments.bool_arg, default=True, help="Should use legacy U-net, or latest version")
    args_parser.add_argument('--output-stride', type=int, default=8, help="Output stride for deeplabv3+")
    args_parser.add_argument('-u', '--unsupervised-training-path', type=str, default='E:\\kaggle\\rsna-intracranial-hemorrhage-detection\\nifti_down_sampled\\', help="Path to unsupervised training data")
    hps = args_parser.parse_args()
    return hps

relu_type = 'normal'
default_fold = 0
start_fold = 0
folds = 5
epochs = 100
N_unsupervised = 0

seed = 123
num_threads = 1
batch_size_supervised = 4

# Set seeds:
torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

sizes = []

transform_u = transforms.Compose([])
transform_s = transforms.Compose([])
transform_t = transforms.Compose([])

def main(hps):
    torch.cuda.empty_cache()
    run_experiment(hps)

def run_experiment(hps):
    device = torch.device('cuda:0')
    color_map = gen_utils.get_color_map(hps.n_classes).to(device)

    start_time = time.time()

    if hps.experiment_type == ExperimentType.CT:
        loaders = create_all_dataloaders_folded(
            N=N_unsupervised,
            num_threads=num_threads,
            fold=default_fold,
            batch_size_s=4,
            batch_size_u=1,
            batch_size_t=1,
            sizes=sizes,
            pin_memory=True,
            resize=0.5,
            transform_u=transform_u,
            transform_s=transform_s,
            transform_t=transform_t,
            image_path='../data/ct_scans/',
            u_image_path=hps.unsupervised_training_path,
            mask_path='../data/multiclass_mask/')

        _, _, _, _, test_loader = loaders
    elif hps.experiment_type == ExperimentType.MNIST:
        loaders = create_dataloaders_from_path(
            n_classes=hps.n_classes,
            batch_size_s=1,
            batch_size_u=1,
            batch_size_t=8,
            transform_u=transform_u,
            transform_s=transform_s,
            transform_t=transform_t,
            pin_memory=True,
            base_path='../data/mnist/mnist_lesion/',
            N_s=1,
            N_u=1,
            N_v=1
        )
        _, _, _, _, test_loader = loaders
    else:
        raise Exception(f'Experiment type {hps.experiment_type} not implemented.')
    
    print(f'Data loading took: {time.time() - start_time}s using {num_threads} threads.')

    if not hps.ensemble:
        print(f"Using fold {default_fold}...")

        save_base_path = f'../experiment_runs/{hps.experiment_name}/evaluation'

        result_dict = {}
        if hps.single_mode:
            models = [hps.model]
        else:
            print(f"Evaluating all models matching {hps.model}")
            matches = glob.glob(hps.model)
            models = [match.replace('\\', '/') for match in matches]

        statistic_evaluator = StatisticEvaluator()
        for model_path in models:
            print(f"Starting to evaluate {model_path}")
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            if hps.model_type == ModelType.UNET:
                full_net = Unet2D(n_input_channels=hps.n_channels, n_classes=hps.n_classes, relu_type=relu_type, legacy=hps.legacy_unet).to(device)
            elif hps.model_type == ModelType.DEEPLABV3_RESNET50:
                raise Exception('Not implemented')
            else:
                raise Exception(f'Model {hps.model_type} is not defined.')
            print(f"Attempting to load model from {model_path}...")
            state_dict = gen_utils.load(model_path)
            net = state_dict['net']
            last_epoch = state_dict['last_epoch']
            test_metrics = state_dict['test_metrics']
            full_net.load_state_dict(net)
            print("Success")

            print(f'The model had been trained for {last_epoch+1} epochs.')

            date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
            writer = SummaryWriter(log_dir=f'../experiment_runs/results/{hps.experiment_name}/{hps.experiment_log_prefix}/{date}')

            print("Starting testing...")
            test_metrics = gen_utils.test(
                net=full_net,
                test_loader=test_loader,
                color_map=color_map,
                visualize_every=1,
                n_classes=hps.n_classes,
                n_channels=hps.n_channels,
                visualization_path=f'../output/tasks/{hps.experiment_name}/evaluation/{hps.experiment_log_prefix}{default_fold}_full')

            gen_utils.extract_to_writer(writer, test_metrics, prefix='test', write_index=default_fold)

            print(f"Testing done. {test_metrics}")

            model_name = model_path.split('/')[-1]
            save_path = os.path.join(save_base_path, f'{hps.experiment_log_prefix}_{model_name}.json')
            save_path = save_path.replace('\\', '/')
        
            if hps.visualize:
                import seaborn as sns
                import matplotlib.pyplot as plt

                fpr = test_metrics['pixel_fpr'][:,1]
                sensitivities = test_metrics['pixel_sensitivities'][:,1]
                sns.lineplot(x=fpr, y=sensitivities)
                sns.lineplot(x=[0,1], y=[0,1], color='red')
                plt.title('ROC')
                plt.show()
                fpr = test_metrics['slice_fpr'][:,1]
                sensitivities = test_metrics['slice_sensitivities'][:,1]
                sns.lineplot(x=fpr, y=sensitivities)
                sns.lineplot(x=[0,1], y=[0,1], color='red')
                plt.title('ROC')
                plt.show()

            # Create table:
            table_dict = {
                'any_blood_dice': test_metrics['any_blood_dice'],
                'any_blood_iou': test_metrics['any_blood_iou'],
                'slice_tpr': test_metrics['slice_sensitivities'][:,1],
                'slice_tnr': test_metrics['slice_specificities'][:,1],
                'pixel_tpr': test_metrics['pixel_sensitivities'][:,1],
                'pixel_tnr': test_metrics['pixel_specificities'][:,1],
            }

            test_metrics['table_dict'] = table_dict
            modify_dict(test_metrics)

            statistic_evaluator.update(test_metrics, ID=model_name)

            with open(save_path, 'w') as f:
                json.dump(test_metrics, f)

        stats_save_path = os.path.join(save_base_path, 'statistics', f'{hps.experiment_log_prefix}_statistics.json')
        stats_save_path = stats_save_path.replace('\\', '/')
        statistic_evaluator.save_as_json(save_path=stats_save_path)

    else:
        import seaborn as sns
        models = []
        print("Ensembling...")

        model_paths = glob.glob(f'{hps.model}*full*.pth')
        print(f"models: {len(model_paths)}")
        for path in model_paths:
            try:
                full_net = Unet2D(n_input_channels=hps.n_channels, n_classes=hps.n_classes, relu_type=relu_type).to(device)
                state_dict = gen_utils.load(path)
                net = state_dict['net']
                net = state_dict['net']
                full_net.load_state_dict(net)
                print("Success")
            except Exception as e:
                print(f"Failed to load previous net {path}")
                print(e)
            
            models.append(full_net)

        test_metrics = gen_utils.test_ensemble(
            nets=models,
            test_loader=test_loader,
            color_map=color_map,
            visualize_every=1,
            n_classes=hps.n_classes,
            n_channels=hps.n_channels,
            visualization_path=f'../output/tasks/{hps.experiment_name}/evaluation/ensemble'
            )

        print(test_metrics)

        sns.lineplot(x=1-np.flip(test_metrics['specificities'][:,1], axis=0), y=np.flip(test_metrics['sensitivities'][:,1], axis=0))
        #sns.lineplot(x=[0,1], y=[0,1])
        plt.show()
        sns.lineplot(y=1-test_metrics['specificities'][:,1], x=np.linspace(0,1,test_metrics['specificities'].shape[0]))
        sns.lineplot(y=test_metrics['sensitivities'][:,1], x=np.linspace(0,1,test_metrics['sensitivities'].shape[0]))
        sns.lineplot(y=test_metrics['specificities'][:,1], x=np.linspace(0,1,test_metrics['specificities'].shape[0]))
        #sns.lineplot(x=[0,1], y=[0,1])
        plt.show()

        fpr = test_metrics['fpr'][5:]
        tpr = test_metrics['tpr'][5:]
        print(test_metrics['specificities'].shape)
        plt.plot(fpr,tpr)
        plt.show()
        print(f"Testing done.")
    
def modify_dict(dictionary):
    for key in dictionary:
        value = dictionary[key]
        if isinstance(value, (np.ndarray)):
            dictionary[key] = value.tolist()
        elif isinstance(value, dict):
             modify_dict(value)

if __name__ == '__main__':
    hps = initialise_arguments()
    main(hps)
