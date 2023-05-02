import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import random
from dataset import MultiViewDataset
from train import trainer
import torch.nn as nn
import wandb
import torchvision.transforms as transforms
from torchvision.io.video import read_video, write_video
from evaluate import EvaluationMetric
from model import MVNetwork
from config.classes import EVENT_DICTIONARY, INVERSE_EVENT_DICTIONARY
from torchvision.models.video import R3D_18_Weights, MC3_18_Weights
from torchvision.models.video import R2Plus1D_18_Weights, S3D_Weights
from torchvision.models.video import MViT_V2_S_Weights, MViT_V1_B_Weights


# Fixing seeds for reproducibility
#torch.manual_seed(55)
#np.random.seed(16)

def checkArguments():

    # args.type
    if args.type != 'Replay' and args.type != 'Both' and args.type != 'Live':
        print("Could not find your desired argument for --type:")
        print("Possible types are: Replay, Both or Live")
        exit()

    # args.num_views
    if args.num_views > 5 or  args.num_views < 1:
        print("Could not find your desired argument for --args.num_views:")
        print("Possible number of views are: 1, 2, 3, 4, 5")
        exit()

    # args.data_aug
    if args.data_aug != 'Yes' and args.data_aug != 'No':
        print("Could not find your desired argument for --args.data_aug:")
        print("Possible arguments are: Yes or No")
        exit()

    # args.pooling_type
    if args.pooling_type != 'max' and args.pooling_type != 'mean':
        print("Could not find your desired argument for --args.pooling_type:")
        print("Possible arguments are: max or mean")
        exit()

    # args.weighted_loss
    if args.weighted_loss != 'Yes' and args.weighted_loss != 'No':
        print("Could not find your desired argument for --args.weighted_loss:")
        print("Possible arguments are: Yes or No")
        exit()

    # args.start_frame
    if args.start_frame > 124 or  args.start_frame < 0 or args.end_frame - args.start_frame < 2:
        print("Could not find your desired argument for --args.start_frame:")
        print("Choose a number between 0 and 124 and smaller as --args.end_frame")
        exit()

    # args.end_frame
    if args.end_frame < 1 or  args.end_frame > 125:
        print("Could not find your desired argument for --args.end_frame:")
        print("Choose a number between 1 and 125 and greater as --args.start_frame")
        exit()

    # args.fps
    if args.fps > 25 or  args.fps < 1:
        print("Could not find your desired argument for --args.fps:")
        print("Possible number for the fps are between 1 and 25")
        exit()

    # args.random_shift
    if args.random_shift < 0 or args.random_shift + args.end_frame > 125 or - args.random_shift + args.end_frame < 0:
        print("Your desired argument for --args.random_shift does not work:")
        print("It should be positive, random_shift + end_frame < 126 and start_frame - random_shift > 0")
        exit()


def main(*args):

    args = args[0]
    LR = args.LR
    gamma = args.gamma
    weight_decay = args.weight_decay
    step_size = args.step_size
    start_frame = args.start_frame
    end_frame = args.end_frame
    random_shift = args.random_shift
        
    model_name = args.model_name
    pre_model = args.pre_model
    num_views = args.num_views
    scheduler = args.scheduler
    fps = args.fps
    number_of_frames = int((args.end_frame - args.start_frame) / ((args.end_frame - args.start_frame) / (((args.end_frame - args.start_frame) / 25) * args.fps)))
    batch_size = args.batch_size
    data_aug = args.data_aug
    path = args.path
    pooling_type = args.pooling_type
    weighted_loss = args.weighted_loss
    patience = args.patience
    max_num_worker = args.max_num_worker
    max_epochs = args.max_epochs
    lr_warmup = args.lr_warmup
    continue_training = args.continue_training

    # Logging information
    numeric_level = getattr(logging, 'INFO'.upper(), None)
    print(numeric_level)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % 'INFO')

    os.makedirs(os.path.join("models", os.path.join(model_name, os.path.join(str(num_views), os.path.join(pre_model, os.path.join(str(LR),
                            "_B" + str(batch_size) + "_F" + str(number_of_frames) + "_S" + str(scheduler) + "_G" + str(gamma) + "_Step" + str(step_size)))))), exist_ok=True)

    best_model_path = os.path.join("models", os.path.join(model_name, os.path.join(str(num_views), os.path.join(pre_model, os.path.join(str(LR),
                            "_B" + str(batch_size) + "_F" + str(number_of_frames) + "_S" + str(scheduler) + "_G" + str(gamma) + "_Step" + str(step_size))))))


    log_path = os.path.join(best_model_path, "logging.log")

    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    # Initialize the data augmentation
    if data_aug == 'Yes':
        transformAug = transforms.Compose([
                                          transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)),
                                          transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                                          transforms.RandomRotation(degrees=5),
                                          transforms.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
                                          transforms.RandomHorizontalFlip()
                                          ])
    else:
        transformAug = None

    if pre_model == "r3d_18":
        transforms_model = R3D_18_Weights.KINETICS400_V1.transforms()        
    elif pre_model == "s3d":
        transforms_model = S3D_Weights.KINETICS400_V1.transforms()       
    elif pre_model == "mc3_18":
        transforms_model = MC3_18_Weights.KINETICS400_V1.transforms()       
    elif pre_model == "r2plus1d_18":
        transforms_model = R2Plus1D_18_Weights.KINETICS400_V1.transforms()
    elif pre_model == "mvit_v2_s":
        transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    else:
        transforms_model = R2Plus1D_18_Weights.KINETICS400_V1.transforms()
        print("Warning: Could not find the desired pretrained model")
        print("Possible options are: r3d_18, s3d, mc3_18, mvit_v2_s and r2plus1d_18")
        print("We continue with r2plus1d_18")


    # Create Train Validation and Test datasets
    dataset_Train = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Train', types='Both', shift=random_shift,
        num_views = num_views, transform=transformAug, transform_model=transforms_model) 
    dataset_Valid2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Valid', types='Live', num_views = 0, 
        transform_model=transforms_model)
    dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', types='Live', num_views = 0, 
        transform_model=transforms_model)
    
    # Create the dataloaders for train validation and test datasets
    train_loader = torch.utils.data.DataLoader(dataset_Train,
        batch_size=batch_size, shuffle=True,
        num_workers=max_num_worker, pin_memory=True)

    val_loader2 = torch.utils.data.DataLoader(dataset_Valid2,
        batch_size=1, shuffle=False,
        num_workers=max_num_worker, pin_memory=True)
    
    test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
        batch_size=1, shuffle=False,
        num_workers=max_num_worker, pin_memory=True)

    model = MVNetwork(net_name=pre_model, agr_type=pooling_type).cuda()
    #model = MVNetwork(net_name=pre_model, agr_type=pooling_type)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, 
                                    betas=(0.9, 0.999), eps=1e-07, 
                                    weight_decay=weight_decay, amsgrad=False)
    if scheduler == 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=500000)
    elif scheduler == 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=patience)
    elif scheduler == 2:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler == 3:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == 4:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma)
    else:
        print("agrs.scheduler not valide")
        print("0: no scheduler, 1: ReduceLROnPlateau, 2: StepLR, 3: ExponentialLR, 4: Decrease LR by gamma if vali loss increases")
        exit()
    epoch_start = 0

    if continue_training:
        path_model = os.path.join(log_path, 'model.pth.tar')
        load = torch.load(path_model)
        model.load_state_dict(load['state_dict'])
        optimizer.load_state_dict(load['optimizer'])
        scheduler.load_state_dict(load['scheduler'])
        epoch_start = load['epoch']


    metric_calculator_offence_severity = EvaluationMetric(4, INVERSE_EVENT_DICTIONARY['offence_severity_class'])
    metric_calculator_action = EvaluationMetric(8, INVERSE_EVENT_DICTIONARY['action_class'])
    metric_calculator = [metric_calculator_offence_severity, metric_calculator_action]


    if weighted_loss == 'Yes':
        criterion_offence_severity = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[0].cuda())
        criterion_action = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[1].cuda())
        #criterion_offence_severity = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[0])
        #criterion_action = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[1])
        criterion = [criterion_offence_severity, criterion_action]
    else:
        criterion_offence_severity = nn.CrossEntropyLoss()
        criterion_action = nn.CrossEntropyLoss()
        criterion = [criterion_offence_severity, criterion_action]

    # Start training
    trainer(train_loader, val_loader2, test_loader2, model, optimizer, scheduler, scheduler, LR, lr_warmup, 
            criterion, metric_calculator, best_model_path, epoch_start, model_name=model_name, max_epochs=max_epochs)

    return 0


if __name__ == '__main__':

    # Load the arguments
    parser = ArgumentParser(description='my method', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--path',   required=True, type=str, help='Path to the dataset folder' )
    parser.add_argument('--max_epochs',   required=False, type=int,   default=60,     help='Maximum number of epochs' )
    parser.add_argument('--model_name',   required=False, type=str,   default="VASM",     help='named of the model to save' )
    parser.add_argument('--batch_size', required=False, type=int,   default=2,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-04, help='Learning Rate' )
    parser.add_argument('--patience', required=False, type=int,   default=5,     help='Patience before reducing LR (ReduceLROnPlateau)' )
    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=1, help='number of worker to load data')
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')
    parser.add_argument("--continue_training", required=False, action='store_true', help="Continue training")
    parser.add_argument("--type", required=False, type=str, default="Replay", help="Replays, Live or both")
    parser.add_argument("--num_views", required=False, type=int, default=1, help="Number of views")
    parser.add_argument("--data_aug", required=False, type=str, default="Yes", help="Data augmentation")
    parser.add_argument("--pre_model", required=False, type=str, default="r2plus1d_18", help="Name of the pretrained model")
    parser.add_argument("--pooling_type", required=False, type=str, default="max", help="Which type of pooling should be done")
    parser.add_argument("--weighted_loss", required=False, type=str, default="Yes", help="If the loss should be weighted")
    parser.add_argument("--start_frame", required=False, type=int, default=0, help="The starting frame")
    parser.add_argument("--end_frame", required=False, type=int, default=125, help="The ending frame")
    parser.add_argument("--fps", required=False, type=int, default=25, help="Number of frames per second")
    parser.add_argument("--random_shift", required=False, type=int, default=0, help="Offset (by how much we shift the frames)")
    parser.add_argument("--weight_decay", required=False, type=float, default=0, help="Weight decay in the optimizer")
    parser.add_argument("--scheduler", required=False, type=int, default=0, help="0: no scheduler, 1: ReduceLROnPlateau, 2: StepLR, 3: ExponentialLR, 4: Decrease LR by gamma if vali loss increases")
    parser.add_argument("--step_size", required=False, type=int, default=3, help="StepLR parameter")
    parser.add_argument("--gamma", required=False, type=float, default=0.1, help="StepLR parameter")
    parser.add_argument("--lr_warmup", required=False, type=int, default=0, help="Learning rate warm up")

    args = parser.parse_args()


    ## Checking if arguments are valid
    checkArguments()

    # Setup the GPU
    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    # Start the main training function
    main(args, False)
