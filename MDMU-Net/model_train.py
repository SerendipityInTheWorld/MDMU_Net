import torch.optim.lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from preconditioning import data_loader,data_transforms
from monai.utils import set_determinism

from network.MDMUNet import MDMUNet
from AblationExperiment.UNetRes import UNetRes
from AblationExperiment.UNetMDMSE import UNetMDMSE
from AblationExperiment.UNetMDMSECA import UNetMDMSECA
from tqdm import tqdm
from monai.losses import DiceCELoss
import os
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser(description='MyNet3D for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root_path', type=str, default=r'', help='Root folder of all your images and labels')
parser.add_argument('--save_path', type=str, default=r'result/MDMU_Net', help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='PancreasTumour', help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='MDMUNet', help='Network models: {MDMUNet,UNetRes,UNetMDMSE,UNetMDMSECA}')
parser.add_argument('--mode', type=str, default='train', help='Training or testing mode')
parser.add_argument('--pretrain', default='False', help='Have pretrained weights or not')

parser.add_argument('--pretrained_weights', default=r'', help='Path of pretrained weights')
parser.add_argument('--crop_sample', type=int, default=2, help='Number of cropped sub-volumes for each subject')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for subject input')
parser.add_argument('--class_num', type=int, default=3, help='Number of label categories')

parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--early_stop',type=int, default=15, help='Stop when the model training changes littles')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
parser.add_argument('--max_iter', type=int, default=40000, help='Maximum iteration steps for training')
parser.add_argument('--eval_step', type=int, default=197, help='Per steps to perform validation')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')


args = parser.parse_args()

def cause_par(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue

    print(f"Total params: {Total_params / 1000000}")
    print(f"Trainable params: {Trainable_params}")
    print(f"Non-trainable params: {NonTrainable_params}")


set_determinism(seed=0)
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_samples, valid_samples, out_classes = data_loader(args)

train_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_samples['images'], train_samples['labels'])
]

val_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(valid_samples['images'], valid_samples['labels'])
]


train_transforms, val_transforms = data_transforms(args)

## Train Pytorch Data Loader and Caching

print('Start caching datasets!')
train_ds = CacheDataset(
    data=train_files, transform=train_transforms,
    cache_rate=args.cache_rate, num_workers=args.num_workers)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          pin_memory=True)

## Valid Pytorch Data Loader and Caching
val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)

val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)


if args.network == 'MDMUNet':
    model = MDMUNet(args.class_num).to(device)
    print('The network load successfully: MDMUNet')

elif args.network == 'UNetRes':
    model = UNetRes(class_num=args.class_num).to(device)
    print('The network load successfully: UNetRes')

elif args.network == 'UNetMDMSE':
    model = UNetMDMSE(class_num=args.class_num).to(device)
    print('The network load successfully: UNetMDMSE')

elif args.network == 'UNetMDMSECA':
    model = UNetMDMSECA(class_num=args.class_num).to(device)
    print('The network load successfully: UNetMDMSECA')

cause_par(model)




if args.pretrain == 'True':
    print('Pretrained weight is found! Start to load weight from: {}'.format(args.pretrained_weights))
    model.load_state_dict(torch.load(args.pretrained_weights))

## Define Loss function and optimizer

# loss_function = DiceCELoss(to_onehot_y=True,softmax=True,lambda_dice=1.2,lambda_ce=0.8)
loss_function = DiceCELoss(to_onehot_y=True,softmax=True)

print('Loss for training: {}'.format('DiceCELoss'))

if args.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
print('Optimizer for training: {}'.format(args.optim))

root_dir = os.path.join(args.save_path)
if os.path.exists(root_dir) == False:
    os.makedirs(root_dir)

t_dir = os.path.join(root_dir, 'tensorboard')
if os.path.exists(t_dir) == False:
    os.makedirs(t_dir)
writer = SummaryWriter(log_dir=t_dir)

def validation(epoch_iterator_val):
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 2, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]

            val_outputs_list = decollate_batch(val_outputs)

            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]

            dice_metric(y_pred=val_output_convert, y=val_labels_convert)

            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (step, len(epoch_iterator_val), dice)
            )
        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)
    writer.add_scalar('ValDice',mean_dice_val,current_step)
    return mean_dice_val

def train(current_step,dice_val_best,train_loader,val_loader,best_step,epoch_num):

    model.train()
    epoch_loss = []
    step = 0
    epoch_tqdm = tqdm(
        train_loader, desc="Training (X / X step) (loss=X.X) (lr:X.X)", dynamic_ncols=True
    )

    for step,batch in enumerate(epoch_tqdm):
        step += 1
        image,label = (batch['image'].to(device),batch['label'].to(device))

        result = model(image)
        loss = loss_function(result, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss.append(loss.item())
        epoch_tqdm.set_description(
            "Training (%d / %d step)  (loss=%2.5f) (lr:%2.7f)" % (current_step, max_iterations, loss, args.lr)
        )
        ave_epoch_loss = np.mean(epoch_loss)
        train_epoch_loss_values.append(ave_epoch_loss)

        if (current_step % eval_num == 0 and current_step != 0) or (current_step == max_iterations):
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)(class1:X class2:X class3:X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            val_metric_values.append(dice_val)

            if dice_val > dice_val_best:
                dice_val_best = dice_val
                best_step = current_step
                torch.save(
                    model.state_dict(), os.path.join(args.save_path, "best_dice_model.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val)
                )

        writer.add_scalar('Loss/train', loss.data,current_step)
        current_step += 1
    writer.add_scalar('Epoch/Losstrain', np.mean(train_epoch_loss_values), epoch_num)
    epoch_num += 1
    return current_step,dice_val_best,best_step,epoch_num

max_iterations = args.max_iter
print('Maximum Iterations for training: {}'.format(str(args.max_iter)))

post_label = AsDiscrete(to_onehot=args.class_num)
post_pred = AsDiscrete(argmax=True, to_onehot=args.class_num)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

epoch_num = 0
eval_num = args.eval_step
dice_val_best_earl = 0.0
dice_val_best = 0.0
earl_stop_step = 0
current_step = 1
best_step = 0
train_epoch_loss_values = []
val_metric_values = []
from datetime import datetime
# get current date
now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
print("time:", formatted_now)

while current_step < max_iterations:
    current_step,dice_val_best,best_step,epoch_num = train(
        current_step,dice_val_best,train_loader,val_loader,best_step,epoch_num
    )
    if dice_val_best_earl < dice_val_best:
        dice_val_best_earl = dice_val_best
        earl_stop_step = 0
    else:
        earl_stop_step += 1

    if earl_stop_step > eval_num:
        print('early stop！')
        break