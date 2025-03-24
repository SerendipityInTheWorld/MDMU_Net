from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from preconditioning import infer_post_transforms
from monai.utils import set_determinism

import time
from network.MDMUNet import *
from AblationExperiment.UNetRes import MMNet1
from AblationExperiment.UNetMDMSE import MMNetBAA
from AblationExperiment.UNetMDMSECA import MMNetBBACA

import torch
# from load_datasets_transforms import data_loader, data_transforms, infer_post_transforms
from preconditioning import data_loader,data_transforms

import os
import argparse
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
parser = argparse.ArgumentParser(description='3D UX-Net inference hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root_path', type=str, default=r'F:\lulian_graduate\pycode\3DUX-Net\3DUX-Net\Pancreatic2024',  help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default=r'result\MDMU_Net/seg', help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='PancreasTumour', help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')
parser.add_argument('--class_num', type=int, default=3, help='Number of label categories')
## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='MDMUNet', help='Network models: {}')
parser.add_argument('--trained_weights', default=r'', help='Path of pretrained/fine-tuned weights')
parser.add_argument('--mode', type=str, default='test', help='Training or testing mode')
parser.add_argument('--sw_batch_size', type=int, default=1, help='Sliding window batch size for inference')
parser.add_argument('--overlap', type=float, default=0.5, help='Sub-volume overlapped percentage')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

test_samples, out_classes = data_loader(args)

test_files = [
    {"image": image_name} for image_name in zip(test_samples['images'])
]

set_determinism(seed=0)

test_transforms = data_transforms(args)
post_transforms = infer_post_transforms(args, test_transforms, out_classes)

## Inference Pytorch Data Loader and Caching
test_ds = CacheDataset(
    data=test_files, transform=test_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers)

## Load Networks
device = torch.device("cuda:0")

if args.network == 'MMNet':
    model = MMNet(class_num=args.class_num).to(device)
    print('The network load successfully: MMN-Net')
elif args.network == 'MMNetAE1':
    model = MMNet1(class_num=args.class_num).to(device)
    print('The network load successfully: MMNetAE1')
elif args.network == 'MMNetBAA':
    model = MMNetBAA(class_num=args.class_num).to(device)
    print('The network load successfully: MMNetBAA')
elif args.network == 'MMNetBAACA':
    model = MMNetBBACA(class_num=args.class_num).to(device)
    print('The network load successfully: MMNetBBACA')


inference_time_list = []

model.load_state_dict(torch.load(args.trained_weights))
model.eval()

with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        images = test_data["image"].to(device)
        roi_size = (96, 96, 96)
        star_time = time.time()
        test_data['pred'] = sliding_window_inference(
            images, roi_size, args.sw_batch_size, model, overlap=args.overlap
        )
        end_time = time.time()
        print(f'same{i}----{end_time - star_time}')
        inference_time_list.append(end_time - star_time)
        test_data = [post_transforms(i) for i in decollate_batch(test_data)]

print('Mean inference time :',np.mean(inference_time_list))