import torch
import os
import numpy as np
import glob
from monai.transforms import (
    AsDiscreted,
    AddChanneld,
    Compose,
    CropForegroundd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    KeepLargestConnectedComponentd,
    Spacingd,
    ToTensord,
    RandAffined,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandRotate90d,
    EnsureTyped,
    Invertd,
    KeepLargestConnectedComponentd,
    SaveImaged,
    Activationsd,
    Resize,
    MapTransform
)

def ensure_z_dim_at_least_96(data_dict, dim_to_adjust=-1, desired_dim_size=96,mode='trilinear'):
    """
    确保数据字典中所有keys对应的数据在指定维度的大小至少为desired_dim_size。
    如果不足，则在该维度进行填充。
    """
    for key, data in data_dict.items():
        if isinstance(data, (torch.Tensor, np.ndarray)) and data.shape[dim_to_adjust] < desired_dim_size:
            # print(data.shape)
            # 获取当前的H和W尺寸
            h, w = data.shape[-3:-1] if dim_to_adjust == -1 else (data.shape[-2], data.shape[-3])
            # 使用Resize变换，确保H和W尺寸不变，仅增加Z轴切片
            resize_transform = Resize(spatial_size=(h, w, desired_dim_size), mode=mode, align_corners=True)
            data_dict[key] = resize_transform(data)
            # print(data.shape)
    return data_dict

def ensure_dims_at_least(data_dict, desired_hw_size=512, desired_d_size=96, mode='trilinear'):
    """
    确保数据字典中所有keys对应的数据在高度、宽度以及深度维度的大小至少为相应的desired尺寸。
    如果不足，则在相应的维度上进行调整。
    """

    for key, data in data_dict.items():
        if isinstance(data, (torch.Tensor, np.ndarray)):
            # 获取当前的H、W和D尺寸
            h, w, d = data.shape[-3:] if len(data.shape) >= 3 else (data.shape[-2], data.shape[-1], 1)

            # 检查H、W和D是否小于期望的大小
            new_h = max(h, desired_hw_size)
            new_w = max(w, desired_hw_size)
            new_d = max(d, desired_d_size)

            # 使用Resize变换，确保H、W和D尺寸至少为desired尺寸
            resize_transform = Resize(spatial_size=(new_h, new_w, new_d), mode=mode,
                                      align_corners=(mode == 'trilinear'))

            # 应用变换
            data_dict[key] = resize_transform(data)

    return data_dict


class PrintShape(MapTransform):
    def __call__(self, data):
        d = dict(data)
        print(f"Image Shape: {d['image'].shape}")
        print(f"Label Shape: {d['label'].shape}")
        return d
def data_transforms(args):
    dataset = args.dataset
    if args.mode == 'train':
        crop_samples = args.crop_sample
    else:
        crop_samples = None

    if dataset == 'PancreasTumour':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    0.9, 0.9, 2.5), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-150, a_max=250,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                # 在CropForegroundd之前插入确保Z轴尺寸的逻辑
                lambda data: ensure_z_dim_at_least_96(data, dim_to_adjust=-1, desired_dim_size=96,mode='trilinear'),
                CropForegroundd(keys=["image", "label"],
                                source_key="image", ),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    0.9, 0.9, 2.5), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),

                ScaleIntensityRanged(
                    keys=["image"], a_min=-150, a_max=250,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(
                    0.9, 0.9, 2.5), mode=("bilinear")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-150, a_max=250,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )
    elif dataset == 'nih':
        train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(
                0.93, 0.93, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-145, a_max=275,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            lambda data: ensure_z_dim_at_least_96(data, dim_to_adjust=-1, desired_dim_size=96,mode='trilinear'),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=crop_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=1.0, spatial_size=(96, 96, 96),
                rotate_range=(0, 0, np.pi / 30),
                scale_range=(0.1, 0.1, 0.1)),
            ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    0.93, 0.93, 1.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-145, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
                ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(
                    0.93, 0.93, 1.0), mode=("bilinear")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-145, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )

    if args.mode == 'train':
        print('Cropping {} sub-volumes for training!'.format(str(crop_samples)))
        print('Performed Data Augmentations for all samples!')
        return train_transforms, val_transforms

    elif args.mode == 'test':
        print('Performed transformations for all samples!')
        return test_transforms

def data_loader(args):
    root_dir = args.root_path
    dataset = args.dataset
    out_classes = args.class_num

    print('Start to load data from directory: {}'.format(root_dir))

    if args.mode == 'train':
        train_samples = {}
        valid_samples = {}


        ## Input training data
        train_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTr', '*.nii.gz')))
        train_label = sorted(glob.glob(os.path.join(root_dir, 'labelsTr', '*.nii.gz')))
        train_samples['images'] = train_img
        train_samples['labels'] = train_label

        ## Input validation data
        valid_img = sorted(glob.glob(os.path.join(root_dir, 'imagesVal', '*.nii.gz')))
        valid_label = sorted(glob.glob(os.path.join(root_dir, 'labelsVal', '*.nii.gz')))
        valid_samples['images'] = valid_img
        valid_samples['labels'] = valid_label


        print('Finished loading all training samples from dataset: {}!'.format(dataset))
        print('Number of classes for segmentation: {}'.format(out_classes))

        return train_samples, valid_samples, out_classes

    elif args.mode == 'test':
        test_samples = {}

        ## Input inference data
        test_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTs', '*.nii.gz')))
        test_samples['images'] = test_img

        print('Finished loading all inference samples from dataset: {}!'.format(dataset))

        return test_samples, out_classes

def infer_post_transforms(args, test_transforms, out_classes):

    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=test_transforms,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
            # then invert `pred` based on this information. we can use same info
            # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
            # for example, may need the `affine` to invert `Spacingd` transform,
            # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
            # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
            # otherwise, no need this arg during inverting
            nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
            # to ensure a smooth output, then execute `AsDiscreted` transform
            to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
        ## If monai version <= 0.6.0:
        AsDiscreted(keys="pred", argmax=True, n_classes=out_classes),
        ## If moani version > 0.6.0:
        # AsDiscreted(keys="pred", argmax=True)
        # KeepLargestConnectedComponentd(keys='pred', applied_labels=[1, 3]),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=args.output,
                   output_postfix="seg", output_ext=".nii.gz", resample=True),
    ])

    return post_transforms