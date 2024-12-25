import torch
from torch.utils.data import random_split
from monai.transforms import Compose, ScaleIntensity, RandCropByPosNegLabel, RandRotate, RandFlip, ToTensor
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, CacheDataset, decollate_batch, Dataset
from monai.apps import DecathlonDataset
from monai.handlers.utils import from_engine
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from models.cascadeNet import CascadeUNet
# import os
from glob import glob  # 注意这是两个glob
from tqdm import tqdm
from torchsummary import summary
from datetime import datetime

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d

def dataload(tr_batchsize=1):
    # 定义数据预处理的转换
    train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            # 指定旋转方向 右前上
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=["image", "label"], roi_size=[160, 160, 128], random_size=False),
            # 沿轴翻转 prop：概率
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # 随机缩放 factors: 强度 prob：概率
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            # 随机偏移 offsets: 偏移量 取值[-0.1, 0.1]
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )

    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=["image", "label"], roi_size=[160, 160, 128], random_size=False),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    test_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=["image", "label"], roi_size=[160, 160, 128], random_size=False),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    trainimgs = sorted(glob('/home/kemove/lt/datasets/brats21/train/images/*.gz'))
    trainlabels = sorted(glob('/home/kemove/lt/datasets/brats21/train/labels/*.gz'))

    testimgs = sorted(glob('/home/kemove/lt/datasets/brats21/test/images/*.gz'))
    testlabels = sorted(glob('/home/kemove/lt/datasets/brats21/test/labels/*.gz'))

    train_data_dict = [{'image': image, 'label': label} for image, label in zip(trainimgs, trainlabels)]
    test_data_dict = [{'image': image, 'label': label} for image, label in zip(testimgs, testlabels)]

    train_ds = Dataset(data=train_data_dict, transform=train_transform)
    test_ds = Dataset(data=test_data_dict, transform=test_transform)
    # print(len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=tr_batchsize, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=12)
    
    return train_loader, test_loader



