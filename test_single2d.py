import torch
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
from dataloader import dataload
import matplotlib.pyplot as plt
import os
import numpy as np
from models.cascade_both import CascadeBoth
from models.fuse2D_ import Cascade2D


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

VAL_AMP = False

# define inference method
def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(160, 160, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

# save image
def save_3d_slices(image, label, prediction, sample_idx, output_dir):
    num_slices = image.shape[2]  # 假设图像形状为 (slices, height, width)
    for slice_idx in range(0, num_slices):
        if np.any(label[:, :, slice_idx] > 0):  # 仅在标签中有类别时保存
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(image[:, :, slice_idx], cmap='gray')
            ax[0].set_title('Input Image')
            ax[1].imshow(label[:, :, slice_idx], cmap='gray')
            ax[1].set_title('Ground Truth')
            ax[2].imshow(prediction[:, :, slice_idx], cmap='gray')
            ax[2].set_title('Prediction')

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_path = os.path.join(output_dir, f'sample_{sample_idx}_slice_{slice_idx}.jpg')
            plt.savefig(output_path)
            plt.close(fig)

# 数据集路径
# root_dir = '/home/kemove/lt/efficientUnet/brats21/test'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 没gpu就用cpu
print(device)

# Tensors for 3D Image Processing in PyTorch
# Batch x Channel x Z x Y x X
# Batch size BY x Number of channels x (BY Z dim) x (BY Y dim) x (BY X dim)
# imglist = sorted(glob('/home/kemove/lt/efficientUnet/brats21/test/image/*.gz'))
# labellist = sorted(glob('/home/kemove/lt/efficientUnet/brats21/test/label/*.gz'))

# data_dict = [{'image': image, 'label': label} for image, label in zip(imglist, labellist)]

# test_ds = Dataset(data=data_dict, transform=test_transform)

# test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=24)

_, test_loader = dataload()
    
# model = UNet3D(num_out_classes=4, input_channels=4, init_feat_channels=8)
model = Cascade2D(num_out_classes=3, input_channels=4, init_coarse_channels=4, init_fine_channels=16, testing=True)
model.to(device)

# summary(model, input_size=(4, 160, 160, 128), device='cuda')

model.load_state_dict(torch.load("/home/kemove/lt/2stage/weights/end07081329.pth"))
model.eval()

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

metric_values = []
outputImage = './visualize'

with torch.no_grad():
    with tqdm(test_loader, desc="Testing", unit="batch") as tqdm_lodaer:
        for sample_idx, test_data in enumerate(tqdm_lodaer):
            
            inputs, labels = test_data["image"].to(device), test_data["label"].to(device)
            
            # coarse_outputs, fine_outputs = inference(inputs)
            # outputs = [post_trans(i) for i in decollate_batch(fine_outputs)]
            
            # coarse_outputs, fine_outputs = model(inputs)
            fine_outputs = model(inputs)

            output = torch.where(fine_outputs > 0.5, 1, 0)
            
            dice_metric(y_pred=output, y=labels)
            dice_metric_batch(y_pred=output, y=labels)
            
            # if sample_idx == 10:
            #     # 可视化并保存部分
            #     for i in range(inputs.shape[0]):  # 对于每个样本
            #         input_image = inputs[i, 0, :, :, :].cpu().numpy()  # 假设输入的第0通道是T1加权图像
            #         label = labels[i, :, :, :, :].cpu().numpy()
            #         prediction = output[i, :, :, :, :].cpu().numpy()
                    # save_3d_slices(input_image, np.max(label, axis=0), np.max(prediction, axis=0), sample_idx * inputs.shape[0] + i, outputImage)
            
            # print(torch.max(output[0]).item())
            # print(torch.median(output[0]).item())
            # print(torch.min(output[0]).item())
            # print(torch.max(labels).item())
            # print(torch.min(labels).item())

            tqdm_lodaer.set_postfix()

        # metric
        metric = dice_metric.aggregate().item()
        metric_values.append(metric)
        metric_batch = dice_metric_batch.aggregate()
        metric_tc = metric_batch[0].item()
        metric_wt = metric_batch[1].item()
        metric_et = metric_batch[2].item()

        dice_metric.reset()
        dice_metric_batch.reset()

print(
    f"current mean dice: {metric:.4f}"
    f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
)

