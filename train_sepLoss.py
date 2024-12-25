import torch
from monai.transforms import Compose, ScaleIntensity, RandCropByPosNegLabel, RandRotate, RandFlip, ToTensor
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, DiceHelper
from monai.data import DataLoader, CacheDataset, list_data_collate, decollate_batch
from monai.apps import DecathlonDataset
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
from models.UNet3D import UNet3D
from models.fineNet import FineNet
import matplotlib.pyplot as plt
import time
import os
import logging
from tqdm import tqdm
from dataloader import dataload
from datetime import datetime
from crop_util import cut_dense_region_centered, restore_tensor

VAL_AMP = False
change = datetime.now().strftime("%m%d%H%M")
print(change)

# define inference method
def inference(input, model):
    return sliding_window_inference(
        inputs=input,
        roi_size=(160, 160, 128),
        sw_batch_size=1,
        predictor=model,
        overlap=0.5,
    )


# about log
logger = logging.getLogger()
handler = logging.FileHandler('./logs/' + change + '.txt')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# # 数据集路径
root_dir = '/home/kemove/lt/2stage/datasets'
model_dir = '/home/kemove/lt/2stage/weights'

train_loader, val_loader = dataload(tr_batchsize=4)

# 定义网络模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.load_state_dict(torch.load("./models/200epoch.pth"))

# separate two stage
model_coarse = UNet3D(num_out_classes=4, input_channels=4, init_feat_channels=4).to(device)
model_fine = FineNet().to(device)

# 定义损失函数、优化器和评价指标
loss_function = DiceLoss(sigmoid=True)
optimizer_coarse = torch.optim.Adam(model_coarse.parameters(), 1e-3)
optimizer_fine = torch.optim.Adam(model_fine.parameters(), 1e-3)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# 训练循环
max_epochs = 200
val_interval = 1
best_metric = -1

coarse_loss_list = []
fine_loss_list = []

metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []
best_metrics_epochs_and_time = [[], [], []]

logger.info(f"this train is about separate Loss cal")

total_start = time.time()
for epoch in range(max_epochs):
    epoch_start = time.time()
    logger.info("-" * 10)
    logger.info(f"Epoch {epoch + 1}/{max_epochs}")
    model_coarse.train()
    model_fine.train()
    
    coarse_losses = 0
    fine_losses = 0
    
    step = 0
            
    with tqdm(train_loader, desc="Training", unit="batch") as tqdm_lodaer:
        for batch_data in tqdm_lodaer:
            step_start = time.time()
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            # coarse train stage
            coarse_outputs = model_coarse(inputs)
            coarse_outputs = coarse_outputs.sum(dim=1, keepdim=True)
            labels_merge = labels.sum(dim=1, keepdim=True)
            
            loss_coarse = loss_function(coarse_outputs, labels_merge)
            
            optimizer_coarse.zero_grad()
            loss_coarse.backward(retain_graph=True)  # retain   
            optimizer_coarse.step()
            
            coarse_losses += loss_coarse.item()
            
            # fine train stage
            cropped_image, cropped_wt, padding_list = cut_dense_region_centered(coarse_outputs, inputs, 128)
            combined_input = torch.cat((cropped_image, cropped_wt), dim=1)
            
            fine_outputs = model_fine(combined_input)
            
            restore_output = restore_tensor(fine_outputs, padding_list)
            
            loss_fine = loss_function(restore_output, labels)
            
            optimizer_fine.zero_grad()
            loss_fine.backward()    
            optimizer_fine.step()
            
            fine_losses += loss_fine.item()
            
            step += 1
            #
            tqdm_lodaer.set_postfix()
        
    # loss 
    coarse_loss_list.append(coarse_losses / step)
    fine_loss_list.append(fine_losses / step)
    
    # val
    if (epoch + 1) % val_interval == 0:
        model_coarse.eval()
        model_fine.eval()
        
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                
                coarse_outputs = inference(val_inputs, model=model_coarse)
                coarse_outputs = coarse_outputs.sum(dim=1, keepdim=True)
                cropped_image, cropped_wt, padding_list = cut_dense_region_centered(coarse_outputs, inputs, 128)
                combined_input = torch.cat((cropped_image, cropped_wt), dim=1)
                
                fine_outputs = inference(combined_input, model=model_fine)
                restore_output = restore_tensor(fine_outputs, padding_list)
            
                # val_outputs = [post_trans(i) for i in decollate_batch(fine_outputs)]
                
                dice_metric(y_pred=restore_output, y=val_labels)
                dice_metric_batch(y_pred=restore_output, y=val_labels)
            
            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            metric_values_tc.append(metric_tc)
            metric_wt = metric_batch[1].item()
            metric_values_wt.append(metric_wt)
            metric_et = metric_batch[2].item()
            metric_values_et.append(metric_et)
            dice_metric.reset()
            dice_metric_batch.reset()

            # save model if it is best metric
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(
                    model_coarse.state_dict(),
                    os.path.join(model_dir, "best" + 'coarse' + change + ".pth"),
                )
                torch.save(
                    model_fine.state_dict(),
                    os.path.join(model_dir, "best" + 'fine' + change + ".pth"),
                )
                logger.info("saved new best metric model")
            
            if epoch+1 == max_epochs:
                # save the end of this train
                torch.save(
                            model_coarse.state_dict(),
                            os.path.join(model_dir, "end" + 'coarse' + change + ".pth"),
                        )
                torch.save(
                            model_fine.state_dict(),
                            os.path.join(model_dir, "end" + 'fine' + change + ".pth"),
                        )
            
            logger.info(f"coarse_Loss: {coarse_losses / step}")
            logger.info(f"fine_Loss: {fine_losses / step}")
            logger.info(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
            
    logger.info(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    
total_time = time.time() - total_start
logger.info(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")


def drawGraphs(loss_list, metric_values, name):
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(loss_list))]
    y = loss_list
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")
    
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [i + 1 for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="green")
    
    plt.savefig('./graphs/loss_metric' + name + change + '.png')

drawGraphs(coarse_loss_list, metric_values, name='coarse')
drawGraphs(fine_loss_list, metric_values, name='fine')






