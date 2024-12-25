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
import matplotlib.pyplot as plt
import time
import os
import logging
from tqdm import tqdm
from dataloader import dataload
from datetime import datetime

# class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
#     """
#     Convert labels to multi channels based on brats classes:
#     label 1 is the peritumoral edema
#     label 2 is the GD-enhancing tumor
#     label 3 is the necrotic and non-enhancing tumor core
#     The possible classes are TC (Tumor core), WT (Whole tumor)
#     and ET (Enhancing tumor).

#     """

#     def __call__(self, data):
#         d = dict(data)
#         for key in self.keys:
#             result = []
#             # merge label 2 and label 3 to construct TC
#             result.append(torch.logical_or(d[key] == 2, d[key] == 3))
#             # merge labels 1, 2 and 3 to construct WT
#             result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
#             # label 2 is ET
#             result.append(d[key] == 2)
#             d[key] = torch.stack(result, axis=0).float()
#         return d

VAL_AMP = False
change = datetime.now().strftime("%m%d%H%M")
print(change)

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

train_loader, val_loader = dataload(tr_batchsize=8)

# 定义网络模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UNet(dimensions=3, in_channels=1, out_channels=1).to(device)
model = CascadeUNet(num_out_classes=3, input_channels=4, init_coarse_channels=4, init_fine_channels=16, testing=True).to(device)
# model.load_state_dict(torch.load("./models/200epoch.pth"))

# 定义损失函数、优化器和评价指标
loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# 训练循环
max_epochs = 200
val_interval = 1
best_metric = -1
loss_list = []
metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []
best_metrics_epochs_and_time = [[], [], []]

logger.info(f"this train is about single stage at 1st")

total_start = time.time()
for epoch in range(max_epochs):
    epoch_start = time.time()
    logger.info("-" * 10)
    logger.info(f"Epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    with tqdm(train_loader, desc="Training", unit="batch") as tqdm_lodaer:
        for batch_data in tqdm_lodaer:
            step_start = time.time()
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            # print('inputs, labels:', inputs.shape, labels.shape)
            # coarse_outputs, fine_outputs = model(inputs)
            
            fine_outputs = model(inputs)

            labels = labels.sum(dim=1, keepdim=True)
            
            loss = loss_function(fine_outputs, labels)
            loss.backward()    
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            #
            tqdm_lodaer.set_postfix()
        
    # loss 
    loss_list.append(epoch_loss / step)
    
    # val
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                # coarse_outputs, fine_outputs = inference(val_inputs)
                fine_outputs = inference(val_inputs)
                # val_outputs = [post_trans(i) for i in decollate_batch(fine_outputs)]
                val_labels = labels.sum(dim=1, keepdim=True)
                
                dice_metric(y_pred=fine_outputs, y=val_labels)
                # dice_metric_batch(y_pred=fine_outputs, y=val_labels)
            
            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            # metric_batch = dice_metric_batch.aggregate()
            # metric_tc = metric_batch[0].item()
            # metric_values_tc.append(metric_tc)
            # metric_wt = metric_batch[1].item()
            # metric_values_wt.append(metric_wt)
            # metric_et = metric_batch[2].item()
            # metric_values_et.append(metric_et)
            
            dice_metric.reset()
            # dice_metric_batch.reset()

            # save model if it is best metric
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(
                    model.state_dict(),
                    os.path.join(model_dir, "best" + change + ".pth"),
                )
                logger.info("saved new best metric model")
            
            if epoch+1 == max_epochs:
                # save the end of this train
                torch.save(
                            model.state_dict(),
                            os.path.join(model_dir, "end" + change + ".pth"),
                        )
            
            logger.info(f"Loss: {epoch_loss / step}")
            logger.info(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
            # logger.info(
            #     f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
            #     f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
            #     f"\nbest mean dice: {best_metric:.4f}"
            #     f" at epoch: {best_metric_epoch}"
            # )
            
    logger.info(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    
total_time = time.time() - total_start
logger.info(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")


def drawGraphs(loss_list, metric_values):
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
    
    plt.savefig('./graphs/loss_metric' + change + '.png')

drawGraphs(loss_list, metric_values)





