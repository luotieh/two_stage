from .UNet3D import UNet3D
from .fineNet import FineNet
from .fuse2d import Fuse2d
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import gc

# 定义级联模型
class Cascade2D(nn.Module):
    def __init__(self, num_out_classes=2, input_channels=1, init_coarse_channels=8, init_fine_channels=8, testing=False):
        super(Cascade2D, self).__init__()
        # 定义第一个U-Net
        # self.coarse_unet = UNet3D(num_out_classes=input_channels, input_channels=input_channels, init_feat_channels=init_coarse_channels)
        
        # 2d in 1st stage
        self.coarse_unet = Fuse2d()
        
        # 定义第二个U-Net
        # self.fine_unet = UNet3D(num_out_classes=num_out_classes, input_channels=input_channels, init_feat_channels=init_fine_channels, testing=testing)
        
        # cascadeUnet
        # self.fine_unet = FineNet()
        # single
        # self.fine_unet = FineNet(embed_dims=[4, 8, 16, 32, 64],)
        
    def forward(self, inputs):
        # 第一个U-Net的前向传播
        B, C, H, W, D = inputs.shape
        reshaped_inputs = inputs.permute(0, 4, 1, 2, 3).reshape(B*D, C, H, W)
        # print(f'reshaped_inputs: {reshaped_inputs.shape}')
        wt_mask = self.coarse_unet(reshaped_inputs)
        # print(f'wt_mask before: {wt_mask.shape}')
        wt_mask = self.reshape_output(wt_mask, B, D)
        # print(f'wt_mask after: {wt_mask.shape}')
        # wt_mask = torch.where(wt_mask > 0.5, torch.tensor(1.0), torch.tensor(1e-3))
        # # self.show(wt_mask)
        # wt_mask = wt_mask.sum(dim=1, keepdim=True)
        # # print('x:', x.shape)
        # # print('wt_shape:', wt_mask.shape)
        # # print('wt_mask', wt_mask)
        # # print('wt_mask:', wt_mask.shape)
        # cropped_image, cropped_wt, padding_list = self.cut_dense_region_centered(wt_mask, inputs, 128)
        # # 第二个U-Net的前向传播，输入为第一个U-Net的输出
        # # self.show(cropped_image)
        # combined_input = torch.cat((cropped_image, cropped_wt), dim=1)
        # # print('cropped_image:', cropped_image.shape)
        # # print('cropped_wt:', cropped_wt.shape)
        # # print('combined_input:', combined_input.shape)
        # fine_output = self.fine_unet(combined_input)
        # # print('fine_output:', fine_output.shape)
        # restore_output = self.restore_tensor(fine_output, padding_list)
        
        return wt_mask
        # return wt_mask
    
    def reshape_output(self, output, batchsize, depth):
        bd, c, h, w = output.shape
        assert bd == batchsize * depth
        
        reshaped_output = output.view(batchsize, depth, c, h, w)
        
        reshaped_output = reshaped_output.permute(0, 2, 3, 4, 1)
        
        return reshaped_output
    
    # 计算非零元素的质心
    def find_centroid(self, tensor):
        non_zero_coords = torch.nonzero(tensor)
        centroid = non_zero_coords.float().mean(dim=0).long()
        return centroid

    # 定义裁切函数
    def cut_dense_region_centered(self, tensorA, tensorB, size):
        batch_size = tensorA.shape[0]
        cut_tensorA_list = []
        cut_tensorB_list = []
        padding_params_list = []

        for batch in range(batch_size):
            centroid = self.find_centroid(tensorA[batch, 0, :, :, :])
            center_i, center_j, center_k = centroid[0].item(), centroid[1].item(), centroid[2].item()
            
            start_i = max(0, center_i - size // 2)
            start_j = max(0, center_j - size // 2)
            start_k = max(0, center_k - size // 2)
            end_i = min(tensorA.shape[2], start_i + size)
            end_j = min(tensorA.shape[3], start_j + size)
            end_k = min(tensorA.shape[4], start_k + size)
            
            # 调整起点以确保裁切区域的大小
            if end_i - start_i < size:
                start_i = end_i - size
            if end_j - start_j < size:
                start_j = end_j - size
            if end_k - start_k < size:
                start_k = end_k - size

            cut_tensorA = tensorA[batch:batch+1, :, start_i:end_i, start_j:end_j, start_k:end_k]
            cut_tensorB = tensorB[batch:batch+1, :, start_i:end_i, start_j:end_j, start_k:end_k]

            padding_params = (
                max(0, tensorA.shape[4] - end_k), start_k,
                max(0, tensorA.shape[3] - end_j), start_j,
                max(0, tensorA.shape[2] - end_i), start_i,
            )

            cut_tensorA_list.append(cut_tensorA)
            cut_tensorB_list.append(cut_tensorB)
            padding_params_list.append(padding_params)

        cut_tensorA = torch.cat(cut_tensorA_list, dim=0)
        cut_tensorB = torch.cat(cut_tensorB_list, dim=0)

        return cut_tensorA, cut_tensorB, padding_params_list

    # 定义还原函数
    def restore_tensor(self, cut_tensor, padding_params_list):
        restored_tensor_list = []
        for batch in range(cut_tensor.shape[0]):
            restored_channels = []
            for channel in range(cut_tensor.shape[1]):
                padding_params = padding_params_list[batch]
                # 注意填充参数的顺序 (left, right, top, bottom, front, back)
                restored_channel = F.pad(cut_tensor[batch, channel], (padding_params[1], padding_params[0], padding_params[3], padding_params[2], padding_params[5], padding_params[4]))
                restored_channels.append(restored_channel.unsqueeze(0))
            restored_tensor = torch.cat(restored_channels, dim=0).unsqueeze(0)
            restored_tensor_list.append(restored_tensor)

        restored_tensor = torch.cat(restored_tensor_list, dim=0)
        return restored_tensor
    
    # # 定义裁切函数
    # def cut_dense_region(slef, tensorA, tensorB, size):
    #     def find_dense_region(tensor, size):
    #         max_sum = -1
    #         best_coords = (0, 0)
    #         for i in range(tensor.shape[0] - size + 1):
    #             for j in range(tensor.shape[1] - size + 1):
    #                 region = tensor[i:i+size, j:j+size]
    #                 region_sum = torch.sum(region != 0).item()
    #                 if region_sum > max_sum:
    #                     max_sum = region_sum
    #                     best_coords = (i, j)
    #         return best_coords

    #     batch_size = tensorA.shape[0]
    #     cut_tensorA_list = []
    #     cut_tensorB_list = []
    #     padding_params_list = []

    #     for batch in range(batch_size):
    #         start_i, start_j = find_dense_region(tensorA[batch, 0, :, :, :], size)
    #         end_i = start_i + size
    #         end_j = start_j + size

    #         cut_tensorA = tensorA[batch:batch+1, :, start_i:end_i, start_j:end_j, :]
    #         cut_tensorB = tensorB[batch:batch+1, :, start_i:end_i, start_j:end_j, :]

    #         padding_params = (0, 0, start_j, tensorA.shape[3] - end_j, start_i, tensorA.shape[2] - end_i)

    #         cut_tensorA_list.append(cut_tensorA)
    #         cut_tensorB_list.append(cut_tensorB)
    #         padding_params_list.append(padding_params)

    #     cut_tensorA = torch.cat(cut_tensorA_list, dim=0)
    #     cut_tensorB = torch.cat(cut_tensorB_list, dim=0)

    #     return cut_tensorA, cut_tensorB, padding_params_list

    # # 定义还原函数
    # def restore_tensor(self, cut_tensor, padding_params_list):
    #     restored_tensor_list = []
    #     for batch in range(cut_tensor.shape[0]):
    #         restored_channels = []
    #         for channel in range(cut_tensor.shape[1]):
    #             padding_params = padding_params_list[batch]
    #             restored_channel = F.pad(cut_tensor[batch, channel], padding_params)
    #             restored_channels.append(restored_channel.unsqueeze(0))
    #         restored_tensor = torch.cat(restored_channels, dim=0).unsqueeze(0)
    #         restored_tensor_list.append(restored_tensor)

    #     restored_tensor = torch.cat(restored_tensor_list, dim=0)
    #     return restored_tensor
    
    def show(self, data):
        # 选择一个batch，例如第一个batch
        data = data.cpu().numpy()
        batch_index = 0
        batch_data = data[batch_index]

        # 选择深度维度的中间切片
        # depth_index = batch_data.shape[3] // 2
        depth_index = 40

        # 从4个模态中分别取出这一切片
        modality_slices = [batch_data[i, :, :, depth_index] for i in range(batch_data.shape[0])]

        # 创建一个用于展示4个模态图像的图
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 显示每个模态的图像
        for i, slice in enumerate(modality_slices):
            axes[i].imshow(slice, cmap="gray")
            axes[i].set_title(f'Modality {i+1}')
            axes[i].axis('off')

        # 显示图像
        plt.suptitle("Middle Slices for Each Modality")
        plt.show()
        
        # plt.close(fig)
        # del fig, axes, modality_slices, batch_data, slice
        # gc.collect()