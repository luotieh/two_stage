import os
import shutil
import random

# 定义原始数据集目录和目标数据集目录
original_dataset_dir = "/home/kemove/lt/datasets/brats21/BraTS2021"
train_dir = "/home/kemove/lt/datasets/brats21/train"
test_dir = "/home/kemove/lt/datasets/brats21/test"

# 创建目标数据集目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 获取原始数据集中所有文件夹的名称
folders = os.listdir(original_dataset_dir)

# 打乱文件夹顺序
random.shuffle(folders)

# 计算划分比例
train_size = int(0.8 * len(folders))
test_size = len(folders) - train_size

# 将文件夹按照比例移动到对应目录中
for i, folder in enumerate(folders):
    if i < train_size:
        destination_dir = train_dir
    else:
        destination_dir = test_dir
    # 构造源文件夹和目标文件夹路径
    source_path = os.path.join(original_dataset_dir, folder)
    destination_path = os.path.join(destination_dir, folder)
    # 移动文件夹
    shutil.move(source_path, destination_path)
    print(f"Moved {folder} to {destination_dir}")

print("Data splitting completed.")
