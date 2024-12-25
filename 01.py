from glob import glob  # 注意这是两个glob
import monai
from monai.data import Dataset

imglist = sorted(glob('/home/kemove/lt/efficientUnet/brats21/test/image/*.gz'))
labellist = sorted(glob('/home/kemove/lt/efficientUnet/brats21/test/label/*.gz'))

data_dict = [{'image': image, 'label': label} for image, label in zip(imglist, labellist)]


