import os
import numpy as np
import pandas as pd
import cv2

import torch
from torch import Tensor

from torch.utils.data import Dataset
import random
from albumentations.augmentations.transforms import Lambda, RandomBrightnessContrast
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate, HorizontalFlip
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.crops.transforms import RandomResizedCrop
from albumentations import Compose
import warnings
import torchvision.transforms as transforms

warnings.filterwarnings("ignore")

seed = 1#seed必须是int，可以自行设置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
torch.cuda.manual_seed_all(seed)#多卡模式下，让所有显卡生成的随机数一致？这个待验证
np.random.seed(seed)#numpy产生的随机数一致
random.seed(seed)

# CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
# 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
torch.backends.cudnn.deterministic = True

# 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
# 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
torch.backends.cudnn.benchmark = False


norm_mean = [0.143]  # 0.458971
norm_std = [0.144]  # 0.225609

RandomErasing = transforms.RandomErasing(scale=(0.02, 0.08), ratio=(0.5, 2), p=0.8)


def randomErase(image, **kwargs):
    return RandomErasing(image)


def sample_normalize(image, **kwargs):
    image = image / 255
    channel = image.shape[2]
    mean, std = image.reshape((-1, channel)).mean(axis=0), image.reshape((-1, channel)).std(axis=0)
    return (image - mean) / (std + 1e-3)


transform_train = Compose([
    # RandomBrightnessContrast(p = 0.8),
    RandomResizedCrop(512, 512, (0.5, 1.0), p=0.5),
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=0.0,
                     p=0.8),
    # HorizontalFlip(p = 0.5),

    # ShiftScaleRotate(shift_limit = 0.2, scale_limit = 0.2, rotate_limit=20, p = 0.8),
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.8, contrast_limit=(-0.3, 0.2)),
    Lambda(image=sample_normalize),
    ToTensorV2(),
    Lambda(image=randomErase)
])


transform_val = Compose([
    Lambda(image=sample_normalize),
    ToTensorV2(),
])


def read_grad(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.reshape((8, 512, 512)).transpose(1, 2, 0)


class BAATrainDataset(Dataset):
    def __init__(self, df, file_path):
        def preprocess_df(df):
            # nomalize boneage distribution
            # df['zscore'] = df['boneage'].map(lambda x: (x - boneage_mean) / boneage_div)
            # change the type of gender, change bool variable to float32
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.df = preprocess_df(df)
        self.file_path = file_path

    def __getitem__(self, index):
        row = self.df.iloc[index]
        num = int(row['id'])
        # return (transform_train(image=read_grad(f"{self.file_path}/{num}.png"))['image'],
        return (transform_train(image=cv2.imread(f"{self.file_path}/{num}.png", cv2.IMREAD_COLOR))['image'],
                Tensor([row['male']])), row['boneage']

    def __len__(self):
        return len(self.df)


class BAAValDataset(Dataset):
    def __init__(self, df, file_path):
        def preprocess_df(df):
            # change the type of gender, change bool variable to float32
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.df = preprocess_df(df)
        self.file_path = file_path

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # return (transform_val(image=read_grad(f"{self.file_path}/{int(row['id'])}.png"))['image'],
        return (transform_train(image=cv2.imread(f"{self.file_path}/{int(row['id'])}.png", cv2.IMREAD_COLOR))['image'],
                Tensor([row['male']])), row['boneage']

    def __len__(self):
        return len(self.df)


def create_data_loader(train_df, val_df, train_root, val_root):
    return BAATrainDataset(train_df, train_root), BAAValDataset(val_df, val_root)


def train_fn(train_loader):
    for i in range(1, 5):
        for batch_idx, data in enumerate(train_loader):
            image, gender = data[0]
            image, gender = image.type(torch.FloatTensor).cuda(), gender.type(torch.FloatTensor).cuda()

            batch_size = len(data[1])
            label = (data[1]-1).type(torch.LongTensor).cuda()

    return 0.


def evaluate_fn(val_loader):
    with torch.no_grad():
        for i in range(1, 5):
            for batch_idx, data in enumerate(val_loader):

                image, gender = data[0]
                image, gender = image.type(torch.FloatTensor).cuda(), gender.type(torch.FloatTensor).cuda()

                label = data[1].cuda()

    return 0.


import time

def map_fn(flags, data_dir, k):
    fold_path = os.path.join(data_dir, f'fold_{k}')
    train_df = pd.read_csv(os.path.join(fold_path, 'train.csv'))
    val_df = pd.read_csv(os.path.join(fold_path, 'valid.csv'))

    train_set, val_set = create_data_loader(train_df, val_df, os.path.join(fold_path, 'train'), os.path.join(fold_path, 'valid'))
    print(train_set.__len__())
    # Creates dataloaders, which load data in batches
    # Note: test loader is not shuffled or sampled
    # train_loader = torch.utils.data.DataLoader(
    #     train_set,
    #     batch_size=flags['batch_size'],
    #     shuffle=True,
    #     num_workers=flags['num_workers'],
    #     drop_last=True,
    #     # pin_memory=True
    # )
    #
    # val_loader = torch.utils.data.DataLoader(
    #     val_set,
    #     batch_size=flags['batch_size'],
    #     shuffle=False,
    #     num_workers=flags['num_workers'],
    #     # pin_memory=True
    # )

    ## Trains
    for epoch in range(flags['num_epochs']):
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=flags['batch_size'],
            shuffle=True,
            num_workers=epoch+1,
            drop_last=True,
            # pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=flags['batch_size'],
            shuffle=False,
            num_workers=epoch+1,
            # pin_memory=True
        )
        start_time = time.time()
        train_fn(train_loader)

        evaluate_fn(val_loader)

        print(f'num_workers={epoch+1}, time : {(time.time() - start_time)/4}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()


    flags = {}
    flags['lr'] = 0.
    flags['batch_size'] = 32
    # flags['num_workers'] = 6
    flags['num_epochs'] = 16
    flags['seed'] = 1

    train_df = pd.read_csv(f'../../archive/boneage-training-dataset.csv')
    # train_ori_dir = '../../../autodl-tmp/grad_4K_fold/'
    # train_ori_dir = '../../archive/grad_1K_fold/'
    train_ori_dir = '../../archive/masked_1K_fold/'
    # only run one fold
    print(f'load Ori')
    map_fn(flags, data_dir=train_ori_dir, k=1)