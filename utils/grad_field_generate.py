import cv2
import torch
from PIL import Image
import numpy as np


def normalize(image):
    non_zero_indices = np.nonzero(image)
    min_value = np.min(image[non_zero_indices])

    # 将非零元素减去最小值
    image[non_zero_indices] = image[non_zero_indices] - min_value

    # 将最大值拉到255
    max_value = np.max(image[non_zero_indices])
    scale_factor = 255 / max_value
    image = image * scale_factor

    # 四舍五入并转换为整数类型
    image = np.round(image).astype(np.uint8)

    return image


def generate_grad_field(path, target_path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    blur_img = cv2.medianBlur(img, ksize=3)
    int_img = np.array(blur_img, dtype=int)

    differ_down = np.zeros_like(int_img)
    differ_down[:511] = np.absolute(int_img[1:] - int_img[:511])

    differ_up = np.zeros_like(int_img)
    differ_up[1:] = np.absolute(int_img[:511] - int_img[1:])

    differ_right = np.zeros_like(int_img)
    differ_right[:, :511] = np.absolute(int_img[:, 1:] - int_img[:, :511])

    differ_left = np.zeros_like(int_img)
    differ_left[:, 1:] = np.absolute(int_img[:, :511] - int_img[:, 1:])

    differ_up_right = np.zeros_like(int_img)
    differ_up_right[1:, :511] = np.absolute(int_img[:511, 1:] - int_img[1:, :511])

    differ_up_left = np.zeros_like(int_img)
    differ_up_left[1:, 1:] = np.absolute(int_img[:511, :511] - int_img[1:, 1:])

    differ_down_right = np.zeros_like(int_img)
    differ_down_right[:511, :511] = np.absolute(int_img[1:, 1:] - int_img[:511, :511])

    differ_down_left = np.zeros_like(int_img)
    differ_down_left[:511, 1:] = np.absolute(int_img[1:, :511] - int_img[:511, 1:])

    norm_up = normalize(differ_up)
    norm_down = normalize(differ_down)
    norm_left = normalize(differ_left)
    norm_right = normalize(differ_right)
    norm_up_left = normalize(differ_up_left)
    norm_up_right = normalize(differ_up_right)
    norm_down_left = normalize(differ_down_left)
    norm_down_right = normalize(differ_down_right)
    total = np.concatenate(
        (norm_up, norm_down, norm_left, norm_right, norm_up_left, norm_up_right, norm_down_left, norm_down_right))
    # norm_all =
    # Image.fromarray(total).save(target_path)
    cv2.imwrite(target_path, total)


import os
import shutil


def extract_files(source_folder, destination_folder):
    # 获取总文件夹中的所有文件名
    os.makedirs(destination_folder, exist_ok=True)
    total_imgs =  os.listdir(source_folder)
    # 遍历每个文件夹
    for filename in total_imgs:
        print(filename)
        source_img_path = os.path.join(source_folder, filename)
        destination_file_path = os.path.join(destination_folder, filename)
        # shutil.copy(ori_filename, destination_file_path)
        generate_grad_field(source_img_path, destination_file_path)
        print(f"复制文件: {filename} 到 {destination_file_path}")



# 用法示例
# source_folder = '../../../autodl-tmp/MaskAll/'  # 源文件夹路径
# total_folder = "../../archive/boneage-training-dataset"  # 总文件夹路径
# destination_folder = '../../../autodl-tmp/gradAll/'  # 目标文件夹路径
source_folder = '../../archive/masked_1K_fold/fold_1/train'  # 源文件夹路径
destination_folder = '../../archive/grad_1K'  # 目标文件夹路径


extract_files(source_folder, destination_folder)
